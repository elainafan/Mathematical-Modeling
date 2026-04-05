"""
flow/flow_simulation.py
基于流仿真的交通网络健壮性指标体系 —— igraph 加速版。

相比前一版本的主要优化
-----------------------
1. compute_node_density  : scipy KDTree sparse_distance_matrix，
                           截断半径 3σ，O(N²) → O(N·K)
2. compute_od_matrix     : KDTree query_pairs 找候选对（O(N log N + P)）
                           + 批量 G.distances(source=batch) 减少 Python→C 次数
                           + 全程 numpy 向量化，无 Python 内层循环
                           + 返回 (D_keys, D_vals, D_total) numpy 数组
3. compute_edge_loads    : output='epath' 直接拿边 id，避免 get_eid 调用
                           + numpy 数组替代 dict 做 id→igraph_idx 映射
4. compute_accessibility : 预建 comp_lookup[json_id] 整型数组
                           → 三行 numpy，仿真循环中每步 O(P) 纯 C 执行
5. compute_congestion_rate: get_edgelist + numpy 索引，避免逐边 Python 迭代
"""

import os
import sys
import time
import heapq
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig
import numpy as np
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_ig import load_graph, rebuild_id2idx

_LARGE_W: float = 1e12   # 封锁用伪无穷权值


# =============================================================================
# 1.  节点密度（KDTree sparse_distance_matrix）
# =============================================================================

def compute_node_density(
    positions: np.ndarray,
    sigma: float,
    trunc_factor: float = 3.0,
) -> np.ndarray:
    r"""
    节点空间密度（截断到 trunc_factor·σ）：

    \[m_i = \sum_{j \neq i,\, \|p_i-p_j\| \le \text{trunc}\cdot\sigma}
            \exp\!\left(-\frac{\|p_i-p_j\|^2}{2\sigma^2}\right)\]

    使用 scipy KDTree ``sparse_distance_matrix``，
    复杂度 O(N·K) 代替 O(N²)，K 为平均邻居数。

    参数
    ----
    positions    : (N, 2) UTM 坐标，单位米
    sigma        : 空间尺度超参数，单位米
    trunc_factor : 截断倍数（默认 3σ，误差 < exp(-4.5) ≈ 1%）

    返回
    ----
    m : (N,) float64，索引与 igraph 内部索引对齐
    """
    tree = cKDTree(positions)
    sdm  = tree.sparse_distance_matrix(
        tree, max_distance=trunc_factor * sigma, output_type='coo_matrix'
    )
    r, c, d = sdm.row, sdm.col, sdm.data
    valid   = r != c                                     # 去除自对（d=0）
    w       = np.exp(-d[valid] ** 2 / (2.0 * sigma ** 2))

    m = np.zeros(len(positions), dtype=np.float64)
    np.add.at(m, r[valid], w)
    return m


# =============================================================================
# 2.  OD 需求矩阵（KDTree + 批量 Dijkstra + 全程 numpy）
# =============================================================================

def compute_od_matrix(
    G_orig: ig.Graph,
    m: np.ndarray,
    beta: float,
    sigma: float,
    cutoff_factor: float = 3.0,
    min_demand: float = 1e-10,
    batch_size: int = 256,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    稀疏 OD 需求矩阵：
    \(D_{ij} = m_i m_j \exp(-\beta d_{ij}^{(0)})\)

    优化流程
    --------
    1. ``cKDTree.query_pairs`` 一次性找出所有欧氏距离 ≤ cutoff 的节点对
    2. 按 batch_size 批量调用 ``G.distances(source=batch)``，
       减少 Python→C 次数（N 次 → N/batch_size 次）
    3. 每批全程 numpy 向量化，无 Python 内层循环

    参数
    ----
    G_orig      : 原始图（节点属性 'node_id'、'pos'；边属性 'weight'）
    m           : (N,) 节点密度
    beta        : OD 距离衰减系数
    sigma       : 空间尺度，单位米
    cutoff_factor: 欧氏截断倍数（默认 3σ）
    min_demand  : 低于此值的 D_ij 不存储
    batch_size  : 每批 Dijkstra 的源节点数（默认 256）
    verbose     : 是否打印进度

    返回
    ----
    D_keys  : (P, 2) int64，每行 (min_json_id, max_json_id)
    D_vals  : (P,) float64，对应 D_ij
    D_total : float，sum(D_vals)
    """
    positions    = np.array([v['pos']     for v in G_orig.vs], dtype=np.float64)
    node_ids_arr = np.array(G_orig.vs['node_id'],               dtype=np.int64)

    # ── 1. KDTree 候选对（i < j）──────────────────────────────────────────
    tree  = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff_factor * sigma, output_type='ndarray')

    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.0

    # ── 2. 按源节点 numpy groupby ─────────────────────────────────────────
    src_arr    = pairs[:, 0]
    tgt_arr    = pairs[:, 1]
    sort_idx   = np.argsort(src_arr, kind='stable')
    srcs_sorted = src_arr[sort_idx]
    tgts_sorted = tgt_arr[sort_idx]
    unique_srcs, g_start, g_count = np.unique(
        srcs_sorted, return_index=True, return_counts=True
    )

    n_batches    = int(np.ceil(len(unique_srcs) / batch_size))
    D_keys_parts: List[np.ndarray] = []
    D_vals_parts: List[np.ndarray] = []
    t0 = time.perf_counter()

    # ── 3. 批量 Dijkstra + numpy 向量化 ──────────────────────────────────
    for b in range(n_batches):
        lo     = b * batch_size
        hi     = min(lo + batch_size, len(unique_srcs))
        b_srcs = unique_srcs[lo:hi].tolist()

        if verbose and b % 20 == 0:
            print(f"  [OD] 批次 {b}/{n_batches}  ({time.perf_counter()-t0:.1f}s)")

        # 一次 C 调用返回 (batch, N) 距离矩阵
        batch_np = np.array(
            G_orig.distances(source=b_srcs, weights='weight'),
            dtype=np.float64,
        )

        for li in range(hi - lo):
            g_idx  = lo + li
            src    = unique_srcs[g_idx]
            sl     = slice(g_start[g_idx], g_start[g_idx] + g_count[g_idx])
            t_arr  = tgts_sorted[sl]               # 候选目标 igraph 索引

            d_net  = batch_np[li, t_arr]           # (K,) 网络距离
            finite = d_net < 1e15
            if not finite.any():
                continue

            t_v   = t_arr[finite]
            d_v   = d_net[finite]
            d_ij  = m[src] * m[t_v] * np.exp(-beta * d_v)   # 全程 numpy

            keep  = d_ij >= min_demand
            if not keep.any():
                continue

            id_s  = node_ids_arr[src]
            id_t  = node_ids_arr[t_v[keep]]
            keys  = np.column_stack([
                np.minimum(id_s, id_t),
                np.maximum(id_s, id_t),
            ]).astype(np.int64)

            D_keys_parts.append(keys)
            D_vals_parts.append(d_ij[keep])

    if not D_keys_parts:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0), 0.0

    D_keys  = np.vstack(D_keys_parts)
    D_vals  = np.concatenate(D_vals_parts)
    D_total = float(D_vals.sum())

    if verbose:
        print(f"  [OD] 完成  P={len(D_keys)}  D_total={D_total:.4e}  "
              f"耗时={time.perf_counter()-t0:.1f}s")
    return D_keys, D_vals, D_total


# =============================================================================
# 3.  k 最短路（Yen 算法）
# =============================================================================

def _yen_spur(
    G_work: ig.Graph,
    spur_node: int,
    target: int,
    blocked_verts: Set[int],
    blocked_eids: Set[int],
    weight: str = 'weight',
) -> Tuple[Optional[List[int]], float]:
    """临时封锁顶点/边（修改权值为 _LARGE_W），求 spur 最短路后恢复。"""
    backup: Dict[int, float] = {}
    for v in blocked_verts:
        for eid in G_work.incident(v):
            if eid not in backup:
                backup[eid] = G_work.es[eid][weight]
            G_work.es[eid][weight] = _LARGE_W
    for eid in blocked_eids:
        if eid not in backup:
            backup[eid] = G_work.es[eid][weight]
        G_work.es[eid][weight] = _LARGE_W

    paths = G_work.get_shortest_paths(
        spur_node, to=target, weights=weight, output='vpath'
    )
    for eid, w in backup.items():
        G_work.es[eid][weight] = w

    if not paths or not paths[0] or paths[0][-1] != target:
        return None, float('inf')
    path = paths[0]
    if any(v in blocked_verts for v in path):
        return None, float('inf')
    try:
        length = sum(
            G_work.es[G_work.get_eid(path[a], path[a + 1])][weight]
            for a in range(len(path) - 1)
        )
    except Exception:
        return None, float('inf')
    return (None, float('inf')) if length >= _LARGE_W / 2 else (path, length)


def yen_k_shortest_paths(
    G: ig.Graph,
    source: int,
    target: int,
    k: int,
    weight: str = 'weight',
) -> List[Tuple[List[int], float]]:
    """Yen 算法，在副本上操作，不修改原图。"""
    if source == target:
        return [([source], 0.0)]
    G_work = G.copy()
    sp = G_work.get_shortest_paths(source, to=target, weights=weight, output='vpath')
    if not sp or not sp[0] or sp[0][-1] != target:
        return []
    p0   = sp[0]
    len0 = sum(G_work.es[G_work.get_eid(p0[a], p0[a+1])][weight]
               for a in range(len(p0) - 1))
    A: List[Tuple[List[int], float]] = [(p0, len0)]
    heap: List[Tuple[float, int, List[int]]] = []
    seen: Set[Tuple[int, ...]] = set()
    seq = 0

    for _ in range(1, k):
        prev, _ = A[-1]
        for i in range(len(prev) - 1):
            spur   = prev[i]
            root   = prev[:i + 1]
            b_eids: Set[int] = set()
            for ap, _ in A:
                if len(ap) > i and ap[:i+1] == root:
                    try:
                        b_eids.add(G_work.get_eid(ap[i], ap[i+1]))
                    except Exception:
                        pass
            b_verts: Set[int] = set(root[:-1])
            sp2, sl2 = _yen_spur(G_work, spur, target, b_verts, b_eids, weight)
            if sp2 is None:
                continue
            rl = sum(G_work.es[G_work.get_eid(root[a], root[a+1])][weight]
                     for a in range(len(root) - 1))
            cand = root[:-1] + sp2
            key  = tuple(cand)
            if key not in seen:
                seen.add(key)
                heapq.heappush(heap, (rl + sl2, seq, cand))
                seq += 1
        if not heap:
            break
        bl, _, bp = heapq.heappop(heap)
        A.append((bp, bl))
    return A


# =============================================================================
# 4.  边负载（epath 输出 + numpy id 映射）
# =============================================================================

def compute_edge_loads(
    G_current: ig.Graph,
    id2idx_current: Dict[int, int],
    D_keys: np.ndarray,
    D_vals: np.ndarray,
    k: int = 1,
    theta: float = 1.0,
    weight: str = 'weight',
    verbose: bool = True,
) -> np.ndarray:
    r"""
    k 最短路 + softmax 流分配。

    \[x_e = 2\sum_{i<j} D_{ij}\sum_r p_{ij}^{(r)}\,\mathbf{1}(e\in P_{ij}^{(r)})\]

    优化点
    ------
    - numpy 数组替代 dict 做 json_id → igraph_idx 映射，向量化过滤失效节点
    - k=1 用 ``output='epath'`` 直接取边 id，消除每步 get_eid 调用

    参数
    ----
    D_keys : (P, 2) int64，OD 对 json_ids（来自 compute_od_matrix）
    D_vals : (P,) float64
    """
    M = G_current.ecount()
    x = np.zeros(M, dtype=np.float64)
    if len(D_keys) == 0:
        return x

    # ── numpy id 映射 ────────────────────────────────────────────────────────
    max_jid    = int(D_keys.max())
    idx_lookup = np.full(max_jid + 1, -1, dtype=np.int64)
    for jid, cidx in id2idx_current.items():
        if jid <= max_jid:
            idx_lookup[jid] = cidx

    cur_src = idx_lookup[D_keys[:, 0]]
    cur_tgt = idx_lookup[D_keys[:, 1]]
    valid   = (cur_src >= 0) & (cur_tgt >= 0)

    sort_idx    = np.argsort(cur_src[valid], kind='stable')
    srcs_sorted = cur_src[valid][sort_idx]
    tgts_sorted = cur_tgt[valid][sort_idx]
    dval_sorted = D_vals[valid][sort_idx]

    unique_srcs, g_start, g_count = np.unique(
        srcs_sorted, return_index=True, return_counts=True
    )

    t0     = time.perf_counter()
    n_srcs = len(unique_srcs)

    for gi, cs in enumerate(unique_srcs.tolist()):
        if verbose and gi % 1000 == 0:
            print(f"  [流分配] {gi}/{n_srcs}  ({time.perf_counter()-t0:.1f}s)")

        sl    = slice(g_start[gi], g_start[gi] + g_count[gi])
        tgts  = tgts_sorted[sl].tolist()
        dvals = dval_sorted[sl]

        if k == 1:
            # output='epath'：直接返回边 id 列表，无 get_eid 调用
            all_ep = G_current.get_shortest_paths(
                cs, to=tgts, weights=weight, output='epath'
            )
            for epath, d_val in zip(all_ep, dvals.tolist()):
                for eid in epath:
                    x[eid] += 2.0 * d_val
        else:
            for ct, d_val in zip(tgts, dvals.tolist()):
                info = yen_k_shortest_paths(G_current, cs, ct, k, weight)
                if not info:
                    continue
                lengths = np.array([ln for _, ln in info])
                lw = -theta * lengths; lw -= lw.max()
                probs = np.exp(lw); probs /= probs.sum()
                for r, (path, _) in enumerate(info):
                    fr = d_val * probs[r]
                    for a in range(len(path) - 1):
                        try:
                            x[G_current.get_eid(path[a], path[a+1])] += 2.0 * fr
                        except Exception:
                            break
    return x


# =============================================================================
# 5.  边容量
# =============================================================================

def compute_capacity(
    G_orig: ig.Graph,
    x0: np.ndarray,
    alpha: float = 0.2,
    eps: float = 1e-3,
) -> Dict[Tuple[int, int], float]:
    r"""\(c_e = (1+\alpha)x_e^{(0)} + \varepsilon\)，key = (min_jid, max_jid)。"""
    nids = G_orig.vs['node_id']
    c: Dict[Tuple[int, int], float] = {}
    for eid in range(G_orig.ecount()):
        e = G_orig.es[eid]
        u, v = nids[e.source], nids[e.target]
        c[(min(u, v), max(u, v))] = (1.0 + alpha) * float(x0[eid]) + eps
    return c


# =============================================================================
# 6.  边拥堵率（get_edgelist + numpy）
# =============================================================================

def compute_congestion_rate(
    G_current: ig.Graph,
    x_current: np.ndarray,
    c_dict: Dict[Tuple[int, int], float],
) -> np.ndarray:
    r"""\(\rho_e = x_e/c_e\)，用 get_edgelist + numpy 批量获取端点 json_id。"""
    if G_current.ecount() == 0:
        return np.zeros(0)
    el      = np.array(G_current.get_edgelist(), dtype=np.int32)  # (M, 2)
    nid_arr = np.array(G_current.vs['node_id'],  dtype=np.int64)
    u_ids   = nid_arr[el[:, 0]]
    v_ids   = nid_arr[el[:, 1]]
    rho     = np.zeros(len(u_ids), dtype=np.float64)
    for eid in range(len(rho)):
        key = (int(min(u_ids[eid], v_ids[eid])), int(max(u_ids[eid], v_ids[eid])))
        c   = c_dict.get(key, 0.0)
        if c > 0.0:
            rho[eid] = x_current[eid] / c
    return rho


# =============================================================================
# 7.  流量特征指标
# =============================================================================

def compute_accessibility(
    G_current: ig.Graph,
    D_keys: np.ndarray,
    D_vals: np.ndarray,
    D_total: float,
    max_json_id: int,
) -> float:
    r"""
    流量可达性：
    \(A(f) = \sum_{i<j,\text{可达}} D_{ij} / D_{\text{total}}\)

    优化：预建 comp_lookup[json_id] 整型数组，避免 Python dict 遍历。

    参数
    ----
    max_json_id : int(D_keys.max())，调用前算一次并缓存
    """
    if D_total == 0.0 or G_current.vcount() == 0 or len(D_keys) == 0:
        return 0.0

    clusters   = G_current.clusters()
    membership = clusters.membership              # list[int]

    # comp_lookup[json_id] = 连通分量 id；不存活节点保持 -1
    comp_lookup = np.full(max_json_id + 1, -1, dtype=np.int32)
    for v in G_current.vs:
        jid = v['node_id']
        if jid <= max_json_id:
            comp_lookup[jid] = membership[v.index]

    comp_i = comp_lookup[D_keys[:, 0]]            # 一次 numpy 索引
    comp_j = comp_lookup[D_keys[:, 1]]
    mask   = (comp_i >= 0) & (comp_i == comp_j)

    return float(D_vals[mask].sum()) / D_total


def compute_herfindahl(x_e: np.ndarray) -> float:
    r"""\(H = \sum_e p_e^2\)，纯 numpy dot 积。"""
    xt = x_e.sum()
    if xt == 0.0:
        return 0.0
    p = x_e / xt
    return float(np.dot(p, p))


# =============================================================================
# 8.  单步指标汇总
# =============================================================================

def compute_flow_metrics(
    G_current: ig.Graph,
    id2idx_current: Dict[int, int],
    D_keys: np.ndarray,
    D_vals: np.ndarray,
    D_total: float,
    c_dict: Dict[Tuple[int, int], float],
    max_json_id: int,
    k: int = 1,
    theta: float = 1.0,
    verbose: bool = False,
) -> Dict:
    """
    当前图的全套流量指标，返回 dict：
    'x', 'rho', 'rho_mean', 'rho_max', 'A', 'H'
    """
    x   = compute_edge_loads(G_current, id2idx_current, D_keys, D_vals,
                              k, theta, verbose=verbose)
    rho = compute_congestion_rate(G_current, x, c_dict)
    A   = compute_accessibility(G_current, D_keys, D_vals, D_total, max_json_id)
    H   = compute_herfindahl(x)
    return {
        'x':        x,
        'rho':      rho,
        'rho_mean': float(rho.mean()) if len(rho) else 0.0,
        'rho_max':  float(rho.max())  if len(rho) else 0.0,
        'A':        A,
        'H':        H,
    }


# =============================================================================
# 快速验证
# =============================================================================

if __name__ == '__main__':
    import random

    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'json_networks', 'Chengdu_Network.json',
    )

    t0 = time.perf_counter()
    G_orig, id2idx, idx2id = load_graph(json_path)
    N = G_orig.vcount()
    print(f"加载  节点={N}  边={G_orig.ecount()}  {time.perf_counter()-t0:.2f}s")

    positions = np.array([v['pos'] for v in G_orig.vs], dtype=np.float64)
    SIGMA, BETA = 1000.0, 0.001

    t0 = time.perf_counter()
    m  = compute_node_density(positions, sigma=SIGMA)
    print(f"[1] 节点密度  [{m.min():.1f}, {m.max():.1f}]  {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    D_keys, D_vals, D_total = compute_od_matrix(
        G_orig, m, beta=BETA, sigma=SIGMA, verbose=True,
    )
    max_jid = int(D_keys.max()) if len(D_keys) else 0
    print(f"[2] OD  P={len(D_keys)}  D_total={D_total:.4e}  {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    x0 = compute_edge_loads(G_orig, id2idx, D_keys, D_vals, k=1, verbose=True)
    print(f"[3] 边负载  总={x0.sum():.4e}  非零={np.count_nonzero(x0)}  "
          f"{time.perf_counter()-t0:.2f}s")

    c_dict = compute_capacity(G_orig, x0)
    A0  = compute_accessibility(G_orig, D_keys, D_vals, D_total, max_jid)
    H0  = compute_herfindahl(x0)
    rho0 = compute_congestion_rate(G_orig, x0, c_dict)
    print(f"[4] 初始指标  A={A0:.4f}  H={H0:.6f}  ρ̄={rho0.mean():.4f}")

    print("\n[5] 随机攻击 3 步：")
    G_sim = G_orig.copy(); id2idx_s = dict(id2idx)
    jids  = list(id2idx.keys()); random.seed(42); random.shuffle(jids)
    for step, nid in enumerate(jids[:3], 1):
        if nid not in id2idx_s:
            continue
        G_sim.delete_vertices(id2idx_s[nid])
        id2idx_s = rebuild_id2idx(G_sim)
        mt = compute_flow_metrics(
            G_sim, id2idx_s, D_keys, D_vals, D_total, c_dict, max_jid,
        )
        print(f"  step {step}: 移除 {nid}  "
              f"A={mt['A']:.4f}  H={mt['H']:.6f}  "
              f"ρ̄={mt['rho_mean']:.4f}  ρ_max={mt['rho_max']:.4f}")
