"""
Q4/q4_rcabs.py
RCABS-BE: Regional Community-Aware Beam Search with Biconnectivity Evaluation

第四问核心算法：把 Q3 的 CABS-BE 升级为区域攻击版本。
  - 每步选择一个靶心节点，摧毁该节点及其半径 r 内所有存活节点（空间级联）
  - 健壮性指标沿用 Q_bc（双连通核结构健壮性），与 Q2/Q3 统一
  - 积分宽度为动态 Δq_t = |A_r(v_t) ∩ V_t| / N_0（因为每步删除节点数不固定）
  - 多半径扫描以讨论故障波及范围对结果的影响

两阶段策略
----------
Phase A — 区域贪心预热（Regional Greedy Warm-up）
    在 LCC 子图上做 Leiden 社区检测，为每个节点计算其圆盘内的社区外部度之和，
    贪心选取区域破坏力最大的靶心，一次性删除整个圆盘。
    快速将图缩减到可供 beam search 精细搜索的规模。

Phase B — 区域集束精细搜索（Regional Beam Search Refinement）
    在缩小后的图上执行 Beam Search：
    - 候选靶心仅从 LCC 内部选取，按区域社区桥梁得分排序
    - 每个候选的评估 = 虚拟删除整个圆盘 → Tarjan 计算 Q_bc
    - 候选评估多线程并行化
    - Δq 动态计算，适应不同密度区域的差异

与 Q3 CABS-BE 的关系
-------------------
Q3: 每步 "删一个点"        → CABS-BE
Q4: 每步 "删一个圆盘 A_r(v)" → RCABS-BE  (本文件)

除攻击粒度和积分宽度外，社区桥梁候选筛选、beam search 剪枝、
Tarjan 双连通评价、Q_bc 指标定义均与 Q3 完全一致。
"""

import os
import sys
import time
import concurrent.futures
import igraph as ig
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# ── 路径设置 ──
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入 Q3 共用的工具函数
try:
    from utils_ig import rebuild_id2idx, compute_performance
except ImportError:
    # ── 本地 fallback ──
    def rebuild_id2idx(G):
        return {G.vs[i]["node_id"]: i for i in range(G.vcount())}

    def compute_performance(G, N0, alpha=0.5, beta=0.5):
        """本地实现 Q_bc 指标，与论文 4.2.1 定义一致。"""
        if G.vcount() == 0:
            return 0.0
        cc = G.connected_components()
        sizes = cc.sizes()
        lcc_size = max(sizes)
        lcc_idx = sizes.index(lcc_size)
        lcc_sub = G.induced_subgraph(cc[lcc_idx])

        # BE: 最大边双连通分量 — 删掉所有桥边后最大连通分量
        if lcc_sub.ecount() > 0:
            bridges = lcc_sub.bridges()
            if bridges:
                tmp = lcc_sub.copy()
                tmp.delete_edges(bridges)
                be_max = max(tmp.connected_components().sizes())
            else:
                be_max = lcc_size
        else:
            be_max = 1 if lcc_size == 1 else 0

        # BV: 最大点双连通分量（block）
        if lcc_sub.vcount() >= 2 and lcc_sub.ecount() > 0:
            blocks = lcc_sub.biconnected_components()
            bv_max = max(len(b) for b in blocks) if blocks else 1
        else:
            bv_max = lcc_sub.vcount()

        return beta * (lcc_size / N0) + (1 - beta) * (
            alpha * (be_max / N0) + (1 - alpha) * (bv_max / N0)
        )


# =====================================================================
# 1. 空间数据加载与预处理
# =====================================================================

def load_spatial_graph(csv_path: str):
    """
    从 CSV 边表加载带空间坐标的 igraph 图。
    返回: (G, id2idx, idx2id)
    """
    df = pd.read_csv(csv_path)
    nodes_df = (
        df[["START_NODE", "XCoord", "YCoord"]]
        .drop_duplicates(subset=["START_NODE"])
        .sort_values("START_NODE")
        .reset_index(drop=True)
    )
    node_ids = nodes_df["START_NODE"].tolist()
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    idx2id = {i: nid for i, nid in enumerate(node_ids)}

    u = df["START_NODE"].map(id2idx).values
    v = df["END_NODE"].map(id2idx).values
    w = df["LENGTH"].values
    mask = (~np.isnan(u)) & (~np.isnan(v)) & (u != v)
    u, v, w = u[mask].astype(int), v[mask].astype(int), w[mask]

    G = ig.Graph(n=len(node_ids), edges=list(zip(u, v)), directed=False)
    G.vs["node_id"] = node_ids
    G.vs["pos"] = list(zip(
        nodes_df["XCoord"].tolist(), nodes_df["YCoord"].tolist()
    ))
    G.es["weight"] = w
    G.simplify(multiple=True, loops=True, combine_edges="first")
    return G, id2idx, idx2id


def precompute_disk_map(G: ig.Graph, radius: float) -> dict:
    """
    预计算每个节点的空间圆盘邻域：node_id → frozenset{node_ids within radius}。
    基于 cKDTree，一次性完成 O(N log N) 查询。
    """
    coords, valid_idxs = [], []
    for i in range(G.vcount()):
        pos = G.vs[i]["pos"]
        if pos and not np.isnan(pos[0]):
            coords.append([pos[0], pos[1]])
            valid_idxs.append(i)

    tree = cKDTree(coords)
    all_nbs = tree.query_ball_point(coords, r=radius)

    node_ids = np.array(G.vs["node_id"])
    valid_arr = np.array(valid_idxs)
    disk_map = {}
    for i, nbs in enumerate(all_nbs):
        center_nid = int(node_ids[valid_arr[i]])
        nb_nids = frozenset(int(node_ids[valid_arr[j]]) for j in nbs)
        disk_map[center_nid] = nb_nids

    return disk_map


# =====================================================================
# 2. 社区检测与候选评估工具
# =====================================================================

def compute_external_degree(G: ig.Graph, membership: list) -> list:
    """计算每个节点的社区外部度 k_ext(v)，O(M)。"""
    ext_deg = [0] * G.vcount()
    for e in G.es:
        if membership[e.source] != membership[e.target]:
            ext_deg[e.source] += 1
            ext_deg[e.target] += 1
    return ext_deg


def get_lcc_info(G: ig.Graph):
    """返回 (lcc_size, lcc_cluster_idx, lcc_vertex_indices)。"""
    if G.vcount() == 0:
        return 0, -1, []
    cc = G.connected_components()
    sizes = cc.sizes()
    lcc_size = max(sizes)
    lcc_idx = sizes.index(lcc_size)
    return lcc_size, lcc_idx, cc[lcc_idx]


def get_lcc_ratio(G: ig.Graph, N0: int) -> float:
    if G.vcount() == 0:
        return 0.0
    return max(G.connected_components().sizes()) / N0


def detect_communities(G: ig.Graph) -> list:
    if G.vcount() <= 2:
        return list(range(G.vcount()))
    try:
        return G.community_leiden(objective_function="modularity").membership
    except AttributeError:
        return G.community_multilevel().membership
    except Exception:
        return [0] * G.vcount()


# =====================================================================
# 3. 区域候选生成（Q4 核心改造点）
# =====================================================================

def get_lcc_candidates_regional(
    G: ig.Graph, K: int, disk_map: dict
) -> list:
    """
    从 LCC 中筛选 top-K 区域攻击靶心。

    评分标准：靶心 v 的圆盘 A_r(v) 在 LCC 内的「社区桥梁总破坏力」
      score(v) = Σ_{u ∈ A_r(v) ∩ LCC} ext_deg(u)
    即：一发炸掉的所有节点所连跨社区边之和。

    这是 Q3 CABS 的自然区域化推广：
    Q3 按单节点 ext_deg 排序，Q4 按圆盘内 ext_deg 之和排序。
    """
    lcc_size, _, lcc_members = get_lcc_info(G)
    if lcc_size == 0:
        return []
    if lcc_size <= K:
        return [(i, G.vs[i]["node_id"]) for i in lcc_members]

    # 提取 LCC 子图，在其上做社区检测
    lcc_sub = G.induced_subgraph(lcc_members)
    membership = detect_communities(lcc_sub)
    ext_deg = compute_external_degree(lcc_sub, membership)
    degrees = lcc_sub.degree()

    # LCC 子图索引 → node_id 映射
    sub_nids = [lcc_sub.vs[j]["node_id"] for j in range(lcc_sub.vcount())]
    nid_to_sub = {nid: j for j, nid in enumerate(sub_nids)}
    lcc_nid_set = set(sub_nids)

    # 为每个 LCC 节点计算区域得分
    scores = []
    for j in range(lcc_sub.vcount()):
        nid = sub_nids[j]
        disk_nids = disk_map.get(nid, frozenset({nid}))
        # 圆盘与 LCC 的交集中，各节点 ext_deg 之和
        regional_ext = sum(
            ext_deg[nid_to_sub[d]]
            for d in disk_nids
            if d in nid_to_sub
        )
        disk_in_lcc = sum(1 for d in disk_nids if d in lcc_nid_set)
        scores.append((regional_ext, disk_in_lcc, degrees[j], j))

    scores.sort(reverse=True)

    # ── 去重：如果两个靶心的圆盘高度重叠 (Jaccard > 0.9)，只保留得分高的 ──
    selected = []
    selected_disks = []
    for _, _, _, sub_idx in scores:
        if len(selected) >= K:
            break
        nid = sub_nids[sub_idx]
        disk = disk_map.get(nid, frozenset({nid})) & lcc_nid_set
        # 检查与已选圆盘是否高度重叠
        redundant = False
        for prev_disk in selected_disks:
            if len(disk & prev_disk) / max(len(disk | prev_disk), 1) > 0.9:
                redundant = True
                break
        if not redundant:
            orig_idx = lcc_members[sub_idx]
            selected.append((orig_idx, nid))
            selected_disks.append(disk)

    return selected


# =====================================================================
# 4. 区域评估函数（线程工作单元）
# =====================================================================

def _evaluate_candidate_regional(args) -> dict:
    """
    评估以 center_nid 为靶心执行区域攻击后的 Q_bc 值。
    核心改造：删除的不是单个节点，而是整个圆盘 A_r(center)。
    """
    (G_b, center_nid, blast_nids, R_b, N0, alpha, beta, b_idx) = args

    blast_count = len(blast_nids)
    if blast_count == 0:
        lcc = get_lcc_ratio(G_b, N0)
        return {"R": R_b, "b_idx": b_idx, "nid": center_nid,
                "blast": 0, "qbc": 0.0, "lcc": lcc}

    # 构建残余子图：保留不在爆炸范围内的所有节点
    keep = [i for i in range(G_b.vcount())
            if G_b.vs[i]["node_id"] not in blast_nids]
    sub = G_b.induced_subgraph(keep)

    qbc = compute_performance(sub, N0, alpha, beta)
    lcc = get_lcc_ratio(sub, N0)

    # 关键：动态积分宽度 Δq = |实际删除节点数| / N0
    delta_q = blast_count / N0

    return {
        "R":     R_b + qbc * delta_q,
        "b_idx": b_idx,
        "nid":   center_nid,
        "blast": blast_count,
        "qbc":   qbc,
        "lcc":   lcc,
    }


# =====================================================================
# 5. Phase A：区域贪心预热
# =====================================================================

def greedy_warmup_regional(
    G:          ig.Graph,
    N0:         int,
    disk_map:   dict,
    target_frac: float = 0.15,
    alpha:      float = 0.5,
    beta:       float = 0.5,
    theta:      float = 0.01,
    leiden_refresh: int = 3,
) -> tuple:
    """
    区域贪心预热：反复选取 LCC 中区域破坏力最大的靶心，删除整个圆盘。
    直到累计删除比例达到 target_frac 或 LCC/N0 ≤ θ。

    返回: (G_out, warmup_seq, warmup_R)
    """
    G_out = G.copy()
    warmup_seq = []
    cumR = 0.0
    total_removed = 0
    t0 = time.perf_counter()

    step = 0
    while total_removed / N0 < target_frac:
        if G_out.vcount() == 0:
            break
        lcc_ratio = get_lcc_ratio(G_out, N0)
        if lcc_ratio <= theta:
            print(f"      [Warmup 提前终止] LCC/N_0={lcc_ratio:.4f} ≤ θ")
            break

        step += 1

        # 在 LCC 上做 Leiden，按区域 ext_deg 之和选最佳靶心
        _, _, lcc_members = get_lcc_info(G_out)
        if len(lcc_members) <= 2:
            # LCC 太小，直接按度排序选靶心
            degrees = G_out.degree()
            best_idx = max(lcc_members, key=lambda i: degrees[i])
            best_nid = G_out.vs[best_idx]["node_id"]
        else:
            lcc_sub = G_out.induced_subgraph(lcc_members)
            membership = detect_communities(lcc_sub)
            ext_deg = compute_external_degree(lcc_sub, membership)
            sub_nids = [lcc_sub.vs[j]["node_id"] for j in range(lcc_sub.vcount())]
            nid_to_sub = {nid: j for j, nid in enumerate(sub_nids)}
            lcc_nid_set = set(sub_nids)

            best_score, best_sub_idx = -1, 0
            for j in range(lcc_sub.vcount()):
                nid = sub_nids[j]
                disk = disk_map.get(nid, frozenset({nid}))
                score = sum(ext_deg[nid_to_sub[d]] for d in disk if d in nid_to_sub)
                if score > best_score:
                    best_score = score
                    best_sub_idx = j
            best_nid = sub_nids[best_sub_idx]

        # 执行区域删除
        alive_nids = set(G_out.vs["node_id"])
        blast = disk_map.get(best_nid, frozenset({best_nid})) & alive_nids
        blast_count = len(blast)
        if blast_count == 0:
            break

        keep = [i for i in range(G_out.vcount())
                if G_out.vs[i]["node_id"] not in blast]
        G_out = G_out.induced_subgraph(keep)

        warmup_seq.append(best_nid)
        total_removed += blast_count

        # 累积积分
        qbc = compute_performance(G_out, N0, alpha, beta)
        delta_q = blast_count / N0
        cumR += qbc * delta_q

        if step % 5 == 0 or step <= 3:
            elapsed = time.perf_counter() - t0
            lcc_now = get_lcc_ratio(G_out, N0)
            print(
                f"      [Warmup] step={step}, removed={total_removed} "
                f"({total_removed/N0*100:.1f}%), LCC/N_0={lcc_now:.4f}, "
                f"R={cumR:.6f}, 耗时={elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    lcc_final = get_lcc_ratio(G_out, N0)
    print(
        f"    [Warmup 完成] {step} 步, 删除 {total_removed} 节点 "
        f"({total_removed/N0*100:.1f}%), LCC/N_0={lcc_final:.4f}, "
        f"R={cumR:.6f}, 耗时={elapsed:.1f}s"
    )
    return G_out, warmup_seq, cumR


# =====================================================================
# 6. Phase B：区域集束搜索
# =====================================================================

def beam_search_phase_regional(
    G_init:     ig.Graph,
    init_seq:   list,
    init_R:     float,
    N0:         int,
    disk_map:   dict,
    W:          int   = 3,
    K:          int   = 20,
    alpha:      float = 0.5,
    beta:       float = 0.5,
    theta:      float = 0.01,
    n_workers:  int   = 0,
    log_interval: int = 5,
) -> tuple:
    """
    区域版 Beam Search。每步从 LCC 中选 K 个候选靶心，
    虚拟执行区域删除后评估 Q_bc，保留 R 最小的 W 条路径。

    返回: (best_seq, best_R)
    """
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 4, 8)

    G0 = G_init.copy()
    lcc0 = get_lcc_ratio(G0, N0)
    beam = [(G0, list(init_seq), init_R, lcc0)]

    step = 0
    t_start = time.perf_counter()

    while True:
        # 全部 beam 都已终止
        if all(s[3] <= theta for s in beam):
            break
        if all(s[0].vcount() == 0 for s in beam):
            break

        step += 1

        # ── 1. 候选生成 + 评估 ──
        eval_tasks = []

        for b_idx, (G_b, seq_b, R_b, P_b) in enumerate(beam):
            if P_b <= theta or G_b.vcount() == 0:
                continue

            # 从 LCC 选 top-K 区域候选靶心
            candidates = get_lcc_candidates_regional(G_b, K, disk_map)

            # 预算当前存活节点集（供所有候选共用）
            alive_nids = set(G_b.vs["node_id"])

            for orig_idx, center_nid in candidates:
                # 计算该靶心实际能炸到的节点集
                blast = disk_map.get(center_nid, frozenset({center_nid}))
                blast_alive = blast & alive_nids
                if len(blast_alive) == 0:
                    continue
                eval_tasks.append((
                    G_b, center_nid, blast_alive,
                    R_b, N0, alpha, beta, b_idx,
                ))

        # 没有可评估的候选
        if not eval_tasks:
            # 对已终止的 beam 做兜底
            remaining = [s for s in beam if s[3] > theta and s[0].vcount() > 0]
            if not remaining:
                break
            # fallback: 在 LCC 中按度数选最大的靶心
            new_beam = []
            for G_b, seq_b, R_b, P_b in remaining:
                _, _, lcc_m = get_lcc_info(G_b)
                if not lcc_m:
                    new_beam.append((G_b, seq_b, R_b, P_b))
                    continue
                deg = G_b.degree()
                best_i = max(lcc_m, key=lambda i: deg[i])
                best_nid = G_b.vs[best_i]["node_id"]
                alive_nids = set(G_b.vs["node_id"])
                blast = disk_map.get(best_nid, frozenset({best_nid})) & alive_nids
                keep = [i for i in range(G_b.vcount())
                        if G_b.vs[i]["node_id"] not in blast]
                G_new = G_b.induced_subgraph(keep)
                qbc = compute_performance(G_new, N0, alpha, beta)
                delta_q = len(blast) / N0
                new_R = R_b + qbc * delta_q
                new_lcc = get_lcc_ratio(G_new, N0)
                new_beam.append((G_new, seq_b + [best_nid], new_R, new_lcc))
            beam = new_beam
            continue

        # 多线程并行评估
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_workers,
        ) as pool:
            results = list(pool.map(_evaluate_candidate_regional, eval_tasks))

        # ── 2. 排序 + 剪枝 ──
        results.sort(key=lambda x: x["R"])
        selected = results[:W]

        # ── 3. 构建新 beam ──
        new_beam = []
        for sel in selected:
            b_idx = sel["b_idx"]
            center_nid = sel["nid"]

            G_old = beam[b_idx][0]
            seq_old = beam[b_idx][1]

            # 实际执行区域删除
            alive_nids = set(G_old.vs["node_id"])
            blast = disk_map.get(center_nid, frozenset({center_nid})) & alive_nids
            keep = [i for i in range(G_old.vcount())
                    if G_old.vs[i]["node_id"] not in blast]
            G_new = G_old.induced_subgraph(keep)

            new_seq = seq_old + [center_nid]
            new_R = sel["R"]
            new_lcc = sel["lcc"]

            new_beam.append((G_new, new_seq, new_R, new_lcc))

        beam = new_beam

        # ── 日志 ──
        if step % log_interval == 0 or step <= 3:
            best = min(beam, key=lambda x: x[2])
            elapsed = time.perf_counter() - t_start
            total_removed = N0 - best[0].vcount()
            q_now = total_removed / N0
            print(
                f"    Step {step:>3d}: centers={len(best[1]):>4d}, "
                f"removed={total_removed:>5d}, q={q_now:.4f}, "
                f"LCC/N_0={best[3]:.4f}, R={best[2]:.6f}, "
                f"耗时={elapsed:.1f}s"
            )

    # 输出最优
    best = min(beam, key=lambda x: x[2])
    elapsed = time.perf_counter() - t_start
    total_removed = N0 - best[0].vcount()
    print(
        f"    [Beam Search 完成] {len(best[1])} 个靶心, "
        f"删除 {total_removed} 节点 ({total_removed/N0*100:.1f}%), "
        f"R_bc={best[2]:.6f}, 耗时={elapsed:.1f}s"
    )
    return best[1], best[2]


# =====================================================================
# 7. 主入口：组合两阶段
# =====================================================================

def estimate_warmup_frac(avg_disk: float, N0: int) -> float:
    """
    根据平均圆盘大小自适应估算合理的预热比例。

    设计逻辑：
    - 预热的目的是用 ~15-25 步贪心快速缩图，把剩余工作留给 beam search。
    - 每步大约删除 avg_disk 个节点。
    - 所以 warmup_frac ≈ target_steps * avg_disk / N0。
    - 小半径（avg_disk ≈ 1-3）→ 行为接近 Q3 逐点贪心 → 给 5%~8% 即可。
    - 大半径（avg_disk ≈ 100+）→ 几步就删很多节点 → 可以给到 25%~35%。
    - 最终 clip 到 [0.05, 0.35] 的安全范围。
    """
    target_steps = 20
    frac = target_steps * avg_disk / N0
    return float(np.clip(frac, 0.05, 0.35))


def rcabs_attack(
    csv_path:     str,
    radius:       float,
    W:            int   = 3,
    K:            int   = 20,
    alpha:        float = 0.5,
    beta:         float = 0.5,
    theta:        float = 0.01,
    warmup_frac:  float = None,
    n_workers:    int   = 0,
    city_name:    str   = "",
) -> tuple:
    """
    RCABS-BE 两阶段区域攻击算法。

    参数
    ----
    csv_path     : CSV 边表路径（含坐标）
    radius       : 故障波及半径（米）
    W            : beam search 宽度
    K            : 每步候选靶心数
    warmup_frac  : 贪心预热阶段累计删除的节点比例上限。
                   传 None 则根据平均圆盘大小自适应计算。

    返回
    ----
    (attack_centers, R_bc)
      attack_centers: 靶心 node_id 列表（按攻击顺序）
      R_bc: 累积结构健壮性积分
    """
    G, _, _ = load_spatial_graph(csv_path)
    N0 = G.vcount()

    # ── 预计算圆盘邻域 ──
    t_pre = time.perf_counter()
    disk_map = precompute_disk_map(G, radius)
    avg_disk = np.mean([len(v) for v in disk_map.values()])
    print(f"  [{city_name}] 圆盘预处理完成: 平均圆盘大小={avg_disk:.1f} 节点, "
          f"耗时={time.perf_counter()-t_pre:.1f}s")

    # ── 自适应 warmup 比例 ──
    if warmup_frac is None:
        warmup_frac = estimate_warmup_frac(avg_disk, N0)

    print(f"\n  [{city_name}] N_0={N0}, M={G.ecount()}, radius={radius}m")
    print(f"  [{city_name}] 超参: W={W}, K={K}, θ={theta}, "
          f"warmup={warmup_frac*100:.1f}% (avg_disk={avg_disk:.1f})")

    # ==== Phase A: 区域贪心预热 ====
    print(f"\n  ---- Phase A: 区域贪心预热 (target ≤ {warmup_frac*100:.1f}%) ----")
    G_warm, warmup_seq, warmup_R = greedy_warmup_regional(
        G, N0, disk_map,
        target_frac=warmup_frac,
        alpha=alpha, beta=beta, theta=theta,
    )

    # 检查预热后是否已达标
    if get_lcc_ratio(G_warm, N0) <= theta:
        print(f"  [{city_name}] 预热阶段已达 LCC/N_0 ≤ θ，跳过 Beam Search!")
        return warmup_seq, warmup_R

    # ==== Phase B: 区域 Beam Search ====
    print(f"\n  ---- Phase B: 区域 Beam Search (W={W}, K={K}) ----")
    full_seq, full_R = beam_search_phase_regional(
        G_warm, warmup_seq, warmup_R, N0, disk_map,
        W=W, K=K, alpha=alpha, beta=beta, theta=theta,
        n_workers=n_workers,
    )

    print(
        f"\n  [{city_name}] RCABS 完成! "
        f"靶心数={len(full_seq)}, R_bc={full_R:.6f}"
    )
    return full_seq, full_R


# =====================================================================
# 8. 回放引擎：按靶心序列重新模拟，输出标准曲线
# =====================================================================

def replay_attack_sequence(
    csv_path:      str,
    target_centers: list,
    radius:        float,
    alpha:         float = 0.5,
    beta:          float = 0.5,
) -> tuple:
    """
    给定攻击靶心序列，完整回放区域级联删除过程。
    输出: (q_list, P_q_list, Q_bc_list, R_final)

    q_list[t]    : 第 t 步后累计失效节点比例
    P_q_list[t]  : 第 t 步后 |LCC| / N_0
    Q_bc_list[t] : 第 t 步后 Q_bc 值
    R_final      : 累积健壮性积分
    """
    G, _, _ = load_spatial_graph(csv_path)
    N0 = G.vcount()
    disk_map = precompute_disk_map(G, radius)

    q_list = [0.0]
    P_q_list = [1.0]
    Q_bc_list = [compute_performance(G, N0, alpha, beta)]

    G_sim = G.copy()
    total_removed = 0

    for center_nid in target_centers:
        alive_nids = set(G_sim.vs["node_id"])
        if center_nid not in alive_nids:
            continue

        blast = disk_map.get(center_nid, frozenset({center_nid})) & alive_nids
        if len(blast) == 0:
            continue

        keep = [i for i in range(G_sim.vcount())
                if G_sim.vs[i]["node_id"] not in blast]
        G_sim = G_sim.induced_subgraph(keep)

        total_removed += len(blast)
        q = total_removed / N0
        pq = get_lcc_ratio(G_sim, N0)
        qbc = compute_performance(G_sim, N0, alpha, beta)

        q_list.append(q)
        P_q_list.append(pq)
        Q_bc_list.append(qbc)

        if pq <= 0.01:
            break

    # 梯形积分
    R_pq = float(np.trapz(P_q_list, q_list))
    R_bc = float(np.trapz(Q_bc_list, q_list))

    return q_list, P_q_list, Q_bc_list, R_pq, R_bc


# =====================================================================
# 9. 主程序：多半径扫描
# =====================================================================

def main():
    # ── 路径配置（根据你的项目结构调整）──
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "B题数据")
    out_dir = os.path.join(base_dir, "Q4", "Results")
    os.makedirs(out_dir, exist_ok=True)

    cities = [
        "Chengdu", "Dalian", "Dongguan", "Harbin",
        "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou",
    ]

    # ── 超参 ──
    W = 3
    K = 20
    alpha, beta = 0.5, 0.5
    theta = 0.01

    # ── 多半径扫描（题目要求讨论半径变化的影响）──
    radii = [100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0]  # 可根据需要扩展

    # ── 每个半径对应的 warmup 比例 ──
    # 设为 None 表示自适应（根据平均圆盘大小自动推算）
    # 也可以手动指定，例如 100: 0.05 表示 r=100m 时只用 5% 预热
    warmup_frac_map = {
        100.0:  0.09,    # 小半径 → 圆盘小 → 接近 Q3 逐点贪心 → 少预热
        250.0:  0.12,
        500.0:  0.15,
        750.0:  0.18,
        1000.0: 0.22,    # 大半径 → 圆盘大 → 几步就删很多 → 可多预热
        1500.0: 0.24,
        2000.0: 0.26,
        3000.0: 0.26,
        # 未列出的半径 → 传 None → 自适应计算
    }

    for radius in radii:
        print(f"\n{'='*70}")
        print(f"  波及半径 r = {int(radius)}m")
        print(f"{'='*70}")

        summary_rows = []

        for city in cities:
            csv_path = os.path.join(data_dir, f"{city}_Edgelist.csv")
            if not os.path.exists(csv_path):
                print(f"  [跳过] 找不到 {csv_path}")
                continue

            print(f"\n{'='*60}")
            print(f"  城市: {city} — RCABS-BE 区域集束搜索拆解 (r={int(radius)}m)")
            print(f"{'='*60}")

            t0 = time.perf_counter()

            # ── 两阶段攻击 ──
            wf = warmup_frac_map.get(radius, None)  # 未配置的半径自动推算
            seq, R_bc = rcabs_attack(
                csv_path, radius,
                W=W, K=K, alpha=alpha, beta=beta, theta=theta,
                warmup_frac=wf,
                city_name=city,
            )

            # ── 保存攻击序列 ──
            seq_csv = os.path.join(
                out_dir, f"{city}_RCABS_Sequence_r{int(radius)}m.csv"
            )
            pd.DataFrame({
                "rank": range(1, len(seq) + 1),
                "center_node_id": seq,
            }).to_csv(seq_csv, index=False, encoding="utf-8-sig")

            # ── 回放标准曲线 ──
            print(f"    回放曲线 ...")
            q_arr, pq_arr, qbc_arr, R_pq, R_bc_replay = replay_attack_sequence(
                csv_path, seq, radius, alpha=alpha, beta=beta,
            )

            elapsed = time.perf_counter() - t0
            print(
                f"    R_pq(LCC)={R_pq:.6f}, R_bc={R_bc_replay:.6f}, "
                f"靶心数={len(seq)}, 总耗时={elapsed:.1f}s"
            )

            # ── 保存回放曲线 ──
            curve_csv = os.path.join(
                out_dir,
                f"Q4_RCABS_Curve_{city}_r{int(radius)}m.csv",
            )
            pd.DataFrame({
                "q": q_arr, "P_q": pq_arr,
                "Q_bc": qbc_arr, "Method": "RCABS-BE",
            }).to_csv(curve_csv, index=False)

            summary_rows.append({
                "City": city,
                "Radius_m": int(radius),
                "R_Pq_LCC": R_pq,
                "R_Qbc": R_bc_replay,
                "Num_Centers": len(seq),
                "Time_s": elapsed,
            })

        # ── 保存该半径下的汇总排名 ──
        if summary_rows:
            df_sum = pd.DataFrame(summary_rows)
            df_sum = df_sum.sort_values("R_Qbc", ascending=False)
            sum_csv = os.path.join(
                out_dir, f"Q4_RCABS_Summary_r{int(radius)}m.csv"
            )
            df_sum.to_csv(sum_csv, index=False, encoding="utf-8-sig")

            print(f"\n  ── 半径 {int(radius)}m 城市健壮性排名（R_bc 降序 = 越难瓦解）──")
            print(df_sum.to_string(index=False))

    print(f"\n{'='*70}")
    print("  全部 RCABS-BE 区域拆解完毕！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
