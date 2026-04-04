"""
utils_ig.py
成都路网图（*_Network.json）工具函数集 —— igraph 版。

相比 NetworkX 版（utils.py），核心算法（连通分量、桥、点双连通）
全部由 C 后端执行，在 1.8 万节点规模下通常快 5～20 倍。

公开接口
--------
load_graph(filepath)          -> (ig.Graph, id2idx, idx2id)
rebuild_id2idx(G)             -> id2idx
compute_performance(G, N, alpha, beta) -> float   # Q_bc 被积函数
"""

import json
import time
import igraph as ig
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 图加载
# ---------------------------------------------------------------------------

def load_graph(filepath: str) -> Tuple[ig.Graph, Dict[int, int], Dict[int, int]]:
    """
    将 JSON 路网文件直接加载为 igraph.Graph 无向图。

    返回
    ----
    G : ig.Graph
        节点属性 ``node_id`` —— 原始 JSON 中的整数节点 ID
        节点属性 ``pos``     —— (x, y) UTM 坐标，单位米
        边属性   ``weight``  —— 两端点间道路距离，单位米
    id2idx : Dict[int, int]
        原始节点 ID  →  igraph 内部连续 0-based 索引
    idx2id : Dict[int, int]
        igraph 内部连续 0-based 索引  →  原始节点 ID

    注意
    ----
    igraph 内部使用紧凑 0-based 整数索引，与 JSON 中的原始 ID 无关。
    每次调用 ``G.delete_vertices(...)`` 后，内部索引会重排，
    必须随即调用 ``rebuild_id2idx(G)`` 更新 ``id2idx``。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw: dict = json.load(f)

    node_list = list(raw.values())
    n = len(node_list)

    # 建立双向映射
    id2idx: Dict[int, int] = {val['id']: i for i, val in enumerate(node_list)}
    idx2id: Dict[int, int] = {i: val['id'] for i, val in enumerate(node_list)}

    # 收集边（无向图：只保留 u < v 的一侧以去重）
    edges:   List[Tuple[int, int]] = []
    weights: List[float]           = []
    for val in node_list:
        u = id2idx[val['id']]
        for nb in val['neighbors']:
            v = id2idx[nb['id']]
            if u < v:
                edges.append((u, v))
                weights.append(nb['distance'])

    # 构造 igraph 对象
    G = ig.Graph(n=n, edges=edges)
    G.vs['node_id'] = [val['id']             for val in node_list]
    G.vs['pos']     = [tuple(val['position']) for val in node_list]
    G.es['weight']  = weights

    return G, id2idx, idx2id


def rebuild_id2idx(G: ig.Graph) -> Dict[int, int]:
    """
    在调用 ``G.delete_vertices(...)`` 之后重建 id2idx 映射。

    igraph 每次删除节点都会将剩余节点重新紧凑编号，
    旧的 id2idx 立即失效，必须通过此函数刷新。

    复杂度：O(N)，N 为当前节点数。

    示例
    ----
    >>> G.delete_vertices(id2idx[nid])
    >>> id2idx = rebuild_id2idx(G)
    """
    return {G.vs[i]['node_id']: i for i in range(G.vcount())}


# ---------------------------------------------------------------------------
# 被积函数 Q_bc
# ---------------------------------------------------------------------------

def compute_performance(
    G:     ig.Graph,
    N:     int,
    alpha: float = 0.5,
    beta:  float = 0.5,
) -> float:
    r"""
    \(Q_{bc}\) 被积函数，用于替换 baseline 里的 ``len(lcc) / N``。

    定义
    ----
    .. math::

        Q_{bc}(f) = \beta \frac{|LCC|}{N}
                  + (1-\beta)\!\left(
                        \alpha \frac{B_E}{N}
                      + (1-\alpha) \frac{B_V}{N}
                    \right)

    其中：

    * \(|LCC|\) —— 当前图最大连通分量的节点数
    * \(B_E\)   —— LCC 内最大**边双连通**分量的节点数（反映桥边瓶颈）
    * \(B_V\)   —— LCC 内最大**点双连通**分量的节点数（反映割点瓶颈）
    * \(N\)     —— 原始网络节点总数（归一化分母，全程不变）

    参数
    ----
    G     : 当前时刻的 igraph.Graph（已删除失效节点）
    N     : 原始网络节点总数
    alpha : \(B_E\) 与 \(B_V\) 之间的权重，\(\in [0,1]\)
    beta  : LCC 项与双连通项之间的权重，\(\in [0,1]\)

    用法
    ----
    baseline 原写法::

        performance = len(max(nx.connected_components(G), key=len)) / N

    替换为::

        performance = compute_performance(G, N, alpha=0.5, beta=0.5)
    """
    if G.vcount() == 0:
        return 0.0

    # ---- 最大连通分量 ----------------------------------------
    clusters  = G.clusters()           # VertexClustering，O(N+M)
    sizes     = clusters.sizes()
    lcc_size  = max(sizes)

    if lcc_size <= 1:
        return lcc_size / N

    lcc_idx = sizes.index(lcc_size)
    lcc     = clusters.subgraph(lcc_idx)   # 子图视图，不复制边数据

    # ---- 最大边双连通分量（B_E）--------------------------------
    # 边双连通分量 ≡ 删去所有桥边后的连通分量
    bridge_eids = lcc.bridges()            # 返回边 ID 列表，O(N+M)
    if bridge_eids:
        tmp = lcc.copy()
        tmp.delete_edges(bridge_eids)
        be_size = max(tmp.clusters().sizes())
    else:
        be_size = lcc_size                 # 无桥 ⇒ 整个 LCC 即为边双连通

    # ---- 最大点双连通分量（B_V）--------------------------------
    bv_comps = lcc.biconnected_components()   # 返回节点 ID 列表的列表，O(N+M)
    bv_size  = max(len(c) for c in bv_comps)

    return (beta * lcc_size / N
            + (1 - beta) * (alpha * be_size / N + (1 - alpha) * bv_size / N))


# ---------------------------------------------------------------------------
# 快速验证（直接运行此文件时执行）
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    import random

    json_path = os.path.join(
        os.path.dirname(__file__),
        r'data\json_networks\Chengdu_Network.json',
    )

    # ---- 加载 ----
    t0 = time.perf_counter()
    G, id2idx, idx2id = load_graph(json_path)
    N = G.vcount()
    print(f"加载耗时:  {time.perf_counter() - t0:.3f}s")
    print(f"节点数:    {N}")
    print(f"边数:      {G.ecount()}")
    print(f"示例 G.vs[0]:  node_id={G.vs[0]['node_id']},  pos={G.vs[0]['pos']}")

    # ---- 初始性能 ----
    p0 = compute_performance(G, N, alpha=0.5, beta=0.5)
    print(f"\n初始 Q_bc (f=0):  {p0:.6f}")

    # ---- 随机攻击前 5 步 ----
    all_ids = [G.vs[i]['node_id'] for i in range(G.vcount())]
    random.seed(42)
    random.shuffle(all_ids)

    print("\n随机攻击前 5 步的 Q_bc：")
    for step, nid in enumerate(all_ids[:5], start=1):
        if nid not in id2idx:
            continue
        G.delete_vertices(id2idx[nid])
        id2idx = rebuild_id2idx(G)          # 删节点后必须刷新映射
        p = compute_performance(G, N, alpha=0.5, beta=0.5)
        print(f"  step {step}: 移除节点 {nid},  Q_bc = {p:.6f}")

    # ---- 500 步随机攻击计时（对比 NetworkX 用） ----
    print("\n===== 500 步随机攻击计时 =====")
    G, id2idx, _ = load_graph(json_path)
    N = G.vcount()
    order = [G.vs[i]['node_id'] for i in range(G.vcount())]
    random.seed(0)
    random.shuffle(order)

    t0 = time.perf_counter()
    for nid in order[:500]:
        if nid not in id2idx:
            continue
        G.delete_vertices(id2idx[nid])
        id2idx = rebuild_id2idx(G)
        compute_performance(G, N)
    print(f"500 步总耗时:  {time.perf_counter() - t0:.2f}s")
