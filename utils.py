"""
utils.py
路网图工具函数集。
"""

import json
import time
import networkx as nx
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 图加载
# ---------------------------------------------------------------------------

def load_graph(filepath: str) -> nx.Graph:
    """
    将 JSON 路网文件直接加载为 NetworkX 无向图。

    节点属性：
        pos    : (x, y) UTM 坐标，单位米
    边属性：
        weight : 两节点间距离，单位米

    参数：
        filepath : JSON 文件路径，例如 'Chengdu_Network.json'
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw: dict = json.load(f)

    G = nx.Graph()
    for val in raw.values():
        nid = val['id']
        G.add_node(nid, pos=tuple(val['position']))
        for nb in val['neighbors']:
            G.add_edge(nid, nb['id'], weight=nb['distance'])
    return G


# ---------------------------------------------------------------------------
# 被积函数：替换 baseline 中的 |LCC| / N
# ---------------------------------------------------------------------------

def compute_performance(G: nx.Graph, N: int,
                        alpha: float = 0.5,
                        beta:  float = 0.5) -> float:
    """
    Q_{bc} 被积函数，直接替换 baseline 里的 performance = len(lcc) / N。

    替换后：
        performance = compute_performance(G, N, alpha, beta)

    公式：
        Q_{bc} = β · |LCC|/N  +  (1−β) · [ α · B_E/N  +  (1−α) · B_V/N ]

    参数：
        G     : 当前时刻的 NetworkX 图（已移除失效节点）
        N     : 原始网络节点总数（归一化分母，全程不变）
        alpha : B_E 与 B_V 之间的权重，∈ [0, 1]
        beta  : LCC 项与双连通项之间的权重，∈ [0, 1]
    """
    if G.number_of_nodes() == 0:
        return 0.0

    lcc_nodes = max(nx.connected_components(G), key=len)
    lcc_size  = len(lcc_nodes)

    if lcc_size <= 1:
        return lcc_size / N

    lcc = G.subgraph(lcc_nodes)
    bridge_edges = list(nx.bridges(lcc))
    if bridge_edges:
        tmp = lcc.copy()
        tmp.remove_edges_from(bridge_edges)
        be_size = max(len(c) for c in nx.connected_components(tmp))
    else:
        be_size = lcc_size
    bv_size = max(len(c) for c in nx.biconnected_components(lcc))

    return (beta * lcc_size / N
            + (1 - beta) * (alpha * be_size / N + (1 - alpha) * bv_size / N))


if __name__ == '__main__':
    import os, random

    json_path = os.path.join(os.path.dirname(__file__), 'data\json_networks\Chengdu_Network.json')

    t0 = time.perf_counter()
    G  = load_graph(json_path)
    N  = G.number_of_nodes()
    print(f"加载耗时: {time.perf_counter()-t0:.3f}s")
    print(f"节点数: {N},  边数: {G.number_of_edges()}")
    print(f"示例节点属性 G.nodes[1]: {G.nodes[1]}")

    p0 = compute_performance(G, N, alpha=0.5, beta=0.5)
    print(f"\n初始 Q_bc (f=0): {p0:.6f}")

    all_ids = list(G.nodes)
    random.seed(42)
    random.shuffle(all_ids)

    print("\n随机攻击前 5 步的 Q_bc：")
    for step, nid in enumerate(all_ids[:5], 1):
        G.remove_node(nid)
        p = compute_performance(G, N, alpha=0.5, beta=0.5)
        print(f"  step {step}: 移除节点 {nid}, Q_bc = {p:.6f}")
