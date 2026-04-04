"""
utils.py
路网图工具函数集
"""

import json
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 类型别名，方便阅读
# ---------------------------------------------------------------------------
Neighbor = Dict[str, float]   # {'id': int, 'distance': float}

class Node:
    """单个路网节点。"""
    __slots__ = ('id', 'position', 'degree', 'neighbors')

    def __init__(self, node_id: int, position: Tuple[float, float],
                 degree: int, neighbors: List[Neighbor]):
        self.id: int = node_id
        self.position: Tuple[float, float] = position   # (x, y)，单位：米（UTM坐标）
        self.degree: int = degree
        self.neighbors: List[Neighbor] = neighbors       # [{'id': ..., 'distance': ...}, ...]

    def __repr__(self) -> str:
        return (f"Node(id={self.id}, pos=({self.position[0]:.2f}, {self.position[1]:.2f}), "
                f"degree={self.degree})")


def load_graph(filepath: str) -> List[Optional[Node]]:
    """
    将 JSON 路网文件加载为以节点 ID 为下标的列表（1-indexed）。

    返回值：
        nodes: List[Optional[Node]]
            nodes[i] 即 id=i 的节点；nodes[0] 为 None（占位，不使用）。
            若某 ID 在 JSON 中缺失，对应位置同样为 None。

    参数：
        filepath: JSON 文件路径，例如 'Chengdu_Network.json'

    复杂度：
        时间 O(N + M)，空间 O(N + M)，N 为节点数，M 为边数（邻居总数）。

    示例：
        >>> nodes = load_graph('Chengdu_Network.json')
        >>> node = nodes[42]
        >>> print(node.position, node.neighbors)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw: Dict[str, dict] = json.load(f)

    if not raw:
        return [None]

    # 确定最大节点 ID，以便预分配定长列表（比动态扩展更快）
    max_id = max(int(k) for k in raw.keys())

    # index 0 用 None 占位，使 nodes[id] 直接对应真实节点
    nodes: List[Optional[Node]] = [None] * (max_id + 1)

    for val in raw.values():
        node_id: int = val['id']
        nodes[node_id] = Node(
            node_id  = node_id,
            position = (val['position'][0], val['position'][1]),
            degree   = val['degree'],
            neighbors= val['neighbors'],   # 原样保留列表，元素为 dict
        )

    return nodes


# ---------------------------------------------------------------------------
# 快速验证（直接运行此文件时执行）
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os, time

    json_path = os.path.join(os.path.dirname(__file__), 'data\json_networks\Chengdu_Network.json')
    t0 = time.perf_counter()
    nodes = load_graph(json_path)
    elapsed = time.perf_counter() - t0

    total  = sum(1 for n in nodes if n is not None)
    total_edges = sum(n.degree for n in nodes if n is not None)

    print(f"加载完成，耗时 {elapsed:.3f}s")
    print(f"节点总数: {total}")
    print(f"有向边总数（度之和）: {total_edges}，无向边数: {total_edges // 2}")
    print(f"数组长度（含占位）: {len(nodes)}")
    print(f"示例 nodes[1]  : {nodes[1]}")
    print(f"示例 nodes[100]: {nodes[100]}")
