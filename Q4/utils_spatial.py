import pandas as pd
import igraph as ig
import numpy as np
from scipy.spatial import cKDTree
import sys
import os

# 将父目录加入路径，以便复用以前写的 utils_ig 中已被验证的计算鲁棒性模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import compute_performance
except ImportError:
    pass  # 如果找不到，后续可以自己重新实现


def load_spatial_graph_from_csv(csv_path: str):
    """
    从带有坐标的边表 (Edgelist) CSV 构建含空间属性的 igraph 实例
    这实现了数据加载的解耦。

    返回:
        G (ig.Graph): 带有 node_id, pos, weight 的图形对象
        id2idx, idx2id (dict): 真实的结点 ID 和 igraph 内部索引的映射
        kdtree (cKDTree): 基于目前所有节点空间坐标构建的 KDTree 查找树
    """
    df = pd.read_csv(csv_path)

    # 提取唯一的节点坐标 (X, Y 对应的是 START_NODE 的坐标)
    nodes_df = df[["START_NODE", "XCoord", "YCoord"]].drop_duplicates(subset=["START_NODE"])
    nodes_df = nodes_df.sort_values(by="START_NODE").reset_index(drop=True)

    node_ids = nodes_df["START_NODE"].tolist()
    x_coords = nodes_df["XCoord"].tolist()
    y_coords = nodes_df["YCoord"].tolist()

    # 构建双向映射
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    idx2id = {i: nid for i, nid in enumerate(node_ids)}

    # 构建边信息 (无向图，去重)
    # pandas iterrows 会比转化为 numpy 慢，这里直接转换为 numpy 操作
    u_vals = df["START_NODE"].map(id2idx).values
    v_vals = df["END_NODE"].map(id2idx).values
    weights = df["LENGTH"].values

    # 过滤掉无法映射（即没出现在 START_NODE）以及自环
    valid_mask = (~np.isnan(u_vals)) & (~np.isnan(v_vals)) & (u_vals != v_vals)
    u_vals = u_vals[valid_mask].astype(int)
    v_vals = v_vals[valid_mask].astype(int)
    weights = weights[valid_mask]

    # 构建图
    G = ig.Graph(n=len(node_ids), edges=list(zip(u_vals, v_vals)), directed=False)
    G.vs["node_id"] = node_ids
    G.vs["pos"] = list(zip(x_coords, y_coords))
    G.es["weight"] = weights

    # 清理多重边 (合并权重/取最小等均可，这里取 first 以加速)
    G.simplify(multiple=True, loops=True, combine_edges="first")

    return G, id2idx, idx2id


def build_kdtree_from_graph(G: ig.Graph):
    """
    从 igraph 实例提取坐标并构建 cKDTree，以供 O(logN) 级别范围查找。
    实现空间查找逻辑的解耦。
    返回值:
        tree (cKDTree): scipy KDTree 对象
        valid_idxs (list): KDTree中索引对应的 igraph idx
    """
    coords = []
    valid_idxs = []
    for i in range(G.vcount()):
        pos = G.vs[i]["pos"]
        if pos and not np.isnan(pos[0]):
            coords.append([pos[0], pos[1]])
            valid_idxs.append(i)

    tree = cKDTree(coords)
    return tree, valid_idxs


def get_nodes_in_radius(tree: cKDTree, valid_idxs: list, center_coords: tuple, radius: float):
    """
    寻找指定经纬度(或者平面坐标)以及半径范围内的所有节点 idx
    """
    # tree.query_ball_point 返回的是 kdtree 内的 indices
    indices = tree.query_ball_point(center_coords, r=radius)
    # 转换为 igraph 的内部连续 idx
    return [valid_idxs[i] for i in indices]
