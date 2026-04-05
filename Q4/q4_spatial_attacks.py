import igraph as ig
import numpy as np
from scipy.spatial import cKDTree
import copy
from Q4.utils_spatial import build_kdtree_from_graph, get_nodes_in_radius


def get_spatial_attack_sequence_degree(G_original: ig.Graph, radius: float, max_steps=None):
    """
    空间度中心性攻击 (S-Degree):
    动态计算方式：在当前存活且有坐标的网络节点中，挑选一个“靶心”，
    使其能一口气炸掉的（即本身及半径内所有的）现存节点初始度数之和最大，
    以此来实现“炸烂最多连边区域”的直观目标。

    返回:
        target_centers: List[int] 每次引爆的靶心 node_id
    """
    G = G_original.copy()
    # 为原图所有节点计算原始度数，挂载到属性里，以免被删之后度数变0
    G.vs["init_deg"] = G.degree()

    target_centers = []

    # 建立一次全量树（如果坐标不移动，全量树可以复用，只是返回的某些点可能在图中已被剔除）
    master_tree, valid_idxs = build_kdtree_from_graph(G)
    valid_idxs_arr = np.array(valid_idxs)

    # 用一个 bool array 记录哪些 node_ids 还活着
    alive_v = np.ones(G.vcount(), dtype=bool)

    # 预计算每个圆盘内部的点（邻居），其实也是一次 O(N logN)
    # 这比每次 while 循环里查快很多
    disk_neighbors = master_tree.query_ball_point(master_tree.data, r=radius)

    init_degs = np.array(G.vs["init_deg"])
    node_ids = np.array(G.vs["node_id"])

    while alive_v.any():
        if max_steps and len(target_centers) >= max_steps:
            break

        # 我们要找一个当前 alive 的点作为靶心，它的存活 disk_neighbors 的 init_deg 之和最高
        best_center_idx = -1
        best_score = -1.0

        # 为了加速，我们可以不再每一轮 O(N) 搜索，直接 O(N) 搜
        # 用 numpy 的数组操作来加速打分计算
        for i in range(len(master_tree.data)):
            if not alive_v[valid_idxs_arr[i]]:
                continue  # 该靶心已被炸飞

            # 此靶心覆盖的所有点
            nbs = disk_neighbors[i]
            # 筛出活着的
            alive_nbs = [valid_idxs_arr[nb] for nb in nbs if alive_v[valid_idxs_arr[nb]]]

            # 简单 S-Degree 权重就是存活者的度数之和（也可以只是存活节点数量）
            score = np.sum(init_degs[alive_nbs])

            if score > best_score:
                best_score = score
                best_center_idx = i

        if best_center_idx == -1:
            break

        # 执行爆破
        real_idx = valid_idxs_arr[best_center_idx]
        target_centers.append(node_ids[real_idx])

        # 批量抹除受灾节点
        blast_nbs = [valid_idxs_arr[nb] for nb in disk_neighbors[best_center_idx]]
        alive_v[blast_nbs] = False

    return target_centers


def get_spatial_attack_sequence_corehd(G_original: ig.Graph, radius: float, max_steps=None):
    """
    空间 2-Core 骨架动态切除 (S-CoreHD): [前沿创新 SOTA]
    与 S-Degree 的解耦体现在：每次挑选靶心时，依据的不是静态拓扑连边数，
    而是每次被炸后【实时更新、动态残余的网络】中的核心闭环构件 (2-Core)。
    它要寻找的，是一个圆盘波及区域内含有【最多冗余 2-Core 节点】的存活节点。
    """
    G = G_original.copy()
    target_centers = []

    master_tree, valid_idxs = build_kdtree_from_graph(G)
    valid_idxs_arr = np.array(valid_idxs)
    disk_neighbors = master_tree.query_ball_point(master_tree.data, r=radius)

    node_ids = np.array(G.vs["node_id"])
    alive_v = np.ones(G.vcount(), dtype=bool)

    while alive_v.any():
        if max_steps and len(target_centers) >= max_steps:
            break

        # 1. 提取当前存活子图并极速计算 k-core
        alive_indices = np.where(alive_v)[0]
        if len(alive_indices) == 0:
            break

        # subgraph 提取动态现存网络
        subG = G.induced_subgraph(alive_indices)
        if subG.vcount() == 0:
            break

        # subG.coreness() 会返回各个节点到底属于几Core
        core_vals = np.array(subG.coreness())
        # Coreness >= 2 代表构成冗余备选桥梁循环，即骨架 (2-Core)
        is_2core_subG = core_vals >= 2

        # 映射回原图的 bool 标记阵列
        is_2core = np.zeros(G.vcount(), dtype=bool)
        is_2core[alive_indices] = is_2core_subG

        # 若网络已被摧残成纯树状分支（再无2-core），则自动降级为保底模式：打存活点最多的区域
        fallback_mode = not is_2core.any()

        best_center_idx = -1
        best_score = -1.0

        # 2. 从存活的靶心中选定轰炸价值最大（包裹最多 2-Core 点）的坐标
        for i in range(len(master_tree.data)):
            real_idx = valid_idxs_arr[i]
            if not alive_v[real_idx]:
                continue

            nbs_local = disk_neighbors[i]
            alive_nbs_real_idx = valid_idxs_arr[nbs_local]

            if fallback_mode:
                # 骨架完全断裂后，看一发导弹波及了多少存活点
                score = np.sum(alive_v[alive_nbs_real_idx])
            else:
                # S-CoreHD 核心判断：看这发导弹能崩碎多少真正维系活命的 2-Core 骨干！
                score = np.sum(is_2core[alive_nbs_real_idx])

            if score > best_score:
                best_score = score
                best_center_idx = i

        if best_center_idx == -1:
            break

        # 3. 选定、记录、毁伤结算
        chosen_real_idx = valid_idxs_arr[best_center_idx]
        target_centers.append(node_ids[chosen_real_idx])

        # 批量抹除该靶心以及半径内波及的所有节点 (连片阵亡)
        blast_nbs = valid_idxs_arr[disk_neighbors[best_center_idx]]
        alive_v[blast_nbs] = False

    return target_centers
