import os
import sys
import numpy as np
import pandas as pd
import igraph as ig

# 确保能找到同级目录和上级目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Q4.utils_spatial import load_spatial_graph_from_csv, build_kdtree_from_graph
from Q4.q4_spatial_attacks import get_spatial_attack_sequence_degree, get_spatial_attack_sequence_corehd


def simulate_spatial_cascade(
    G_original: ig.Graph, id2idx: dict, target_centers: list, radius: float, kdtree, valid_idxs: list
):
    """
    负责执行真实的物理模拟：按给定的靶心顺序，依次引爆并伴随周边半径内的级联瘫痪。
    一直执行直到网络最大连通分量 P(q) 跌落到原生规模的 0.01（即 1%）为止。
    """
    num_init_nodes = G_original.vcount()
    alive_v = np.ones(num_init_nodes, dtype=bool)

    q_list = [0.0]
    P_q_list = [1.0]  # 初始 100% 完整
    valid_idxs_arr = np.array(valid_idxs)

    # 获取原始坐标集序列
    node_coords = np.array([G_original.vs[i]["pos"] for i in range(num_init_nodes)])

    for center_id in target_centers:
        if center_id not in id2idx:
            continue

        center_idx = id2idx[center_id]
        if not alive_v[center_idx]:
            # 已经被之前的连带爆炸波及了，跳过，去打下一个靶子
            continue

        # 寻找爆炸中心给定半径内的所有受灾点索引（包括中心点自身）
        # 这里为了复用 KDTree 极速运算
        center_pos = node_coords[center_idx]
        raw_kdtree_idx = kdtree.query_ball_point(center_pos, r=radius)
        blast_idxs = valid_idxs_arr[raw_kdtree_idx]

        # 挑出里面还活着的，给它们发阵亡通知
        alive_blast_idxs = blast_idxs[alive_v[blast_idxs]]
        if len(alive_blast_idxs) == 0:
            continue

        # 物理割裂
        alive_v[alive_blast_idxs] = False

        # 核心：计算幸存道路网的最大连通分量 P(q)
        alive_indices = np.where(alive_v)[0]
        if len(alive_indices) == 0:
            q_list.append(1.0)
            P_q_list.append(0.0)
            break

        subG = G_original.induced_subgraph(alive_indices)
        if subG.vcount() > 0:
            lcc_size = max(subG.connected_components().sizes())
        else:
            lcc_size = 0

        P_q = lcc_size / num_init_nodes
        q = 1.0 - (len(alive_indices) / num_init_nodes)  # 已瘫痪节点比例

        q_list.append(q)
        P_q_list.append(P_q)

        # Q4 题目判停条件要求：使得当最大连通分量是原始网络规模的 0.01 时
        if P_q <= 0.01:
            break

    # 计算鲁棒性曲线积分 R (曲线下面积)，使用 numpy 梯形积分法则
    robustness_R = np.trapezoid(P_q_list, q_list)
    return q_list, P_q_list, robustness_R


def evaluate_all_cities(radius=2000.0):
    data_dir = os.path.join("data", "B题数据")
    # 待测 8 个城市
    cities = ["Chengdu", "Dalian", "Dongguan", "Harbin", "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou"]
    results_dir = os.path.join("Q4", "Results")

    # 讨论波及半径在多大范围变化

    summary_records = []

    for city in cities:
        print(f"\n[ 正在评估城市: {city} | 波及半径: {radius}m ]")
        csv_path = os.path.join(data_dir, f"{city}_Edgelist.csv")

        G, id2idx, idx2id = load_spatial_graph_from_csv(csv_path)
        kdtree, valid_idxs = build_kdtree_from_graph(G)

        # 1. 模拟空间度攻击 (S-Degree)
        print(f"  --> 计算 S-Degree 绝杀序列...")
        seq_deg = get_spatial_attack_sequence_degree(G, radius)
        q_deg, Pq_deg, R_deg = simulate_spatial_cascade(G, id2idx, seq_deg, radius, kdtree, valid_idxs)

        # 2. 模拟空间前沿算法 (S-CoreHD)
        print(f"  --> 计算 S-CoreHD 绝杀序列...")
        seq_core = get_spatial_attack_sequence_corehd(G, radius)
        q_core, Pq_core, R_core = simulate_spatial_cascade(G, id2idx, seq_core, radius, kdtree, valid_idxs)

        print(f"  [评估完毕] S-Degree 鲁棒性 R = {R_deg:.5f}  |  S-CoreHD 鲁棒性 R = {R_core:.5f}")

        # 保存坠落轨迹数据给画图用
        df_deg = pd.DataFrame({"q": q_deg, "P_q": Pq_deg, "Method": "S-Degree"})
        df_core = pd.DataFrame({"q": q_core, "P_q": Pq_core, "Method": "S-CoreHD"})
        df_combined = pd.concat([df_deg, df_core])
        df_combined.to_csv(os.path.join(results_dir, f"Q4_Simulation_{city}_r{int(radius)}m.csv"), index=False)

        summary_records.append(
            {
                "City": city,
                "Radius_m": radius,
                "R_S_Degree": R_deg,
                "R_S_CoreHD": R_core,
                "Collapse_q_S_Degree": q_deg[-1],
                "Collapse_q_S_CoreHD": q_core[-1],
            }
        )

    summary_df = pd.DataFrame(summary_records)
    # 按 S-CoreHD 最强打击下的抵抗力（健壮性最大）从小到大排列
    summary_df = summary_df.sort_values(by="R_S_CoreHD", ascending=False)
    summary_df.to_csv(os.path.join(results_dir, f"Q4_Robustness_Summary_r{int(radius)}m.csv"), index=False)

    print("\n================ [ Q4 最终兵棋推演榜单 (Radius=2km) ] ================")
    print(summary_df.to_string(index=False))
    print("=====================================================================")


if __name__ == "__main__":
    # 讨论故障波及半径在多大范围变化，结果的变化
    # 补充细化探测：下限到100m, 并加入中间粒度
    test_radii = [100.0, 250.0, 750.0, 1500.0]
    for r in test_radii:
        evaluate_all_cities(radius=r)
