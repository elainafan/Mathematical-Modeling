import os
import json
import igraph as ig
import pandas as pd
import numpy as np
import sys

# 导入上一级目录的 utils_ig (用于加载图以及队友独立健壮度计算)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import load_graph, rebuild_id2idx, compute_performance
except ImportError:
    print("Warning: utils_ig.py Not Found")
    sys.exit(1)


# =====================================================================
# 评价函数 (官方 LCC P(q) 遗及 队友双连通 Q_bc)
# =====================================================================
def eval_official_p_q(G_temp, N0):
    if G_temp.vcount() == 0:
        return 0.0
    connected_components = G_temp.connected_components()
    max_comp_size = max(connected_components.sizes()) if len(connected_components) > 0 else 0
    return max_comp_size / N0


def eval_teammate_q_bc(G_temp, N0):
    return compute_performance(G_temp, N0, alpha=0.5, beta=0.5)


# =====================================================================
# 进阶算法生成序列 (CI & CoreHD)
# =====================================================================
def get_attack_sequence_ci(G, city_name=None, l=2):
    """
    计算 Collective Influence 攻击序列 (静态影响圈预测)
    """
    ci_scores = {}
    degrees = G.degree()
    # 保证使用准确的原始节点ID
    name_attr = "node_id" if "node_id" in G.vs.attributes() else ("name" if "name" in G.vs.attributes() else None)

    for i, v in enumerate(G.vs):
        vid = v[name_attr] if name_attr else i
        ci_scores[vid] = 0
        if degrees[i] <= 1:
            continue

        # 获取距离恰好为 l 的边界层
        try:
            neigh_l = set(G.neighborhood(i, order=l))
            neigh_l_minus_1 = set(G.neighborhood(i, order=l - 1))
        except Exception:
            neigh_l = set([i])
            neigh_l_minus_1 = set([i])

        frontier = neigh_l - neigh_l_minus_1

        sum_kj = sum((degrees[j] - 1) for j in frontier)
        ci_scores[vid] = (degrees[i] - 1) * sum_kj

    # 基于 CI 值降序排列作为“斩首名单”
    return sorted(ci_scores.keys(), key=lambda k: ci_scores[k], reverse=True)


def get_attack_sequence_corehd(G, city_name=None):
    """
    计算 CoreHD 动态 2-Core 网络核最大度剥离降维序列
    """
    g_temp = G.copy()
    name_attr = "node_id" if "node_id" in G.vs.attributes() else ("name" if "name" in G.vs.attributes() else None)
    if not name_attr:
        g_temp.vs["node_id"] = [v.index for v in g_temp.vs]
        name_attr = "node_id"

    seq_front = []  # 存放在前排处决的 2-Core 最大度骨架枢纽
    seq_back = []  # 存在放在队尾忽略的末梢节点

    while g_temp.vcount() > 0:
        degrees = g_temp.degree()
        # 1. 递归剥洋葱：抽出度不足 2 的树干末梢（叶子）
        leaves = [i for i, d in enumerate(degrees) if d < 2]

        if leaves:
            leaf_names = [g_temp.vs[i][name_attr] for i in leaves]
            seq_back.extend(leaf_names)
            g_temp.delete_vertices(leaves)
            continue

        # 2. 定点爆破骨架：在纯净的 2-Core 结构里面寻找连接数最多的点摧毁
        degrees = g_temp.degree()
        max_deg = max(degrees)
        max_idx = degrees.index(max_deg)

        seq_front.append(g_temp.vs[max_idx][name_attr])
        g_temp.delete_vertices(max_idx)

    # 最外围率先剥离的树干，被保留到最后才攻击
    seq_back.reverse()
    return seq_front + seq_back


# =====================================================================
# 通用滑动窗口仿真引擎
# =====================================================================
def simulate_targeted_attack(G, id2idx, sorted_nodes, step_size=0.01, city_name="", attack_type="", metric="P_q"):
    N0 = G.vcount()
    G_temp = G.copy()
    current_id2idx = id2idx.copy()

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    vals = []

    if metric == "P_q":
        vals.append(eval_official_p_q(G_temp, N0))
    else:
        vals.append(eval_teammate_q_bc(G_temp, N0))

    current_idx = 0
    reached_critical = False

    for i in range(1, len(q_points)):
        q = q_points[i]
        target_remove_count = int(N0 * q)
        num_to_remove_now = target_remove_count - current_idx

        nodes_to_remove = []
        if num_to_remove_now > 0:
            nodes_to_remove = sorted_nodes[current_idx : min(current_idx + num_to_remove_now, len(sorted_nodes))]

            indices_to_remove = [current_id2idx[nid] for nid in nodes_to_remove if nid in current_id2idx]

            G_temp.delete_vertices(indices_to_remove)
            current_id2idx = rebuild_id2idx(G_temp)
            current_idx += num_to_remove_now

        if metric == "P_q":
            val = eval_official_p_q(G_temp, N0)
        else:
            val = eval_teammate_q_bc(G_temp, N0)
        vals.append(val)

        # 记录首次跌破 1% 的临界批次
        if not reached_critical and val <= 0.01:
            reached_critical = True
            print(f"      [{metric} 崩溃] q={q:.2f} 时指标降至 {val:.4f} (<= 1%)！")
            total_removed = len(sorted_nodes[:current_idx])
            print(f"      [临界信息] 仅用 {total_removed} 个节点摧毁防线！致命批次(前5): {nodes_to_remove[:5]} ...")

            import os

            record = pd.DataFrame(
                [
                    {
                        "City": city_name,
                        "Attack_Metric": attack_type,
                        "Critical_q": f"{q:.2f}",
                        "Total_Removed": total_removed,
                        metric: f"{val:.4f}",
                        "Fatal_Nodes_Array": str(nodes_to_remove),
                    }
                ]
            )
            f_csv = f"d:/Project/Model/Q3/Results/Fatal_Nodes_Advanced_{metric}.csv"
            record.to_csv(f_csv, mode="a", header=not os.path.exists(f_csv), index=False, encoding="utf-8-sig")

    return q_points, vals


def main():
    base_dir = r"d:\Project\Model"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    out_dir = os.path.join(base_dir, "Q3", "Results")
    os.makedirs(out_dir, exist_ok=True)

    strategy_map = {"CI_Radius2": get_attack_sequence_ci, "CoreHD": get_attack_sequence_corehd}

    step_size = 0.01
    all_data_pq = []
    all_data_qbc = []

    for filename in os.listdir(json_dir):
        if filename.endswith("_Network.json"):
            city = filename.split("_")[0]
            print(f"\n======== 开始【高级智能攻击导弹】(CI & CoreHD): {city} ========")
            json_path = os.path.join(json_dir, filename)

            G, id2idx, idx2id = load_graph(json_path)

            for strategy_name, sequence_func in strategy_map.items():
                print(f"  -> {strategy_name} 高维演化排序中...")
                seq = sequence_func(G, city)

                print("    - 对 传统大连通指标 P(q) 执行毁伤测试...")
                q_pts, p_arr = simulate_targeted_attack(
                    G, id2idx, seq, step_size=step_size, city_name=city, attack_type=strategy_name, metric="P_q"
                )
                for q, p in zip(q_pts, p_arr):
                    all_data_pq.append({"City": city, "Strategy": strategy_name, "q": q, "P_q_Official": p})

                print("    - 对 队友双连通指标 Q_bc 执行毁伤测试...")
                q_pts, qbc_arr = simulate_targeted_attack(
                    G, id2idx, seq, step_size=step_size, city_name=city, attack_type=strategy_name, metric="Q_bc"
                )
                for q, qbc in zip(q_pts, qbc_arr):
                    all_data_qbc.append({"City": city, "Strategy": strategy_name, "q": q, "Q_bc_Teammate": qbc})

    df_pq = pd.DataFrame(all_data_pq)
    df_pq.to_csv(os.path.join(out_dir, "Q3_Advanced_Attacks_Data_IG_Official.csv"), index=False, encoding="utf-8-sig")

    df_qbc = pd.DataFrame(all_data_qbc)
    df_qbc.to_csv(os.path.join(out_dir, "Q3_Advanced_Attacks_Data_IG_Teammate.csv"), index=False, encoding="utf-8-sig")
    print("\n======== ✅ 所有进阶算法(CI & CoreHD) 双维指标测算已生成完毕！ ========")


if __name__ == "__main__":
    main()
