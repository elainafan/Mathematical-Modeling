import os
import json
import igraph as ig
import pandas as pd
import numpy as np
import sys

# 导入上一级目录的 utils_ig (用于计算新的 Q_{bc} 指标)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import load_graph, rebuild_id2idx, compute_performance
except ImportError:
    print("Warning: utils_ig.py Not Found")
    sys.exit(1)


# =====================================================================
# 模块一：健壮度计算函数 (仅解耦原生 P_q 指标)
# =====================================================================
def eval_official_p_q(G_temp, N0):
    """健壮度 1：(赛题原生) 最大连通规模占比 LCC/N0"""
    if G_temp.vcount() > 0:
        lcc_size = max(G_temp.connected_components().sizes())
        return lcc_size / N0
    return 0.0


# =====================================================================
# 模块二：攻击序列生成函数 (三种中心性排序维度解耦)
# =====================================================================
def get_attack_sequence_degree(G, city_name=None):
    """策略 1：度中心性攻击序列"""
    # igraph 中计算度极快
    degrees = G.degree()
    node_ids = G.vs["node_id"]
    deg_dict = {nid: deg for nid, deg in zip(node_ids, degrees)}
    return sorted(deg_dict.keys(), key=lambda n: deg_dict[n], reverse=True)


def get_attack_sequence_betweenness(G, city_name):
    """策略 2：介数中心性攻击序列 (从 Q1 导出的 CSV 光速读取)"""
    csv_path = os.path.join(r"D:\Project\Model\Q1\betweenness", f"betweenness_{city_name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.sort_values(by="排名")["节点ID"].tolist()
    else:
        print(f"  [警告] 未找到 {city_name} 的介数中心性 CSV，被迫重新进行超漫长的计算...")
        betweenness = G.betweenness()
        node_ids = G.vs["node_id"]
        bet_dict = {nid: bet for nid, bet in zip(node_ids, betweenness)}
        return sorted(bet_dict.keys(), key=lambda n: bet_dict[n], reverse=True)


def get_attack_sequence_closeness(G, city_name):
    """策略 3：接近度中心性攻击序列 (从 Q1 导出的 CSV 光速读取)"""
    csv_path = os.path.join(r"D:\Project\Model\Q1\closeness", f"closeness_{city_name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.sort_values(by="排名")["节点ID"].tolist()
    else:
        print(f"  [警告] 未找到 {city_name} 的接近度 CSV，被迫进行重算...")
        closeness = G.closeness()
        node_ids = G.vs["node_id"]
        clo_dict = {nid: clo for nid, clo in zip(node_ids, closeness)}
        return sorted(clo_dict.keys(), key=lambda n: clo_dict[n], reverse=True)


# =====================================================================
# 模块三：泛化仿真引擎 (接收独立攻击序列，调用独立评估函数)
# =====================================================================
def simulate_targeted_attack(G, id2idx, sorted_nodes, step_size=0.01, city_name="", attack_type=""):
    """
    通用毁伤引擎：按照给定名单分批拆除，仅追踪LCC
    """
    N0 = G.vcount()
    G_temp = G.copy()
    current_id2idx = id2idx.copy()

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    P_q_vals = []

    # 存入初始分 (q=0)
    P_q_vals.append(eval_official_p_q(G_temp, N0))

    current_idx = 0
    reached_critical = False

    for i in range(1, len(q_points)):
        q = q_points[i]
        target_remove_count = int(N0 * q)
        num_to_remove_now = target_remove_count - current_idx

        nodes_to_remove = []
        if num_to_remove_now > 0:
            nodes_to_remove = sorted_nodes[current_idx : min(current_idx + num_to_remove_now, len(sorted_nodes))]

            # 必须用映射将原节点ID转换为 igraph 当前内部连续索引！
            indices_to_remove = [current_id2idx[nid] for nid in nodes_to_remove if nid in current_id2idx]

            G_temp.delete_vertices(indices_to_remove)
            # ibraph 删除后，所有索引重新排布，必须马上刷新图层映射！
            current_id2idx = rebuild_id2idx(G_temp)
            current_idx += num_to_remove_now

        # 触发独立评价函数
        p_val = eval_official_p_q(G_temp, N0)
        P_q_vals.append(p_val)

        # 记录 LCC 首次跌破 1% 的临界批次
        if not reached_critical and p_val <= 0.01:
            reached_critical = True
            print(f"      [网络崩溃] q={q:.2f} 时最大连通分量(LCC)降至 {p_val:.4f} (<= 1%)！")
            total_removed = len(sorted_nodes[:current_idx])
            print(f"      [临界信息] 此时总计删除了 {total_removed} 个节点。")
            print(f"      [致命批次] 这是压死骆驼的最后一批节点ID(前5个): {nodes_to_remove[:5]} ...")
            import pandas as pd
            import os

            record = pd.DataFrame(
                [
                    {
                        "City": city_name,
                        "Attack_Metric": attack_type,
                        "Critical_q": f"{q:.2f}",
                        "Total_Removed": total_removed,
                        "P_q": f"{p_val:.4f}",
                        "Fatal_Nodes_Array": str(nodes_to_remove),
                    }
                ]
            )
            f_csv = "d:/Project/Model/Q3/Results/Fatal_Nodes_Official_Pq.csv"
            record.to_csv(f_csv, mode="a", header=not os.path.exists(f_csv), index=False, encoding="utf-8-sig")

    return q_points, P_q_vals


def main():
    base_dir = r"d:\Project\Model"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    out_dir = os.path.join(base_dir, "Q3", "Results")
    os.makedirs(out_dir, exist_ok=True)

    strategy_map = {
        "Degree": get_attack_sequence_degree,
        "Betweenness": get_attack_sequence_betweenness,
        "Closeness": get_attack_sequence_closeness,
    }

    all_data = []
    step_size = 0.01  # 设置细粒度破坏批次频率 (1% 即 0.01)

    for filename in os.listdir(json_dir):
        if filename.endswith("_Network.json"):
            city = filename.split("_")[0]
            print(f"\n======== 开始处理靶标城市 (igraph加速): {city} ========")
            json_path = os.path.join(json_dir, filename)

            # 使用 igraph 重构版的拉取逻辑
            G, id2idx, idx2id = load_graph(json_path)

            for strategy_name, sequence_func in strategy_map.items():
                print(f"  -> 装填【{strategy_name}】针对性破袭名单...")

                if strategy_name == "Degree":
                    seq = sequence_func(G)
                else:
                    seq = sequence_func(G, city)

                print("    - 执行滑动窗口接力处决 (仅跟踪 native LCC)...")
                q_pts, p_arr = simulate_targeted_attack(
                    G, id2idx, seq, step_size=step_size, city_name=city, attack_type=strategy_name
                )

                for q, p in zip(q_pts, p_arr):
                    all_data.append({"City": city, "Strategy": strategy_name, "q": q, "P_q_Official": p})

    df_all = pd.DataFrame(all_data)
    csv_out = os.path.join(out_dir, "Q3_Baseline_Attacks_Data_IG_Official.csv")
    df_all.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n全部 Q3 Baseline 降维打击测试(仅原生指标)完毕，数据已存放至: {csv_out}")


if __name__ == "__main__":
    main()
