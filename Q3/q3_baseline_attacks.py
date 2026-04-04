import os
import json
import networkx as nx
import pandas as pd
import numpy as np

# 导入同伴的 utils (用于计算新的 Q_{bc} 指标)
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import compute_performance
except ImportError:
    print("Warning: utils.py Not Found, using baseline only")


# =====================================================================
# 模块一：健壮度计算函数 (两种评价标准解耦)
# =====================================================================
def eval_official_p_q(G_temp, N0):
    """健壮度 1：(赛题原生) 最大连通规模占比 LCC/N0"""
    if G_temp.number_of_nodes() > 0:
        lcc = max(nx.connected_components(G_temp), key=len)
        return len(lcc) / N0
    return 0.0


def eval_teammate_q_bc(G_temp, N0):
    """健壮度 2：(同伴优化) 双连通核结构健壮性 Q_{bc}(f)"""
    return compute_performance(G_temp, N0, alpha=0.5, beta=0.5)


# =====================================================================
# 模块二：攻击序列生成函数 (三种中心性排序维度解耦)
# =====================================================================
def get_attack_sequence_degree(G, city_name=None):
    """策略 1：度中心性攻击序列"""
    # Degree 计算复杂度极低 O(V+E)，直接秒出，无需读文件
    centrality = dict(G.degree())
    return sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)


def get_attack_sequence_betweenness(G, city_name):
    """策略 2：介数中心性攻击序列 (从 Q1 导出的 CSV 光速读取)"""
    csv_path = os.path.join(r"D:\Project\Model\Q1\betweenness", f"betweenness_{city_name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # 表头结构中有 "节点ID" 和 "排名"
        return df.sort_values(by="排名")["节点ID"].tolist()
    else:
        print(f"  [警告] 未找到 {city_name} 的介数中心性 CSV，被迫重新进行超漫长的计算...")
        centrality = nx.betweenness_centrality(G, weight=None, normalized=True)
        return sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)


def get_attack_sequence_closeness(G, city_name):
    """策略 3：接近度中心性攻击序列 (从 Q1 导出的 CSV 光速读取)"""
    csv_path = os.path.join(r"D:\Project\Model\Q1\closeness", f"closeness_{city_name}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.sort_values(by="排名")["节点ID"].tolist()
    else:
        print(f"  [警告] 未找到 {city_name} 的接近度 CSV，被迫进行重算...")
        centrality = nx.closeness_centrality(G)
        return sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)


# =====================================================================
# 模块三：泛化仿真引擎 (接收独立攻击序列，调用独立评估函数)
# =====================================================================
def build_graph_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.Graph()
    for node_str, info in data.items():
        u = int(info["id"])
        G.add_node(u)
        for neighbor in info.get("neighbors", []):
            v = int(neighbor["id"])
            G.add_edge(u, v)
    return G


def simulate_targeted_attack(G, sorted_nodes, step_size=0.01):
    """
    通用毁伤引擎：按照给定名单分批拆除，并同时追踪多重健壮度
    """
    N0 = G.number_of_nodes()
    G_temp = G.copy()

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    P_q_vals = []
    Q_bc_vals = []

    # 存入初始分 (q=0)
    P_q_vals.append(eval_official_p_q(G_temp, N0))
    Q_bc_vals.append(eval_teammate_q_bc(G_temp, N0))

    current_idx = 0

    for i in range(1, len(q_points)):
        q = q_points[i]
        target_remove_count = int(N0 * q)
        num_to_remove_now = target_remove_count - current_idx

        if num_to_remove_now > 0:
            # 兼容性处理：防止越界
            nodes_to_remove = sorted_nodes[current_idx : min(current_idx + num_to_remove_now, len(sorted_nodes))]
            G_temp.remove_nodes_from(nodes_to_remove)
            current_idx += num_to_remove_now

        # 触发独立的评价函数算分
        P_q_vals.append(eval_official_p_q(G_temp, N0))
        Q_bc_vals.append(eval_teammate_q_bc(G_temp, N0))

    return q_points, P_q_vals, Q_bc_vals


def main():
    base_dir = r"d:\Project\Model"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    out_dir = os.path.join(base_dir, "Q3", "Results")
    os.makedirs(out_dir, exist_ok=True)

    # 将"获得序列"与"算法名"解耦映射
    strategy_map = {
        "Degree": get_attack_sequence_degree,
        "Betweenness": get_attack_sequence_betweenness,
        "Closeness": get_attack_sequence_closeness,
    }

    all_data = []
    step_size = 0.05  # 设置细粒度破坏批次频率

    for filename in os.listdir(json_dir):
        if filename.endswith("_Network.json"):
            city = filename.split("_")[0]
            print(f"\n======== 开始处理靶标城市: {city} ========")
            json_path = os.path.join(json_dir, filename)
            G = build_graph_from_json(json_path)

            for strategy_name, sequence_func in strategy_map.items():
                print(f"  -> 装填【{strategy_name}】针对性破袭名单...")

                # 如果生成函数需要城市名来读表，则传入城市名
                if strategy_name == "Degree":
                    seq = sequence_func(G)
                else:
                    seq = sequence_func(G, city)

                print(f"    - 执行滑动窗口接力处决 (跟踪LCC与Q_bc)...")
                q_pts, p_arr, qbc_arr = simulate_targeted_attack(G, seq, step_size=step_size)

                for q, p, qbc in zip(q_pts, p_arr, qbc_arr):
                    all_data.append(
                        {"City": city, "Strategy": strategy_name, "q": q, "P_q_Official": p, "Q_bc_Teammate": qbc}
                    )

    df_all = pd.DataFrame(all_data)
    csv_out = os.path.join(out_dir, "Q3_Baseline_Attacks_Data.csv")
    df_all.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n全部 Q3 Baseline 降维打击测试完毕，多维数据已存放至: {csv_out}")


if __name__ == "__main__":
    main()
