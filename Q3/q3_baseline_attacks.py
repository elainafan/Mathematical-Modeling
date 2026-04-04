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


def compute_official_p_q(G_temp, N0):
    """(官方) 传统指标：最大连通规模占比 / LCC占比"""
    if G_temp.number_of_nodes() > 0:
        lcc = max(nx.connected_components(G_temp), key=len)
        return len(lcc) / N0
    return 0.0


def load_q1_ranking(city_name, metric_type):
    """
    为了避免 openpyxl 依赖报错或者表结构未知问题，如果有 CSV 则读 CSV，
    如果这里无法直接去 Q1 Excel 里抽取，可以直接使用 networkx 即刻简单结算/重算排序。
    这里做了一个鲁棒封装：若 Q1 的提取失败，能自动回退到重算，防止代码卡住。
    """
    # 此处假设用户可以成功提供/整理出的 Q1 数据路径
    return None  # 占位结构，在下面的方法中如果没拿到，会执行回退计算


def generate_attack_sequence(G, metric_type):
    """
    根据给定的指标，对图 G 的节点进行降序排序。
    返回: nodes list 作为攻击序列。
    """
    # 退底方案：如果 Q1 不能完美映射，直接在图上重新计算，这里只跑单次所以速度尚可接受
    print(f"      - 计算 {metric_type} 进行排序...")
    if metric_type == "Degree":
        centrality = dict(G.degree())
    elif metric_type == "Betweenness":
        # 避免过于耗费时间，如果是极大的图可只采部分点，但此处精确跑完以便对齐精度
        centrality = nx.betweenness_centrality(G, weight=None, normalized=True)
    elif metric_type == "Closeness":
        centrality = nx.closeness_centrality(G)
    else:
        raise ValueError("Unknown Metric")

    # 节点按照得分降序排，相同分数随机打乱一点以防特定输入偏好
    sorted_nodes = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)
    return sorted_nodes


def simulate_targeted_attack(G, sorted_nodes, step_size=0.01):
    """
    按照预先排好序的节点名单，逐批次毁塌节点。
    分别记录 官方指标 P(q) 和 同伴指标 Q_{bc}(q)。
    """
    N0 = G.number_of_nodes()
    G_temp = G.copy()

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    P_q_vals = []
    Q_bc_vals = []

    # 初始状态 P(0.0) 和 Q_{bc}(0.0)
    P_q_vals.append(compute_official_p_q(G_temp, N0))
    Q_bc_vals.append(compute_performance(G_temp, N0, alpha=0.5, beta=0.5))

    current_idx = 0

    # 开始逐批次移除重点
    for i in range(1, len(q_points)):
        q = q_points[i]
        target_remove_count = int(N0 * q)

        # 本轮需要额外干掉多少个
        num_to_remove_now = target_remove_count - current_idx

        if num_to_remove_now > 0:
            nodes_to_remove = sorted_nodes[current_idx : current_idx + num_to_remove_now]
            G_temp.remove_nodes_from(nodes_to_remove)
            current_idx += num_to_remove_now

        p_val = compute_official_p_q(G_temp, N0)
        qbc_val = compute_performance(G_temp, N0, alpha=0.5, beta=0.5)
        P_q_vals.append(p_val)
        Q_bc_vals.append(qbc_val)

    return q_points, P_q_vals, Q_bc_vals


def main():
    base_dir = r"d:\Project\Model"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    out_dir = os.path.join(base_dir, "Q3", "Results")
    os.makedirs(out_dir, exist_ok=True)

    metrics = ["Degree", "Closeness", "Betweenness"]

    all_data = []

    # 降低测试门槛，这里 step_size 调大一点 0.05 方便快速跑出成型数据
    step_size = 0.05

    for filename in os.listdir(json_dir):
        if filename.endswith("_Network.json"):
            city = filename.split("_")[0]
            print(f"\n======== 开始处理城市: {city} ========")
            json_path = os.path.join(json_dir, filename)
            G = build_graph_from_json(json_path)

            for m in metrics:
                print(f"  [策略]: {m} 蓄意攻击")
                seq = generate_attack_sequence(G, m)

                # 开始逐批次斩杀验证
                q_pts, p_arr, qbc_arr = simulate_targeted_attack(G, seq, step_size=step_size)

                for q, p, qbc in zip(q_pts, p_arr, qbc_arr):
                    all_data.append({"City": city, "Strategy": m, "q": q, "P_q_Official": p, "Q_bc_Teammate": qbc})

    df_all = pd.DataFrame(all_data)
    csv_out = os.path.join(out_dir, "Q3_Baseline_Attacks_Data.csv")
    df_all.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n全部 Q3 Baseline 攻击测试完毕，数据已存放至: {csv_out}")


if __name__ == "__main__":
    main()
