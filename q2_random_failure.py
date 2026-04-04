import os
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# 设置中文字体，防止图表中文字符显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def build_graph_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    G = nx.Graph()
    for node_str, info in data.items():
        u = int(info['id'])
        G.add_node(u)
        for neighbor in info.get('neighbors', []):
            v = int(neighbor['id'])
            G.add_edge(u, v)
    return G

def simulate_random_failures(G, num_simulations=10, step_size=0.01):
    initial_nodes = list(G.nodes())
    N0 = len(initial_nodes)

    q_points = np.arange(0.0, 1.0 + step_size/2, step_size)
    P_q_avg = np.zeros(len(q_points))

    initial_nodes_set = set(initial_nodes)

    for i, q in enumerate(q_points):
        target_remove_count = int(N0 * q)
        P_q_sum = 0.0

        for _ in range(num_simulations):
            nodes_to_remove = random.sample(initial_nodes, target_remove_count)
            nodes_to_keep = initial_nodes_set.difference(nodes_to_remove)
            G_temp = G.subgraph(nodes_to_keep)

            if G_temp.number_of_nodes() > 0:
                largest_cc_size = len(max(nx.connected_components(G_temp), key=len))
                P_q_sum += largest_cc_size / N0

        P_q_avg[i] = P_q_sum / num_simulations
        print(f'    - 完成 q={q:.2f} 的测算')

    return q_points, P_q_avg

def main():
    base_dir = r"d:\Project\Model"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    output_dir = os.path.join(base_dir, "data", "B题数据", "Q2_Results")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for filename in os.listdir(json_dir):
        if filename.endswith("_Network.json"):
            city_name = filename.split("_")[0]
            print(f"开始处理城市: {city_name} ...")

            json_path = os.path.join(json_dir, filename)
            G = build_graph_from_json(json_path)

            q_points, P_q_avg = simulate_random_failures(G, num_simulations=20, step_size=0.01)

            R = np.trapezoid(P_q_avg, x=q_points)
            diffs = np.diff(P_q_avg)
            steepest_drop_idx = np.argmin(diffs)
            q_critical = q_points[steepest_drop_idx]

            print(f"  --> {city_name}: 健壮性R = {R:.4f}, 显著崩溃节点比例 q = {q_critical:.2f}")
            results.append({"城市": city_name, "健壮性积分(R)": R, "网络显著崩溃比例(q)": q_critical})

            plt.figure(figsize=(8, 6))
            plt.plot(q_points, P_q_avg, marker='.', linestyle='-', color='b', linewidth=2)
            plt.axvline(x=q_critical, color='r', linestyle='--', label=f"显著变化点 (q={q_critical:.2f})")
            plt.xlabel("破坏节点比例 (q)")
            plt.ylabel("最大连通分量占比 P(q)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plot_path = os.path.join(output_dir, f"{city_name}_Robustness_Random.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="健壮性积分(R)", ascending=False)
    csv_out = os.path.join(output_dir, "Q2_Robustness_Summary.csv")
    df_results.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f"\n全部处理完毕！汇总数据已保存至: {csv_out}")

if __name__ == "__main__":
    main()