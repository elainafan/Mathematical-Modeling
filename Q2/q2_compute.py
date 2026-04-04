import os
import json
import networkx as nx
import numpy as np
import pandas as pd
import random
import concurrent.futures


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


def calculate_network_metric(G_temp, N0):
    """计算网络性能指标，当前指标为最大连通子图节点数与其初始节点数之比 P(q)"""
    if G_temp.number_of_nodes() > 0:
        largest_cc_size = len(max(nx.connected_components(G_temp), key=len))
        return largest_cc_size / N0
    return 0.0


def _simulate_single_q(args):
    q, G, initial_nodes, N0, num_simulations = args
    initial_nodes_set = set(initial_nodes)
    target_remove_count = int(N0 * q)
    P_q_sum = 0.0

    for _ in range(num_simulations):
        nodes_to_remove = random.sample(initial_nodes, target_remove_count)
        nodes_to_keep = initial_nodes_set.difference(nodes_to_remove)
        G_temp = G.subgraph(nodes_to_keep)

        P_q_sum += calculate_network_metric(G_temp, N0)

    avg = P_q_sum / num_simulations
    print(f"    - 完成横轴 q={q:.2f} 的 {num_simulations} 次独立采样测算")
    return avg


def simulate_random_failures(G, num_simulations=20, step_size=0.01):
    initial_nodes = list(G.nodes())
    N0 = len(initial_nodes)

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    P_q_avg = np.zeros(len(q_points))

    tasks = [(q, G, initial_nodes, N0, num_simulations) for q in q_points]
    print(f"    - 启动多进程并发计算 (按横轴 q 并行，每个 q 计算 {num_simulations} 次)...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_simulate_single_q, tasks))

    for i, res in enumerate(results):
        P_q_avg[i] = res

    return q_points, P_q_avg


def main():
    base_dir = r"d:\Project\Model"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    output_dir = os.path.join(base_dir, "data", "B题数据", "Q2_Results")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_simulation_data = []

    for filename in os.listdir(json_dir):
        if filename.endswith("_Network.json"):
            city_name = filename.split("_")[0]
            print(f"开始处理城市 (仅计算并保存数据): {city_name} ...")

            json_path = os.path.join(json_dir, filename)
            G = build_graph_from_json(json_path)

            step_size = 0.01
            q_points, P_q_avg = simulate_random_failures(G, num_simulations=20, step_size=step_size)

            # 将这个城市的离散数据全部记录下来
            for q, p in zip(q_points, P_q_avg):
                all_simulation_data.append({"City": city_name, "q": q, "P_q_avg": p})

    # 输出为一个大 CSV 进行原始数据归档
    df_all_data = pd.DataFrame(all_simulation_data)
    csv_raw_out = os.path.join(output_dir, "Q2_Raw_Simulation_Data.csv")
    df_all_data.to_csv(csv_raw_out, index=False, encoding="utf-8-sig")
    print(f"\n全部计算完毕！各城市的 P(q) 原始数据已成功解耦并保存至: {csv_raw_out}")
    print("接下来可以运行 q2_plot_results.py 读取该文件直接绘图并计算健壮性积分。")


if __name__ == "__main__":
    main()
