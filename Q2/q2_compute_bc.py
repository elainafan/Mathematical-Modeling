import os
import sys
import json
import networkx as nx
import numpy as np
import pandas as pd
import random
import concurrent.futures

# 引入上一级目录的 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils import compute_performance
except ImportError:
    print("找不到 utils.py，请检查路径。")
    sys.exit(1)


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


def _simulate_single_q_bc(args):
    q, G, initial_nodes, N0, num_simulations = args
    initial_nodes_set = set(initial_nodes)
    target_remove_count = int(N0 * q)
    Q_bc_sum = 0.0

    for _ in range(num_simulations):
        nodes_to_remove = random.sample(initial_nodes, target_remove_count)
        nodes_to_keep = initial_nodes_set.difference(nodes_to_remove)
        G_temp = G.subgraph(nodes_to_keep).copy()  # 建立副本以确保不会在双连通计算时不小心改动

        # 调用同伴定义的基于双连通核的结构健壮性指标
        Q_bc_sum += compute_performance(G_temp, N0, alpha=0.5, beta=0.5)

    avg = Q_bc_sum / num_simulations
    print(f"    - 完成横轴 q={q:.2f} 的 {num_simulations} 次独立采样测算 (双连通指标)")
    return avg


def simulate_random_failures_bc(G, num_simulations=20, step_size=0.01):
    initial_nodes = list(G.nodes())
    N0 = len(initial_nodes)

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    Q_bc_avg = np.zeros(len(q_points))

    tasks = [(q, G, initial_nodes, N0, num_simulations) for q in q_points]
    print(f"    - 启动多线程并发计算 Q_bc (避开多进程内存死锁，按横轴 q 并行，每个 q 计算 {num_simulations} 次)...")

    # 使用 ThreadPoolExecutor，不涉及跨进程的 Pickle 拷贝，能有效规避之前的大图卡死问题
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_simulate_single_q_bc, tasks))

    for i, res in enumerate(results):
        Q_bc_avg[i] = res

    return q_points, Q_bc_avg


def main():
    base_dir = r"d:\Project\Model"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    output_dir = os.path.join(base_dir, "Q2", "Q2_Results")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_simulation_data = []

    for filename in os.listdir(json_dir):
        if filename.endswith("_Network.json"):
            city_name = filename.split("_")[0]
            print(f"开始处理城市 (双连通 Q_bc): {city_name} ...")

            json_path = os.path.join(json_dir, filename)
            G = build_graph_from_json(json_path)

            step_size = 0.01
            # 仿真计算双连通指标下的健壮度，降低复杂度以防多进程死锁卡死
            q_points, Q_bc_avg = simulate_random_failures_bc(G, num_simulations=20, step_size=step_size)

            for q, p in zip(q_points, Q_bc_avg):
                all_simulation_data.append({"City": city_name, "q": q, "Q_bc_avg": p})

    df_all_data = pd.DataFrame(all_simulation_data)
    # 储存为独立文件
    csv_raw_out = os.path.join(output_dir, "Q2_Raw_Simulation_Data_BC.csv")
    df_all_data.to_csv(csv_raw_out, index=False, encoding="utf-8-sig")
    print(f"\n全部计算完毕！各城市的 Q_bc(q) 原始数据已成功保存至: {csv_raw_out}")


if __name__ == "__main__":
    main()
