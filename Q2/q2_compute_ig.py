import os
import sys
import numpy as np
import pandas as pd
import random
import concurrent.futures

# 引入上一级目录的 utils_ig
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import load_graph
except ImportError:
    print("找不到 utils_ig.py，请检查路径。")
    sys.exit(1)


def calculate_network_metric(G_temp, N0):
    """计算网络性能指标，当前指标为最大连通子图节点数与其初始节点数之比 P(q)"""
    if G_temp.vcount() > 0:
        largest_cc_size = max(G_temp.connected_components().sizes())
        return largest_cc_size / N0
    return 0.0


def _simulate_single_q(args):
    q, G, N0, num_simulations = args
    target_remove_count = int(N0 * q)
    P_q_sum = 0.0

    # 这里的 initial_nodes 是原内部索引列表 0 到 N0-1
    initial_nodes = list(range(N0))

    for _ in range(num_simulations):
        # 随机挑选要删除的节点的内部索引
        nodes_to_remove = random.sample(initial_nodes, target_remove_count)

        # 在 igraph 中复制十分高效
        G_temp = G.copy()
        # 删除节点（基于内部索引，批量删除）
        G_temp.delete_vertices(nodes_to_remove)

        P_q_sum += calculate_network_metric(G_temp, N0)

    avg = P_q_sum / num_simulations
    print(f"    - 完成横轴 q={q:.2f} 的 {num_simulations} 次独立采样测试")
    return avg


def simulate_random_failures(G, num_simulations=20, step_size=0.01):
    N0 = G.vcount()

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    P_q_avg = np.zeros(len(q_points))

    tasks = [(q, G, N0, num_simulations) for q in q_points]
    print(
        f"    - 启动多线程并发计算 (使用 ThreadPoolExecutor 提速并规避死锁，按横轴 q 并行，每个 q 计算 {num_simulations} 次) ..."
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(_simulate_single_q, tasks))

    for i, res in enumerate(results):
        P_q_avg[i] = res

    return q_points, P_q_avg


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
            print(f"开始处理城市 (原生健壮度 P_q, 使用 igraph): {city_name} ...")

            json_path = os.path.join(json_dir, filename)
            G, id2idx, idx2id = load_graph(json_path)

            step_size = 0.01
            q_points, P_q_avg = simulate_random_failures(G, num_simulations=20, step_size=step_size)

            for q, p in zip(q_points, P_q_avg):
                all_simulation_data.append({"City": city_name, "q": q, "P_q_avg": p})

    df_all_data = pd.DataFrame(all_simulation_data)
    csv_raw_out = os.path.join(output_dir, "Q2_Raw_Simulation_Data_IG.csv")
    df_all_data.to_csv(csv_raw_out, index=False, encoding="utf-8-sig")
    print(f"\n全部计算完毕！各城市的 P(q) 原始数据已成功保存至: {csv_raw_out}")


if __name__ == "__main__":
    main()
