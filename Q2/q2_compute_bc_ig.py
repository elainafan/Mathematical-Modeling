"""
q2_compute_bc_ig.py
题目二：基于双连通核指标 Q_bc 的随机节点故障健壮性仿真（igraph 版）。

相比 NetworkX 版（q2_compute_bc.py）的改动：
  - 图加载与 compute_performance 全部走 utils_ig（igraph C 后端）
  - 子图创建改用 G.induced_subgraph(indices)，不再需要 rebuild_id2idx
  - 原始图 G 全程只读，ThreadPoolExecutor 多线程安全，无需加锁
  - 去掉 build_graph_from_json（由 utils_ig.load_graph 替代）

输出与 NetworkX 版完全兼容：
  Q2/Q2_Results/bc/Q2_Raw_Simulation_Data_BC.csv
  Q2/Q2_Results/bc/Q2_Robustness_Summary_BC.csv
  Q2/Q2_Results/bc/<City>_Robustness_Random_BC.png  （由 q2_plot_bc.py 生成）
"""

import os
import sys
import random
import concurrent.futures

import numpy as np
import pandas as pd

# ── 引入上一级目录的 utils_ig ─────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import load_graph, compute_performance
except ImportError:
    print("找不到 utils_ig.py，请检查路径。")
    sys.exit(1)


# =============================================================================
# 单个 q 点的仿真任务（线程工作函数）
# =============================================================================

def _simulate_single_q(args):
    """
    对给定的故障比例 q，独立重复 num_simulations 次采样并返回 Q_bc 均值。

    参数
    ----
    args : tuple
        (q, G, all_indices, N0, num_simulations)
        q               : 故障节点占比，∈ [0, 1]
        G               : igraph.Graph，原始图（只读，所有线程共享）
        all_indices     : list[int]，G 中所有节点的 igraph 内部索引（0-based）
        N0              : int，原始网络节点总数（归一化分母）
        num_simulations : int，该 q 下的独立重复次数

    返回
    ----
    float：num_simulations 次仿真的 Q_bc 均值

    注意
    ----
    induced_subgraph 创建新的 igraph 对象，不会修改 G，线程安全。
    igraph 大部分 C 实现会释放 GIL，ThreadPoolExecutor 可以获得真正的并行收益。
    """
    q, G, all_indices, N0, num_simulations = args

    remove_count = int(N0 * q)
    keep_count   = N0 - remove_count

    total = 0.0
    for _ in range(num_simulations):
        # 随机抽取要保留的节点索引（不修改原图）
        keep_indices = random.sample(all_indices, keep_count)
        sub = G.induced_subgraph(keep_indices)   # O(N+M)，新对象，线程安全
        total += compute_performance(sub, N0, alpha=0.5, beta=0.5)

    avg = total / num_simulations
    print(f"    q={q:.2f}  完成 {num_simulations} 次采样  Q_bc_avg={avg:.6f}")
    return avg


# =============================================================================
# 单城市随机故障仿真
# =============================================================================

def simulate_random_failures_bc(
    json_path:       str,
    num_simulations: int   = 20,
    step_size:       float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单个城市路网，在 q ∈ [0, 1] 上均匀采样，计算每个故障比例下 Q_bc 的均值。

    参数
    ----
    json_path       : 城市 JSON 路网文件路径
    num_simulations : 每个 q 点的独立重复次数（默认 20）
    step_size       : q 的采样间隔（默认 0.01，即 101 个点）

    返回
    ----
    (q_points, Q_bc_avg)
    q_points  : shape (K,)，故障比例数组
    Q_bc_avg  : shape (K,)，对应的 Q_bc 均值
    """
    # ── 加载图（只做一次）─────────────────────────────────────────────────────
    G, id2idx, _ = load_graph(json_path)
    N0           = G.vcount()
    all_indices  = list(range(N0))   # igraph 0-based 索引，全程不变

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)

    # ── 构造任务列表 ──────────────────────────────────────────────────────────
    tasks = [
        (q, G, all_indices, N0, num_simulations)
        for q in q_points
    ]

    print(f"    启动 ThreadPoolExecutor（{os.cpu_count()} 线程），"
          f"共 {len(q_points)} 个 q 点，每点 {num_simulations} 次仿真 …")

    # ── 并发执行（线程池，igraph 释放 GIL，可获得真正并行）──────────────────
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count()
    ) as executor:
        results = list(executor.map(_simulate_single_q, tasks))

    return q_points, np.array(results)


# =============================================================================
# 主程序
# =============================================================================

def main():
    # ── 路径配置 ─────────────────────────────────────────────────────────────
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)                    # Q2 的上一级
    json_dir    = os.path.join(project_dir, "data", "json_networks")
    output_dir  = os.path.join(script_dir,  "Q2_Results", "bc")

    os.makedirs(output_dir, exist_ok=True)

    # ── 遍历所有城市 ──────────────────────────────────────────────────────────
    all_raw:     list[dict] = []
    summary_rows: list[dict] = []

    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith("_Network.json"):
            continue

        city      = filename.split("_")[0]
        json_path = os.path.join(json_dir, filename)
        print(f"\n{'='*60}")
        print(f"城市: {city}")
        print(f"{'='*60}")

        q_pts, q_avg = simulate_random_failures_bc(
            json_path,
            num_simulations=20,
            step_size=0.01,
        )

        # 梯形积分得到健壮性标量 R_bc
        R_bc = float(np.trapz(q_avg, q_pts))

        # 临界点：Q_bc 首次低于 0.5 对应的 q（若未触及则为 NaN）
        below = np.where(q_avg < 0.5)[0]
        q_critical = float(q_pts[below[0]]) if len(below) else float("nan")

        print(f"  R_bc = {R_bc:.6f}    q_critical(Q<0.5) = {q_critical:.2f}")

        for q, p in zip(q_pts, q_avg):
            all_raw.append({"City": city, "q": round(float(q), 4), "Q_bc_avg": float(p)})

        summary_rows.append({
            "City":       city,
            "R_bc":       R_bc,
            "q_critical": q_critical,
        })

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    raw_csv = os.path.join(output_dir, "Q2_Raw_Simulation_Data_BC.csv")
    pd.DataFrame(all_raw).to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print(f"\n原始仿真数据已保存至: {raw_csv}")

    summary_csv = os.path.join(output_dir, "Q2_Robustness_Summary_BC.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"健壮性汇总已保存至:   {summary_csv}")

    # ── 打印排名 ──────────────────────────────────────────────────────────────
    df_sum = pd.DataFrame(summary_rows).sort_values("R_bc", ascending=False)
    print("\n各城市 R_bc 排名（降序）：")
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
