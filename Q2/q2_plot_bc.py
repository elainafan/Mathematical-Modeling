"""
q2_plot_bc.py
题目二：双连通核指标 Q_bc 结果可视化。

读取 q2_compute_bc_ig.py（igraph 版）输出的原始仿真数据，
生成各城市健壮性曲线图并汇总积分摘要。

数据来源：Q2/Q2_Results/bc/Q2_Raw_Simulation_Data_BC.csv
输出目标：Q2/Q2_Results/bc/<City>_Robustness_Random_BC.png
          Q2/Q2_Results/bc/Q2_Robustness_Summary_BC.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def trapz_custom(y, x):
    """
    手动梯形积分，兼容各版本 numpy（规避 np.trapz / np.trapezoid 的版本差异）。
    """
    total = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        total += (y[i] + y[i - 1]) * 0.5 * dx
    return total


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    # ── 路径：相对脚本文件自动定位，无需硬编码 base_dir ────────────────────
    script_dir  = os.path.dirname(os.path.abspath(__file__))   # Q2/
    results_dir = os.path.join(script_dir, "Q2_Results", "bc") # Q2/Q2_Results/bc/

    raw_data_file = os.path.join(results_dir, "Q2_Raw_Simulation_Data_BC.csv")

    if not os.path.exists(raw_data_file):
        print(f"找不到原始数据文件：{raw_data_file}")
        print("请先执行 q2_compute_bc_ig.py。")
        return

    print(f"数据源: {raw_data_file}")
    df_all = pd.read_csv(raw_data_file)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    cities = df_all["City"].unique()
    robustness_results = []

    for city in sorted(cities):
        df_city = df_all[df_all["City"] == city].sort_values(by="q")
        q_vals    = df_city["q"].values
        Q_bc_vals = df_city["Q_bc_avg"].values

        # ── 积分 R_bc ─────────────────────────────────────────────────────
        R_bc = trapz_custom(Q_bc_vals, q_vals)

        # ── 显著崩溃点：Q_bc 下降最陡的 q ────────────────────────────────
        diffs = np.diff(Q_bc_vals)
        steepest_drop_idx = int(np.argmin(diffs))
        q_critical = float(q_vals[steepest_drop_idx])

        robustness_results.append({
            "城市":              city,
            "健壮性积分(R_bc)":   R_bc,
            "网络显著崩溃比例(q)": q_critical,
        })

        # ── 绘图 ──────────────────────────────────────────────────────────
        plt.figure(figsize=(8, 6))
        plt.plot(q_vals, Q_bc_vals,
                 marker=".", linestyle="-", color="b", linewidth=2)
        plt.axvline(x=q_critical, color="r", linestyle="--",
                    label=f"显著变化点 (q={q_critical:.2f})")
        plt.xlabel("破坏节点比例 (q)")
        plt.ylabel("双连通核结构健壮性指标 Q_bc(q)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        plot_path = os.path.join(results_dir, f"{city}_Robustness_Random_BC.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[{city}] -> {plot_path}")

    # ── 汇总 CSV ──────────────────────────────────────────────────────────
    df_results  = pd.DataFrame(robustness_results)
    summary_path = os.path.join(results_dir, "Q2_Robustness_Summary_BC.csv")
    df_results.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存至 -> {summary_path}")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    main()
