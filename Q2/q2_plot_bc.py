import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def trapz_custom(y, x):
    """
    手动实现梯形积分，兼容各个版本的 numpy，避免 np.trapz vs np.trapezoid 的版本冲突。
    """
    total = 0.0
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        dy = (y[i] + y[i - 1]) / 2.0
        total += dx * dy
    return total


def main():
    base_dir = r"d:\Project\Model"
    results_dir = os.path.join(base_dir, "Q2", "Q2_Results")
    raw_data_file = os.path.join(results_dir, "Q2_Raw_Simulation_Data_BC.csv")

    if not os.path.exists(raw_data_file):
        print(f"找不到原始数据文件：{raw_data_file}")
        print("请先执行 q2_compute_bc.py。")
        return

    print(f"开始处理纯绘图脚本... 数据源: {raw_data_file}")
    df_all = pd.read_csv(raw_data_file)

    cities = df_all["City"].unique()
    robustness_results = []

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    for city in cities:
        df_city = df_all[df_all["City"] == city].sort_values(by="q")
        q_vals = df_city["q"].values
        Q_bc_vals = df_city["Q_bc_avg"].values

        # 1. 积分计算 R_bc
        R_bc = trapz_custom(Q_bc_vals, q_vals)

        # 2. 计算显著崩溃点 (与 baseline 的逻辑对齐)
        diffs = np.diff(Q_bc_vals)
        steepest_drop_idx = np.argmin(diffs)
        q_critical = q_vals[steepest_drop_idx]

        robustness_results.append({"城市": city, "健壮性积分(R_bc)": R_bc, "网络显著崩溃比例(q)": q_critical})

        # 3. 绘图 (格式完全对齐 baseline)
        plt.figure(figsize=(8, 6))
        plt.plot(q_vals, Q_bc_vals, marker=".", linestyle="-", color="b", linewidth=2)
        plt.axvline(x=q_critical, color="r", linestyle="--", label=f"显著变化点 (q={q_critical:.2f})")
        plt.xlabel("破坏节点比例 (q)")
        plt.ylabel("双连通核结构健壮性指标 Q_bc(q)")
        # 根据学术论文排版习惯，去除图表上方的标题（表头），使用正文图注说明
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        plot_path = os.path.join(results_dir, f"{city}_Robustness_Random_BC.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[{city}] 的双连通健壮性散点图已导出 -> {plot_path}")

    df_results = pd.DataFrame(robustness_results)
    summary_path = os.path.join(results_dir, "Q2_Robustness_Summary_BC.csv")
    df_results.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n全部完成！各城市的积分摘要已保存至 -> {summary_path}")


if __name__ == "__main__":
    main()
