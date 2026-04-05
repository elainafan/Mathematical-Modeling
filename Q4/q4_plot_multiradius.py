import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_pq_curves_by_method(method, radii, cities, results_dir):
    """
    画图1：针对某一特定方法，绘制不同波及半径下，各城市 q vs P(q) 的衰减曲线。
    8个城市以 2x4 的子图呈现，每张子图包含横跨不同半径的连线。
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # 采用颜色渐变映射不同半径 (例如：从细小半径的浅绿色到暴大半径的深紫色)
    colors = cm.turbo(np.linspace(0, 0.9, len(radii)))

    for idx, city in enumerate(cities):
        ax = axes[idx]
        for r_idx, r in enumerate(radii):
            csv_path = os.path.join(results_dir, f"Q4_Simulation_{city}_r{int(r)}m.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df_method = df[df["Method"] == method]
                if not df_method.empty:
                    ax.plot(
                        df_method["q"],
                        df_method["P_q"],
                        color=colors[r_idx],
                        linewidth=2.0,
                        label=f"Radius = {int(r)}m",
                    )

        ax.set_title(f"{city} [{method}]", fontsize=16, fontweight="bold")
        ax.set_xlabel("Fraction of nodes removed ($q$)", fontsize=14)
        ax.set_ylabel("Max Connected Component $P(q)$", fontsize=14)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.7)
        # 放置图例
        ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    out_pdf = os.path.join(results_dir, f"Q4_MultiRadius_Pq_{method}.pdf")
    plt.savefig(out_pdf, dpi=300)
    plt.close()
    print(f"✅ 生成 P(q) 衰减对比图: {out_pdf}")


def plot_r_vs_radius_by_method(method, radii, cities, results_dir):
    """
    画图2：针对某一特定方法，绘制不同波及半径下鲁棒性积分 R 的折线趋势图。
    横轴为半径 Radius，纵轴为积分 R。可以看出相变点和雪崩阈值。
    """
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # 首先读取所有的 summary 文件
    summary_dfs = {}
    for r in radii:
        sum_path = os.path.join(results_dir, f"Q4_Robustness_Summary_r{int(r)}m.csv")
        if os.path.exists(sum_path):
            summary_dfs[r] = pd.read_csv(sum_path).set_index("City")

    col_name = "R_S_Degree" if method == "S-Degree" else "R_S_CoreHD"

    for idx, city in enumerate(cities):
        ax = axes[idx]

        plot_r = []
        plot_R_val = []

        for r in radii:
            if r in summary_dfs and city in summary_dfs[r].index:
                val = summary_dfs[r].loc[city, col_name]
                if not pd.isna(val):
                    plot_r.append(r)
                    plot_R_val.append(val)

        if plot_r:
            line_color = "crimson" if method == "S-CoreHD" else "royalblue"
            ax.plot(
                plot_r, plot_R_val, marker="o", color=line_color, linewidth=3.0, markersize=8, markeredgecolor="black"
            )

        ax.set_title(f"{city} [{method}]", fontsize=16, fontweight="bold")
        ax.set_xlabel("Damage Radius $r$ (meters)", fontsize=14)
        ax.set_ylabel("Robustness Integral ($R$)", fontsize=14)

        # 让横轴清晰显示所有的半径刻度
        ax.set_xticks(radii)
        ax.tick_params(axis="x", rotation=45)

        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_pdf = os.path.join(results_dir, f"Q4_MultiRadius_RIntegral_{method}.pdf")
    plt.savefig(out_pdf, dpi=300)
    plt.close()
    print(f"✅ 生成 R 积分趋势图: {out_pdf}")


if __name__ == "__main__":
    radii = [100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0]
    cities = ["Chengdu", "Dalian", "Dongguan", "Harbin", "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou"]
    results_dir = os.path.join("Q4", "Results")

    if not os.path.exists(results_dir):
        print(f"找不到结果文件夹 {results_dir}")
    else:
        for method in ["S-Degree", "S-CoreHD"]:
            plot_pq_curves_by_method(method, radii, cities, results_dir)
            plot_r_vs_radius_by_method(method, radii, cities, results_dir)
