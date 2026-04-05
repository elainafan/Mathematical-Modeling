import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_q4_results(radius=2000.0):
    """
    绘制 Q4 空间级联瘫痪下，8个城市的鲁棒性跌落曲线 P(q)。
    对比 S-Degree 与前沿解 S-CoreHD 的降维打击效果。
    """
    results_dir = os.path.join("Q4", "Results")
    cities = ["Chengdu", "Dalian", "Dongguan", "Harbin", "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou"]

    # 尝试加载 summary 文件，提取出 R 积分值，方便写进图例
    summary_path = os.path.join(results_dir, f"Q4_Robustness_Summary_r{int(radius)}m.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path).set_index("City")
    else:
        print(f"未找到汇总文件 {summary_path}，请先运行 q4_simulate_attacks.py")
        return

    # 创建 2x4 的网格画板
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, city in enumerate(cities):
        csv_path = os.path.join(results_dir, f"Q4_Simulation_{city}_r{int(radius)}m.csv")
        if not os.path.exists(csv_path):
            print(f"警告：未找到 {city} 的仿真数据记录。")
            continue

        df = pd.read_csv(csv_path)
        ax = axes[idx]

        # 将两种方法的曲线进行叠图绘制 (q 为破坏比例, P_q 为幸存最大连通分量百分比)
        # 1. 经典空间度中心性攻击
        df_deg = df[df["Method"] == "S-Degree"]
        r_deg = summary_df.loc[city, "R_S_Degree"]
        ax.plot(df_deg["q"], df_deg["P_q"], label=f"S-Degree ($R$={r_deg:.3f})", color="royalblue", linewidth=2.5)

        # 2. 前沿空间 2-Core 骨架动态剥离攻击
        df_core = df[df["Method"] == "S-CoreHD"]
        r_core = summary_df.loc[city, "R_S_CoreHD"]
        ax.plot(
            df_core["q"],
            df_core["P_q"],
            label=f"S-CoreHD ($R$={r_core:.3f})",
            color="crimson",
            linewidth=2.5,
            linestyle="--",
        )

        # 图表美化设置
        ax.set_title(f"{city} (AoE Radius = {int(radius)}m)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Fraction of nodes removed ($q$)", fontsize=14)
        ax.set_ylabel("Max Connected Component $P(q)$", fontsize=14)

        # 限制坐标轴范围 0-1
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)

        # 美化网格线
        ax.grid(True, linestyle=":", alpha=0.7)
        # 设置图例避免遮挡曲线
        ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    out_pdf = os.path.join(results_dir, f"Q4_Spatial_Robustness_r{int(radius)}m.pdf")
    plt.savefig(out_pdf, dpi=300)
    print(f"✅ Q4 空间打击鲁棒性对比PDF高清图已生成！保存在：{out_pdf}")


if __name__ == "__main__":
    radii = [100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0]
    for r in radii:
        # 画图时捕获可能因为还没计算完毕或没数据的半径
        try:
            plot_q4_results(radius=r)
        except Exception as e:
            print(f"画图跳过 r={r}m: {e}")
