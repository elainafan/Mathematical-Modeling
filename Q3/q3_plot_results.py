import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置学术绘图风格，避免中文乱码，并且去除所有标题（Title）以符合论文要求
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


def plot_combined_strategies(csv_paths, metric_col, out_pdf, y_label):
    """
    绘制指定城市的不同维度破袭策略对比图，输出为带有子图的高清 PDF。
    支持合并多个 CSV 文件（例如 Baseline + Advanced 算法）进行同台展示。
    """
    dfs = []
    for cp in csv_paths:
        if os.path.exists(cp):
            dfs.append(pd.read_csv(cp))
        else:
            print(f"  [警告] 找不到数据文件: {cp}，跳过该文件。")
    if not dfs:
        return

    df = pd.concat(dfs, ignore_index=True)
    cities = sorted(df["City"].unique())
    num_cities = len(cities)

    # 行列自动适配，假设有 8 个城市，则为 2行4列
    cols = 4
    rows = int(np.ceil(num_cities / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # 确保 axes 可迭代 (即便只有一行)
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    strategies = df["Strategy"].unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    line_styles = ["-", "--", "-.", ":", "-"]

    for idx, city in enumerate(cities):
        ax = axes[idx]
        city_df = df[df["City"] == city]

        # 分别绘制各个攻击策略的下降曲线
        for st_idx, strat in enumerate(strategies):
            strat_df = city_df[city_df["Strategy"] == strat].sort_values("q")

            # numpy > 2.0 uses np.trapezoid instead of np.trapz
            integral_R = np.trapezoid(strat_df[metric_col], strat_df["q"])

            label_text = f"{strat} (R={integral_R:.3f})"
            ax.plot(
                strat_df["q"],
                strat_df[metric_col],
                label=label_text,
                color=colors[st_idx % len(colors)],
                linestyle=line_styles[st_idx % len(line_styles)],
                linewidth=2.0,
                alpha=0.85,
            )

        # 将城市名字作为子图标题放到整个图的上方，避免挡住任何曲线
        ax.set_title(city, fontsize=14, fontweight="bold")

        ax.set_xlabel("$q$ (Fraction of nodes removed)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.05])

        # 将图例固定放到右上角
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.7)

    # 隐藏多余的空白子图
    for i in range(num_cities, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> 成功汇总图表至 PDF: {out_pdf}")


def main():
    base_dir = r"d:\Project\Model"
    res_dir = os.path.join(base_dir, "Q3", "Results")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print("开始生成 Q3 的靶向攻击下降路径汇总矩阵图 (PDF 格式)...")

    # 1. 原生 LCC 健壮度 P(q) 绘图 (合并基础与高阶攻击)
    csv_official_base = os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Official.csv")
    csv_official_adv = os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Official.csv")
    pdf_official = os.path.join(res_dir, "Q3_Attacks_Optimal_Path_Pq.pdf")
    plot_combined_strategies([csv_official_base, csv_official_adv], "P_q_Official", pdf_official, "$P(q)$")

    # 2. 队友定义结构健壮度 Q_{bc}(f) 绘图 (合并基础与高阶攻击)
    csv_teammate_base = os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Teammate.csv")
    csv_teammate_adv = os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Teammate.csv")
    pdf_teammate = os.path.join(res_dir, "Q3_Attacks_Optimal_Path_Qbc.pdf")
    plot_combined_strategies([csv_teammate_base, csv_teammate_adv], "Q_bc_Teammate", pdf_teammate, "$Q_{bc}(f)$")

    print("\n全部绘图已完成。寻找【最优攻击路径】可以直接看图中 'R' (曲线下面积) 最小的维度策略。")


if __name__ == "__main__":
    main()
