import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 学术绘图风格
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

# ── 配色方案：支持 7+ 种策略 ──────────────────────────────────────────────
# Baseline: Degree, Betweenness, Closeness
# Advanced: CI_Radius2, CoreHD
# CABS:     CABS_W3_K30_B5 (Q_bc 优化), CABS_LCC_W3_K30_B5 (LCC 优化)
STRATEGY_STYLES = {
    "Degree":            {"color": "#1f77b4", "ls": "-",  "lw": 1.5},
    "Betweenness":       {"color": "#ff7f0e", "ls": "-",  "lw": 1.5},
    "Closeness":         {"color": "#2ca02c", "ls": "-",  "lw": 1.5},
    "CI_Radius2":        {"color": "#9467bd", "ls": "--", "lw": 1.8},
    "CoreHD":            {"color": "#8c564b", "ls": "--", "lw": 1.8},
    "CABS_W3_K30_B5":    {"color": "#d62728", "ls": "-.", "lw": 2.2},
    "CABS_LCC_W3_K30_B5":{"color": "#e377c2", "ls": "-.", "lw": 2.2},
}

# 图例中显示的简短标签
STRATEGY_LABELS = {
    "Degree":             "Degree",
    "Betweenness":        "Betweenness",
    "Closeness":          "Closeness",
    "CI_Radius2":         "CI(r=2)",
    "CoreHD":             "CoreHD",
    "CABS_W3_K30_B5":     "CABS(Qbc)",
    "CABS_LCC_W3_K30_B5": "CABS(LCC)",
}

# 回退用的默认颜色/线型
FALLBACK_COLORS = [
    "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2",
]
FALLBACK_LS = ["-", "--", "-.", ":", "-"]


def get_style(strategy, idx):
    """获取策略的绘图样式，优先用预定义的，否则回退。"""
    if strategy in STRATEGY_STYLES:
        return STRATEGY_STYLES[strategy]
    return {
        "color": FALLBACK_COLORS[idx % len(FALLBACK_COLORS)],
        "ls": FALLBACK_LS[idx % len(FALLBACK_LS)],
        "lw": 1.5,
    }


def get_label(strategy):
    """获取图例标签。"""
    return STRATEGY_LABELS.get(strategy, strategy)


def plot_combined_strategies(csv_paths, metric_col, out_pdf, y_label):
    """
    绘制所有城市 × 所有攻击策略的对比子图矩阵，输出 PDF。
    合并多个 CSV 文件（Baseline + Advanced + CABS）。
    """
    dfs = []
    for cp in csv_paths:
        if os.path.exists(cp):
            dfs.append(pd.read_csv(cp))
        else:
            print(f"  [警告] 找不到数据文件: {cp}，跳过。")
    if not dfs:
        print("  [错误] 没有找到任何有效数据文件，无法绘图。")
        return

    df = pd.concat(dfs, ignore_index=True)

    # 去重（Advanced 的 CI_Radius2 可能有重复行）
    df = df.drop_duplicates(subset=["City", "Strategy", "q"], keep="first")

    cities = sorted(df["City"].unique())
    num_cities = len(cities)

    cols = 4
    rows = int(np.ceil(num_cities / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    strategies = df["Strategy"].unique()

    # 按预定义顺序排列策略（已知的靠前，未知的靠后）
    known_order = [
        "Degree", "Betweenness", "Closeness",
        "CI_Radius2", "CoreHD",
        "CABS_W3_K30_B5", "CABS_LCC_W3_K30_B5",
    ]
    ordered = [s for s in known_order if s in strategies]
    ordered += [s for s in strategies if s not in ordered]

    for idx, city in enumerate(cities):
        ax = axes[idx]
        city_df = df[df["City"] == city]

        for st_idx, strat in enumerate(ordered):
            strat_df = city_df[city_df["Strategy"] == strat].sort_values("q")
            if strat_df.empty:
                continue

            # 兼容 numpy >= 2.0
            try:
                integral_R = np.trapezoid(strat_df[metric_col], strat_df["q"])
            except AttributeError:
                integral_R = np.trapz(strat_df[metric_col], strat_df["q"])

            style = get_style(strat, st_idx)
            label_text = f"{get_label(strat)} (R={integral_R:.3f})"

            ax.plot(
                strat_df["q"],
                strat_df[metric_col],
                label=label_text,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                alpha=0.85,
            )

        ax.set_title(city, fontsize=14, fontweight="bold")
        ax.set_xlabel("$q$ (Fraction of nodes removed)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.05])
        ax.legend(loc="upper right", fontsize=7, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.7)

    for i in range(num_cities, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> 已保存: {out_pdf}")


def main():
    base_dir = r"D:\college education\math\model\2026.4\2026\B-codes\Mathematical-Modeling"
    res_dir = os.path.join(base_dir, "Q3", "Results")
    os.makedirs(res_dir, exist_ok=True)

    print("=" * 64)
    print("  Q3 全策略对比绘图（Baseline + Advanced + CABS）")
    print("=" * 64)

    # ══════════════════════════════════════════════════════════════
    # 1. P(q) 曲线对比（原生 LCC 健壮性指标）
    # ══════════════════════════════════════════════════════════════
    pq_csvs = [
        os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Official.csv"),
        os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Official.csv"),
        os.path.join(res_dir, "Q3_CABS_BeamSearch_Data_Pq.csv"),
        os.path.join(res_dir, "Q3_CABS_LCC_Data_Pq.csv"),
    ]
    pdf_pq = os.path.join(res_dir, "Q3_All_Attacks_Pq.pdf")
    print("\n[1] 绘制 P(q) = LCC/N_0 曲线对比 ...")
    plot_combined_strategies(pq_csvs, "P_q_Official", pdf_pq, "$P(q) = |LCC|/N_0$")

    # ══════════════════════════════════════════════════════════════
    # 2. Q_bc(f) 曲线对比（双连通核指标）
    # ══════════════════════════════════════════════════════════════
    qbc_csvs = [
        os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Teammate.csv"),
        os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Teammate.csv"),
        os.path.join(res_dir, "Q3_CABS_BeamSearch_Data_Qbc.csv"),
        os.path.join(res_dir, "Q3_CABS_LCC_Data_Qbc.csv"),
    ]
    pdf_qbc = os.path.join(res_dir, "Q3_All_Attacks_Qbc.pdf")
    print("\n[2] 绘制 Q_bc(f) 曲线对比 ...")
    plot_combined_strategies(qbc_csvs, "Q_bc_Teammate", pdf_qbc, "$Q_{bc}(f)$")

    # ══════════════════════════════════════════════════════════════
    # 3. 汇总 R 值排名表
    # ══════════════════════════════════════════════════════════════
    print("\n[3] 汇总各策略 R 值排名 ...")

    all_csvs_pq = pq_csvs
    dfs = [pd.read_csv(cp) for cp in all_csvs_pq if os.path.exists(cp)]
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["City", "Strategy", "q"], keep="first")

        rows = []
        for (city, strat), g in df_all.groupby(["City", "Strategy"]):
            g = g.sort_values("q")
            try:
                R = np.trapezoid(g["P_q_Official"], g["q"])
            except AttributeError:
                R = np.trapz(g["P_q_Official"], g["q"])
            rows.append({"City": city, "Strategy": strat, "R_Pq": round(R, 6)})

        df_rank = pd.DataFrame(rows)

        # 每个城市找最优策略
        best = df_rank.loc[df_rank.groupby("City")["R_Pq"].idxmin()]
        print("\n  各城市最优攻击策略（R_Pq 最小 = 攻击最有效）：")
        print(best.to_string(index=False))

        # 按最优策略的 R_Pq 排名城市
        best_sorted = best.sort_values("R_Pq", ascending=False)
        print("\n  城市健壮性排名（R_Pq 最大 = 最难瓦解）：")
        print(best_sorted.to_string(index=False))

        # 保存完整排名表
        rank_csv = os.path.join(res_dir, "Q3_All_Strategies_R_Ranking.csv")
        pivot = df_rank.pivot(index="City", columns="Strategy", values="R_Pq")
        pivot.to_csv(rank_csv, encoding="utf-8-sig")
        print(f"\n  完整排名表已保存: {rank_csv}")

    print("\n" + "=" * 64)
    print("  绘图完成！")
    print("=" * 64)


if __name__ == "__main__":
    main()
