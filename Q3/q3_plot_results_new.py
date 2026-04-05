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
    #    CABS 只用面向 LCC 优化的版本 (new_algorithm_origin.py)
    # ══════════════════════════════════════════════════════════════
    pq_csvs = [
        os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Official.csv"),
        os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Official.csv"),
        os.path.join(res_dir, "Q3_CABS_LCC_Data_Pq.csv"),
    ]
    pdf_pq = os.path.join(res_dir, "Q3_All_Attacks_Pq.pdf")
    print("\n[1] 绘制 P(q) = LCC/N_0 曲线对比 ...")
    plot_combined_strategies(pq_csvs, "P_q_Official", pdf_pq, "$P(q) = |LCC|/N_0$")

    # ══════════════════════════════════════════════════════════════
    # 2. Q_bc(f) 曲线对比（双连通核指标）
    #    CABS 只用面向 Q_bc 优化的版本 (new_algorithm.py)
    # ══════════════════════════════════════════════════════════════
    qbc_csvs = [
        os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Teammate.csv"),
        os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Teammate.csv"),
        os.path.join(res_dir, "Q3_CABS_BeamSearch_Data_Qbc.csv"),
    ]
    pdf_qbc = os.path.join(res_dir, "Q3_All_Attacks_Qbc.pdf")
    print("\n[2] 绘制 Q_bc(f) 曲线对比 ...")
    plot_combined_strategies(qbc_csvs, "Q_bc_Teammate", pdf_qbc, "$Q_{bc}(f)$")

    # ══════════════════════════════════════════════════════════════
    # 3. 综合分析：所有策略 × 两个指标 R_lcc & R_Qbc + Speedup
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 64)
    print("  [3] 综合 R 值排名 + Speedup 分析（R_lcc & R_Qbc）")
    print("=" * 64)

    def safe_trapz(y, x):
        try:
            return float(np.trapezoid(y, x))
        except AttributeError:
            return float(np.trapz(y, x))

    def load_and_integrate(csv_list, metric_col):
        """加载多个 CSV 并计算每 (City, Strategy) 的 R 值。"""
        dfs_local = [pd.read_csv(cp) for cp in csv_list if os.path.exists(cp)]
        if not dfs_local:
            return pd.DataFrame()
        df = pd.concat(dfs_local, ignore_index=True)
        df = df.drop_duplicates(subset=["City", "Strategy", "q"], keep="first")
        rows = []
        for (city, strat), g in df.groupby(["City", "Strategy"]):
            g = g.sort_values("q")
            R = safe_trapz(g[metric_col], g["q"])
            rows.append({"City": city, "Strategy": strat, "R": R})
        return pd.DataFrame(rows)

    # ---- 加载所有 P_q 数据（所有策略都有 P_q）----
    all_pq_csvs = [
        os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Official.csv"),
        os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Official.csv"),
        os.path.join(res_dir, "Q3_CABS_BeamSearch_Data_Pq.csv"),
        os.path.join(res_dir, "Q3_CABS_LCC_Data_Pq.csv"),
    ]
    df_rlcc = load_and_integrate(all_pq_csvs, "P_q_Official")
    if not df_rlcc.empty:
        df_rlcc = df_rlcc.rename(columns={"R": "R_lcc"})

    # ---- 加载所有 Q_bc 数据（所有策略都有 Q_bc）----
    all_qbc_csvs = [
        os.path.join(res_dir, "Q3_Baseline_Attacks_Data_IG_Teammate.csv"),
        os.path.join(res_dir, "Q3_Advanced_Attacks_Data_IG_Teammate.csv"),
        os.path.join(res_dir, "Q3_CABS_BeamSearch_Data_Qbc.csv"),
        os.path.join(res_dir, "Q3_CABS_LCC_Data_Qbc.csv"),
    ]
    df_rqbc = load_and_integrate(all_qbc_csvs, "Q_bc_Teammate")
    if not df_rqbc.empty:
        df_rqbc = df_rqbc.rename(columns={"R": "R_Qbc"})

    # ---- 合并两个指标到同一张表 ----
    if not df_rlcc.empty and not df_rqbc.empty:
        df_merged = df_rlcc.merge(df_rqbc, on=["City", "Strategy"], how="outer")
    elif not df_rlcc.empty:
        df_merged = df_rlcc.copy()
    elif not df_rqbc.empty:
        df_merged = df_rqbc.copy()
    else:
        print("  [错误] 无可用数据文件。")
        return

    # ---- 策略列排序 ----
    known_order = [
        "Degree", "Betweenness", "Closeness",
        "CI_Radius2", "CoreHD",
        "CABS_W3_K30_B5", "CABS_LCC_W3_K30_B5",
    ]

    # ---- 打印完整 R 值表 ----
    for metric in ["R_lcc", "R_Qbc"]:
        if metric not in df_merged.columns:
            continue
        print(f"\n  ┌───────────────────────────────────────────────────┐")
        print(f"  │  {metric} 各策略 R 值（越小 = 攻击越有效）          │")
        print(f"  └───────────────────────────────────────────────────┘")
        pivot = df_merged.pivot(index="City", columns="Strategy", values=metric)
        col_order = [c for c in known_order if c in pivot.columns]
        col_order += [c for c in pivot.columns if c not in col_order]
        pivot = pivot[col_order]
        print(pivot.round(4).to_string())

        # 保存
        csv_path = os.path.join(res_dir, f"Q3_All_Strategies_{metric}_Ranking.csv")
        pivot.to_csv(csv_path, encoding="utf-8-sig")

    # ---- Speedup 计算：CABS vs 最优 baseline/advanced ----
    is_cabs = df_merged["Strategy"].str.contains("CABS", case=False)
    df_cabs = df_merged[is_cabs].copy()
    df_other = df_merged[~is_cabs].copy()

    if df_cabs.empty or df_other.empty:
        print("\n  [!] CABS 或 baseline 数据缺失，跳过 Speedup 计算。")
    else:
        for metric in ["R_lcc", "R_Qbc"]:
            if metric not in df_other.columns:
                continue

            # 跳过 NaN
            df_other_valid = df_other.dropna(subset=[metric])
            df_cabs_valid = df_cabs.dropna(subset=[metric])
            if df_other_valid.empty or df_cabs_valid.empty:
                continue

            # 每城市非 CABS 中的最优 R
            best_other = df_other_valid.groupby("City").apply(
                lambda g: g.loc[g[metric].idxmin()]
            )[["City", "Strategy", metric]].reset_index(drop=True)
            best_other.columns = ["City", "Best_Other", f"{metric}_other"]

            # 合并 CABS
            df_sp = df_cabs_valid[["City", "Strategy", metric]].merge(
                best_other, on="City"
            )
            df_sp["Speedup"] = df_sp[f"{metric}_other"] / df_sp[metric]
            df_sp["Improv%"] = (
                (df_sp[f"{metric}_other"] - df_sp[metric])
                / df_sp[f"{metric}_other"] * 100
            )

            print(f"\n  ┌──────────────────────────────────────────────────────────┐")
            print(f"  │  CABS vs 最优 Baseline/Advanced — Speedup ({metric})       │")
            print(f"  └──────────────────────────────────────────────────────────┘")
            print(f"  Speedup = R_best_other / R_CABS  (>1 表示 CABS 更优)")
            print(f"  Improv% = (R_best_other - R_CABS) / R_best_other × 100%\n")

            df_disp = pd.DataFrame({
                "City":        df_sp["City"],
                "Best_Other":  df_sp["Best_Other"],
                f"{metric}_other": df_sp[f"{metric}_other"].round(4),
                "CABS":        df_sp["Strategy"],
                f"{metric}_CABS":  df_sp[metric].round(4),
                "Speedup":     df_sp["Speedup"].round(2),
                "Improv%":     df_sp["Improv%"].round(1),
            }).sort_values("City")
            print(df_disp.to_string(index=False))

            avg_sp = df_sp["Speedup"].mean()
            avg_im = df_sp["Improv%"].mean()
            print(f"\n  >>> 平均 Speedup = {avg_sp:.2f}x，"
                  f"平均改进率 = {avg_im:.1f}%")

    # ---- 城市健壮性排名（按全局最优 R）----
    for metric in ["R_lcc", "R_Qbc"]:
        if metric not in df_merged.columns:
            continue
        all_best = df_merged.dropna(subset=[metric]).groupby("City")[metric].min()
        all_best = all_best.reset_index().sort_values(metric, ascending=False)
        print(f"\n  城市健壮性排名（{metric}，R 最大 = 最难瓦解）：")
        for _, row in all_best.iterrows():
            print(f"    {row['City']:>12s}  {metric} = {row[metric]:.4f}")

    print("\n" + "=" * 64)
    print("  全部完成！")
    print("=" * 64)


if __name__ == "__main__":
    main()
