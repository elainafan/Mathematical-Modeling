"""
Q3/sensitivity_analysis.py
R_k 敏感性分析：不同连通阈值 (k_v, k_e) 下的健壮性指标对比

思路
----
1. 读取已有的 CABS 攻击序列（固定不变）
2. 对每种 (k_v, k_e) 配置，回放攻击过程计算 Q_k(f)
3. 积分得到 R_k
4. 比较不同阈值下城市排名是否稳定 → 证明指标选择的鲁棒性

(k_v, k_e) 配置
----------------
(1,1) : Q = |LCC|/N                              （退化为原生指标）
(1,2) : Q = β|LCC|/N + (1-β)|B_E|/N              （仅边双连通核）
(2,1) : Q = β|LCC|/N + (1-β)|B_V|/N              （仅点双连通核）
(2,2) : Q = β|LCC|/N + (1-β)|B_{2,2}|/N          （点+边双连通核）
(3,3) : Q = β|LCC|/N + (1-β)|3-core largest CC|/N （近似3连通核）

主实验用 (1,2)+(2,1) 加权，此脚本对所有配置做敏感性分析。
"""

import os
import sys
import time
import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import load_graph, rebuild_id2idx
except ImportError:
    print("错误: 找不到 utils_ig.py，请检查路径。")
    sys.exit(1)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


# =====================================================================
# k-连通核计算函数
# =====================================================================

def get_lcc_subgraph(G: ig.Graph) -> ig.Graph:
    """提取最大连通分支子图。"""
    if G.vcount() == 0:
        return G
    cc = G.connected_components()
    sizes = cc.sizes()
    lcc_idx = sizes.index(max(sizes))
    return G.induced_subgraph(cc[lcc_idx])


def max_2edge_connected_size(G_lcc: ig.Graph) -> int:
    """
    B_E: LCC 中最大 2-边连通分量的规模。
    方法：删掉所有桥边，找剩余图的最大连通分支。
    """
    if G_lcc.vcount() <= 1:
        return G_lcc.vcount()

    bridge_ids = G_lcc.bridges()
    if not bridge_ids:
        return G_lcc.vcount()  # 整个 LCC 已经是 2-边连通的

    # 删掉桥边后找最大 CC
    G_tmp = G_lcc.copy()
    G_tmp.delete_edges(bridge_ids)
    if G_tmp.vcount() == 0:
        return 0
    return max(G_tmp.connected_components().sizes())


def max_biconnected_size(G_lcc: ig.Graph) -> int:
    """
    B_V: LCC 中最大点双连通分量的规模。
    igraph.biconnected_components() 返回各分量的顶点列表。
    """
    if G_lcc.vcount() <= 2:
        return G_lcc.vcount()

    bcc = G_lcc.biconnected_components()
    if not bcc:
        return 0
    return max(len(comp) for comp in bcc)


def max_2v2e_connected_size(G_lcc: ig.Graph) -> int:
    """
    B_{2,2}: LCC 中最大同时满足 2-点连通和 2-边连通的分量规模。

    对于节点数 ≥ 3 的图，2-点连通 ⟹ 2-边连通，
    因此 B_{2,2} = B_V（最大点双连通分量）。
    """
    return max_biconnected_size(G_lcc)


def max_3connected_approx_size(G_lcc: ig.Graph) -> int:
    """
    B_{3,3} 的近似计算：最大 3-连通核。

    精确算法（SPQR 树分解）对万级节点图太慢。
    近似方案：
    1. 取 3-core（迭代删除度 < 3 的节点）
    2. 在 3-core 上找最大连通分支
    3. 这是 3-连通分量的上界（3-core 不一定 3-连通，但在实际交通网络中是合理近似）

    理论依据：对于平面稀疏图，3-core 的最大 CC 通常接近最大 3-连通子图。
    """
    if G_lcc.vcount() < 4:
        return 0

    # igraph 的 k-core 分解
    coreness = G_lcc.coreness()
    vertices_3core = [i for i in range(G_lcc.vcount()) if coreness[i] >= 3]

    if not vertices_3core:
        return 0

    sub_3core = G_lcc.induced_subgraph(vertices_3core)
    if sub_3core.vcount() == 0:
        return 0

    return max(sub_3core.connected_components().sizes())


# =====================================================================
# 通用 Q_k 计算
# =====================================================================

def compute_Q_k(G: ig.Graph, N0: int, kv: int, ke: int, beta: float = 0.5) -> float:
    """
    计算 Q_k(f) = β * |LCC|/N + (1-β) * B_{kv,ke}/N

    特殊情况：
    - (1,1): Q = |LCC|/N（忽略 β，直接退化为原生指标）
    - (1,2): B = 最大 2-边连通
    - (2,1): B = 最大点双连通
    - (2,2): B = 最大 (2v,2e)-连通
    - (3,3): B = 3-core 最大 CC（近似）
    """
    if G.vcount() == 0:
        return 0.0

    lcc_sub = get_lcc_subgraph(G)
    lcc_size = lcc_sub.vcount()
    lcc_ratio = lcc_size / N0

    if kv <= 1 and ke <= 1:
        # (1,1): 退化为纯 LCC 指标
        return lcc_ratio

    # 计算 B_{kv,ke}
    if kv <= 1 and ke == 2:
        B = max_2edge_connected_size(lcc_sub)
    elif kv == 2 and ke <= 1:
        B = max_biconnected_size(lcc_sub)
    elif kv == 2 and ke == 2:
        B = max_2v2e_connected_size(lcc_sub)
    elif kv >= 3 or ke >= 3:
        B = max_3connected_approx_size(lcc_sub)
    else:
        B = lcc_size

    B_ratio = B / N0
    return beta * lcc_ratio + (1 - beta) * B_ratio


# =====================================================================
# 回放引擎
# =====================================================================

def replay_with_configs(
    json_path: str,
    sorted_nodes: list[int],
    configs: list[tuple[int, int, str]],
    step_size: float = 0.01,
    beta: float = 0.5,
) -> dict[str, tuple[np.ndarray, list[float], float]]:
    """
    用同一个攻击序列，同时回放多种 (k_v, k_e) 配置。

    参数
    ----
    configs : [(kv, ke, label), ...]  例如 [(1,1,"k11"), (2,1,"k21"), ...]

    返回
    ----
    {label: (q_points, Q_values, R_integral)}
    """
    G, _, _ = load_graph(json_path)
    N0 = G.vcount()
    G_temp = G.copy()
    current_id2idx = rebuild_id2idx(G_temp)

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)

    # 初始化各配置的值列表
    results: dict[str, list[float]] = {}
    for kv, ke, label in configs:
        q0 = compute_Q_k(G_temp, N0, kv, ke, beta)
        results[label] = [q0]

    current_idx = 0

    for i in range(1, len(q_points)):
        q = q_points[i]
        target = int(N0 * q)
        num = target - current_idx
        if num > 0:
            batch = sorted_nodes[current_idx: min(current_idx + num, len(sorted_nodes))]
            indices = [current_id2idx[nid] for nid in batch if nid in current_id2idx]
            if indices:
                G_temp.delete_vertices(indices)
                current_id2idx = rebuild_id2idx(G_temp)
            current_idx += num

        for kv, ke, label in configs:
            val = compute_Q_k(G_temp, N0, kv, ke, beta)
            results[label].append(val)

    # 积分
    output = {}
    for kv, ke, label in configs:
        vals = results[label]
        try:
            R = float(np.trapezoid(vals, q_points))
        except AttributeError:
            R = float(np.trapz(vals, q_points))
        output[label] = (q_points, vals, R)

    return output


# =====================================================================
# 主程序
# =====================================================================

def main():
    base_dir = r"D:\college education\math\model\2026.4\2026\B-codes\Mathematical-Modeling"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    res_dir = os.path.join(base_dir, "Q3", "Results")
    os.makedirs(res_dir, exist_ok=True)

    beta = 0.5
    step_size = 0.01

    # ── 敏感性分析的 (k_v, k_e) 配置 ──
    configs = [
        (1, 1, "k(1,1)"),
        (1, 2, "k(1,2)"),
        (2, 1, "k(2,1)"),
        (2, 2, "k(2,2)"),
        (3, 3, "k(3,3)"),
    ]

    # ── 主实验配置：(1,2) + (2,1) 加权 ──
    # Q_bc = β|LCC|/N + (1-β)[α|B_E|/N + (1-α)|B_V|/N]
    # 这等价于 α * Q_{(1,2)} + (1-α) * Q_{(2,1)} 在内核部分
    # 为简化敏感性分析，我们直接比较各单一配置

    print("=" * 70)
    print("  R_k 敏感性分析：不同 (k_v, k_e) 连通阈值下的健壮性指标")
    print("=" * 70)

    # 收集所有城市的结果
    all_results = {}      # {city: {label: (q_pts, vals, R)}}
    summary_rows = []     # 汇总表

    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith("_Network.json"):
            continue

        city = filename.split("_")[0]
        json_path = os.path.join(json_dir, filename)

        # 读取 CABS 攻击序列（优先用 Q_bc 优化的版本）
        seq_csv = os.path.join(res_dir, f"{city}_CABS_Attack_Sequence.csv")
        if not os.path.exists(seq_csv):
            seq_csv = os.path.join(res_dir, f"{city}_CABS_LCC_Attack_Sequence.csv")
        if not os.path.exists(seq_csv):
            print(f"  [!] {city}: 找不到 CABS 攻击序列，跳过。")
            continue

        df_seq = pd.read_csv(seq_csv)
        sorted_nodes = df_seq["node_id"].tolist()

        print(f"\n  {city}: 回放 {len(sorted_nodes)} 节点攻击序列 × {len(configs)} 种配置 ...")
        t0 = time.perf_counter()

        city_results = replay_with_configs(
            json_path, sorted_nodes, configs,
            step_size=step_size, beta=beta,
        )
        all_results[city] = city_results

        elapsed = time.perf_counter() - t0
        print(f"    耗时 {elapsed:.1f}s")

        row = {"City": city}
        for kv, ke, label in configs:
            _, _, R = city_results[label]
            row[f"R_{label}"] = round(R, 6)
            print(f"    {label}: R = {R:.6f}")
        summary_rows.append(row)

    if not summary_rows:
        print("  没有可用数据，退出。")
        return

    df_summary = pd.DataFrame(summary_rows)

    # ── 输出汇总表 ──
    print("\n" + "=" * 70)
    print("  各城市 × 各 (k_v,k_e) 配置的 R_k 值")
    print("=" * 70)
    print(df_summary.to_string(index=False))

    # ── 排名稳定性分析 ──
    print("\n" + "=" * 70)
    print("  排名稳定性分析（各配置下城市排名）")
    print("=" * 70)

    r_cols = [c for c in df_summary.columns if c.startswith("R_")]
    rank_df = df_summary[["City"]].copy()
    for col in r_cols:
        # R 越大 → 健壮性越大 → 排名靠前（rank=1 最健壮）
        rank_df[col + "_rank"] = df_summary[col].rank(ascending=False).astype(int)

    print(rank_df.to_string(index=False))

    try:
        from scipy.stats import kendalltau
    except ImportError:
        print("  [!] scipy 未安装，跳过 Kendall τ 分析。")
        print("      安装方式: pip install scipy")
        kendalltau = None

    if kendalltau is not None:
        print("\n  Kendall τ 排名相关性矩阵：")
        rank_cols = [c for c in rank_df.columns if c.endswith("_rank")]
        n_configs = len(rank_cols)
        tau_matrix = np.ones((n_configs, n_configs))
        labels_short = [c.replace("R_", "").replace("_rank", "") for c in rank_cols]

        for i in range(n_configs):
            for j in range(i + 1, n_configs):
                tau, pval = kendalltau(
                    rank_df[rank_cols[i]], rank_df[rank_cols[j]]
                )
                tau_matrix[i, j] = tau
                tau_matrix[j, i] = tau

        df_tau = pd.DataFrame(tau_matrix, index=labels_short, columns=labels_short)
        print(df_tau.round(3).to_string())

        avg_tau = tau_matrix[np.triu_indices(n_configs, k=1)].mean()
        print(f"\n  >>> 平均 Kendall τ = {avg_tau:.3f}")
        if avg_tau > 0.7:
            print("  >>> 结论：城市排名在不同 (k_v,k_e) 配置下高度一致，"
                  "指标选择对结论影响有限。")
        elif avg_tau > 0.4:
            print("  >>> 结论：城市排名在不同配置下中度一致，"
                  "具体排名存在一定波动但总体趋势稳定。")
        else:
            print("  >>> 结论：城市排名对 (k_v,k_e) 选择较敏感，"
                  "需谨慎选择连通阈值。")
    else:
        df_tau = None

    # ── 绘图：每城市一子图，各配置一条曲线 ──
    cities = list(all_results.keys())
    num_cities = len(cities)
    cols = 4
    rows = int(np.ceil(num_cities / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for idx, city in enumerate(cities):
        ax = axes[idx]
        city_res = all_results[city]

        for ci, (kv, ke, label) in enumerate(configs):
            q_pts, vals, R = city_res[label]
            ax.plot(
                q_pts, vals,
                label=f"{label} (R={R:.3f})",
                color=colors[ci % len(colors)],
                linestyle=line_styles[ci % len(line_styles)],
                linewidth=1.8, alpha=0.85,
            )

        ax.set_title(city, fontsize=14, fontweight="bold")
        ax.set_xlabel("$q$ (Fraction removed)", fontsize=11)
        ax.set_ylabel("$Q_k(f)$", fontsize=11)
        ax.set_xlim([0, 0.5])  # 前 50% 最有信息量
        ax.set_ylim([0, 1.05])
        ax.legend(loc="upper right", fontsize=7, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.7)

    for i in range(num_cities, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    pdf_path = os.path.join(res_dir, "Q3_Sensitivity_Qk_Curves.pdf")
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  曲线图已保存: {pdf_path}")

    # ── 保存汇总 CSV ──
    summary_csv = os.path.join(res_dir, "Q3_Sensitivity_Rk_Summary.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    rank_csv = os.path.join(res_dir, "Q3_Sensitivity_Rank_Stability.csv")
    rank_df.to_csv(rank_csv, index=False, encoding="utf-8-sig")

    if df_tau is not None:
        tau_csv = os.path.join(res_dir, "Q3_Sensitivity_KendallTau.csv")
        df_tau.to_csv(tau_csv, encoding="utf-8-sig")
        print(f"  Kendall τ 矩阵已保存: {tau_csv}")

    print(f"  汇总表已保存: {summary_csv}")
    print(f"  排名表已保存: {rank_csv}")

    print("\n" + "=" * 70)
    print("  敏感性分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
