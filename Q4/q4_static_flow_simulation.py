import os
import sys
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt

# 添加上级目录以导入必要的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_ig import load_graph, rebuild_id2idx
from flow.flow_simulation import compute_node_density, compute_od_matrix, compute_capacity, compute_flow_metrics

# 中文绘图支持
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_weighted_graph(filepath: str):
    import json

    with open(filepath, "r", encoding="utf-8") as f:
        raw: dict = json.load(f)

    node_list = list(raw.values())
    n = len(node_list)

    id2idx = {val["id"]: i for i, val in enumerate(node_list)}
    idx2id = {i: val["id"] for i, val in enumerate(node_list)}

    edges, weights, flows, node_weights = [], [], [], []
    for val in node_list:
        node_weights.append(val["node_weight"])
        u = id2idx[val["id"]]
        for nb in val["neighbors"]:
            v = id2idx[nb["id"]]
            if u < v:
                edges.append((u, v))
                weights.append(nb["distance"])
                flows.append(nb["flow"])

    G = ig.Graph(n=n, edges=edges)
    G.vs["node_id"] = [val["id"] for val in node_list]
    G.vs["pos"] = [tuple(val["position"]) for val in node_list]
    G.vs["node_weight"] = node_weights
    G.es["weight"] = weights
    G.es["flow"] = flows

    return G, id2idx, idx2id


def simulate_static_deletion_flow(city_name: str, delete_fractions: np.ndarray, metric_type="degree"):
    """
    基于网络拓扑特性的静态删除模型。
    在攻击开始前，评估一次节点重要性（如Degree），然后按此顺序逐步移除节点，
    观察流量系统的崩溃过程（A降低，Gini变大，Max_rho上升）。
    """
    json_path = os.path.join("flow", "data", f"{city_name}_Network_weighted.json")
    print(f"[{city_name}] 开始执行静态 {metric_type} 删除网络流模拟...")

    # 加载带有位置信息、流和权重的预计算图形
    G_orig, id2idx, idx2id = load_weighted_graph(json_path)
    N_orig = G_orig.vcount()

    # 1. 计算基准无损状态网络条件
    positions = np.array([v["pos"] for v in G_orig.vs], dtype=np.float64)
    SIGMA, BETA = 1000.0, 0.001

    # 我们直接使用预先计算好的 m（node_weight）来计算 OD 矩阵需求
    m = np.array(G_orig.vs["node_weight"], dtype=np.float64)
    D_keys, D_vals, D_total = compute_od_matrix(G_orig, m, beta=BETA, sigma=SIGMA, verbose=True)
    max_jid = int(np.max(D_keys)) if len(D_keys) > 0 else 0

    from flow.flow_simulation import compute_edge_loads

    # 跳过 Yen 最短路径，直接使用团队跑出来的数据！
    x0 = np.array(G_orig.es["flow"], dtype=np.float64)
    c_dict = compute_capacity(G_orig, x0)

    # 2. 静态评估：确定节点的移除顺序
    if metric_type == "initial_flow":
        node_flows = np.zeros(N_orig, dtype=np.float64)
        for eid, edge in enumerate(G_orig.es):
            node_flows[edge.source] += x0[eid]
            node_flows[edge.target] += x0[eid]
        sorted_indices = np.argsort(node_flows)[::-1]
    elif metric_type == "degree":
        degrees = G_orig.degree()
        sorted_indices = np.argsort(degrees)[::-1]
    else:
        # random
        sorted_indices = np.arange(N_orig)
        np.random.shuffle(sorted_indices)

    # 保存要按顺序删除的目标 node_id
    target_node_ids = [G_orig.vs[int(i)]["node_id"] for i in sorted_indices]

    results = []
    # 创建模拟状态及映射
    G_sim = G_orig.copy()
    id2idx_s = dict(id2idx)

    # 初始指标 (P = 0)
    metrics0 = compute_flow_metrics(G_sim, id2idx_s, D_keys, D_vals, D_total, c_dict, max_jid, verbose=True)
    print(
        f"[{city_name}] 初始 0.0% -> A:{metrics0['A']:.4f}, Rho_Max:{metrics0['rho_max']:.2f}, Gini:{metrics0['H']:.4f}"
    )
    results.append(
        {
            "Fraction": 0.0,
            "Accessibility": metrics0["A"],
            "Gini": metrics0["H"],
            "Rho_Max": metrics0["rho_max"],
            "Rho_Mean": metrics0["rho_mean"],
        }
    )

    prev_idx = 0
    # 3. 按指定的累积比例进行删除
    for frac in delete_fractions:
        if frac <= 0.0:
            continue

        target_n_deleted = int(frac * N_orig)
        nodes_to_delete_now = target_node_ids[prev_idx:target_n_deleted]

        ig_indices_to_delete = []
        for nid in nodes_to_delete_now:
            if nid in id2idx_s:
                ig_indices_to_delete.append(id2idx_s[nid])

        # 降序排列以避免下标偏移
        ig_indices_to_delete.sort(reverse=True)
        G_sim.delete_vertices(ig_indices_to_delete)
        id2idx_s = rebuild_id2idx(G_sim)

        # 当网络断开或图太破碎时，指标能反映这点
        metrics = compute_flow_metrics(G_sim, id2idx_s, D_keys, D_vals, D_total, c_dict, max_jid, verbose=True)
        print(
            f"[{city_name}] 损毁 {frac * 100:.1f}% -> A:{metrics['A']:.4f}, Rho_Max:{metrics['rho_max']:.2f}, Gini:{metrics['H']:.4f}"
        )

        results.append(
            {
                "Fraction": frac,
                "Accessibility": metrics["A"],
                "Gini": metrics["H"],
                "Rho_Max": metrics["rho_max"],
                "Rho_Mean": metrics["rho_mean"],
            }
        )
        prev_idx = target_n_deleted

        # 提前终止阈值 (可达性过低)
        if metrics["A"] < 0.01:
            break

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    cities = ["Dalian", "Dongguan", "Harbin", "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou"]
    # 从 2% 采到 30%，因为超过 30% 可能图就全散了
    fractions = np.linspace(0.02, 0.30, 15)

    out_dir = os.path.join(os.path.dirname(__file__), "Results")
    os.makedirs(out_dir, exist_ok=True)

    for city in cities:
        print(f"\n{'=' * 50}\n====== Processing {city} ======\n{'=' * 50}")
        # 基于初始流量重要性摧毁
        df_flow = simulate_static_deletion_flow(city, fractions, metric_type="initial_flow")

        df_flow.to_csv(os.path.join(out_dir, f"StaticFlow_{city}_Flow.csv"), index=False)

        # -- 开始绘图 --
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1) Accessibility (A)
        axes[0].plot(df_flow["Fraction"] * 100, df_flow["Accessibility"], marker="o", label="Static (Initial Flow)")
        axes[0].set_title(f"Accessibility $A$ Decay ({city})")
        axes[0].set_xlabel("Nodes Deleted (%)")
        axes[0].set_ylabel("Accessibility A")
        axes[0].legend()
        axes[0].grid(True, linestyle="--")

        # 2) 拥堵上升 (Rho_Max)
        axes[1].plot(df_flow["Fraction"] * 100, df_flow["Rho_Max"], marker="o", label="Max $\\rho$ (Initial Flow)")
        axes[1].set_title(f"Max Congestion $\\rho$ ({city})")
        axes[1].set_xlabel("Nodes Deleted (%)")
        axes[1].set_ylabel("Max Congestion $\\rho$")
        axes[1].legend()
        axes[1].grid(True, linestyle="--")

        # 3) Gini 指数 (流量不平等指标)
        axes[2].plot(df_flow["Fraction"] * 100, df_flow["Gini"], marker="o", label="Gini (Initial Flow)")
        axes[2].set_title(f"Flow Gini Coefficient ({city})")
        axes[2].set_xlabel("Nodes Deleted (%)")
        axes[2].set_ylabel("Gini Index")
        axes[2].legend()
        axes[2].grid(True, linestyle="--")

        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"Q4_Static_Flow_Deletion_{city}.pdf")
        plt.savefig(pdf_path)
        plt.close(fig)  # 释放内存
        print(f"\n模型运行完毕，图表已妥善保存至: {pdf_path}")
