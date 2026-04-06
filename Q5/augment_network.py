"""
Q5/augment_network.py
基于双连通结构增量评估的贪心加边算法 (VGRA)

策略：
  阶段一 — 代理贪心加边
    对每条候选边虚拟加入后，用 Tarjan 算法在 O(N+M) 内计算
    Δ|BE|, Δ|BV|, Δbridges，以 Gain/cost 为排序指标贪心选边。
    每加 n_check 条边执行一次完整 CABS 精确复核。

  阶段二 — 反向删边精简
    对已加入的边逆序检查：若删去后目标仍满足则永久删去，降低总成本。

用法:
    python -m Q5.augment_network
"""

import os
import sys
import time
import json
import csv
import math
import igraph as ig
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import load_graph, rebuild_id2idx, compute_performance
except ImportError:
    print("错误: 找不到 utils_ig.py，请检查路径。")
    sys.exit(1)


# =====================================================================
# 图加载与结构指标
# =====================================================================

def load_graph_from_json(json_path: str) -> ig.Graph:
    """从 json_networks 的 JSON 加载 igraph 图。"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    node_list = list(raw.values())
    n = len(node_list)
    id2idx = {val["id"]: i for i, val in enumerate(node_list)}

    edges = []
    weights = []
    for val in node_list:
        u = id2idx[val["id"]]
        for nb in val["neighbors"]:
            v = id2idx.get(nb["id"])
            if v is not None and u < v:
                edges.append((u, v))
                weights.append(nb["distance"])

    G = ig.Graph(n=n, edges=edges, directed=False)
    G.vs["node_id"] = [val["id"] for val in node_list]
    G.vs["pos"] = [val["position"] for val in node_list]
    if weights:
        G.es["weight"] = weights

    return G


def compute_structural_metrics(G: ig.Graph):
    """
    计算图的双连通结构指标。
    返回: (n_bridges, n_artic, be_size, bv_size)
      n_bridges : 桥边数
      n_artic   : 割点数
      be_size   : 最大边双连通分量的节点数
      bv_size   : 最大点双连通分量的节点数
    """
    if G.vcount() == 0:
        return 0, 0, 0, 0

    # 桥边数
    bridges = G.bridges()
    n_bridges = len(bridges)

    # 割点数
    artic = G.articulation_points()
    n_artic = len(artic)

    # 边双连通分量（删掉所有桥后的连通分量）
    if n_bridges > 0:
        G_temp = G.copy()
        G_temp.delete_edges(bridges)
        be_sizes = G_temp.connected_components().sizes()
        be_size = max(be_sizes)
    else:
        be_size = max(G.connected_components().sizes())

    # 点双连通分量
    try:
        bicon = G.biconnected_components()
        bv_size = max(len(comp) for comp in bicon) if bicon else 0
    except Exception:
        bv_size = 0

    return n_bridges, n_artic, be_size, bv_size


def get_lcc_size(G: ig.Graph) -> int:
    """返回 LCC 的节点数。"""
    if G.vcount() == 0:
        return 0
    return max(G.connected_components().sizes())


# =====================================================================
# 候选边加载
# =====================================================================

def load_candidates(csv_path: str) -> list[dict]:
    """
    从 CSV 加载候选边。
    返回 list[{u, v, source, cost, ...}]
    """
    candidates = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # 自动检测距离列名
        header = reader.fieldnames
        dist_col = None
        for name in ["euclidean_dist", "eucl_dist", "dist", "distance",
                      "cost", "length", "euclid_dist"]:
            if name in header:
                dist_col = name
                break
        if dist_col is None:
            # 找包含 "dist" 的列
            for name in header:
                if "dist" in name.lower():
                    dist_col = name
                    break
        if dist_col is None:
            raise KeyError(
                f"无法在 CSV 中找到距离列。可用列名: {header}"
            )

        for row in reader:
            candidates.append({
                "u": int(row["u"]),
                "v": int(row["v"]),
                "source": row["source"],
                "cost": float(row[dist_col]),
            })
    return candidates


# =====================================================================
# 代理指标计算
# =====================================================================

def evaluate_edge_gain(
    G: ig.Graph,
    u_idx: int,
    v_idx: int,
    baseline: dict,
    lam_bridge: float = 2.0,
) -> dict:
    """
    虚拟加入边 (u_idx, v_idx)，计算代理收益。

    策略：原地加边 → Tarjan → 原地删边。避免复制图。

    参数
    ----
    baseline : {'n_bridges', 'be_size', 'bv_size', 'lcc_size'}
    lam_bridge : 桥边消除数的权重系数

    返回
    ----
    {'delta_be', 'delta_bv', 'delta_bridge', 'gain'}
    """
    # 加边
    eid_new = G.ecount()
    G.add_edge(u_idx, v_idx)

    try:
        # 计算新指标
        bridges_new = G.bridges()
        n_bridges_new = len(bridges_new)

        # 边双连通
        if n_bridges_new > 0:
            G_temp = G.copy()
            G_temp.delete_edges(G_temp.bridges())
            be_size_new = max(G_temp.connected_components().sizes())
        else:
            be_size_new = max(G.connected_components().sizes())

        # 点双连通
        try:
            bicon = G.biconnected_components()
            bv_size_new = max(len(comp) for comp in bicon) if bicon else 0
        except Exception:
            bv_size_new = 0

        # LCC
        lcc_size_new = max(G.connected_components().sizes())

        delta_be = be_size_new - baseline["be_size"]
        delta_bv = bv_size_new - baseline["bv_size"]
        delta_bridge = baseline["n_bridges"] - n_bridges_new  # 消除的桥数
        delta_lcc = lcc_size_new - baseline["lcc_size"]

        gain = delta_be + delta_bv + lam_bridge * delta_bridge + delta_lcc

    finally:
        # 删边还原
        G.delete_edges(eid_new)

    return {
        "delta_be": delta_be,
        "delta_bv": delta_bv,
        "delta_bridge": delta_bridge,
        "delta_lcc": delta_lcc,
        "gain": gain,
    }


def evaluate_edge_gain_fast(
    G: ig.Graph,
    u_idx: int,
    v_idx: int,
    baseline: dict,
    lam_bridge: float = 2.0,
) -> dict:
    """
    快速版本：只计算桥数变化和 LCC 变化，跳过边双连通的 copy 操作。
    用于候选集 > 2000 时的加速。
    """
    eid_new = G.ecount()
    G.add_edge(u_idx, v_idx)

    try:
        n_bridges_new = len(G.bridges())
        lcc_size_new = max(G.connected_components().sizes())

        delta_bridge = baseline["n_bridges"] - n_bridges_new
        delta_lcc = lcc_size_new - baseline["lcc_size"]

        # 粗估 delta_be ≈ delta_bridge（消除一条桥大约合并两个边双连通分量）
        gain = delta_bridge * (1 + lam_bridge) + delta_lcc

    finally:
        G.delete_edges(eid_new)

    return {
        "delta_be": 0,
        "delta_bv": 0,
        "delta_bridge": delta_bridge,
        "delta_lcc": delta_lcc,
        "gain": gain,
    }


# =====================================================================
# 轻量级 CABS 调用（直接操作 Graph 对象）
# =====================================================================

def run_cabs_on_graph(
    G: ig.Graph,
    W: int = 3,
    K: int = 30,
    batch_size: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    theta: float = 0.01,
    warmup_frac: float = 0.05,
    city_name: str = "",
) -> tuple[list[int], float]:
    """
    在给定 Graph 对象上运行 CABS 攻击算法。
    直接导入 Q3 的模块，避免通过 JSON 中转。

    返回: (attack_seq, R_bc)
    """
    try:
        from Q3.new_algorithm import (
            greedy_warmup, beam_search_phase, get_lcc_ratio,
        )
    except ImportError:
        # 回退：导入同目录
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Q3"
        ))
        from new_algorithm import (
            greedy_warmup, beam_search_phase, get_lcc_ratio,
        )

    N0 = G.vcount()
    G_work = G.copy()

    print(f"      [CABS 精确复核] N={N0}, M={G_work.ecount()}")

    warmup_count = int(N0 * warmup_frac)

    # Phase A
    G_warm, warmup_seq, warmup_R = greedy_warmup(
        G_work, N0, warmup_count,
        alpha=alpha, beta=beta, theta=theta,
    )

    if get_lcc_ratio(G_warm, N0) <= theta:
        return warmup_seq, warmup_R

    # Phase B
    full_seq, full_R = beam_search_phase(
        G_warm, warmup_seq, warmup_R, N0,
        W=W, K=K, batch_size=batch_size,
        alpha=alpha, beta=beta, theta=theta,
    )

    return full_seq, full_R


# =====================================================================
# 主算法：贪心加边 + 精确复核 + 反向精简
# =====================================================================

def augment_greedy(
    G: ig.Graph,
    candidates: list[dict],
    target_R: float,
    n_check: int = 5,
    max_add: int = 200,
    lam_bridge: float = 2.0,
    cost_exponent: float = 0.3,
    cabs_params: dict = None,
    use_fast_eval: bool = False,
) -> tuple[list[dict], float, float]:
    """
    代理贪心加边算法。

    参数
    ----
    G            : 原始图（会被就地修改）
    candidates   : 候选边列表 [{u, v, source, cost}]
    target_R     : 目标健壮性下界 T
    n_check      : 每加 n_check 条边做一次 CABS 精确复核
    max_add      : 最多加边条数
    lam_bridge   : 桥边消除权重
    cabs_params  : CABS 超参字典
    use_fast_eval: 是否使用快速评估（跳过 BE/BV 精确计算）

    返回
    ----
    added_edges  : 已加入的边列表
    total_cost   : 总成本
    final_R      : 最终精确 R_bc
    """
    if cabs_params is None:
        cabs_params = {}

    N0 = G.vcount()
    id2idx = {G.vs[i]["node_id"]: i for i in range(N0)}

    # 预处理候选边：映射到 igraph 索引，过滤掉已有边
    valid_candidates = []
    existing_edges = set()
    for e in G.es:
        u, v = e.source, e.target
        existing_edges.add((min(u, v), max(u, v)))

    for cand in candidates:
        u_nid, v_nid = cand["u"], cand["v"]
        if u_nid not in id2idx or v_nid not in id2idx:
            continue
        u_idx, v_idx = id2idx[u_nid], id2idx[v_nid]
        edge_key = (min(u_idx, v_idx), max(u_idx, v_idx))
        if edge_key in existing_edges:
            continue
        valid_candidates.append({
            **cand,
            "u_idx": u_idx,
            "v_idx": v_idx,
            "edge_key": edge_key,
        })

    print(f"    有效候选边: {len(valid_candidates)} / {len(candidates)}")

    # 评估函数（只对预筛后的少量候选使用）
    eval_fn = evaluate_edge_gain_fast if use_fast_eval else evaluate_edge_gain

    # 预筛参数：每步最多对 TOP_N 条候选跑完整 Tarjan
    TOP_N = 200

    added_edges = []
    total_cost = 0.0
    current_R = None
    t_start = time.perf_counter()

    for step in range(1, max_add + 1):
        if not valid_candidates:
            print("    候选边已用完！")
            break

        # ---- 计算当前基线 ----
        n_bridges, n_artic, be_size, bv_size = compute_structural_metrics(G)
        lcc_size = get_lcc_size(G)
        baseline = {
            "n_bridges": n_bridges,
            "n_artic": n_artic,
            "be_size": be_size,
            "bv_size": bv_size,
            "lcc_size": lcc_size,
        }

        # ---- O(N+M) 预筛：2-边连通分量 + 连通分量 ----
        t_eval = time.perf_counter()

        # 计算连通分量标签
        cc_membership = G.connected_components().membership

        # 计算 2-边连通分量标签（删掉所有桥后的连通分量）
        bridge_eids = G.bridges()
        if bridge_eids:
            G_no_br = G.copy()
            G_no_br.delete_edges(bridge_eids)
            ec_membership = G_no_br.connected_components().membership
        else:
            ec_membership = cc_membership  # 无桥 → 整个连通分量就是 2EC

        # O(1) 分类每条候选
        priority = []  # 跨 2EC 或跨 CC 的候选（会消除桥或合并分量）
        for i, cand in enumerate(valid_candidates):
            u, v = cand["u_idx"], cand["v_idx"]
            cost_penalty = max(cand["cost"], 1.0) ** cost_exponent
            if cc_membership[u] != cc_membership[v]:
                # 合并不同连通分量 → 最高优先
                score = 100.0 / cost_penalty
                priority.append((score, i))
            elif ec_membership[u] != ec_membership[v]:
                # 跨 2-边连通分量 → 消除桥边
                score = 10.0 / cost_penalty
                priority.append((score, i))
            # else: 同一个 2EC 内部 → 不消除任何桥，跳过

        # 按预筛分数降序，只取 top-N 做完整 Tarjan
        priority.sort(key=lambda x: x[0], reverse=True)
        to_evaluate = [valid_candidates[idx] for _, idx in priority[:TOP_N]]

        if not to_evaluate:
            print(f"    Step {step}: 无跨瓶颈候选边，停止。")
            break

        # ---- 只对 top-N 候选做完整 Tarjan 评估 ----
        best_ratio = -1e30
        best_cand_ref = None
        best_info = None

        for cand in to_evaluate:
            info = eval_fn(
                G, cand["u_idx"], cand["v_idx"],
                baseline, lam_bridge,
            )
            cost_penalty = max(cand["cost"], 1.0) ** cost_exponent
            ratio = info["gain"] / cost_penalty

            if ratio > best_ratio:
                best_ratio = ratio
                best_cand_ref = cand
                best_info = info

        eval_time = time.perf_counter() - t_eval

        # 找到 best_cand_ref 在 valid_candidates 中的索引
        best_idx = -1
        if best_cand_ref is not None:
            for i, c in enumerate(valid_candidates):
                if c is best_cand_ref:
                    best_idx = i
                    break

        if best_idx < 0 or best_info["gain"] <= 0:
            print(f"    Step {step}: 无正收益候选边，停止。")
            break

        # ---- 加入最佳边 ----
        best_cand = valid_candidates.pop(best_idx)
        G.add_edge(best_cand["u_idx"], best_cand["v_idx"])

        # 更新已有边集（避免后续重复）
        existing_edges.add(best_cand["edge_key"])

        added_edges.append(best_cand)
        total_cost += best_cand["cost"]

        elapsed = time.perf_counter() - t_start
        print(
            f"    Step {step}: 加边 ({best_cand['u']}→{best_cand['v']}), "
            f"来源={best_cand['source']}, 成本={best_cand['cost']:.0f}m, "
            f"Gain={best_info['gain']:.1f}, ΔBridge={best_info['delta_bridge']}, "
            f"预筛={len(priority)}/{len(valid_candidates)+1}→评估{len(to_evaluate)}, "
            f"{eval_time:.1f}s, 总={elapsed:.1f}s"
        )

        # ---- 精确复核 ----
        if step % n_check == 0:
            print(f"\n    === 精确复核 (已加 {step} 条边) ===")
            t_cabs = time.perf_counter()
            _, current_R = run_cabs_on_graph(G, **cabs_params)
            cabs_time = time.perf_counter() - t_cabs
            print(
                f"    R_bc = {current_R:.6f} "
                f"(目标 ≥ {target_R:.6f}), "
                f"CABS 耗时 {cabs_time:.1f}s\n"
            )

            if current_R >= target_R:
                print(f"    >>> 达标! R_bc = {current_R:.6f} ≥ T = {target_R:.6f}")
                break

    # 最终复核（如果没在循环中做过）
    if current_R is None or step % n_check != 0:
        print(f"\n    === 最终精确复核 ===")
        _, current_R = run_cabs_on_graph(G, **cabs_params)
        print(f"    最终 R_bc = {current_R:.6f} (目标 ≥ {target_R:.6f})")

    return added_edges, total_cost, current_R


def reverse_pruning(
    G: ig.Graph,
    added_edges: list[dict],
    target_R: float,
    cabs_params: dict = None,
) -> tuple[list[dict], float, float]:
    """
    反向删边精简：尝试删除已加入的边，若目标仍满足则永久删去。

    参数
    ----
    G            : 已增强的图（会被就地修改）
    added_edges  : 按加入顺序排列的边列表
    target_R     : 目标健壮性下界
    cabs_params  : CABS 超参

    返回
    ----
    kept_edges   : 保留的边列表
    final_cost   : 最终总成本
    final_R      : 最终 R_bc
    """
    if cabs_params is None:
        cabs_params = {}

    print(f"\n  ==== 阶段二：反向删边精简 ({len(added_edges)} 条) ====")

    kept = list(added_edges)
    removed_count = 0

    # 逆序遍历
    for i in range(len(kept) - 1, -1, -1):
        edge = kept[i]
        u_idx, v_idx = edge["u_idx"], edge["v_idx"]

        # 找到这条边的 eid
        eid = G.get_eid(u_idx, v_idx, error=False)
        if eid < 0:
            continue

        # 虚拟删除
        G.delete_edges(eid)

        # 精确复核
        _, R_check = run_cabs_on_graph(G, **cabs_params)

        if R_check >= target_R:
            # 仍达标，永久删除
            kept.pop(i)
            removed_count += 1
            print(
                f"    删除边 ({edge['u']}→{edge['v']}): "
                f"R_bc={R_check:.6f} ≥ T, 节省 {edge['cost']:.0f}m"
            )
        else:
            # 不达标，加回来
            G.add_edge(u_idx, v_idx)
            print(
                f"    保留边 ({edge['u']}→{edge['v']}): "
                f"R_bc={R_check:.6f} < T, 必须保留"
            )

    final_cost = sum(e["cost"] for e in kept)

    # 最终复核
    _, final_R = run_cabs_on_graph(G, **cabs_params)
    print(
        f"\n    反向精简完成: 删除 {removed_count} 条冗余边, "
        f"保留 {len(kept)} 条, 总成本 {final_cost:.0f}m, "
        f"R_bc = {final_R:.6f}"
    )

    return kept, final_cost, final_R


# =====================================================================
# 结果保存
# =====================================================================

def save_augmented_graph(G: ig.Graph, raw_json_path: str, out_path: str,
                         added_edges: list[dict]):
    """
    将增强后的图保存为 JSON（在原始 JSON 基础上添加新边）。
    """
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 添加新边到 neighbors
    node_dict = {}
    for key, val in raw.items():
        node_dict[val["id"]] = (key, val)

    for edge in added_edges:
        u_nid, v_nid = edge["u"], edge["v"]
        cost = edge["cost"]

        if u_nid in node_dict:
            key_u, val_u = node_dict[u_nid]
            val_u["neighbors"].append({
                "id": v_nid,
                "distance": cost,
                "augmented": True,
            })

        if v_nid in node_dict:
            key_v, val_v = node_dict[v_nid]
            val_v["neighbors"].append({
                "id": u_nid,
                "distance": cost,
                "augmented": True,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)


def save_results_csv(added_edges: list[dict], total_cost: float,
                     R_before: float, R_after: float,
                     out_path: str):
    """保存加边结果到 CSV。"""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "u", "v", "source", "cost_m",
            "R_before", "R_after", "total_cost_m", "n_added",
        ])
        for i, edge in enumerate(added_edges):
            writer.writerow([
                i + 1, edge["u"], edge["v"], edge["source"],
                f"{edge['cost']:.1f}",
                f"{R_before:.6f}" if i == 0 else "",
                f"{R_after:.6f}" if i == len(added_edges) - 1 else "",
                f"{total_cost:.1f}" if i == len(added_edges) - 1 else "",
                len(added_edges) if i == len(added_edges) - 1 else "",
            ])


# =====================================================================
# 主程序
# =====================================================================

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_dir = os.path.join(base_dir, "data", "json_networks")
    cand_dir = os.path.join(base_dir, "Q5", "data")
    out_dir = os.path.join(base_dir, "Q5", "results")
    os.makedirs(out_dir, exist_ok=True)

    CITIES = [
        "Chengdu", "Dalian", "Dongguan", "Harbin",
        "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou",
    ]

    # ── 超参 ──
    # 目标: 所有城市的 R_bc 提升到最强城市（沈阳）的水平
    # 需要先运行 Q3 得到各城市的 R_bc 基线值
    # 这里先设置一个占位值，运行时替换为实际值
    TARGET_R = 0.10  # TODO: 替换为 max(各城市 R_bc)

    # CABS 超参（精确复核用，可适当降低精度以加速）
    CABS_PARAMS = {
        "W": 2,           # beam 宽度（复核时可用较小值）
        "K": 20,          # 候选数
        "batch_size": 5,
        "alpha": 0.5,
        "beta": 0.5,
        "theta": 0.01,
        "warmup_frac": 0.1,
    }

    N_CHECK = 40        # 每加 N_CHECK 条边精确复核一次
    MAX_ADD = 200       # 单城市最多加边数
    LAM_BRIDGE = 2.0    # 桥边消除权重
    COST_EXPONENT = 0.1 # 成本指数：ratio = Gain / cost^γ，γ越小越不在意成本
    DO_PRUNING = True   # 是否做反向删边精简

    # ── 第一步：获取各城市基线 R_bc ──
    # 检查是否有已存在的 Q3 结果
    q3_summary = os.path.join(base_dir, "Q3", "Results",
                              "Q3_CABS_BeamSearch_Summary.csv")
    city_R = {}
    if os.path.exists(q3_summary):
        import pandas as pd
        df = pd.read_csv(q3_summary)
        for _, row in df.iterrows():
            city_R[row["City"]] = row["R_bc_trapz"]
        TARGET_R = max(city_R.values())
        print(f"  从 Q3 结果加载基线 R_bc：")
        for c, r in sorted(city_R.items(), key=lambda x: x[1]):
            print(f"    {c}: R_bc = {r:.6f}")
        print(f"  目标 T = {TARGET_R:.6f} (最强城市)")
    else:
        print(f"  [警告] 未找到 Q3 结果，使用默认目标 T = {TARGET_R:.6f}")
        print(f"  请先运行 Q3/new_algorithm.py 获取各城市基线 R_bc")

    print(f"\n{'='*60}")
    print(f"  Q5 贪心加边增强")
    print(f"  目标 R_bc ≥ {TARGET_R:.6f}")
    print(f"{'='*60}")

    summary_rows = []

    for city in CITIES:
        json_path = os.path.join(json_dir, f"{city}_Network.json")
        cand_path = os.path.join(cand_dir, f"{city}_candidates_selected.csv")
        if not os.path.exists(cand_path):
            cand_path = os.path.join(cand_dir, f"{city}_candidates.csv")

        if not os.path.exists(json_path):
            print(f"\n  [跳过] {city}: 找不到 {json_path}")
            continue
        if not os.path.exists(cand_path):
            print(f"\n  [跳过] {city}: 找不到候选边，请先运行 generate + select")
            continue

        # 如果该城市已达标，跳过
        R_baseline = city_R.get(city, 0.0)
        if R_baseline >= TARGET_R:
            print(f"\n  [跳过] {city}: R_bc={R_baseline:.6f} ≥ T={TARGET_R:.6f}，已达标")
            summary_rows.append({
                "City": city,
                "R_before": R_baseline,
                "R_after": R_baseline,
                "n_added": 0,
                "total_cost_m": 0,
                "status": "already_met",
            })
            continue

        print(f"\n  {'='*50}")
        print(f"  城市: {city}")
        print(f"  基线 R_bc = {R_baseline:.6f}, 差距 = {TARGET_R - R_baseline:.6f}")
        print(f"  {'='*50}")

        # ---- 加载 ----
        G = load_graph_from_json(json_path)
        candidates = load_candidates(cand_path)
        N0 = G.vcount()
        M0 = G.ecount()

        print(f"    N={N0}, M={M0}, 候选边={len(candidates)}")

        # 结构基线
        n_br, n_ap, be, bv = compute_structural_metrics(G)
        print(
            f"    结构基线: bridges={n_br}, artic_points={n_ap}, "
            f"|BE|={be}, |BV|={bv}"
        )

        # 判断是否用快速评估
        use_fast = len(candidates) > 3000
        if use_fast:
            print(f"    候选数 > 3000，启用快速评估模式")

        # ---- 阶段一：代理贪心加边 ----
        print(f"\n  ==== 阶段一：代理贪心加边 ====")
        added, cost, R_after = augment_greedy(
            G, candidates,
            target_R=TARGET_R,
            n_check=N_CHECK,
            max_add=MAX_ADD,
            lam_bridge=LAM_BRIDGE,
            cost_exponent=COST_EXPONENT,
            cabs_params=CABS_PARAMS,
            use_fast_eval=use_fast,
        )

        # ---- 阶段二：反向删边精简（可选）----
        if DO_PRUNING and len(added) > 1 and R_after >= TARGET_R:
            kept, cost, R_after = reverse_pruning(
                G, added, TARGET_R, CABS_PARAMS,
            )
        else:
            kept = added

        # ---- 保存结果 ----
        # 加边结果 CSV
        csv_out = os.path.join(out_dir, f"{city}_augmentation.csv")
        save_results_csv(kept, cost, R_baseline, R_after, csv_out)
        print(f"    结果已保存 → {csv_out}")

        # 增强后的图 JSON
        json_out = os.path.join(out_dir, f"{city}_Network_augmented.json")
        save_augmented_graph(G, json_path, json_out, kept)
        print(f"    增强图已保存 → {json_out}")

        # 增强后结构
        n_br2, n_ap2, be2, bv2 = compute_structural_metrics(G)
        print(
            f"    增强后结构: bridges={n_br2} (Δ={n_br2-n_br}), "
            f"artic_points={n_ap2} (Δ={n_ap2-n_ap}), "
            f"|BE|={be2} (Δ={be2-be}), |BV|={bv2} (Δ={bv2-bv})"
        )

        summary_rows.append({
            "City": city,
            "R_before": R_baseline,
            "R_after": R_after,
            "n_added": len(kept),
            "total_cost_m": cost,
            "status": "met" if R_after >= TARGET_R else "not_met",
            "bridges_before": n_br,
            "bridges_after": n_br2,
            "BE_before": be,
            "BE_after": be2,
            "BV_before": bv,
            "BV_after": bv2,
        })

    # ── 全局汇总 ──
    print(f"\n{'='*60}")
    print(f"  Q5 全局汇总")
    print(f"{'='*60}")

    summary_csv = os.path.join(out_dir, "Q5_Augmentation_Summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    for row in summary_rows:
        status = "✓" if row["status"] in ("met", "already_met") else "✗"
        print(
            f"  {status} {row['City']:>10s}: "
            f"R {row['R_before']:.6f} → {row['R_after']:.6f}, "
            f"+{row['n_added']}条, 成本={row['total_cost_m']:.0f}m"
        )

    print(f"\n  汇总已保存 → {summary_csv}")


if __name__ == "__main__":
    main()
