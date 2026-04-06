"""
Q5/augment_network.py
CABS 引导的批量加边增强算法（LCC 快速版）

加速措施：
  1. 直接加载 Q3 已有的攻击序列做疫苗排序，不重新跑 CABS
  2. 精确复核用 new_algorithm_origin.py（LCC 版），不算双连通，快 3~5 倍

用法:
    python -m Q5.augment_network
"""

import os
import sys
import time
import json
import csv
import igraph as ig
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# 图加载
# =====================================================================

def load_graph_from_json(json_path: str) -> ig.Graph:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    node_list = list(raw.values())
    n = len(node_list)
    id2idx = {val["id"]: i for i, val in enumerate(node_list)}
    edges, weights = [], []
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


def load_candidates(csv_path: str) -> list[dict]:
    candidates = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        dist_col = None
        for name in ["euclid_dist", "euclidean_dist", "dist", "distance"]:
            if name in header:
                dist_col = name
                break
        if dist_col is None:
            for name in header:
                if "dist" in name.lower():
                    dist_col = name
                    break
        if dist_col is None:
            raise KeyError(f"找不到距离列。可用列: {header}")
        for row in reader:
            candidates.append({
                "u": int(row["u"]),
                "v": int(row["v"]),
                "source": row.get("source", "unknown"),
                "cost": float(row[dist_col]),
            })
    return candidates


def load_attack_sequence(csv_path: str) -> list[int]:
    """从 Q3 结果加载已有的攻击序列。"""
    df = pd.read_csv(csv_path)
    return df["node_id"].tolist()


# =====================================================================
# CABS-LCC 调用（快速版，不算双连通）
# =====================================================================

def run_cabs_lcc(
    G: ig.Graph,
    W: int = 3, K: int = 30, batch_size: int = 5,
    theta: float = 0.01, warmup_frac: float = 0.05,
    **kwargs,
) -> tuple[list[int], float]:
    """
    用 LCC 版 CABS 跑攻击，返回 (attack_seq, R_lcc)。
    如果 beam search 结束后 LCC 仍 > theta，继续用贪心补完直到达标。
    """
    try:
        from Q3.new_algorithm_origin import (
            greedy_warmup, beam_search_phase, get_lcc_ratio,
        )
    except ImportError:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Q3"))
        from new_algorithm_origin import (
            greedy_warmup, beam_search_phase, get_lcc_ratio,
        )

    N0 = G.vcount()
    G_work = G.copy()
    warmup_count = int(N0 * warmup_frac)

    # Phase A: 贪心预热
    G_warm, warmup_seq, warmup_R = greedy_warmup(
        G_work, N0, warmup_count, theta=theta,
    )
    if get_lcc_ratio(G_warm, N0) <= theta:
        return warmup_seq, warmup_R

    # Phase B: Beam Search
    full_seq, full_R = beam_search_phase(
        G_warm, warmup_seq, warmup_R, N0,
        W=W, K=K, batch_size=batch_size, theta=theta,
    )
    return full_seq, full_R


# =====================================================================
# 疫苗式候选排序
# =====================================================================

def detect_communities(G: ig.Graph) -> list[int]:
    if G.vcount() <= 2:
        return list(range(G.vcount()))
    try:
        return G.community_leiden(objective_function="modularity").membership
    except AttributeError:
        return G.community_multilevel().membership
    except Exception:
        return [0] * G.vcount()


def rank_candidates_by_vaccination(
    G: ig.Graph,
    candidates: list[dict],
    attack_seq: list[int],
    top_frac: float = 0.05,
) -> list[dict]:
    """
    根据 CABS 攻击序列，为每条候选边计算「疫苗分数」。
    优先选择能为关键攻击节点提供跨社区旁路的边。
    """
    N0 = G.vcount()
    id2idx = {G.vs[i]["node_id"]: i for i in range(N0)}

    n_critical = max(20, int(len(attack_seq) * top_frac))
    critical_idxs = set()
    for nid in attack_seq[:n_critical]:
        if nid in id2idx:
            critical_idxs.add(id2idx[nid])

    print(f"    关键攻击节点: {len(critical_idxs)} 个")

    membership = detect_communities(G)

    # 每个关键节点的邻居集
    critical_neighbors = {}
    for c_idx in critical_idxs:
        critical_neighbors[c_idx] = set(G.neighbors(c_idx))

    # 关键区域（关键节点 + 其邻居）
    critical_zone = set()
    for c_idx in critical_idxs:
        critical_zone.add(c_idx)
        critical_zone.update(critical_neighbors[c_idx])

    # 已有边集
    existing_edges = set()
    for e in G.es:
        s, t = e.source, e.target
        existing_edges.add((min(s, t), max(s, t)))

    scored = []
    for cand in candidates:
        u_nid, v_nid = cand["u"], cand["v"]
        if u_nid not in id2idx or v_nid not in id2idx:
            continue
        u_idx, v_idx = id2idx[u_nid], id2idx[v_nid]
        edge_key = (min(u_idx, v_idx), max(u_idx, v_idx))
        if edge_key in existing_edges:
            continue

        vaccine_score = 0.0
        cross_community = membership[u_idx] != membership[v_idx]

        for c_idx in critical_idxs:
            nbs = critical_neighbors[c_idx]
            u_near = (u_idx == c_idx or u_idx in nbs)
            v_near = (v_idx == c_idx or v_idx in nbs)

            if u_near and v_near and cross_community:
                vaccine_score += 5.0
            elif u_near and v_near:
                vaccine_score += 2.0
            elif (u_near or v_near) and cross_community:
                vaccine_score += 1.0

        if vaccine_score == 0 and cross_community:
            if u_idx in critical_zone or v_idx in critical_zone:
                vaccine_score += 0.5

        scored.append({
            **cand,
            "u_idx": u_idx,
            "v_idx": v_idx,
            "edge_key": edge_key,
            "vaccine_score": vaccine_score,
        })

    scored.sort(key=lambda x: x["vaccine_score"], reverse=True)

    n_pos = sum(1 for s in scored if s["vaccine_score"] > 0)
    print(f"    有效候选: {len(scored)}, 正分候选: {n_pos}")
    if scored and scored[0]["vaccine_score"] > 0:
        print(f"    最高分: {scored[0]['vaccine_score']:.1f}, "
              f"前10平均: {np.mean([s['vaccine_score'] for s in scored[:10]]):.1f}")

    return scored


# =====================================================================
# 回放引擎（在 Graph 对象上，和 Q3 的 replay 保持一致）
# =====================================================================

def replay_on_graph(
    G: ig.Graph,
    attack_seq: list[int],
    step_size: float = 0.01,
) -> float:
    """
    用攻击序列回放，梯形积分算 R_lcc。
    和 Q3 的 replay_attack_sequence 完全一致的积分方式。
    """
    from utils_ig import rebuild_id2idx

    N0 = G.vcount()
    G_temp = G.copy()
    id2idx = rebuild_id2idx(G_temp)

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    p_vals = [max(G_temp.connected_components().sizes()) / N0
              if G_temp.vcount() > 0 else 0.0]

    current_idx = 0
    for i in range(1, len(q_points)):
        q = q_points[i]
        target = int(N0 * q)
        num = target - current_idx
        if num > 0:
            batch = attack_seq[current_idx: min(current_idx + num, len(attack_seq))]
            indices = [id2idx[nid] for nid in batch if nid in id2idx]
            if indices:
                G_temp.delete_vertices(indices)
                id2idx = rebuild_id2idx(G_temp)
            current_idx += num
        if G_temp.vcount() > 0:
            p_vals.append(max(G_temp.connected_components().sizes()) / N0)
        else:
            p_vals.append(0.0)

    return float(np.trapz(p_vals, q_points))


# =====================================================================
# 批量加边 + CABS 验证
# =====================================================================

def augment_with_verification(
    G: ig.Graph,
    ranked_candidates: list[dict],
    target_R: float,
    batch_size: int = 10,
    max_batches: int = 20,
    cabs_params: dict = None,
) -> tuple[list[dict], float, float]:
    if cabs_params is None:
        cabs_params = {}

    added_edges = []
    total_cost = 0.0
    cursor = 0
    best_R = 0.0
    t_start = time.perf_counter()

    for batch_round in range(1, max_batches + 1):
        if cursor >= len(ranked_candidates):
            print(f"    候选边用完。")
            break

        batch_added = []
        while cursor < len(ranked_candidates) and len(batch_added) < batch_size:
            cand = ranked_candidates[cursor]
            cursor += 1
            u_idx, v_idx = cand["u_idx"], cand["v_idx"]
            eid = G.get_eid(u_idx, v_idx, error=False)
            if eid >= 0:
                continue
            G.add_edge(u_idx, v_idx)
            batch_added.append(cand)

        if not batch_added:
            continue

        added_edges.extend(batch_added)
        batch_cost = sum(c["cost"] for c in batch_added)
        total_cost += batch_cost
        avg_score = np.mean([c["vaccine_score"] for c in batch_added])

        print(f"\n    === 第 {batch_round} 轮: +{len(batch_added)} 条 "
              f"(累计 {len(added_edges)}), 疫苗分 {avg_score:.1f} ===")

        t_cabs = time.perf_counter()
        attack_seq, R_beam = run_cabs_lcc(G, **cabs_params)
        # 用 replay + trapz 计算 R_lcc，和 Q3 基线保持一致
        R_new = replay_on_graph(G, attack_seq)
        cabs_time = time.perf_counter() - t_cabs
        elapsed = time.perf_counter() - t_start

        delta_R = R_new - best_R if best_R > 0 else 0
        print(f"    R_lcc = {R_new:.6f} (目标 ≥ {target_R:.6f}), "
              f"ΔR = {delta_R:+.6f}, CABS {cabs_time:.0f}s, 总 {elapsed:.0f}s")

        if R_new > best_R:
            best_R = R_new

        if best_R >= target_R:
            print(f"\n    >>> 达标! R_lcc = {best_R:.6f} ≥ T = {target_R:.6f}")
            break

    return added_edges, total_cost, best_R


# =====================================================================
# 结果保存
# =====================================================================

def save_augmented_graph(G, raw_json_path, out_path, added_edges):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    node_dict = {val["id"]: val for val in raw.values()}
    for edge in added_edges:
        u_nid, v_nid, cost = edge["u"], edge["v"], edge["cost"]
        if u_nid in node_dict:
            node_dict[u_nid]["neighbors"].append(
                {"id": v_nid, "distance": cost, "augmented": True})
        if v_nid in node_dict:
            node_dict[v_nid]["neighbors"].append(
                {"id": u_nid, "distance": cost, "augmented": True})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)


def save_results_csv(added_edges, total_cost, R_before, R_after, out_path):
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "u", "v", "source", "cost_m", "vaccine_score"])
        for i, edge in enumerate(added_edges):
            writer.writerow([
                i + 1, edge["u"], edge["v"], edge["source"],
                f"{edge['cost']:.1f}", f"{edge.get('vaccine_score', 0):.1f}",
            ])


# =====================================================================
# 主程序
# =====================================================================

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_dir = os.path.join(base_dir, "data", "json_networks")
    cand_dir = os.path.join(base_dir, "Q5", "data")
    q3_dir = os.path.join(base_dir, "Q3", "Results")
    out_dir = os.path.join(base_dir, "Q5", "results")
    os.makedirs(out_dir, exist_ok=True)

    CITIES = [
        #"Chengdu", "Dalian", "Dongguan", "Harbin",
        "Qingdao", "Quanzhou", #"Shenyang", "Zhengzhou",
    ]

    # ── 超参 ──
    CABS_PARAMS = {
        "W": 3,
        "K": 30,
        "batch_size": 5,
        "theta": 0.01,
        "warmup_frac": 0.05,
    }

    BATCH_SIZE = 40
    MAX_BATCHES = 20
    TOP_FRAC = 0.05

    # ── 加载 Q3 LCC 基线 ──
    lcc_summary_path = os.path.join(q3_dir, "Q3_CABS_LCC_Summary.csv")
    bc_summary_path = os.path.join(q3_dir, "Q3_CABS_BeamSearch_Summary.csv")

    city_R_lcc = {}
    if os.path.exists(lcc_summary_path):
        df = pd.read_csv(lcc_summary_path)
        for _, row in df.iterrows():
            # 尝试多种可能的列名
            r_val = None
            for col in ["R_lcc_trapz", "R_bc_trapz", "R_lcc_beam", "R_lcc"]:
                if col in df.columns:
                    r_val = row[col]
                    break
            if r_val is not None:
                city_R_lcc[row["City"]] = r_val
        print(f"  从 Q3 LCC 结果加载基线:")
    elif os.path.exists(bc_summary_path):
        df = pd.read_csv(bc_summary_path)
        for _, row in df.iterrows():
            city_R_lcc[row["City"]] = row["R_bc_trapz"]
        print(f"  从 Q3 BC 结果加载基线 (回退):")
    else:
        print(f"  [警告] 未找到 Q3 结果")

    if city_R_lcc:
        for c, r in sorted(city_R_lcc.items(), key=lambda x: x[1]):
            print(f"    {c}: R = {r:.6f}")
        TARGET_R = max(city_R_lcc.values())
        print(f"  目标 T = {TARGET_R:.6f}")
    else:
        TARGET_R = 0.040556

    print(f"\n{'='*60}")
    print(f"  Q5 CABS-LCC 引导批量加边增强")
    print(f"  目标 R_lcc ≥ {TARGET_R:.6f}")
    print(f"{'='*60}")

    summary_rows = []

    for city in CITIES:
        json_path = os.path.join(json_dir, f"{city}_Network.json")

        # 候选边：优先 vaccine 版，可合并 selected 版
        cand_paths = []
        vaccine_path = os.path.join(cand_dir, f"{city}_candidates_vaccine.csv")
        selected_path = os.path.join(cand_dir, f"{city}_candidates_selected.csv")
        raw_path = os.path.join(cand_dir, f"{city}_candidates.csv")
        if os.path.exists(vaccine_path):
            cand_paths.append(vaccine_path)
        if os.path.exists(selected_path):
            cand_paths.append(selected_path)
        elif os.path.exists(raw_path):
            cand_paths.append(raw_path)

        # 攻击序列：优先 LCC 版，回退 BC 版
        seq_path = os.path.join(q3_dir, f"{city}_CABS_LCC_Attack_Sequence.csv")
        if not os.path.exists(seq_path):
            seq_path = os.path.join(q3_dir, f"{city}_CABS_Attack_Sequence.csv")

        if not os.path.exists(json_path) or not cand_paths:
            print(f"\n  [跳过] {city}: 文件不存在")
            continue

        R_baseline = city_R_lcc.get(city, 0.0)
        if R_baseline >= TARGET_R:
            print(f"\n  [跳过] {city}: R={R_baseline:.6f} ≥ T，已达标")
            summary_rows.append({
                "City": city, "R_before": R_baseline, "R_after": R_baseline,
                "n_added": 0, "total_cost_m": 0, "status": "already_met",
            })
            continue

        print(f"\n  {'='*50}")
        print(f"  城市: {city}")
        print(f"  基线 R = {R_baseline:.6f}, 差距 = {TARGET_R - R_baseline:.6f}")
        print(f"  {'='*50}")

        G = load_graph_from_json(json_path)

        # 合并多个候选源，去重
        candidates = []
        seen_pairs = set()
        for cp in cand_paths:
            for c in load_candidates(cp):
                pair = (min(c["u"], c["v"]), max(c["u"], c["v"]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidates.append(c)
            print(f"    加载 {os.path.basename(cp)}: 累计 {len(candidates)} 条")
        print(f"    N={G.vcount()}, M={G.ecount()}, 候选边={len(candidates)}")

        # ---- 第一步：加载已有攻击序列（不重新跑 CABS）----
        if os.path.exists(seq_path):
            print(f"\n  >> 加载 Q3 攻击序列: {os.path.basename(seq_path)} <<")
            attack_seq = load_attack_sequence(seq_path)
            print(f"    序列长度: {len(attack_seq)}")
        else:
            print(f"\n  >> 未找到攻击序列，现场跑 CABS-LCC <<")
            t0 = time.perf_counter()
            attack_seq, R_original = run_cabs_lcc(G, **CABS_PARAMS)
            print(f"    R_lcc = {R_original:.6f}, 耗时 {time.perf_counter()-t0:.0f}s")

        # ---- 第二步：疫苗式排序 ----
        print(f"\n  >> 疫苗式排序候选边 <<")
        ranked = rank_candidates_by_vaccination(
            G, candidates, attack_seq, top_frac=TOP_FRAC,
        )
        if not ranked:
            print(f"    无有效候选边。")
            continue

        # ---- 第三步：批量加边 + CABS-LCC 验证 ----
        print(f"\n  >> 批量加边 + CABS-LCC 验证 <<")
        added, cost, R_after = augment_with_verification(
            G, ranked,
            target_R=TARGET_R,
            batch_size=BATCH_SIZE,
            max_batches=MAX_BATCHES,
            cabs_params=CABS_PARAMS,
        )

        # ---- 保存 ----
        csv_out = os.path.join(out_dir, f"{city}_augmentation.csv")
        save_results_csv(added, cost, R_baseline, R_after, csv_out)

        json_out = os.path.join(out_dir, f"{city}_Network_augmented.json")
        save_augmented_graph(G, json_path, json_out, added)

        print(f"\n    结果: +{len(added)} 条边, 成本 {cost:.0f}m, "
              f"R {R_baseline:.6f} → {R_after:.6f}")

        summary_rows.append({
            "City": city, "R_before": R_baseline, "R_after": R_after,
            "n_added": len(added), "total_cost_m": cost,
            "status": "met" if R_after >= TARGET_R else "not_met",
        })

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  Q5 全局汇总")
    print(f"{'='*60}")

    summary_csv = os.path.join(out_dir, "Q5_Augmentation_Summary.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    for row in summary_rows:
        status = "✓" if row["status"] in ("met", "already_met") else "✗"
        print(f"  {status} {row['City']:>10s}: "
              f"R {row['R_before']:.6f} → {row['R_after']:.6f}, "
              f"+{row['n_added']}条, 成本={row['total_cost_m']:.0f}m")

    print(f"\n  汇总 → {summary_csv}")


if __name__ == "__main__":
    main()
