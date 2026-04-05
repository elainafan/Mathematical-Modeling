"""
Q3/new_algorithm_origin.py
CABS-LCC: Community-Aware Beam Search for Network Dismantling
面向赛题原生指标 Q_{lcc} = |LCC(G)|/N_0 的两阶段拆解算法。

与 new_algorithm.py 的区别
--------------------------
优化目标从 Q_bc（双连通核指标）替换为 Q_lcc = LCC/N_0（赛题原生指标）。
由于不需要计算桥边和点双连通分量，每步评估开销大幅降低，速度显著更快。

两阶段策略
----------
Phase A — 贪心预热（Greedy Warm-up）
    用 Leiden 社区外部度贪心逐节点删除。
    候选节点仅从当前 LCC 内选取。

Phase B — 集束精细搜索（Beam Search Refinement）
    在缩小后的图上执行 Beam Search：
    - 候选节点仅从 LCC 内部选取
    - 候选评估多线程并行化
    - 支持批量删除（每步删 batch_size 个节点）

输出
----
回放时同时输出 P_q（LCC/N_0）和 Q_bc 两条曲线，便于与 new_algorithm.py 对比。
"""

import os
import sys
import time
import concurrent.futures
import igraph as ig
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils_ig import load_graph, rebuild_id2idx, compute_performance
except ImportError:
    print("错误: 找不到 utils_ig.py，请检查路径。")
    sys.exit(1)


# =====================================================================
# 辅助函数
# =====================================================================

def compute_external_degree(G: ig.Graph, membership: list[int]) -> list[int]:
    """计算每个节点的社区外部度 k_ext(v)，复杂度 O(M)。"""
    ext_deg = [0] * G.vcount()
    for e in G.es:
        s, t = e.source, e.target
        if membership[s] != membership[t]:
            ext_deg[s] += 1
            ext_deg[t] += 1
    return ext_deg


def get_lcc_info(G: ig.Graph) -> tuple[int, int, list[int]]:
    """
    返回 (lcc_size, lcc_idx_in_clusters, lcc_vertex_indices)。
    如果图为空返回 (0, -1, [])。
    """
    if G.vcount() == 0:
        return 0, -1, []
    cc = G.connected_components()
    sizes = cc.sizes()
    lcc_size = max(sizes)
    lcc_idx = sizes.index(lcc_size)
    lcc_members = cc[lcc_idx]
    return lcc_size, lcc_idx, lcc_members


def get_lcc_ratio(G: ig.Graph, N0: int) -> float:
    """计算当前图的 LCC / N_0。"""
    if G.vcount() == 0:
        return 0.0
    return max(G.connected_components().sizes()) / N0


def detect_communities(G: ig.Graph) -> list[int]:
    """
    Leiden 社区检测，保证每个社区是连通子图。
    若 igraph 版本不支持则回退 Louvain。
    """
    if G.vcount() <= 2:
        return list(range(G.vcount()))
    try:
        return G.community_leiden(objective_function="modularity").membership
    except AttributeError:
        return G.community_multilevel().membership
    except Exception:
        return [0] * G.vcount()


def get_lcc_candidates(G: ig.Graph, K: int) -> list[tuple[int, int]]:
    """
    从当前图的 LCC 中筛选 top-K 候选攻击节点。

    步骤：
    1. 找到 LCC 的节点集合
    2. 提取 LCC 子图
    3. 在 LCC 子图上做 Leiden 社区检测
    4. 在 LCC 子图上计算社区外部度
    5. 按 (ext_deg↓, degree↓) 排序取 top-K
    6. 将子图索引映射回原图索引

    返回
    ----
    list[(原图内部索引, 原始节点ID)]，按优先级降序，长度 ≤ K
    """
    lcc_size, lcc_idx, lcc_members = get_lcc_info(G)

    if lcc_size == 0:
        return []

    if lcc_size <= K:
        return [(i, G.vs[i]["node_id"]) for i in lcc_members]

    # 提取 LCC 子图
    lcc_sub = G.induced_subgraph(lcc_members)

    # 在 LCC 子图上做社区检测
    membership = detect_communities(lcc_sub)
    ext_deg = compute_external_degree(lcc_sub, membership)
    degrees = lcc_sub.degree()

    # 按 (ext_deg↓, degree↓) 排序取 top-K
    indexed = [
        (ext_deg[j], degrees[j], j)
        for j in range(lcc_sub.vcount())
    ]
    indexed.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # 映射回原图索引
    result = []
    for _, _, sub_idx in indexed[:K]:
        orig_idx = lcc_members[sub_idx]
        nid = G.vs[orig_idx]["node_id"]
        result.append((orig_idx, nid))

    return result


def _evaluate_candidate(args) -> dict:
    """
    线程工作函数：评估删除单个候选节点后的 LCC/N_0。

    与 new_algorithm.py 的区别：用 get_lcc_ratio 替代 compute_performance，
    不需要计算桥边和点双连通，速度更快。
    """
    G_b, v_idx, nid, R_b, N0, delta_f, b_idx = args
    keep = [i for i in range(G_b.vcount()) if i != v_idx]
    sub = G_b.induced_subgraph(keep)
    lcc = get_lcc_ratio(sub, N0)
    return {
        "R": R_b + lcc * delta_f,
        "b_idx": b_idx, "nid": nid, "v_idx": v_idx,
        "lcc": lcc,
    }


# =====================================================================
# Phase A: 贪心预热（候选限定在 LCC 内）
# =====================================================================

def greedy_warmup(
    G:       ig.Graph,
    N0:      int,
    target_count: int,
    leiden_refresh: int = 50,
    theta:   float = 0.01,
) -> tuple[ig.Graph, list[int], float]:
    """
    用 Leiden 外部度贪心删除 target_count 个节点。
    候选节点仅从 LCC 内部选取。
    累积积分基于 Q_lcc = LCC/N_0。

    返回
    ----
    G_out       : 删除后的图（副本）
    warmup_seq  : 被删节点 ID 列表（按删除顺序）
    warmup_R    : 预热阶段累积积分 Σ (LCC/N_0) · Δf
    """
    G_out = G.copy()
    delta_f = 1.0 / N0
    warmup_seq: list[int] = []
    cumR = 0.0

    t0 = time.perf_counter()

    # 预生成的删除名单（node_id 列表）
    planned_nids: list[int] = []
    plan_cursor = 0

    def refresh_plan():
        """在 LCC 子图上做 Leiden，生成下一批删除名单。"""
        nonlocal planned_nids, plan_cursor

        _, _, lcc_members = get_lcc_info(G_out)
        if len(lcc_members) <= 2:
            degrees = G_out.degree()
            ranked = sorted(lcc_members, key=lambda i: degrees[i], reverse=True)
            planned_nids = [G_out.vs[i]["node_id"] for i in ranked]
        else:
            lcc_sub = G_out.induced_subgraph(lcc_members)
            membership = detect_communities(lcc_sub)
            ext_deg = compute_external_degree(lcc_sub, membership)
            degrees = lcc_sub.degree()

            indexed = sorted(
                range(lcc_sub.vcount()),
                key=lambda j: (ext_deg[j], degrees[j]),
                reverse=True,
            )
            planned_nids = [lcc_sub.vs[j]["node_id"] for j in indexed]

        plan_cursor = 0

    # 初始名单
    refresh_plan()
    id2idx = rebuild_id2idx(G_out)

    for step in range(1, target_count + 1):
        if G_out.vcount() == 0:
            break

        lcc_ratio = get_lcc_ratio(G_out, N0)
        if lcc_ratio <= theta:
            print(f"      [Warmup 提前终止] step={step}, LCC/N_0={lcc_ratio:.4f} <= θ")
            break

        # 定期刷新名单
        if plan_cursor >= len(planned_nids) or (step - 1) % leiden_refresh == 0:
            refresh_plan()
            id2idx = rebuild_id2idx(G_out)

        # 从名单中找下一个仍在图中的节点
        found = False
        while plan_cursor < len(planned_nids):
            nid = planned_nids[plan_cursor]
            plan_cursor += 1
            if nid in id2idx:
                found = True
                break

        if not found:
            refresh_plan()
            id2idx = rebuild_id2idx(G_out)
            while plan_cursor < len(planned_nids):
                nid = planned_nids[plan_cursor]
                plan_cursor += 1
                if nid in id2idx:
                    found = True
                    break

        if not found:
            break

        # 删除节点
        idx = id2idx[nid]
        warmup_seq.append(nid)
        G_out.delete_vertices(idx)
        id2idx = rebuild_id2idx(G_out)

        # 累积积分：使用 LCC/N_0
        lcc_val = get_lcc_ratio(G_out, N0)
        cumR += lcc_val * delta_f

        if step % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"      [Warmup] step={step}/{target_count}, "
                f"LCC/N_0={lcc_val:.4f}, R_lcc={cumR:.6f}, "
                f"耗时={elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - t0
    lcc = get_lcc_ratio(G_out, N0)
    print(
        f"    [Warmup 完成] 删除 {len(warmup_seq)} 节点, "
        f"LCC/N_0={lcc:.4f}, R_lcc={cumR:.6f}, 耗时={elapsed:.1f}s"
    )

    return G_out, warmup_seq, cumR


# =====================================================================
# Phase B: Beam Search（候选限定在 LCC 内，目标函数为 LCC/N_0）
# =====================================================================

def beam_search_phase(
    G_init:     ig.Graph,
    init_seq:   list[int],
    init_R:     float,
    N0:         int,
    W:          int   = 3,
    K:          int   = 30,
    batch_size: int   = 5,
    theta:      float = 0.01,
    n_workers:  int   = 0,
    log_interval: int = 20,
) -> tuple[list[int], float]:
    """
    在已完成预热的图上执行 Beam Search。
    目标函数：最小化 R_lcc = Σ (LCC/N_0) · Δf。
    候选节点仅从 LCC 内部选取。

    返回
    ----
    best_seq : 完整攻击序列（含预热 + beam search）
    best_R   : 累积积分 R_lcc
    """
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 4, 8)

    delta_f = 1.0 / N0

    G0 = G_init.copy()
    lcc0 = get_lcc_ratio(G0, N0)
    beam = [(G0, list(init_seq), init_R, lcc0)]

    step = 0
    t_start = time.perf_counter()

    while True:
        # 终止条件
        if all(s[3] <= theta for s in beam):
            break
        if all(s[0].vcount() == 0 for s in beam):
            break

        step += 1

        # ---- 候选生成（仅 LCC 内）+ 多线程评估 ----
        scored: list[dict] = []
        eval_tasks: list[tuple] = []

        for b_idx, (G_b, seq_b, R_b, P_b) in enumerate(beam):
            if P_b <= theta or G_b.vcount() == 0:
                scored.append({
                    "R": R_b, "b_idx": b_idx, "nid": None,
                    "v_idx": None, "lcc": P_b,
                })
                continue

            # 从 LCC 中选 top-K 候选
            candidates = get_lcc_candidates(G_b, K)

            for orig_idx, nid in candidates:
                eval_tasks.append((
                    G_b, orig_idx, nid,
                    R_b, N0, delta_f, b_idx,
                ))

        # 多线程评估
        if eval_tasks:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers,
            ) as pool:
                results = list(pool.map(_evaluate_candidate, eval_tasks))
            scored.extend(results)

        if not scored:
            break

        # ---- 排序 + 剪枝 ----
        scored.sort(key=lambda x: x["R"])
        selected = scored[:W]

        # ---- 构建新 beam（含批量删除）----
        new_beam = []
        for sel in selected:
            b_idx = sel["b_idx"]
            nid = sel["nid"]

            if nid is None:
                new_beam.append(beam[b_idx])
                continue

            G_old = beam[b_idx][0]
            seq_old = beam[b_idx][1]

            # 第一个节点：beam search 选出的最优候选
            v_idx = sel["v_idx"]
            keep = [i for i in range(G_old.vcount()) if i != v_idx]
            G_new = G_old.induced_subgraph(keep)
            new_seq = seq_old + [nid]
            new_R = sel["R"]

            # 追加 (batch_size - 1) 个节点：从 LCC 中贪心选择（用度数）
            for _ in range(batch_size - 1):
                if G_new.vcount() <= 1:
                    break
                lcc_check = get_lcc_ratio(G_new, N0)
                if lcc_check <= theta:
                    break

                _, _, lcc_members = get_lcc_info(G_new)
                if not lcc_members:
                    break

                degrees_g = G_new.degree()
                best_g = max(lcc_members, key=lambda i: degrees_g[i])
                greedy_nid = G_new.vs[best_g]["node_id"]
                G_new.delete_vertices(best_g)
                new_seq.append(greedy_nid)

                # 累积积分：使用 LCC/N_0
                lcc_g = get_lcc_ratio(G_new, N0)
                new_R += lcc_g * delta_f

            lcc_new = get_lcc_ratio(G_new, N0)
            new_beam.append((G_new, new_seq, new_R, lcc_new))

        beam = new_beam

        # ---- 日志 ----
        if step % log_interval == 0 or step <= 3:
            best = min(beam, key=lambda x: x[2])
            elapsed = time.perf_counter() - t_start
            removed = len(best[1])
            q_now = removed / N0
            print(
                f"    Step {step:>4d}: removed={removed:>5d}, "
                f"q={q_now:.4f}, LCC/N_0={best[3]:.4f}, "
                f"R_lcc={best[2]:.6f}, 耗时={elapsed:.1f}s"
            )

    # 输出最优
    best = min(beam, key=lambda x: x[2])
    elapsed = time.perf_counter() - t_start
    print(
        f"    [Beam Search 完成] 序列长度={len(best[1])}, "
        f"R_lcc={best[2]:.6f}, 耗时={elapsed:.1f}s"
    )
    return best[1], best[2]


# =====================================================================
# 主入口：组合两阶段
# =====================================================================

def cabs_attack(
    json_path:    str,
    W:            int   = 3,
    K:            int   = 30,
    batch_size:   int   = 5,
    theta:        float = 0.01,
    warmup_frac:  float = 0.05,
    leiden_refresh: int = 50,
    n_workers:    int   = 0,
    city_name:    str   = "",
) -> tuple[list[int], float]:
    """
    CABS-LCC 两阶段攻击算法（面向原生 LCC 指标）。

    参数
    ----
    warmup_frac    : 贪心预热阶段删除的节点比例（默认 5%）
    batch_size     : beam search 阶段每步实际删除的节点数
    leiden_refresh : 贪心阶段每隔多少步重算 Leiden
    其余参数同 beam_search_phase
    """
    G, _, _ = load_graph(json_path)
    N0 = G.vcount()

    print(f"\n  [{city_name}] N_0 = {N0}, M = {G.ecount()}")
    print(f"  [{city_name}] 超参: W={W}, K={K}, θ={theta}, "
          f"batch={batch_size}, workers={n_workers or min(os.cpu_count() or 4, 8)}")
    print(f"  [{city_name}] 目标指标: R_lcc = ∫(LCC/N_0)df（赛题原生）")
    print(f"  [{city_name}] 贪心预热: {warmup_frac*100:.0f}% = "
          f"{int(N0*warmup_frac)} 节点")

    # ==== Phase A: 贪心预热 ====
    warmup_count = int(N0 * warmup_frac)
    print(f"\n  ---- Phase A: 贪心预热 ({warmup_count} 节点) ----")
    G_warm, warmup_seq, warmup_R = greedy_warmup(
        G, N0, warmup_count,
        leiden_refresh=leiden_refresh,
        theta=theta,
    )

    # 检查预热后是否已达标
    if get_lcc_ratio(G_warm, N0) <= theta:
        print(f"  [{city_name}] 预热阶段已达 LCC/N_0 ≤ θ，跳过 Beam Search!")
        return warmup_seq, warmup_R

    # ==== Phase B: Beam Search ====
    print(f"\n  ---- Phase B: Beam Search (W={W}, K={K}, batch={batch_size}) ----")
    full_seq, full_R = beam_search_phase(
        G_warm, warmup_seq, warmup_R, N0,
        W=W, K=K, batch_size=batch_size,
        theta=theta,
        n_workers=n_workers,
    )

    print(
        f"\n  [{city_name}] 全部完成! "
        f"序列长度={len(full_seq)}, R_lcc={full_R:.6f}"
    )

    return full_seq, full_R


# =====================================================================
# 回放引擎（与 baseline 对齐，同时输出 P_q 和 Q_bc）
# =====================================================================

def replay_attack_sequence(
    json_path:   str,
    sorted_nodes: list[int],
    step_size:   float = 0.01,
    alpha:       float = 0.5,
    beta:        float = 0.5,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    用已知攻击序列回放，输出标准步长的 P_q 和 Q_bc 曲线。
    P_q 即本文件的优化目标 LCC/N_0，Q_bc 用于与 new_algorithm.py 的结果对比。
    """
    G, _, _ = load_graph(json_path)
    N0 = G.vcount()
    G_temp = G.copy()
    current_id2idx = rebuild_id2idx(G_temp)

    q_points = np.arange(0.0, 1.0 + step_size / 2, step_size)
    P_q_vals = [get_lcc_ratio(G_temp, N0)]
    Q_bc_vals = [compute_performance(G_temp, N0, alpha, beta)]

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
        P_q_vals.append(get_lcc_ratio(G_temp, N0))
        Q_bc_vals.append(compute_performance(G_temp, N0, alpha, beta))

    return q_points, P_q_vals, Q_bc_vals


# =====================================================================
# 主程序
# =====================================================================

def main():
    base_dir = r"D:\college education\math\model\2026.4\2026\B-codes\Mathematical-Modeling"
    json_dir = os.path.join(base_dir, "data", "json_networks")
    out_dir = os.path.join(base_dir, "Q3", "Results")
    os.makedirs(out_dir, exist_ok=True)

    # ── 超参 ──
    W = 3
    K = 30
    batch_size = 5
    warmup_frac = 0.05
    theta = 0.01
    step_size = 0.01
    alpha, beta = 0.5, 0.5  # 仅用于回放时计算 Q_bc 对比

    strategy_name = f"CABS_LCC_W{W}_K{K}_B{batch_size}"

    all_data_pq = []
    all_data_qbc = []
    summary_rows = []

    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith("_Network.json"):
            continue

        city = filename.split("_")[0]
        json_path = os.path.join(json_dir, filename)
        print(f"\n{'='*64}")
        print(f"  城市: {city} — CABS-LCC 社区桥梁集束搜索（原生指标）")
        print(f"{'='*64}")

        t0 = time.perf_counter()

        # ---- 两阶段攻击 ----
        seq, R_lcc_beam = cabs_attack(
            json_path,
            W=W, K=K, batch_size=batch_size,
            theta=theta,
            warmup_frac=warmup_frac,
            city_name=city,
        )

        # ---- 保存攻击序列 ----
        seq_csv = os.path.join(out_dir, f"{city}_CABS_LCC_Attack_Sequence.csv")
        pd.DataFrame({"rank": range(1, len(seq) + 1), "node_id": seq}).to_csv(
            seq_csv, index=False, encoding="utf-8-sig",
        )

        # ---- 回放标准步长曲线（同时输出 P_q 和 Q_bc）----
        print(f"    回放 P_q / Q_bc 曲线 ...")
        q_pts, p_arr, qbc_arr = replay_attack_sequence(
            json_path, seq,
            step_size=step_size, alpha=alpha, beta=beta,
        )

        # 梯形积分
        R_lcc_replay = float(np.trapz(p_arr, q_pts))
        R_bc_replay = float(np.trapz(qbc_arr, q_pts))

        # 临界点：P_q 首次跌破 0.5
        below_pq = np.where(np.array(p_arr) < 0.5)[0]
        q_critical_pq = float(q_pts[below_pq[0]]) if len(below_pq) else float("nan")

        # 临界点：Q_bc 首次跌破 0.5（对比参考）
        below_qbc = np.where(np.array(qbc_arr) < 0.5)[0]
        q_critical_qbc = float(q_pts[below_qbc[0]]) if len(below_qbc) else float("nan")

        elapsed = time.perf_counter() - t0
        print(
            f"    R_lcc(trapz)={R_lcc_replay:.6f}, "
            f"R_bc(trapz)={R_bc_replay:.6f}, "
            f"q_crit(P<0.5)={q_critical_pq:.2f}, "
            f"总耗时={elapsed:.1f}s"
        )

        for q, p, qbc in zip(q_pts, p_arr, qbc_arr):
            all_data_pq.append({
                "City": city, "Strategy": strategy_name,
                "q": q, "P_q_Official": p,
            })
            all_data_qbc.append({
                "City": city, "Strategy": strategy_name,
                "q": q, "Q_bc_Teammate": qbc,
            })

        summary_rows.append({
            "City": city,
            "Strategy": strategy_name,
            "R_lcc_trapz": R_lcc_replay,
            "R_lcc_beam": R_lcc_beam,
            "R_bc_trapz": R_bc_replay,
            "q_critical_Pq_05": q_critical_pq,
            "q_critical_Qbc_05": q_critical_qbc,
            "Nodes_Removed": len(seq),
            "Time_seconds": elapsed,
        })

    # ── 保存 ──
    pq_csv = os.path.join(out_dir, "Q3_CABS_LCC_Data_Pq.csv")
    pd.DataFrame(all_data_pq).to_csv(pq_csv, index=False, encoding="utf-8-sig")

    qbc_csv = os.path.join(out_dir, "Q3_CABS_LCC_Data_Qbc.csv")
    pd.DataFrame(all_data_qbc).to_csv(qbc_csv, index=False, encoding="utf-8-sig")

    summary_csv = os.path.join(out_dir, "Q3_CABS_LCC_Summary.csv")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'='*64}")
    print("  全部城市 CABS-LCC 拆解完毕！")
    print(f"{'='*64}")
    print("\n各城市 R_lcc 排名（升序 = 攻击越有效）：")
    print(df_summary.sort_values("R_lcc_trapz").to_string(index=False))


if __name__ == "__main__":
    main()
