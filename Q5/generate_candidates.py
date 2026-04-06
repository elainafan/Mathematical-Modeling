"""
Q5/generate_candidates.py
生成第五问的候选新增道路集合。

三个来源：
  1. 跨双连通瓶颈：桥边两侧 / 割点分割的不同分量之间的空间邻近点对
  2. 跨社区稀疏连接：Leiden 社区之间连接边数过少的社区对之间的空间邻近点对
  3. 高绕行比：空间上近、拓扑上远的点对（从瓶颈节点附近采样）

统一过滤后输出候选边 CSV。

用法：
    python -m Q5.generate_candidates
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import igraph as ig
from scipy.spatial import cKDTree
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CITIES = [
    "Chengdu", "Dalian", "Dongguan", "Harbin",
    "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou",
]


# =====================================================================
# 1. 图加载
# =====================================================================

def load_graph(json_path: str):
    """
    从 json_networks 的 JSON 加载 igraph 图。
    返回: G, id2idx, idx2id, positions(N×2)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    node_list = list(raw.values())
    n = len(node_list)
    id2idx = {val["id"]: i for i, val in enumerate(node_list)}
    idx2id = {i: val["id"] for i, val in enumerate(node_list)}

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
    G.es["weight"] = weights
    G.simplify(multiple=True, loops=True, combine_edges="first")

    positions = np.array([val["position"] for val in node_list], dtype=np.float64)
    return G, id2idx, idx2id, positions


def load_community_labels(json_path: str) -> dict:
    """
    从 Q5/data/ 的社区 JSON 加载 node_id → community 映射。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {val["id"]: val["community"] for val in raw.values()}


# =====================================================================
# 2. 构建已有边集（用于快速排除）
# =====================================================================

def build_edge_set(G: ig.Graph) -> set:
    """返回 frozenset{(min_id, max_id)} 形式的已有边集合。"""
    edge_set = set()
    for e in G.es:
        u_id = G.vs[e.source]["node_id"]
        v_id = G.vs[e.target]["node_id"]
        edge_set.add((min(u_id, v_id), max(u_id, v_id)))
    return edge_set


# =====================================================================
# 3. 来源一：跨双连通瓶颈
# =====================================================================

def generate_bridge_candidates(
    G: ig.Graph, positions: np.ndarray,
    existing_edges: set, L_max: float = 2000.0,
    per_bridge: int = 1,
) -> list:
    """
    对每条桥边：删掉后图分成两个分量，
    在两侧各找空间上最邻近的节点对作为候选边。
    """
    candidates = []
    bridge_eids = G.bridges()
    if not bridge_eids:
        return candidates

    print(f"      桥边数: {len(bridge_eids)}")

    tree = cKDTree(positions)

    for eid in bridge_eids:
        e = G.es[eid]
        u, v = e.source, e.target

        # 删掉这条桥后，标记两侧分量
        G_tmp = G.copy()
        G_tmp.delete_edges(eid)
        cc = G_tmp.connected_components()
        memb = cc.membership

        side_u = set(i for i in range(G.vcount()) if memb[i] == memb[u])
        side_v = set(i for i in range(G.vcount()) if memb[i] == memb[v])

        for anchor, other_side in [(u, side_v), (v, side_u)]:
            anchor_pos = positions[anchor]
            nearby_idxs = tree.query_ball_point(anchor_pos, r=L_max)
            partner_idxs = [i for i in nearby_idxs if i in other_side]

            if not partner_idxs:
                continue

            dists = [np.linalg.norm(positions[i] - anchor_pos)
                     for i in partner_idxs]
            sorted_pairs = sorted(zip(dists, partner_idxs))

            count = 0
            for dist, partner in sorted_pairs:
                if count >= per_bridge:
                    break
                a_id = G.vs[anchor]["node_id"]
                b_id = G.vs[partner]["node_id"]
                key = (min(a_id, b_id), max(a_id, b_id))
                if key not in existing_edges and dist > 10.0:
                    candidates.append({
                        "u": key[0], "v": key[1],
                        "euclid_dist": dist,
                        "source": "bridge",
                    })
                    count += 1

    return candidates


def generate_articulation_candidates(
    G: ig.Graph, positions: np.ndarray,
    existing_edges: set, L_max: float = 2000.0,
    per_ap: int = 3,
) -> list:
    """
    快速版：用 biconnected_components 一次性拿到 block-cut 结构，
    割点是同时属于多个 block 的节点。
    对每个割点，从它所属的不同 block 中各选空间邻近的非割点节点配对。
    不需要任何 copy/delete 操作。
    """
    candidates = []

    aps_set = set(G.articulation_points())
    if not aps_set:
        return candidates

    print(f"      割点数: {len(aps_set)}")

    # 一次性计算所有 biconnected components（blocks）
    blocks = G.biconnected_components()

    # 建立 割点 → 它所属的 block 列表
    ap_to_blocks = defaultdict(list)
    for bid, block_verts in enumerate(blocks):
        for v in block_verts:
            if v in aps_set:
                ap_to_blocks[v].append(bid)

    tree = cKDTree(positions)

    # 快速版无需采样：biconnected_components 已一次性算完，
    # 遍历所有割点只是查表 + KDTree 查询，开销很小
    ap_list = list(aps_set)

    for ap in ap_list:
        block_ids = ap_to_blocks.get(ap, [])
        if len(block_ids) < 2:
            continue

        ap_pos = positions[ap]

        # 从每个 block 中提取非割点节点
        block_nodes = []
        for bid in block_ids:
            non_ap = [v for v in blocks[bid] if v != ap and v not in aps_set]
            if non_ap:
                block_nodes.append(non_ap)

        if len(block_nodes) < 2:
            continue

        # 对每对不同 block，用 KDTree 找最近的节点对
        for bi in range(len(block_nodes)):
            for bj in range(bi + 1, len(block_nodes)):
                nodes_i = block_nodes[bi]
                nodes_j = block_nodes[bj]

                # 用较小集合建 KDTree
                if len(nodes_i) > len(nodes_j):
                    nodes_i, nodes_j = nodes_j, nodes_i

                pos_i = positions[nodes_i]
                tree_i = cKDTree(pos_i)

                seen = set()
                count = 0
                for nj in nodes_j:
                    if count >= per_ap:
                        break
                    d, idx = tree_i.query(positions[nj])
                    if d < 10.0 or d > L_max:
                        continue
                    ni = nodes_i[idx]
                    a_id = G.vs[ni]["node_id"]
                    b_id = G.vs[nj]["node_id"]
                    key = (min(a_id, b_id), max(a_id, b_id))
                    if key not in existing_edges and key not in seen:
                        candidates.append({
                            "u": key[0], "v": key[1],
                            "euclid_dist": d,
                            "source": "articulation",
                        })
                        seen.add(key)
                        count += 1

    return candidates


# =====================================================================
# 4. 来源二：跨社区稀疏连接
# =====================================================================

def generate_community_candidates(
    G: ig.Graph, positions: np.ndarray,
    community_map: dict, existing_edges: set,
    L_max: float = 2000.0,
    max_inter_edges: int = 3,
    per_pair: int = 10,
) -> list:
    """
    找连接边数 ≤ max_inter_edges 的社区对，
    在两个社区之间找空间最邻近的点对作为候选。
    """
    candidates = []

    # 统计每对社区之间的现有边数
    inter_count = defaultdict(int)
    for e in G.es:
        u_id = G.vs[e.source]["node_id"]
        v_id = G.vs[e.target]["node_id"]
        cu = community_map.get(u_id, -1)
        cv = community_map.get(v_id, -1)
        if cu != cv and cu >= 0 and cv >= 0:
            pair = (min(cu, cv), max(cu, cv))
            inter_count[pair] += 1

    # 找稀疏社区对
    sparse_pairs = [(p, cnt) for p, cnt in inter_count.items()
                    if cnt <= max_inter_edges]

    # 也找完全断开的社区对（如果它们空间上接近）
    all_comms = set(community_map.values())
    idx_to_comm = {i: community_map.get(G.vs[i]["node_id"], -1)
                   for i in range(G.vcount())}

    # 按社区分组节点
    comm_nodes = defaultdict(list)
    for i in range(G.vcount()):
        c = idx_to_comm[i]
        if c >= 0:
            comm_nodes[c].append(i)

    print(f"      社区数: {len(comm_nodes)}, 稀疏社区对(≤{max_inter_edges}边): {len(sparse_pairs)}")

    # 对每个稀疏社区对，找空间最邻近点对
    for (ca, cb), cnt in sparse_pairs:
        nodes_a = comm_nodes.get(ca, [])
        nodes_b = comm_nodes.get(cb, [])
        if not nodes_a or not nodes_b:
            continue

        # 用较小社区建 KDTree，在较大社区中查询
        if len(nodes_a) > len(nodes_b):
            nodes_a, nodes_b = nodes_b, nodes_a

        pos_a = positions[nodes_a]
        tree_a = cKDTree(pos_a)

        best = []
        for nj in nodes_b:
            d, idx = tree_a.query(positions[nj])
            if d <= L_max and d > 10.0:
                ni = nodes_a[idx]
                best.append((d, ni, nj))

        best.sort()
        # 去重 + 取 top
        seen = set()
        count = 0
        for d, ni, nj in best:
            if count >= per_pair:
                break
            a_id = G.vs[ni]["node_id"]
            b_id = G.vs[nj]["node_id"]
            key = (min(a_id, b_id), max(a_id, b_id))
            if key not in existing_edges and key not in seen:
                candidates.append({
                    "u": key[0], "v": key[1],
                    "euclid_dist": d,
                    "source": "community",
                })
                seen.add(key)
                count += 1

    return candidates


# =====================================================================
# 5. 来源三：高绕行比（从瓶颈节点附近采样）
# =====================================================================

def generate_detour_candidates(
    G: ig.Graph, positions: np.ndarray,
    existing_edges: set,
    euclid_min: float = 200.0,
    euclid_max: float = 2000.0,
    detour_threshold: float = 5.0,
    seed_count: int = 1000,
    per_seed: int = 10,
) -> list:
    """
    从桥边端点和割点附近采样，找空间上近但拓扑上远的点对。

    策略：不做全图 all-pairs shortest paths（太贵），
    只从 "结构重要" 的种子节点出发跑单源 Dijkstra。
    """
    candidates = []

    # 收集种子节点：桥边端点 + 割点
    seeds = set()
    for eid in G.bridges():
        e = G.es[eid]
        seeds.add(e.source)
        seeds.add(e.target)
    seeds.update(G.articulation_points())

    # 如果种子太少，补充高度数节点
    if len(seeds) < seed_count:
        degrees = G.degree()
        ranked = sorted(range(G.vcount()), key=lambda i: degrees[i], reverse=True)
        for i in ranked:
            if len(seeds) >= seed_count:
                break
            seeds.add(i)

    # 如果种子太多，随机采样
    seeds = list(seeds)
    if len(seeds) > seed_count:
        rng = np.random.RandomState(42)
        seeds = list(rng.choice(seeds, seed_count, replace=False))

    print(f"      绕行比种子数: {len(seeds)}")

    tree = cKDTree(positions)

    # 预建邻居集（用于快速判断是否直连）
    neighbor_sets = [set(G.neighbors(i)) for i in range(G.vcount())]

    for seed in seeds:
        seed_pos = positions[seed]

        # 找空间上 [euclid_min, euclid_max] 范围内的非邻居节点
        nearby = tree.query_ball_point(seed_pos, r=euclid_max)
        targets = []
        for j in nearby:
            if j == seed:
                continue
            if j in neighbor_sets[seed]:
                continue
            d_euclid = np.linalg.norm(positions[j] - seed_pos)
            if d_euclid >= euclid_min:
                targets.append((j, d_euclid))

        if not targets:
            continue

        # 跑单源 Dijkstra
        sp = G.shortest_paths(source=seed, weights="weight")[0]

        count = 0
        for j, d_euclid in targets:
            if count >= per_seed:
                break
            d_graph = sp[j]
            if d_graph == float("inf") or d_euclid < 1.0:
                continue
            detour = d_graph / d_euclid
            if detour >= detour_threshold:
                a_id = G.vs[seed]["node_id"]
                b_id = G.vs[j]["node_id"]
                key = (min(a_id, b_id), max(a_id, b_id))
                if key not in existing_edges:
                    candidates.append({
                        "u": key[0], "v": key[1],
                        "euclid_dist": d_euclid,
                        "detour_ratio": detour,
                        "source": "detour",
                    })
                    count += 1

    return candidates


# =====================================================================
# 6. 全局去重与过滤
# =====================================================================

def deduplicate_and_filter(candidates: list, L_max: float = 2000.0) -> pd.DataFrame:
    """
    全局去重，过滤过短或过长的边。
    同一条边可能被多个来源同时生成，保留来源优先级最高的那条。
    """
    if not candidates:
        return pd.DataFrame()

    df = pd.DataFrame(candidates)

    # 统一排序 key
    df["edge_key"] = df.apply(
        lambda r: (min(r["u"], r["v"]), max(r["u"], r["v"])), axis=1
    )

    # 去重：同一条边保留第一个来源（优先级: bridge > articulation > community > detour）
    source_priority = {"bridge": 0, "articulation": 1, "community": 2, "detour": 3}
    df["priority"] = df["source"].map(source_priority)
    df = df.sort_values("priority")
    df = df.drop_duplicates(subset="edge_key", keep="first")
    df = df.drop(columns=["priority", "edge_key"])

    # 过滤
    df = df[df["euclid_dist"] >= 10.0]     # 太短的没意义
    df = df[df["euclid_dist"] <= L_max]     # 太长的成本过高

    df = df.sort_values("euclid_dist").reset_index(drop=True)
    return df


# =====================================================================
# 7. 主程序
# =====================================================================

def process_city(city: str, graph_dir: str, community_dir: str,
                 out_dir: str, L_max: float = 2000.0):
    print(f"\n  {'='*50}")
    print(f"  城市: {city}")
    print(f"  {'='*50}")

    # 加载图
    json_path = os.path.join(graph_dir, f"{city}_Network.json")
    if not os.path.exists(json_path):
        print(f"  [跳过] 找不到 {json_path}")
        return
    G, id2idx, idx2id, positions = load_graph(json_path)
    print(f"    N={G.vcount()}, M={G.ecount()}")

    # 加载社区标签
    comm_path = os.path.join(community_dir, f"{city}_Network_community.json")
    if os.path.exists(comm_path):
        community_map = load_community_labels(comm_path)
        print(f"    已加载社区标签")
    else:
        print(f"    [警告] 未找到社区文件，跳过社区候选")
        community_map = None

    existing_edges = build_edge_set(G)
    all_candidates = []

    # ── 来源 1：桥边候选 ──
    t0 = time.perf_counter()
    print(f"    来源1: 跨桥边候选...")
    bridge_cands = generate_bridge_candidates(
        G, positions, existing_edges, L_max=L_max,
    )
    print(f"      → {len(bridge_cands)} 条, 耗时 {time.perf_counter()-t0:.1f}s")
    all_candidates.extend(bridge_cands)

    # ── 来源 2：割点候选 ──
    t0 = time.perf_counter()
    print(f"    来源2: 跨割点候选...")
    ap_cands = generate_articulation_candidates(
        G, positions, existing_edges, L_max=L_max,
    )
    print(f"      → {len(ap_cands)} 条, 耗时 {time.perf_counter()-t0:.1f}s")
    all_candidates.extend(ap_cands)

    # ── 来源 3：跨社区候选 ──
    if community_map:
        t0 = time.perf_counter()
        print(f"    来源3: 跨社区稀疏连接候选...")
        comm_cands = generate_community_candidates(
            G, positions, community_map, existing_edges, L_max=L_max,
        )
        print(f"      → {len(comm_cands)} 条, 耗时 {time.perf_counter()-t0:.1f}s")
        all_candidates.extend(comm_cands)

    # ── 来源 4：高绕行比候选 ──
    t0 = time.perf_counter()
    print(f"    来源4: 高绕行比候选...")
    detour_cands = generate_detour_candidates(
        G, positions, existing_edges,
    )
    print(f"      → {len(detour_cands)} 条, 耗时 {time.perf_counter()-t0:.1f}s")
    all_candidates.extend(detour_cands)

    # ── 去重过滤 ──
    df = deduplicate_and_filter(all_candidates, L_max=L_max)
    print(f"\n    去重过滤后: {len(df)} 条候选边")

    if len(df) > 0:
        print(f"    来源分布:")
        print(df["source"].value_counts().to_string(header=False))
        print(f"    欧氏距离: min={df['euclid_dist'].min():.0f}m, "
              f"max={df['euclid_dist'].max():.0f}m, "
              f"median={df['euclid_dist'].median():.0f}m")

    # 保存
    out_path = os.path.join(out_dir, f"{city}_candidates.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"    已保存 → {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_dir = os.path.join(base_dir, "data", "json_networks")
    community_dir = os.path.join(base_dir, "Q5", "data")
    out_dir = os.path.join(base_dir, "Q5", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  Q5 候选新增道路生成")
    print("=" * 60)

    for city in CITIES:
        process_city(city, graph_dir, community_dir, out_dir)

    print(f"\n{'='*60}")
    print("  全部城市候选边生成完毕!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
