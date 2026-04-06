"""
Q5/generate_vaccine_candidates.py
围绕 CABS 攻击序列中的关键节点，定向生成跨社区候选边。

思路：
  对每个关键攻击节点 w，在 w 的邻域（2 跳）内寻找所有
  跨社区、300m ≤ 距离 ≤ 5000m、原图不存在的节点对。

输出：Q5/data/{City}_candidates_vaccine.csv

用法：
    python -m Q5.generate_vaccine_candidates
"""

import os
import sys
import time
import json
import csv
import numpy as np
from scipy.spatial import cKDTree
import igraph as ig
import pandas as pd

CITIES = [
    "Chengdu", "Dalian", "Dongguan", "Harbin",
    "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou",
]

MIN_DIST = 300.0
MAX_DIST = 5000.0
TOP_FRAC = 0.10   # 取攻击序列前 10%（比 augment 的 5% 更宽）


def load_graph_from_json(json_path):
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


def detect_communities(G):
    if G.vcount() <= 2:
        return list(range(G.vcount()))
    try:
        return G.community_leiden(objective_function="modularity").membership
    except AttributeError:
        return G.community_multilevel().membership
    except Exception:
        return [0] * G.vcount()


def generate_for_city(city, json_path, seq_path, out_path):
    print(f"\n  {city}")

    G = load_graph_from_json(json_path)
    N = G.vcount()
    print(f"    N={N}, M={G.ecount()}")

    # 加载攻击序列
    df_seq = pd.read_csv(seq_path)
    attack_seq = df_seq["node_id"].tolist()
    id2idx = {G.vs[i]["node_id"]: i for i in range(N)}

    # 关键节点
    n_critical = max(30, int(len(attack_seq) * TOP_FRAC))
    critical_idxs = []
    for nid in attack_seq[:n_critical]:
        if nid in id2idx:
            critical_idxs.append(id2idx[nid])
    print(f"    关键攻击节点: {len(critical_idxs)} 个 (前 {TOP_FRAC*100:.0f}%)")

    # 社区
    membership = detect_communities(G)
    n_comm = len(set(membership))
    print(f"    社区数: {n_comm}")

    # 坐标和 KDTree
    coords = np.array(G.vs["pos"], dtype=np.float64)
    tree = cKDTree(coords)

    # 已有边集
    existing = set()
    for e in G.es:
        s, t = e.source, e.target
        existing.add((min(s, t), max(s, t)))

    # 关键节点的 2 跳邻域
    critical_zone = set()
    for c in critical_idxs:
        critical_zone.add(c)
        for nb1 in G.neighbors(c):
            critical_zone.add(nb1)
            for nb2 in G.neighbors(nb1):
                critical_zone.add(nb2)

    print(f"    关键区域节点数: {len(critical_zone)} "
          f"({len(critical_zone)/N*100:.1f}%)")

    # 在关键区域内找所有跨社区、距离合适的节点对
    zone_list = sorted(critical_zone)
    zone_coords = coords[zone_list]
    zone_tree = cKDTree(zone_coords)

    # 查询所有 ≤ MAX_DIST 的对
    pairs = zone_tree.query_pairs(r=MAX_DIST)

    candidates = []
    seen = set()

    t0 = time.perf_counter()
    for i_local, j_local in pairs:
        i_global = zone_list[i_local]
        j_global = zone_list[j_local]

        # 距离过滤
        dist = np.linalg.norm(coords[i_global] - coords[j_global])
        if dist < MIN_DIST:
            continue

        # 去重
        edge_key = (min(i_global, j_global), max(i_global, j_global))
        if edge_key in existing or edge_key in seen:
            continue
        seen.add(edge_key)

        # 跨社区
        if membership[i_global] == membership[j_global]:
            continue

        u_nid = G.vs[i_global]["node_id"]
        v_nid = G.vs[j_global]["node_id"]

        candidates.append({
            "u": u_nid,
            "v": v_nid,
            "euclid_dist": f"{dist:.1f}",
            "source": "vaccine",
        })

    elapsed = time.perf_counter() - t0
    print(f"    跨社区候选对: {len(pairs)} → 有效: {len(candidates)}, "
          f"耗时 {elapsed:.1f}s")

    # 保存
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["u", "v", "euclid_dist", "source"])
        writer.writeheader()
        writer.writerows(candidates)

    print(f"    已保存 → {out_path}")
    return len(candidates)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_dir = os.path.join(base_dir, "data", "json_networks")
    q3_dir = os.path.join(base_dir, "Q3", "Results")
    out_dir = os.path.join(base_dir, "Q5", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 50)
    print(f"  Q5 疫苗式候选边生成")
    print(f"  距离: {MIN_DIST}m - {MAX_DIST}m, 攻击序列前 {TOP_FRAC*100:.0f}%")
    print("=" * 50)

    for city in CITIES:
        json_path = os.path.join(json_dir, f"{city}_Network.json")
        seq_path = os.path.join(q3_dir, f"{city}_CABS_LCC_Attack_Sequence.csv")
        if not os.path.exists(seq_path):
            seq_path = os.path.join(q3_dir, f"{city}_CABS_Attack_Sequence.csv")
        out_path = os.path.join(out_dir, f"{city}_candidates_vaccine.csv")

        if not os.path.exists(json_path) or not os.path.exists(seq_path):
            print(f"\n  [跳过] {city}")
            continue

        generate_for_city(city, json_path, seq_path, out_path)

    print(f"\n{'='*50}")
    print("  全部完成!")


if __name__ == "__main__":
    main()
