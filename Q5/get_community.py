"""
Q5/get_community.py
对 data/json_networks/ 中的 8 个城市网络做 Leiden 社区检测，
为每个节点添加 community 属性，输出到 Q5/data/{City}_Network_community.json。
"""

import os
import sys
import json
import igraph as ig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CITIES = [
    "Chengdu", "Dalian", "Dongguan", "Harbin",
    "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou",
]


def load_graph_from_json(json_path: str):
    """
    从 json_networks 的 JSON 文件加载 igraph 图。
    返回: (G, raw_data)
      G        : igraph.Graph，带 node_id / pos / weight 属性
      raw_data : 原始 dict（保留用于后续写回）
    """
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

    return G, raw


def detect_communities(G: ig.Graph) -> list:
    """
    Leiden 社区检测，回退 Louvain。
    返回: membership 列表（长度 = G.vcount()）
    """
    if G.vcount() <= 2:
        return list(range(G.vcount()))
    try:
        return G.community_leiden(objective_function="modularity").membership
    except AttributeError:
        return G.community_multilevel().membership
    except Exception:
        return [0] * G.vcount()


def process_city(city: str, input_dir: str, output_dir: str):
    json_path = os.path.join(input_dir, f"{city}_Network.json")
    if not os.path.exists(json_path):
        print(f"  [跳过] 找不到 {json_path}")
        return

    print(f"  加载 {city} ...")
    G, raw = load_graph_from_json(json_path)

    print(f"    N={G.vcount()}, M={G.ecount()}")
    membership = detect_communities(G)
    n_comm = len(set(membership))
    print(f"    Leiden 社区数: {n_comm}")

    # 将 community 属性写回 raw dict
    node_list = list(raw.values())
    id2idx = {val["id"]: i for i, val in enumerate(node_list)}

    for val in node_list:
        idx = id2idx[val["id"]]
        val["community"] = membership[idx]

    # 保存
    out_path = os.path.join(output_dir, f"{city}_Network_community.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)

    print(f"    已保存 → {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, "data", "json_networks")
    output_dir = os.path.join(base_dir, "Q5", "data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("  Q5 社区检测 — Leiden")
    print("=" * 50)

    for city in CITIES:
        process_city(city, input_dir, output_dir)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
