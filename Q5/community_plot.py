"""
Q5/community_plot.py
读取 Q5/data/{City}_Network_community.json，
为每个城市绘制社区分配图（节点按社区着色），保存为 PDF。

用法：
    python -m Q5.community_plot
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CITIES = [
    "Chengdu", "Dalian", "Dongguan", "Harbin",
    "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou",
]

def generate_distinct_colors(n: int):
    """
    生成 n 种视觉可区分的颜色。
    用 HSV 色环均匀采样，饱和度和明度随机扰动以增加辨识度。
    """
    if n <= 20:
        # 少量社区直接用 tab20
        cmap = plt.cm.get_cmap("tab20", n)
        return [cmap(i) for i in range(n)]

    rng = np.random.RandomState(42)
    colors = []
    for i in range(n):
        h = i / n
        s = 0.55 + 0.40 * rng.random()
        v = 0.65 + 0.30 * rng.random()
        rgb = mcolors.hsv_to_rgb([h, s, v])
        colors.append(rgb)
    return colors


def load_community_graph(json_path: str):
    """
    加载带 community 属性的 JSON 文件。
    返回: positions (N×2), communities (N,), edges list[(i,j)]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    node_list = list(raw.values())
    id2idx = {val["id"]: i for i, val in enumerate(node_list)}

    positions = np.array([val["position"] for val in node_list], dtype=np.float64)
    communities = np.array([val["community"] for val in node_list], dtype=int)

    edges = []
    for val in node_list:
        u = id2idx[val["id"]]
        for nb in val["neighbors"]:
            v = id2idx.get(nb["id"])
            if v is not None and u < v:
                edges.append((u, v))

    return positions, communities, edges


def plot_community_map(
    positions:   np.ndarray,
    communities: np.ndarray,
    edges:       list,
    out_path:    str,
    node_size:   float = 1.5,
    edge_alpha:  float = 0.08,
    dpi:         int   = 300,
):
    """
    绘制社区分配图。
    - 边：浅灰色细线，作为路网骨架
    - 节点：按社区着色，小圆点
    - 白色背景，无标题，无图例，无坐标轴
    """
    unique_comms = np.unique(communities)
    n_comm = len(unique_comms)
    palette = generate_distinct_colors(n_comm)
    comm_to_color = {c: palette[i] for i, c in enumerate(unique_comms)}

    node_colors = [comm_to_color[c] for c in communities]

    fig, ax = plt.subplots(figsize=(6, 8), facecolor="white")
    ax.set_facecolor("white")

    # 画边（路网骨架）
    xs, ys = positions[:, 0], positions[:, 1]
    for u, v in edges:
        ax.plot(
            [xs[u], xs[v]], [ys[u], ys[v]],
            color="#cccccc", linewidth=0.15, alpha=edge_alpha, zorder=1,
        )

    # 画节点
    ax.scatter(
        xs, ys,
        c=node_colors, s=node_size, edgecolors="none",
        zorder=2, rasterized=True,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"    已保存 → {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Q5", "data")
    out_dir = os.path.join(base_dir, "Q5", "results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 50)
    print("  Q5 社区分配图绘制")
    print("=" * 50)

    for city in CITIES:
        json_path = os.path.join(data_dir, f"{city}_Network_community.json")
        if not os.path.exists(json_path):
            print(f"  [跳过] 找不到 {json_path}，请先运行 get_community.py")
            continue

        print(f"  绘制 {city} ...")
        positions, communities, edges = load_community_graph(json_path)
        n_comm = len(np.unique(communities))
        print(f"    N={len(positions)}, 社区数={n_comm}, 边数={len(edges)}")

        out_path = os.path.join(out_dir, f"{city}_community.pdf")
        plot_community_map(positions, communities, edges, out_path)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
