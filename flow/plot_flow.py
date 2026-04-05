"""
flow/plot_flow.py
读取已由 get_weight.py 增强的 *_Network.json，绘制流量热力图。

配色方案（深色背景 / 冷暖分离）
节点：  navy(#1B4F72) → 冰蓝(#85C1E9)，沉在底层(zorder=1)，密集区呈蓝雾
边  ：  plasma 色图（深紫→洋红→橙→亮黄），浮在节点上方(zorder=2)
背景：  近黑(#0D0D0D)，高流量路网发光，低流量路网深紫隐约可见

用法
----
python plot_flow.py Chengdu_Network.json
python plot_flow.py Chengdu_Network.json --out chengdu_flow.pdf --dpi 200
"""

import argparse
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import numpy as np


# =============================================================================
# 颜色映射
# =============================================================================

def _make_cmap(from_hex: str, to_hex: str, name: str) -> mcolors.LinearSegmentedColormap:
    """从两个十六进制颜色构造线性渐变 colormap。"""
    return mcolors.LinearSegmentedColormap.from_list(
        name, [from_hex, to_hex], N=256
    )

# 节点：深蓝 → 冰蓝（冷色调，不与边竞争）
NODE_CMAP = _make_cmap('#1B4F72', '#85C1E9', 'node_blue')

# 边：深紫 → 洋红 → 橙 → 亮黄（"城市夜光"效果）
EDGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'flow_plasma',
    ['#2D1B69', '#8B1A8B', '#C0392B', '#E67E22', '#F9E79F'],
    N=512,
)


# =============================================================================
# 数据读取
# =============================================================================

def load_augmented_json(json_path: str):
    """
    读取已添加 node_weight / flow 的 JSON，返回：
    positions  : (N, 2) float64，UTM 坐标
    weights    : (N,) float64，节点密度权重
    node_ids   : (N,) int，json_id
    edges_uv   : (M, 2) int，端点 json_id 对（已去重，u < v）
    flows      : (M,) float64，边流量

    若 node_weight / flow 字段不存在，抛出 KeyError 并提示先运行 get_weight.py。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)

    sample = next(iter(data.values()))
    if 'node_weight' not in sample:
        raise KeyError(
            "JSON 中缺少 'node_weight' 字段，请先运行 get_weight.py。"
        )

    # ── 节点 ─────────────────────────────────────────────────────────────────
    node_ids  = []
    positions = []
    weights   = []
    for node_data in data.values():
        node_ids.append(int(node_data['id']))
        positions.append(node_data['position'])
        weights.append(float(node_data.get('node_weight', 0.0)))

    node_ids  = np.array(node_ids,  dtype=np.int64)
    positions = np.array(positions, dtype=np.float64)
    weights   = np.array(weights,   dtype=np.float64)

    # ── 边（去重：只保留 u < v 的一侧）──────────────────────────────────────
    id_to_pos = {int(d['id']): np.array(d['position']) for d in data.values()}

    seen:  set  = set()
    edges_uv:  list = []
    flows_list: list = []

    for node_data in data.values():
        u = int(node_data['id'])
        for nb in node_data.get('neighbors', []):
            v   = int(nb['id'])
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            edges_uv.append(key)
            flows_list.append(float(nb.get('flow', 0.0)))

    edges_uv = np.array(edges_uv,   dtype=np.int64)   # (M, 2)
    flows    = np.array(flows_list, dtype=np.float64)  # (M,)

    return positions, weights, node_ids, edges_uv, flows, id_to_pos


# =============================================================================
# 绘图
# =============================================================================

def plot_flow(
    json_path:   str,
    out_path:    str  = None,
    dpi:         int   = 150,
    figsize:     tuple = (14, 12),
    node_size:   float = 1.5,   # 缩小：密集区不遮挡边
    edge_lw:     float = 0.55,
    edge_alpha:  float = 0.85,
    node_alpha:  float = 0.55,  # 半透明：密集区呈蓝雾而非实色
) -> None:
    """
    绘制流量热力图并保存（或显示）。

    参数
    ----
    json_path  : 已增强的 *_Network.json
    out_path   : 输出图片路径；None 则直接 plt.show()
    dpi        : 输出分辨率
    figsize    : 画布尺寸（英寸）
    node_size  : scatter 点大小
    edge_lw    : 边线宽
    edge_alpha : 边透明度
    node_alpha : 节点透明度
    """
    city = os.path.basename(json_path).split('_')[0]
    print(f"读取数据: {json_path}")

    positions, weights, node_ids, edges_uv, flows, id_to_pos = \
        load_augmented_json(json_path)

    N = len(positions)
    M = len(edges_uv)
    print(f"节点={N}  边={M}")

    # ── 坐标归一化（保持纵横比）──────────────────────────────────────────────
    x = positions[:, 0]
    y = positions[:, 1]

    # ── 颜色归一化 ────────────────────────────────────────────────────────────
    # 节点：线性归一化
    w_min, w_max = weights.min(), weights.max()
    node_norm = mcolors.Normalize(vmin=w_min, vmax=w_max)

    # 边：对数归一化（零流量边单独处理）
    pos_flows  = flows[flows > 0]
    if len(pos_flows) > 0:
        f_min = pos_flows.min()
        f_max = pos_flows.max()
    else:
        f_min, f_max = 1.0, 2.0
    edge_norm  = mcolors.LogNorm(vmin=f_min, vmax=f_max)

    # ── 构建 LineCollection 数据 ──────────────────────────────────────────────
    id_to_xy = {int(nid): (positions[i, 0], positions[i, 1])
                for i, nid in enumerate(node_ids)}

    segments  = []
    edge_vals = []
    for k in range(M):
        u, v  = int(edges_uv[k, 0]), int(edges_uv[k, 1])
        if u not in id_to_xy or v not in id_to_xy:
            continue
        segments.append([id_to_xy[u], id_to_xy[v]])
        edge_vals.append(flows[k])

    segments  = np.array(segments,  dtype=np.float64)   # (M, 2, 2)
    edge_vals = np.array(edge_vals, dtype=np.float64)   # (M,)

    # 零流量边用最深色（深紫），流量越高越亮
    edge_vals_safe = np.where(edge_vals > 0, edge_vals, f_min)
    edge_colors = EDGE_CMAP(edge_norm(edge_vals_safe))

    # ── 画图 ──────────────────────────────────────────────────────────────────
    BG = '#0D0D0D'
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # 边（LineCollection，远快于逐条 plot）
    lc = LineCollection(
        segments,
        colors=edge_colors,
        linewidths=edge_lw,
        alpha=edge_alpha,
        zorder=2,   # 边浮在节点上方
    )
    ax.add_collection(lc)

    # 节点（底层，密集区呈冷色蓝雾）
    node_colors = NODE_CMAP(node_norm(weights))
    ax.scatter(
        x, y,
        s=node_size,
        c=node_colors,
        alpha=node_alpha,
        linewidths=0,
        zorder=1,
    )

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')

    # ── 颜色条 ────────────────────────────────────────────────────────────────
    # 边流量色条（右侧）
    edge_sm = ScalarMappable(cmap=EDGE_CMAP, norm=edge_norm)
    edge_sm.set_array([])
    cbar_edge = fig.colorbar(
        edge_sm, ax=ax,
        fraction=0.025, pad=0.01, aspect=30,
        location='right',
    )
    cbar_edge.set_label('边流量（对数刻度）', fontsize=10, color='white')
    cbar_edge.ax.tick_params(labelsize=8, colors='white')
    cbar_edge.outline.set_edgecolor('white')

    # 节点权重色条（左侧）
    node_sm = ScalarMappable(cmap=NODE_CMAP, norm=node_norm)
    node_sm.set_array([])
    cbar_node = fig.colorbar(
        node_sm, ax=ax,
        fraction=0.025, pad=0.01, aspect=30,
        location='left',
    )
    cbar_node.set_label('节点密度权重', fontsize=10, color='white')
    cbar_node.ax.tick_params(labelsize=8, colors='white')
    cbar_node.outline.set_edgecolor('white')

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_title(f'{city} 交通网络流量热力图', fontsize=13, pad=10, color='white')

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"已保存: {out_path}")
    else:
        plt.show()

    plt.close(fig)


# =============================================================================
# CLI 入口
# =============================================================================

def _resolve_json_path(raw: str) -> str:
    """
    将用户传入的任意形式解析为实际存在的 weighted JSON 路径。

    接受以下几种写法（均等价）：
      Chengdu
      Chengdu_Network.json
      ../data/json_networks/Chengdu_Network.json
      flow/data/Chengdu_Network_weighted.json
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    # 路径直接存在则直接用
    if os.path.isfile(raw):
        return raw

    # 从任意形式中提取城市名
    basename = os.path.splitext(os.path.basename(raw))[0]
    city     = basename.split('_')[0]

    candidate = os.path.join(data_dir, f'{city}_Network_weighted.json')
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(
        f"找不到文件：{raw}\n"
        f"也找不到：{candidate}\n"
        f"请先运行：python get_weight.py ../data/json_networks/{city}_Network.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='绘制交通网络流量热力图（需先运行 get_weight.py）'
    )
    parser.add_argument(
        'json_path',
        help='城市名（Chengdu）、原始 JSON 或 weighted JSON 路径均可',
    )
    parser.add_argument('--out',  default=None, help='输出图片路径（默认存到 flow/data/）')
    parser.add_argument('--dpi',  type=int,   default=150)
    parser.add_argument('--lw',   type=float, default=0.5,  dest='edge_lw')
    parser.add_argument('--ns',   type=float, default=3.0,  dest='node_size')
    args = parser.parse_args()

    resolved = _resolve_json_path(args.json_path)
    city     = os.path.basename(resolved).split('_')[0]

    if args.out is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(data_dir, exist_ok=True)
        args.out = os.path.join(data_dir, f'{city}_flow_heatmap.pdf')

    plot_flow(
        json_path = resolved,
        out_path  = args.out,
        dpi       = args.dpi,
        edge_lw   = args.edge_lw,
        node_size = args.node_size,
    )


if __name__ == '__main__':
    main()
