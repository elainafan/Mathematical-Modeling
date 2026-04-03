"""
网络直径严格计算 —— 方法1：欧式直线距离
定义：LCC 中所有节点对坐标直线距离的最大值
     sqrt((x1-x2)^2 + (y1-y2)^2)

算法：先求凸包，再对凸包顶点做暴力两两比较（凸包极大缩小候选集）
"""

import json, os, glob
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_DIR   = os.path.join(BASE_DIR, 'data', 'json_networks')
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

JSON_FILES = sorted(glob.glob(os.path.join(JSON_DIR, '*.json')))
CITIES     = [os.path.basename(p).replace('_Network.json', '') for p in JSON_FILES]

COLORS = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
          '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']


def build_hop_adj(data):
    adj = defaultdict(list)
    for node in data.values():
        u = node['id']
        for nb in node['neighbors']:
            v = nb['id']
            adj[u].append(v)
            adj[v].append(u)
    return adj


def largest_cc(adj, all_nodes):
    visited, best = set(), set()
    for s in all_nodes:
        if s in visited:
            continue
        comp, q = {s}, deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp.add(v); q.append(v)
        visited |= comp
        if len(comp) > len(best):
            best = comp
    return best


def euclidean_diameter(pos_array):
    """
    严格计算点集中最大两点欧式距离：
    1. 求凸包（凸包上的点才可能是最远对）
    2. 对凸包顶点暴力两两计算，取最大值
    """
    hull   = ConvexHull(pos_array)
    hull_pts = pos_array[hull.vertices]
    h = len(hull_pts)

    max_d = 0.0
    u_idx = v_idx = 0
    for i in range(h):
        for j in range(i + 1, h):
            d = np.linalg.norm(hull_pts[i] - hull_pts[j])
            if d > max_d:
                max_d = d
                u_idx, v_idx = i, j

    return max_d, hull_pts[u_idx], hull_pts[v_idx], hull_pts


# ── 计算 ──────────────────────────────────────────────────────
summary_rows = []

for path, city in zip(JSON_FILES, CITIES):
    print(f'\n{city} ...', flush=True)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    pos_map   = {node['id']: node['position'] for node in data.values()}
    all_nodes = list(pos_map.keys())
    adj  = build_hop_adj(data)
    lcc  = largest_cc(adj, all_nodes)

    pos_array = np.array([pos_map[u] for u in lcc])
    diam, pt_u, pt_v, hull_pts = euclidean_diameter(pos_array)

    print(f'  凸包顶点数={len(hull_pts)}  直径={diam:.2f} m  ({diam/1000:.3f} km)')

    summary_rows.append({
        '城市': city, '节点数': len(all_nodes),
        'LCC节点数': len(lcc),
        '凸包顶点数': len(hull_pts),
        '直径(m)': round(diam, 2),
        '直径(km)': round(diam / 1000, 3),
    })


df = pd.DataFrame(summary_rows)

# ── 绘图 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(df['城市'], df['直径(km)'], color=COLORS,
              alpha=0.85, edgecolor='white', zorder=2)
for bar, val in zip(bars, df['直径(km)']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('各城市交通网络直径（欧式直线距离，严格计算）', fontsize=14, fontweight='bold')
ax.set_xlabel('城市', fontsize=12)
ax.set_ylabel('直径（km）', fontsize=12)
ax.set_ylim(0, df['直径(km)'].max() * 1.15)
ax.grid(axis='y', alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'diameter_euclidean.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图已保存: diameter_euclidean.png')

df.to_excel(os.path.join(OUT_DIR, 'diameter_euclidean.xlsx'), index=False)
print('Excel已保存: diameter_euclidean.xlsx')
print('\n── 汇总 ──')
print(df.to_string(index=False))
