"""
网络直径严格计算 —— 方法2：Dijkstra 加权距离
定义：最大连通分量中所有节点对加权最短路（边 distance 之和）的最大值
对每个节点做一次 Dijkstra，取所有最远距离的最大值
"""

import json, os, glob, heapq
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_DIR   = os.path.join(BASE_DIR, 'data', 'json_networks')
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

JSON_FILES = sorted(glob.glob(os.path.join(JSON_DIR, '*.json')))
CITIES     = [os.path.basename(p).replace('_Network.json', '') for p in JSON_FILES]

COLORS = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
          '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']


def build_adj(data):
    """无权邻接表（用于找LCC）和有权邻接表（用于Dijkstra）"""
    hop_adj  = defaultdict(list)
    dist_adj = defaultdict(list)   # [(neighbor, weight)]
    for node in data.values():
        u = node['id']
        for nb in node['neighbors']:
            v, w = nb['id'], nb['distance']
            hop_adj[u].append(v);  hop_adj[v].append(u)
            dist_adj[u].append((v, w)); dist_adj[v].append((u, w))
    return hop_adj, dist_adj


def largest_cc(hop_adj, all_nodes):
    visited, best = set(), set()
    for s in all_nodes:
        if s in visited:
            continue
        comp, q = {s}, deque([s])
        while q:
            u = q.popleft()
            for v in hop_adj[u]:
                if v not in comp:
                    comp.add(v); q.append(v)
        visited |= comp
        if len(comp) > len(best):
            best = comp
    return best


def dijkstra_max(dist_adj, source):
    """Dijkstra 返回从 source 出发的最大加权距离"""
    dist = {source: 0.0}
    heap = [(0.0, source)]
    max_d = 0.0
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        if d > max_d:
            max_d = d
        for v, w in dist_adj[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return max_d


# ── 计算 ──────────────────────────────────────────────────────
summary_rows = []

for path, city in zip(JSON_FILES, CITIES):
    print(f'\n{city} 开始...', flush=True)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    all_nodes = [node['id'] for node in data.values()]
    hop_adj, dist_adj = build_adj(data)
    lcc   = largest_cc(hop_adj, all_nodes)
    nodes = list(lcc)
    n     = len(nodes)

    diam = 0.0
    for i, u in enumerate(nodes):
        d = dijkstra_max(dist_adj, u)
        if d > diam:
            diam = d
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f'  {i+1}/{n}  直径下界={diam:.1f} m', flush=True)

    summary_rows.append({
        '城市': city, '节点数': len(all_nodes),
        'LCC节点数': n, '直径(m)': round(diam, 1),
        '直径(km)': round(diam / 1000, 3)
    })
    print(f'  => 直径 = {diam:.1f} m  ({diam/1000:.3f} km)')


df = pd.DataFrame(summary_rows)

# ── 绘图 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(df['城市'], df['直径(km)'], color=COLORS,
              alpha=0.85, edgecolor='white', zorder=2)
for bar, val in zip(bars, df['直径(km)']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('各城市交通网络直径（Dijkstra加权距离，严格计算）', fontsize=14, fontweight='bold')
ax.set_xlabel('城市', fontsize=12)
ax.set_ylabel('直径（km）', fontsize=12)
ax.set_ylim(0, df['直径(km)'].max() * 1.15)
ax.grid(axis='y', alpha=0.3, zorder=1)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'diameter_weighted.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图已保存: diameter_weighted.png')

df.to_excel(os.path.join(OUT_DIR, 'diameter_weighted.xlsx'), index=False)
print('Excel已保存: diameter_weighted.xlsx')
print('\n── 汇总 ──')
print(df.to_string(index=False))
