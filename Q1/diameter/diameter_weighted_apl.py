"""
直径计算（二）：Dijkstra 加权直径 + 加权平均路径长度
=====================================================
定义：
  加权直径   —— LCC 中所有节点对 Dijkstra 最短路（边权=路段长度/m）的最大值。
  平均路径长度（APL）—— LCC 中所有节点对 Dijkstra 最短路的均值。
              APL = (1 / (n*(n-1))) * Σ_{u≠v} d(u,v)
              其中 d(u,v) 为节点 u 到 v 的加权最短路长度（米）。

算法：
  对 LCC 中每个节点做一次 Dijkstra，记录从该节点出发到所有可达节点的距离。
  · 直径   = max(所有节点对最短路)
  · APL    = mean(所有节点对最短路)，排除不可达节点对

  时间复杂度 O(n * (m + n) * log n)，对于数万节点耗时较长。

输出：
  diameter_weighted_apl.png   — 左轴：加权直径(km)，右轴：APL(km)，双轴对比柱状图
  diameter_weighted_apl.xlsx  — 汇总表（直径米/km、APL米/km）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, glob, os, heapq, time
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
COLORS     = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
              '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def build_adj(data):
    hop_adj  = defaultdict(list)
    dist_adj = defaultdict(list)
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


def dijkstra_all(dist_adj, source):
    """从 source 出发 Dijkstra，返回到所有可达节点的距离字典"""
    dist  = {source: 0.0}
    heap  = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in dist_adj[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


# ════════════════════════════════════════════════════════════
# 主计算
# ════════════════════════════════════════════════════════════
summary_rows = []

for path, city in zip(JSON_FILES, CITIES):
    print(f'\n{city} 开始...', flush=True)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    all_nodes         = [node['id'] for node in data.values()]
    hop_adj, dist_adj = build_adj(data)
    lcc               = largest_cc(hop_adj, all_nodes)
    nodes             = list(lcc)
    n                 = len(nodes)

    t0           = time.time()
    diam         = 0.0
    total_dist   = 0.0
    pair_count   = 0

    for i, u in enumerate(nodes):
        d_map = dijkstra_all(dist_adj, u)
        for v, dv in d_map.items():
            if v != u and v in lcc:
                if dv > diam:
                    diam = dv
                total_dist += dv
                pair_count += 1

        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(f'  {i+1}/{n}  直径下界={diam/1000:.2f}km  '
                  f'耗时={elapsed:.0f}s', flush=True)

    # APL：有序对均值（除以2得无序对均值结果相同）
    apl = total_dist / pair_count if pair_count > 0 else 0.0

    print(f'  => 直径={diam:.1f}m ({diam/1000:.3f}km)  '
          f'APL={apl:.1f}m ({apl/1000:.3f}km)')

    summary_rows.append({
        '城市'      : city,
        '总节点数'  : len(all_nodes),
        'LCC节点数' : n,
        '直径(m)'   : round(diam, 1),
        '直径(km)'  : round(diam / 1000, 3),
        'APL(m)'    : round(apl, 1),
        'APL(km)'   : round(apl / 1000, 3),
    })

df = pd.DataFrame(summary_rows)
df.to_excel(os.path.join(OUT_DIR, 'diameter_weighted_apl.xlsx'), index=False)
print('\nExcel已保存: diameter_weighted_apl.xlsx')
print('\n── 汇总 ──')
print(df.to_string(index=False))


# ════════════════════════════════════════════════════════════
# 绘图：从 Excel 读取，单纵坐标分组柱状图（单位 km）
# ════════════════════════════════════════════════════════════
df = pd.read_excel(os.path.join(OUT_DIR, 'diameter_weighted_apl.xlsx'))

x = np.arange(len(df))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 6))

bars1 = ax.bar(x - w/2, df['直径(km)'], w, color=COLORS, alpha=0.85,
               edgecolor='white', label='加权直径', zorder=2)
bars2 = ax.bar(x + w/2, df['APL(km)'],  w, color=COLORS, alpha=0.45,
               edgecolor='white', hatch='//', label='加权平均路径长度 (APL)', zorder=2)

for bar, val in zip(bars1, df['直径(km)']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, df['APL(km)']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)

ax.set_title('各城市交通网络加权直径与加权平均路径长度（Dijkstra，边权=路段长度）',
             fontsize=13, fontweight='bold')
ax.set_xlabel('城市', fontsize=12)
ax.set_ylabel('距离 (km)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(df['城市'], fontsize=11)
ax.set_ylim(0, max(df['直径(km)'].max(), df['APL(km)'].max()) * 1.18)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, zorder=1)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'diameter_weighted_apl.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图已保存: diameter_weighted_apl.png')
print('\n── 汇总 ──')
print(df.to_string(index=False))
