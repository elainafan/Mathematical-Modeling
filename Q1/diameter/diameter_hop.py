"""
直径与平均路径长度（一）：跳数定义
=====================================================
定义：
  跳数直径    —— LCC 中所有节点对 BFS 最短跳数的最大值。
  跳数 APL    —— LCC 中所有节点对 BFS 最短跳数的均值。
              APL = (1 / (n*(n-1))) * Σ_{u≠v} d_hop(u,v)

算法：
  直径：2-sweep BFS（两次 BFS 得精确或接近精确的直径，适合路网）。
        第1次 BFS 从任意节点出发，取最远节点 u；
        第2次 BFS 从 u 出发，最远距离即为直径估计值。

  APL：对 LCC 中每个节点做一次完整 BFS，累加所有可达节点距离，
        最后除以有序节点对数 n*(n-1)。
        时间复杂度 O(n*(n+m))，路网稀疏，可行。

输出：
  diameter_hop.png   — 左轴：跳数直径，右轴：跳数 APL，双轴对比柱状图
  diameter_hop.xlsx  — 汇总表
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, glob, os, time
from collections import deque, defaultdict

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


def bfs_all(adj, source, nodes_set):
    """BFS 返回从 source 到 nodes_set 内所有可达节点的跳数字典"""
    dist  = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v in nodes_set and v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def two_sweep_diameter(adj, lcc):
    """2-sweep BFS 估计直径"""
    start    = next(iter(lcc))
    dist1    = bfs_all(adj, start, lcc)
    u        = max(dist1, key=dist1.get)
    dist2    = bfs_all(adj, u, lcc)
    diameter = max(dist2.values())
    return diameter


# ════════════════════════════════════════════════════════════
# 主计算
# ════════════════════════════════════════════════════════════
summary_rows = []

for path, city in zip(JSON_FILES, CITIES):
    print(f'\n{city} 开始...', flush=True)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    all_nodes = [node['id'] for node in data.values()]
    adj       = build_hop_adj(data)
    lcc       = largest_cc(adj, all_nodes)
    n         = len(lcc)

    # ── 直径（2-sweep）────────────────────────────────────
    diam = two_sweep_diameter(adj, lcc)
    print(f'  跳数直径={diam}', flush=True)

    # ── APL（全节点 BFS）─────────────────────────────────
    t0          = time.time()
    total_hops  = 0
    pair_count  = 0

    for i, u in enumerate(lcc):
        dist_map = bfs_all(adj, u, lcc)
        for v, d in dist_map.items():
            if v != u:
                total_hops += d
                pair_count += 1

        if (i + 1) % 1000 == 0 or (i + 1) == n:
            print(f'  BFS {i+1}/{n}  耗时={time.time()-t0:.0f}s', flush=True)

    apl = total_hops / pair_count if pair_count > 0 else 0.0
    print(f'  APL={apl:.4f} 跳')

    summary_rows.append({
        '城市'      : city,
        '总节点数'  : len(all_nodes),
        'LCC节点数' : n,
        '跳数直径'  : diam,
        '跳数APL'   : round(apl, 4),
    })

df = pd.DataFrame(summary_rows)
df.to_excel(os.path.join(OUT_DIR, 'diameter_hop.xlsx'), index=False)
print('\nExcel已保存: diameter_hop.xlsx')
print('\n── 汇总 ──')
print(df.to_string(index=False))


# ════════════════════════════════════════════════════════════
# 绘图：从 Excel 读取，单纵坐标分组柱状图
# ════════════════════════════════════════════════════════════
df = pd.read_excel(os.path.join(OUT_DIR, 'diameter_hop.xlsx'))

x = np.arange(len(df))
w = 0.38

fig, ax = plt.subplots(figsize=(13, 6))

bars1 = ax.bar(x - w/2, df['跳数直径'], w, color=COLORS, alpha=0.85,
               edgecolor='white', label='跳数直径', zorder=2)
bars2 = ax.bar(x + w/2, df['跳数APL'],  w, color=COLORS, alpha=0.45,
               edgecolor='white', hatch='//', label='跳数平均路径长度 (APL)', zorder=2)

for bar, val in zip(bars1, df['跳数直径']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, df['跳数APL']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=9)

ax.set_title('各城市交通网络跳数直径与跳数平均路径长度', fontsize=13, fontweight='bold')
ax.set_xlabel('城市', fontsize=12)
ax.set_ylabel('跳数', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(df['城市'], fontsize=11)
ax.set_ylim(0, max(df['跳数直径'].max(), df['跳数APL'].max()) * 1.18)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, zorder=1)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'diameter_hop.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图已保存: diameter_hop.png')
