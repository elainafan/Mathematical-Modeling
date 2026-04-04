"""
度分布统计与绘图
数据来源：data/json_networks/*.json
每个节点结构：{"id":..., "position":[x,y], "degree":..., "neighbors":[...]}
"""

import json
import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 字体 ─────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR   = os.path.join(BASE_DIR, 'data', 'json_networks')
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))

JSON_FILES = sorted(glob.glob(os.path.join(JSON_DIR, '*.json')))
CITIES     = [os.path.basename(p).replace('_Network.json', '') for p in JSON_FILES]

COLORS = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
          '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']


# ════════════════════════════════════════════════════════════
# 1. 读数据，提取度序列
# ════════════════════════════════════════════════════════════
city_degrees: dict[str, list[int]] = {}

for path, city in zip(JSON_FILES, CITIES):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    degrees = [node['degree'] for node in data.values()]
    city_degrees[city] = degrees
    print(f'{city:12s}  节点数={len(degrees):6d}  '
          f'平均度={np.mean(degrees):.3f}  最大度={max(degrees)}')


# ════════════════════════════════════════════════════════════
# 2. 计算度分布
# ════════════════════════════════════════════════════════════
def degree_dist(degrees):
    cnt   = Counter(degrees)
    k_arr = np.array(sorted(cnt), dtype=float)
    prob  = np.array([cnt[k] for k in k_arr.astype(int)], dtype=float)
    prob  = prob / prob.sum()
    return k_arr, prob

dist_results = {city: degree_dist(degs) for city, degs in city_degrees.items()}


# ════════════════════════════════════════════════════════════
# 3. 图1：各城市度分布柱状图（2×4 网格）
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('各城市交通网络度分布', fontsize=16, fontweight='bold', y=1.01)

for ax, city, color in zip(axes.flat, CITIES, COLORS):
    k_arr, prob = dist_results[city]
    ax.bar(k_arr, prob, color=color, alpha=0.8, width=0.6, zorder=2)
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('度 k', fontsize=9)
    ax.set_ylabel('P(k)', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(left=0)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'degree_dist_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图1 已保存: degree_dist_bar.png')


# ════════════════════════════════════════════════════════════
# 4. 图2：双对数坐标（2×4 网格）
# ════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 8))
fig2.suptitle('各城市度分布（双对数坐标）', fontsize=16, fontweight='bold', y=1.01)

for ax, city, color in zip(axes2.flat, CITIES, COLORS):
    k_arr, prob = dist_results[city]
    mask = (k_arr >= 1) & (prob > 0)
    ax.scatter(k_arr[mask], prob[mask], color=color, s=40, alpha=0.9, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('度 k (log)', fontsize=9)
    ax.set_ylabel('P(k) (log)', fontsize=9)
    ax.grid(True, which='both', alpha=0.2)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'degree_dist_loglog.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 已保存: degree_dist_loglog.png')


# ════════════════════════════════════════════════════════════
# 5. 保存度分布数据到 Excel
# ════════════════════════════════════════════════════════════
excel_path = os.path.join(OUT_DIR, 'degree_distribution.xlsx')
rows = []
for city in CITIES:
    degs  = city_degrees[city]
    cnt   = Counter(degs)
    total = len(degs)
    for k in sorted(cnt):
        rows.append({'城市': city, '度k': k,
                     '节点数': cnt[k], 'P(k)': round(cnt[k] / total, 6)})

pd.DataFrame(rows).to_excel(excel_path, sheet_name='度分布', index=False)
print(f'Excel 已保存: degree_distribution.xlsx')
