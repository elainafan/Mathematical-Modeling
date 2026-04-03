"""
聚类系数计算与绘图
数据来源：data/json_networks/*.json
"""

import json, os, glob
from collections import defaultdict

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


# ════════════════════════════════════════════════════════════
# 工具：从 JSON 构建邻接集，计算各节点局部聚类系数
# ════════════════════════════════════════════════════════════
def compute_clustering(data: dict):
    """
    局部聚类系数 C(v) = 实际三角形数 / 可能三角形数
                      = 2*T / (k*(k-1))
    其中 k = 节点度数，T = 邻居之间存在的边数。
    度数 < 2 的节点定义为 0。
    返回 {node_id: C(v)} 字典。
    """
    # 构建无向邻接集
    adj: dict[int, set[int]] = defaultdict(set)
    for node in data.values():
        u = node['id']
        for nb in node['neighbors']:
            v = nb['id']
            adj[u].add(v)
            adj[v].add(u)

    cc = {}
    for node in data.values():
        u   = node['id']
        nbs = adj[u]
        k   = len(nbs)
        if k < 2:
            cc[u] = 0.0
            continue
        # 邻居之间的边数
        triangles = sum(1 for v in nbs for w in nbs if w in adj[v] and v < w)
        cc[u] = 2 * triangles / (k * (k - 1))
    return cc


# ════════════════════════════════════════════════════════════
# 计算各城市聚类系数
# ════════════════════════════════════════════════════════════
city_cc   : dict[str, dict] = {}
summary_rows = []

for path, city in zip(JSON_FILES, CITIES):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    cc = compute_clustering(data)
    city_cc[city] = cc

    values = list(cc.values())
    avg    = np.mean(values)
    med    = np.median(values)
    std    = np.std(values)
    nonzero_avg = np.mean([v for v in values if v > 0]) if any(v > 0 for v in values) else 0.0

    summary_rows.append({
        '城市'         : city,
        '节点数'        : len(values),
        '平均聚类系数'   : round(avg, 6),
        '中位数'        : round(med, 6),
        '标准差'        : round(std, 6),
        '非零节点平均'   : round(nonzero_avg, 6),
        '聚类系数=0比例' : round(sum(1 for v in values if v == 0) / len(values), 4),
    })
    print(f'{city:12s}  avg={avg:.6f}  median={med:.6f}  std={std:.6f}')


# ════════════════════════════════════════════════════════════
# 图1：各城市局部聚类系数分布直方图（2×4）
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('各城市节点局部聚类系数分布', fontsize=16, fontweight='bold', y=1.01)

for ax, city, color in zip(axes.flat, CITIES, COLORS):
    values = list(city_cc[city].values())
    nonzero = [v for v in values if v > 0]
    avg = np.mean(values)

    ax.hist(nonzero, bins=30, color=color, alpha=0.8, edgecolor='white', linewidth=0.4)
    ax.axvline(avg, color='black', linestyle='--', linewidth=1.2,
               label=f'均值 {avg:.4f}')
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('聚类系数 C(v)', fontsize=9)
    ax.set_ylabel('节点数', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'clustering_hist.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图1 已保存: clustering_hist.png')


# ════════════════════════════════════════════════════════════
# 图2：8城市平均聚类系数横向对比柱状图
# ════════════════════════════════════════════════════════════
df_sum = pd.DataFrame(summary_rows).set_index('城市')

fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.bar(df_sum.index, df_sum['平均聚类系数'],
               color=COLORS, alpha=0.85, edgecolor='white', linewidth=0.5)
ax2.errorbar(df_sum.index, df_sum['平均聚类系数'],
             yerr=df_sum['标准差'], fmt='none', color='black',
             capsize=4, linewidth=1.2, label='±1 标准差')
for bar, val in zip(bars, df_sum['平均聚类系数']):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
ax2.set_title('各城市平均聚类系数对比', fontsize=14, fontweight='bold')
ax2.set_xlabel('城市', fontsize=11)
ax2.set_ylabel('平均聚类系数', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'clustering_compare.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 已保存: clustering_compare.png')


# ════════════════════════════════════════════════════════════
# 保存 Excel
# ════════════════════════════════════════════════════════════
excel_path = os.path.join(OUT_DIR, 'clustering_coefficient.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='汇总', index=False)

    # 各城市节点明细
    detail_rows = []
    for city in CITIES:
        for node_id, val in city_cc[city].items():
            detail_rows.append({'城市': city, '节点ID': node_id,
                                '聚类系数': round(val, 6)})
    pd.DataFrame(detail_rows).to_excel(writer, sheet_name='节点明细', index=False)

print(f'Excel 已保存: clustering_coefficient.xlsx')
print('\n── 汇总 ──')
print(pd.DataFrame(summary_rows).to_string(index=False))
