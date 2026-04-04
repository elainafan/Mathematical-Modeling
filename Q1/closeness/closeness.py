"""
节点接近中心性统计分析
数据来源：data/json_networks/*.json

定义：C(v) = (N-1) / Σ_{u≠v} d(v,u)
即到所有其他节点最短路径平均值的倒数（N-1条路径）
对于非连通图，仅在最大连通分量内计算。
"""

import json, glob, os, time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_DIR  = os.path.join(BASE_DIR, 'data', 'json_networks')
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

JSON_FILES = sorted(glob.glob(os.path.join(JSON_DIR, '*.json')))
CITIES     = [os.path.basename(p).replace('_Network.json', '') for p in JSON_FILES]
COLORS     = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
              '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']


# ════════════════════════════════════════════════════════════
# 计算各城市接近中心性（在最大连通分量上，非加权BFS）
# ════════════════════════════════════════════════════════════
city_cc      = {}   # city -> np.array of closeness values（LCC节点顺序）
city_cc_dict = {}   # city -> {node_id: closeness_value}
summary_rows = []

for path, city in zip(JSON_FILES, CITIES):
    print(f'\n[{city}] 构建图...', flush=True)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    G = nx.Graph()
    for node in data.values():
        u = node['id']
        for nb in node['neighbors']:
            G.add_edge(u, nb['id'])

    # 取最大连通分量
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc     = G.subgraph(lcc_nodes).copy()
    n_lcc     = G_lcc.number_of_nodes()

    print(f'  总节点={G.number_of_nodes()}  LCC节点={n_lcc}  开始计算...', flush=True)
    t0 = time.time()

    # networkx closeness_centrality 在 LCC 上计算
    # 公式：C(v) = (n_lcc-1) / Σ d(v,u)，u∈LCC, u≠v
    cc = nx.closeness_centrality(G_lcc)

    elapsed = time.time() - t0
    print(f'  完成，耗时 {elapsed:.1f}s', flush=True)

    vals = np.array(list(cc.values()))
    city_cc[city]      = vals
    city_cc_dict[city] = cc

    summary_rows.append({
        '城市'      : city,
        '总节点数'  : G.number_of_nodes(),
        'LCC节点数' : n_lcc,
        '均值'      : round(vals.mean(), 6),
        '中位数'    : round(np.median(vals), 6),
        '标准差'    : round(vals.std(), 6),
        '最大值'    : round(vals.max(), 6),
        '最小值'    : round(vals.min(), 6),
        '耗时(s)'   : round(elapsed, 1),
    })
    print(f'  均值={vals.mean():.6f}  最大={vals.max():.6f}  最小={vals.min():.6f}')

df_sum = pd.DataFrame(summary_rows)


# ════════════════════════════════════════════════════════════
# 图1：各城市接近中心性分布直方图（2×4）
# ════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 9))
fig1.suptitle('各城市节点接近中心性分布', fontsize=15, fontweight='bold', y=1.01)

for ax, city, color in zip(axes1.flat, CITIES, COLORS):
    vals   = city_cc[city]
    mean_v = vals.mean()

    ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axvline(mean_v, color='black', linestyle='--', linewidth=1.3,
               label=f'均值 {mean_v:.4f}')
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('接近中心性 C(v)', fontsize=9)
    ax.set_ylabel('节点数', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'closeness_hist.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图1 已保存: closeness_hist.png')


# ════════════════════════════════════════════════════════════
# 图2：8城市均值/最大值/标准差对比柱状图
# ════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
fig2.suptitle('各城市接近中心性统计对比', fontsize=13, fontweight='bold')

metrics = [('均值', '平均接近中心性'), ('最大值', '最大接近中心性'), ('标准差', '标准差')]
for ax, (col, title) in zip(axes2, metrics):
    vals_plot = df_sum[col].values
    bars = ax.bar(CITIES, vals_plot, color=COLORS, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals_plot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(CITIES)))
    ax.set_xticklabels(CITIES, rotation=25, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'closeness_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 已保存: closeness_summary.png')


# ════════════════════════════════════════════════════════════
# 图3：8城市接近中心性箱线图（横向对比分布形态）
# ════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(12, 6))
data_box = [city_cc[c] for c in CITIES]
bp = ax3.boxplot(data_box, tick_labels=CITIES, patch_artist=True,
                 medianprops=dict(color='black', linewidth=1.8),
                 whiskerprops=dict(linewidth=1.2),
                 capprops=dict(linewidth=1.2),
                 flierprops=dict(marker='.', markersize=2, alpha=0.4))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_title('各城市接近中心性分布箱线图', fontsize=14, fontweight='bold')
ax3.set_xlabel('城市', fontsize=12)
ax3.set_ylabel('接近中心性 C(v)', fontsize=12)
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'closeness_boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图3 已保存: closeness_boxplot.png')


# ════════════════════════════════════════════════════════════
# 保存 Excel（汇总 + 节点明细）
# ════════════════════════════════════════════════════════════
excel_path = os.path.join(OUT_DIR, 'closeness.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1：汇总统计
    df_sum.to_excel(writer, sheet_name='汇总统计', index=False)

    # Sheet 2~9：各城市节点明细，按接近中心性降序
    for city in CITIES:
        cc_dict = city_cc_dict[city]
        with open([p for p in JSON_FILES if city in p][0], encoding='utf-8') as f:
            raw = json.load(f)

        rows = []
        sorted_ids = sorted(cc_dict, key=cc_dict.get, reverse=True)
        rank_map   = {nid: r + 1 for r, nid in enumerate(sorted_ids)}

        for node in raw.values():
            nid = node['id']
            cv  = cc_dict.get(nid, None)   # LCC外节点无值
            rows.append({
                '排名'       : rank_map.get(nid, '-'),
                '节点ID'     : nid,
                'X坐标'      : round(node['position'][0], 4),
                'Y坐标'      : round(node['position'][1], 4),
                '度数'       : node['degree'],
                '接近中心性' : round(cv, 8) if cv is not None else None,
            })

        df_city = pd.DataFrame(rows).sort_values(
            '接近中心性', ascending=False, na_position='last'
        )
        df_city.to_excel(writer, sheet_name=city, index=False)
        print(f'  Sheet [{city}] 已写入 {len(df_city)} 行')

print(f'\nExcel 已保存: closeness.xlsx')
print('\n── 汇总 ──')
print(df_sum.to_string(index=False))
