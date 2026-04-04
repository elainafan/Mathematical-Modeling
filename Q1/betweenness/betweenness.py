"""
节点介数中心性统计分析（精确算法）
数据来源：data/json_networks/*.json
"""

import json, glob, os, time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from scipy.optimize import curve_fit

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
# 计算各城市介数中心性（精确，非加权）
# ════════════════════════════════════════════════════════════
city_bc      = {}   # city -> np.array of bc values（按节点顺序）
city_bc_dict = {}   # city -> {node_id: bc_value}
city_json    = {}   # city -> raw json data（用于坐标/度数）
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

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f'  节点={n_nodes}  边={n_edges}  开始精确计算...', flush=True)

    t0 = time.time()
    bc = nx.betweenness_centrality(G, normalized=True, weight=None)
    elapsed = time.time() - t0
    print(f'  完成，耗时 {elapsed:.1f}s', flush=True)

    vals = np.array(list(bc.values()))
    city_bc[city]      = vals
    city_bc_dict[city] = bc
    city_json[city]    = data

    summary_rows.append({
        '城市'    : city,
        '节点数'  : n_nodes,
        '边数'    : n_edges,
        '均值'    : round(vals.mean(), 6),
        '中位数'  : round(np.median(vals), 6),
        '标准差'  : round(vals.std(), 6),
        '最大值'  : round(vals.max(), 6),
        '最小值'  : round(vals.min(), 6),
        '零值比例': round((vals == 0).mean(), 4),
        '耗时(s)' : round(elapsed, 1),
    })
    print(f'  均值={vals.mean():.6f}  最大={vals.max():.6f}  零值比例={(vals==0).mean():.2%}')

df_sum = pd.DataFrame(summary_rows)


# ════════════════════════════════════════════════════════════
# 图1：各城市介数分布直方图（log y 轴，2×4）
# ════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 9))
fig1.suptitle('各城市节点介数中心性分布（排除零值）', fontsize=15, fontweight='bold', y=1.01)

for ax, city, color in zip(axes1.flat, CITIES, COLORS):
    vals    = city_bc[city]
    nonzero = vals[vals > 0]
    mean_v  = vals.mean()

    ax.hist(nonzero, bins=60, color=color, alpha=0.8, edgecolor='white', linewidth=0.3, log=True)
    ax.axvline(mean_v, color='black', linestyle='--', linewidth=1.3,
               label=f'均值 {mean_v:.5f}')
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('介数中心性 B(v)', fontsize=9)
    ax.set_ylabel('节点数（log）', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'betweenness_hist.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图1 已保存: betweenness_hist.png')


# ════════════════════════════════════════════════════════════
# 图2：双对数坐标下的互补累积分布（CCDF）
# 道路网介数通常呈幂律或对数正态分布，CCDF 最直观
# ════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 9))
fig2.suptitle('介数中心性互补累积分布 CCDF（双对数坐标）', fontsize=15, fontweight='bold', y=1.01)

fit_rows = []

for ax, city, color in zip(axes2.flat, CITIES, COLORS):
    vals    = city_bc[city]
    nonzero = np.sort(vals[vals > 0])
    n       = len(nonzero)
    ccdf    = 1 - np.arange(1, n + 1) / n

    ax.loglog(nonzero, ccdf, '.', color=color, markersize=2, alpha=0.7, label='实测 CCDF')

    # 在尾部（前5%高值）拟合幂律斜率
    tail_mask = nonzero >= np.percentile(nonzero, 70)
    x_tail = nonzero[tail_mask]
    y_tail = ccdf[tail_mask & (ccdf > 0)]
    x_tail = x_tail[:len(y_tail)]

    if len(x_tail) > 5:
        log_x = np.log10(x_tail)
        log_y = np.log10(y_tail + 1e-10)
        slope, intercept, r, *_ = stats.linregress(log_x, log_y)
        x_fit = np.logspace(np.log10(x_tail.min()), np.log10(x_tail.max()), 50)
        y_fit = 10 ** (intercept + slope * np.log10(x_fit))
        ax.loglog(x_fit, y_fit, 'k--', linewidth=1.5,
                  label=f'幂律斜率 α={-slope:.2f}\n(R²={r**2:.3f})')
        fit_rows.append({'城市': city, '幂律斜率α': round(-slope, 3),
                         'R²': round(r**2, 4)})

    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('介数中心性 B(v)', fontsize=9)
    ax.set_ylabel('P(B > b)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.2)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'betweenness_ccdf.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 已保存: betweenness_ccdf.png')


# ════════════════════════════════════════════════════════════
# 图3：8城市均值/最大值/零值比例对比
# ════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle('各城市介数中心性统计对比', fontsize=13, fontweight='bold')

metrics = [('均值', '平均介数中心性'), ('最大值', '最大介数中心性'), ('零值比例', '零介数节点占比')]
for ax, (col, title) in zip(axes3, metrics):
    vals_plot = df_sum[col].values
    bars = ax.bar(CITIES, vals_plot, color=COLORS, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals_plot):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticklabels(CITIES, rotation=25, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'betweenness_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图3 已保存: betweenness_summary.png')


# ════════════════════════════════════════════════════════════
# 保存 Excel（汇总 + 幂律拟合 + 各城市节点明细）
# ════════════════════════════════════════════════════════════
excel_path = os.path.join(OUT_DIR, 'betweenness.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1：汇总统计
    df_sum.to_excel(writer, sheet_name='汇总统计', index=False)

    # Sheet 2：幂律拟合（CCDF 线性回归斜率）
    if fit_rows:
        pd.DataFrame(fit_rows).to_excel(writer, sheet_name='幂律拟合', index=False)

    # Sheet 3：全城市节点明细（一张总表）
    all_detail = []
    for city in CITIES:
        bc_dict = city_bc_dict[city]
        data    = city_json[city]
        bc_vals = np.array(list(bc_dict.values()))
        # 按介数从大到小排名
        sorted_ids = sorted(bc_dict, key=bc_dict.get, reverse=True)
        rank_map   = {nid: r + 1 for r, nid in enumerate(sorted_ids)}
        for node in data.values():
            nid = node['id']
            bv  = bc_dict.get(nid, 0.0)
            all_detail.append({
                '城市'    : city,
                '节点ID'  : nid,
                'X坐标'   : round(node['position'][0], 4),
                'Y坐标'   : round(node['position'][1], 4),
                '度数'    : node['degree'],
                '介数中心性': round(bv, 8),
                '城市内排名': rank_map[nid],
            })
    pd.DataFrame(all_detail).to_excel(writer, sheet_name='节点明细(全部)', index=False)

    # Sheet 4~11：每座城市单独一张 sheet，按介数降序
    for city in CITIES:
        bc_dict = city_bc_dict[city]
        data    = city_json[city]
        rows = []
        for node in data.values():
            nid = node['id']
            bv  = bc_dict.get(nid, 0.0)
            rows.append({
                '节点ID'  : nid,
                'X坐标'   : round(node['position'][0], 4),
                'Y坐标'   : round(node['position'][1], 4),
                '度数'    : node['degree'],
                '介数中心性': round(bv, 8),
            })
        df_city = pd.DataFrame(rows).sort_values('介数中心性', ascending=False)
        df_city.insert(0, '排名', range(1, len(df_city) + 1))
        df_city.to_excel(writer, sheet_name=city, index=False)
        print(f'  Sheet [{city}] 已写入 {len(df_city)} 行')

print(f'\nExcel 已保存: betweenness.xlsx')
print('\n── 汇总 ──')
print(df_sum.to_string(index=False))
