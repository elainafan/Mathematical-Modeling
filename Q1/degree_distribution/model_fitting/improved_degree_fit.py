"""
改进度 (Improved Degree) 建模与幂律拟合
=====================================================
参考文献：
  Wang et al. (2017) Physica A 469, 256-264

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 改进度的定义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本数据集无车道数，改用路段长度作代理：
    K_i = (1/k_i) * Σ_{j∈Γ(i)} L_ij / L̄
  · L_ij  : 节点 i 到邻居 j 的路段长度（来自 JSON neighbor.distance）
  · L̄    : 全网所有路段长度的均值（归一化，使 K 量纲无关）
  · k_i   : 节点 i 的原始度（连接路段数）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 梯度分级与幂律拟合流程
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1. 计算每个节点的改进度 K_i（路长加权均值）。

Step 2. 梯度化：以 Δ=0.5 为步长对 K_i 取整归档，
        r(K_i) = round(K_i / 0.5) * 0.5

Step 3. 统计各梯度的概率：p(r) = 该梯度节点数 / 总节点数。

Step 4. 幂律拟合：
        p(r(K)) = C · [r(K)]^{-γ}
        在双对数坐标下呈线性，用非线性最小二乘（Levenberg-Marquardt）
        求解参数 C 和 γ。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 输出文件说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
图1  hierarchy_fit_linear.png   梯度分布柱状图 + 幂律拟合曲线（线性坐标）
图2  hierarchy_fit_loglog.png   梯度分布散点 + 拟合线（双对数坐标，验证线性）
图3  improved_degree_hist.png   各城市改进度 K 连续分布直方图

Excel  improved_degree_fit.xlsx
  Sheet1: 拟合结果汇总（γ、R² 等核心指标）
  Sheet2: 节点改进度明细（每节点的 k, K, r(K)）
  Sheet3: 梯度分布与拟合值（实测 p(r) vs 拟合 p(r)，含残差）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, glob, os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR  = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
JSON_DIR  = os.path.join(BASE_DIR, 'data', 'json_networks')
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))

JSON_FILES = sorted(glob.glob(os.path.join(JSON_DIR, '*.json')))
CITIES     = [os.path.basename(p).replace('_Network.json', '') for p in JSON_FILES]
COLORS     = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
              '#9C27B0', '#00BCD4', '#FF5722', '#607D8B']

BIN_WIDTH = 0.5


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def r_squared_log(y_obs, y_pred):
    """对数空间 R²，适用于幂律拟合质量评估"""
    log_obs  = np.log(y_obs)
    log_pred = np.log(y_pred)
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - log_obs.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def powerlaw_func(x, C, gamma):
    """p(r) = C * r^{-gamma}"""
    return C * x ** (-gamma)


def fit_powerlaw(r_arr, p_arr):
    """
    对数空间幂律拟合：在 log-log 坐标下做线性回归。
    等价于对 log(p) = log(C) - gamma*log(r) 做最小二乘。
    返回 (C, gamma, R²_log)，R² 在对数空间计算。
    """
    if len(r_arr) < 3 or np.any(p_arr <= 0):
        return np.nan, np.nan, np.nan
    try:
        log_r = np.log(r_arr)
        log_p = np.log(p_arr)
        # 线性回归：log_p = a + b * log_r，其中 b = -gamma，a = log(C)
        coeffs = np.polyfit(log_r, log_p, 1)
        gamma  = -coeffs[0]
        C      = np.exp(coeffs[1])
        pred   = powerlaw_func(r_arr, C, gamma)
        r2     = r_squared_log(p_arr, pred)
        return C, gamma, r2
    except Exception:
        return np.nan, np.nan, np.nan


# ════════════════════════════════════════════════════════════
# 改进度计算
# ════════════════════════════════════════════════════════════

def compute_improved_degree(data):
    """
    K_i = (1/k_i) * sum_{j in Gamma(i)} L_ij / L_mean
    返回：K_dict {node_id: K}, L_mean
    """
    all_len = [nb['distance'] for v in data.values() for nb in v['neighbors']]
    L_mean  = np.mean(all_len)

    K_dict = {}
    for node in data.values():
        u  = node['id']
        ki = node['degree']
        if ki == 0:
            K_dict[u] = 0.0
            continue
        total_len = sum(nb['distance'] for nb in node['neighbors'])
        K_dict[u] = total_len / ki / L_mean
    return K_dict, L_mean


MIN_COUNT = 3   # 梯度内节点数低于此值时过滤，避免稀疏尾部噪声


def make_hierarchy(K_vals, bin_width=BIN_WIDTH):
    """
    按 bin_width 步长对 K 取整归档，统计各梯度概率。
    过滤节点数 < MIN_COUNT 的稀疏梯度。
    返回：(r_arr, p_arr)
    """
    rounded = np.round(K_vals / bin_width) * bin_width
    rounded = np.maximum(rounded, bin_width)
    cnt     = Counter(rounded.tolist())
    r_arr   = np.array([r for r in sorted(cnt.keys()) if cnt[r] >= MIN_COUNT])
    p_arr   = np.array([cnt[r] / len(K_vals) for r in r_arr])
    return r_arr, p_arr


# ════════════════════════════════════════════════════════════
# 主循环：计算 + 拟合
# ════════════════════════════════════════════════════════════

store   = {}
results = []

print('=' * 60)
print('改进度梯度幂律拟合（Wang et al. 2017）')
print('K_i = (1/k_i) * Σ L_ij / L_mean')
print('=' * 60)

for path, city in zip(JSON_FILES, CITIES):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    K_dict, L_mean = compute_improved_degree(data)
    K_arr    = np.array(list(K_dict.values()))
    k_orig   = np.array([v['degree'] for v in data.values()])
    node_ids = [v['id'] for v in data.values()]

    r_arr, p_arr = make_hierarchy(K_arr)
    C, gamma, r2 = fit_powerlaw(r_arr, p_arr)

    store[city] = dict(
        K_arr=K_arr, k_orig=k_orig, node_ids=node_ids,
        K_dict=K_dict, L_mean=L_mean,
        r_arr=r_arr, p_arr=p_arr,
        C=C, gamma=gamma, r2=r2,
    )

    print(f'\n{city}  (L_mean={L_mean:.1f}m  K∈[{K_arr.min():.2f},{K_arr.max():.2f}]  梯度数={len(r_arr)})')
    print(f'  γ={gamma:.4f}  R²={r2:.4f}')

    results.append({
        '城市'       : city,
        '节点数'     : len(k_orig),
        'L_mean(m)'  : round(L_mean, 1),
        'K均值'      : round(K_arr.mean(), 4),
        'K最大值'    : round(K_arr.max(), 4),
        '梯度数'     : len(r_arr),
        'C'          : round(C, 6) if not np.isnan(C) else None,
        'γ'          : round(gamma, 4) if not np.isnan(gamma) else None,
        'R²'         : round(r2, 4) if not np.isnan(r2) else None,
    })

df = pd.DataFrame(results)


# ════════════════════════════════════════════════════════════
# 图1：梯度分布 + 幂律拟合（线性坐标）
# ════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
fig1.suptitle(r'改进度梯度分布与幂律拟合  $p(r(K)) \sim [r(K)]^{-\gamma}$（线性坐标）',
              fontsize=13, fontweight='bold', y=1.01)

for ax, city, color in zip(axes1.flat, CITIES, COLORS):
    d   = store[city]
    row = df[df['城市'] == city].iloc[0]

    ax.bar(d['r_arr'], d['p_arr'], BIN_WIDTH * 0.8,
           color=color, alpha=0.85, label='实测', zorder=2)

    if not np.isnan(d['C']):
        rl = np.linspace(d['r_arr'].min(), d['r_arr'].max(), 300)
        ax.plot(rl, powerlaw_func(rl, d['C'], d['gamma']),
                'k-', lw=2.2, zorder=3,
                label=f"$\\gamma$={row['γ']:.3f}\n$R^2$={row['R²']:.4f}")

    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'梯度 $r(K)$', fontsize=9)
    ax.set_ylabel(r'$p(r(K))$', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'hierarchy_fit_linear.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n图1 已保存: hierarchy_fit_linear.png')


# ════════════════════════════════════════════════════════════
# 图2：双对数坐标（验证幂律的线性关系）
# ════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
fig2.suptitle(r'改进度梯度分布（双对数坐标）— 验证幂律线性关系',
              fontsize=13, fontweight='bold', y=1.01)

for ax, city, color in zip(axes2.flat, CITIES, COLORS):
    d   = store[city]
    row = df[df['城市'] == city].iloc[0]

    ax.scatter(d['r_arr'], d['p_arr'], s=55, color=color,
               edgecolors='white', lw=0.5, zorder=4, label='实测')

    if not np.isnan(d['C']):
        rl = np.linspace(d['r_arr'].min(), d['r_arr'].max(), 300)
        ax.plot(rl, powerlaw_func(rl, d['C'], d['gamma']),
                'k--', lw=2, zorder=5,
                label=f"$\\gamma$={row['γ']:.3f}  $R^2$={row['R²']:.4f}")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$r(K)$ (log)', fontsize=9)
    ax.set_ylabel(r'$p(r(K))$ (log)', fontsize=9)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, which='both', alpha=0.2)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'hierarchy_fit_loglog.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 已保存: hierarchy_fit_loglog.png')


# ════════════════════════════════════════════════════════════
# 图3：各城市改进度 K 连续分布直方图
# ════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
fig3.suptitle('各城市改进度 $K$ 连续分布直方图',
              fontsize=13, fontweight='bold', y=1.01)

for ax, city, color in zip(axes3.flat, CITIES, COLORS):
    d = store[city]
    ax.hist(d['K_arr'], bins=40, color=color, alpha=0.8,
            edgecolor='white', linewidth=0.3)
    ax.axvline(d['K_arr'].mean(), color='black', linestyle='--', linewidth=1.3,
               label=f"均值 {d['K_arr'].mean():.3f}")
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('改进度 $K$', fontsize=9)
    ax.set_ylabel('节点数', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'improved_degree_hist.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图3 已保存: improved_degree_hist.png')


# ════════════════════════════════════════════════════════════
# 保存 Excel（3个 Sheet）
# ════════════════════════════════════════════════════════════
excel_path = os.path.join(OUT_DIR, 'improved_degree_fit.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1：拟合结果汇总
    df.to_excel(writer, sheet_name='拟合结果汇总', index=False)

    # Sheet 2：节点改进度明细
    detail_rows = []
    for city in CITIES:
        d = store[city]
        for nid, ko, kv in zip(d['node_ids'], d['k_orig'], d['K_arr']):
            detail_rows.append({
                '城市'    : city,
                '节点ID'  : nid,
                '原始度k' : int(ko),
                '改进度K' : round(float(kv), 6),
                '梯度r(K)': round(round(float(kv) / BIN_WIDTH) * BIN_WIDTH, 2),
            })
    pd.DataFrame(detail_rows).to_excel(writer, sheet_name='节点改进度明细', index=False)

    # Sheet 3：梯度分布与拟合值
    grad_rows = []
    for city in CITIES:
        d = store[city]
        for r, p in zip(d['r_arr'], d['p_arr']):
            pred = powerlaw_func(r, d['C'], d['gamma']) \
                   if not np.isnan(d['C']) else None
            grad_rows.append({
                '城市'     : city,
                '梯度r(K)' : round(float(r), 2),
                'p_实测'   : round(float(p), 8),
                'p_拟合'   : round(float(pred), 8) if pred is not None else None,
                '残差'     : round(float(p) - float(pred), 8) if pred is not None else None,
            })
    pd.DataFrame(grad_rows).to_excel(writer, sheet_name='梯度分布与拟合值', index=False)

print(f'\nExcel 已保存: improved_degree_fit.xlsx')

print('\n' + '=' * 60)
print('  拟合结果汇总')
print('=' * 60)
print(df[['城市', 'γ', 'R²', '梯度数', 'K均值', 'K最大值']].to_string(index=False))
