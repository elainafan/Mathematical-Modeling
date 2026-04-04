"""
改进度 (Improved Degree) 建模与幂律拟合
=====================================================
参考文献：
  Wang et al. (2017) Physica A 469, 256-264
  "The improved degree of urban road traffic network:
   A case study of Xiamen, China"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 为何原始度拟合失败
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
城市路网的原始度 k 通常只取 {1,2,3,4,5,6} 这几个整数，且 k=3（T形路口）
和 k=4（十字路口）占绝大多数，分布严重非单调。
尝试用经典双幂律 P(k)=A*(1+a*k)^{-b} 拟合时：
  · R² 仅 0.24~0.38，拟合效果差
  · 参数 b 退化到数千，失去物理意义
根本原因：k 取值范围过窄（只有 6 个离散值），无法体现幂律分布
所需的"连续长尾"特征。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 改进度的定义（论文 Eq.6）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
论文原始定义（有向网络，需要车道数据）：
    K_i = (1/k_i) * Σ_{j∈Γ(i)} E_j
  其中 E_j = k_j^out + k_j^in 为路段 j 的双向车道总数。
  物理意义：节点 i 每条连接路段平均承载的双向车道数，
            反映交叉口的真实通行能力等级。

本数据集无车道数，改用路段长度作代理（路段越长通常等级越高）：
    K_i = (1/k_i) * Σ_{j∈Γ(i)} L_ij / L̄
  · L_ij  : 节点 i 到邻居 j 的路段长度（来自 JSON neighbor.distance）
  · L̄    : 全网所有路段长度的均值（归一化，使 K 量纲无关）
  · k_i   : 节点 i 的原始度（连接路段数）

与原始度的本质区别：
  · 原始度只计"连接数"，无法区分"连接了3条主干道"和"连接了3条小巷"
  · 改进度用路段长度加权，捕捉了节点周围道路的空间等级信息
  · 改进度为连续变量，取值范围远大于 {1..6}，适合幂律建模

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 梯度分级与幂律拟合流程（复现论文 Fig.5b）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1. 计算每个节点的改进度 K_i（路长加权均值）。

Step 2. 梯度化：以 Δ=0.5 为步长对 K_i 取整归档，
        r(K_i) = round(K_i / 0.5) * 0.5
        将连续的 K 值离散化为一系列等间距梯度 r ∈ {0.5, 1.0, 1.5, ...}。
        论文发现改进度呈现明显的梯度化层次结构。

Step 3. 统计各梯度的概率：p(r) = 该梯度节点数 / 总节点数。

Step 4. 去除第1梯度（论文明确操作）：
        第1梯度（最小 r 值）对应路段极短的特殊节点（如死巷端点），
        它们是结构异常点，会拉低幂律拟合效果，论文在 Fig.5b 中
        明确标注 "p(-1)" 表示去除第1梯度后的分布。

Step 5. 幂律拟合：对剩余梯度拟合
        p(r(K)) = C · [r(K)]^{-γ}
        在双对数坐标下呈线性，用非线性最小二乘（Levenberg-Marquardt）
        求解参数 C 和 γ。

Step 6. 奇偶度分类拟合：
        · 奇数度节点（k=1,3,5）—— 主要是 T 形路口（局部汇聚）
        · 偶数度节点（k=2,4,6）—— 主要是十字路口（骨干转换）
        对两类节点分别做改进度梯度幂律拟合，揭示不同路口类型
        的等级化差异。
        论文参考值：γ_odd = 1.68，γ_even = 1.81。

Step 7. 度-度相关性分析：
        计算每个节点的平均邻居改进度 K_nn(K)，
        考察 K_nn 与 K 的线性关系，判断网络是否存在同配性
        （assortative mixing）及分段特征。

Step 8. 输出图表（6张）与 Excel（4个Sheet）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 输出文件说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
图1  hierarchy_fit_linear.png   梯度分布柱状图 + 幂律拟合曲线（线性坐标）
图2  hierarchy_fit_loglog.png   梯度分布散点 + 拟合线（双对数坐标，验证线性）
图3  odd_even_fit.png           奇偶度节点分类幂律拟合（双对数）
图4  r2_gamma_summary.png       R² 三方对比柱状图 + 各城市 γ 折线图
图5  degree_degree_corr.png     K_nn(K) 度-度相关性散点 + 线性回归
图6  improved_degree_hist.png   各城市改进度 K 连续分布直方图

Excel  improved_degree_fit.xlsx
  Sheet1: 拟合结果汇总（γ、R²、R²提升等核心指标）
  Sheet2: 节点改进度明细（每节点的 k, K, r(K), 奇偶）
  Sheet3: 梯度分布与拟合值（实测 p(r) vs 拟合 p(r)，含残差）
  Sheet4: 原始度分布（对照用）
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, glob, os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
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

BIN_WIDTH = 0.5   # 论文使用 0.5 步长分级


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def r_squared(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def powerlaw_func(x, C, gamma):
    """p(r) = C * r^{-gamma}"""
    return C * x ** (-gamma)


def fit_powerlaw(r_arr, p_arr):
    """幂律拟合，返回 (C, gamma, R²)"""
    if len(r_arr) < 3 or np.any(p_arr <= 0):
        return np.nan, np.nan, np.nan
    try:
        popt, _ = curve_fit(
            powerlaw_func, r_arr, p_arr,
            p0=[p_arr[0] * r_arr[0], 1.7],
            bounds=([1e-8, 0.3], [np.inf, 8.0]),
            maxfev=10000
        )
        pred = powerlaw_func(r_arr, *popt)
        return popt[0], popt[1], r_squared(p_arr, pred)
    except Exception:
        return np.nan, np.nan, np.nan


def double_powerlaw(k, A, a, b):
    """P(k) = A*(1+a*k)^{-b}，用于原始度对照拟合"""
    return A * (1.0 + a * k) ** (-b)


def fit_double_powerlaw(k_arr, p_arr):
    try:
        popt, _ = curve_fit(
            double_powerlaw, k_arr, p_arr,
            p0=[p_arr.max(), 1.0, 2.0],
            bounds=(0, [np.inf, np.inf, np.inf]),
            method='trf', maxfev=10000
        )
        pred = double_powerlaw(k_arr, *popt)
        return *popt, r_squared(p_arr, pred)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ════════════════════════════════════════════════════════════
# 改进度计算（路长代理 Eq.6）
# ════════════════════════════════════════════════════════════

def compute_improved_degree(data):
    """
    K_i = (1/k_i) * sum_{j in Gamma(i)} L_ij / L_mean
    返回：K_dict {node_id: K}, L_mean, k_map {node_id: degree}
    """
    k_map   = {v['id']: v['degree'] for v in data.values()}
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
        K_dict[u] = total_len / ki / L_mean   # 每条路段的平均相对长度
    return K_dict, L_mean, k_map


def make_hierarchy(K_vals, bin_width=BIN_WIDTH, skip_first=True):
    """
    按 bin_width 步长对 K 取整归档，统计各梯度概率。
    skip_first=True：去掉第1梯度（论文做法）。
    返回：(r_all, p_all, r_fit, p_fit)
    """
    rounded = np.round(K_vals / bin_width) * bin_width
    rounded = np.maximum(rounded, bin_width)   # 最小梯度 = bin_width
    cnt     = Counter(rounded.tolist())
    r_all   = np.array(sorted(cnt.keys()))
    p_all   = np.array([cnt[r] / len(K_vals) for r in r_all])
    if skip_first and len(r_all) > 1:
        return r_all, p_all, r_all[1:], p_all[1:]
    return r_all, p_all, r_all, p_all


# ════════════════════════════════════════════════════════════
# 主循环：计算 + 拟合
# ════════════════════════════════════════════════════════════

store   = {}
results = []

print('=' * 72)
print('改进度梯度幂律拟合（路长代理 Wang et al. 2017）')
print('K_i = (1/k_i) * Σ L_ij / L_mean')
print('=' * 72)

for path, city in zip(JSON_FILES, CITIES):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    K_dict, L_mean, k_map = compute_improved_degree(data)
    K_arr  = np.array(list(K_dict.values()))
    k_orig = np.array([v['degree'] for v in data.values()])
    node_ids = [v['id'] for v in data.values()]
    odd_mask  = (k_orig % 2 == 1)
    even_mask = (k_orig % 2 == 0)

    # ── 原始度对照拟合 ───────────────────────────────────────
    cnt_k  = Counter(k_orig.tolist())
    total  = len(k_orig)
    k_arr  = np.array(sorted(cnt_k), dtype=float)
    p_orig = np.array([cnt_k[int(k)] / total for k in k_arr])
    A, a, b, r2_orig = fit_double_powerlaw(k_arr, p_orig)

    # ── 全体改进度梯度拟合 ───────────────────────────────────
    r_all, p_all, r_fit, p_fit = make_hierarchy(K_arr, skip_first=True)
    C_all, g_all, r2_all = fit_powerlaw(r_fit, p_fit)
    # 不去1st作对照
    _, _, r_ns, p_ns = make_hierarchy(K_arr, skip_first=False)
    _, g_ns, r2_ns   = fit_powerlaw(r_ns, p_ns)

    # ── 奇数度节点 ───────────────────────────────────────────
    r_ao, p_ao, r_fo, p_fo = make_hierarchy(K_arr[odd_mask])
    C_odd, g_odd, r2_odd = fit_powerlaw(r_fo, p_fo)

    # ── 偶数度节点 ───────────────────────────────────────────
    r_ae, p_ae, r_fe, p_fe = make_hierarchy(K_arr[even_mask])
    C_even, g_even, r2_even = fit_powerlaw(r_fe, p_fe)

    # ── 度-度相关性：K_nn(K) ─────────────────────────────────
    adj = defaultdict(list)
    for node in data.values():
        for nb in node['neighbors']:
            adj[node['id']].append(nb['id'])
    Knn = {u: np.mean([K_dict.get(v, 0) for v in nbrs]) if nbrs else 0
           for u, nbrs in adj.items()}

    store[city] = dict(
        K_arr=K_arr, k_orig=k_orig, node_ids=node_ids,
        K_dict=K_dict, Knn=Knn, L_mean=L_mean,
        odd_mask=odd_mask, even_mask=even_mask,
        # 原始度
        k_arr=k_arr, p_orig=p_orig, A=A, a=a, b=b, r2_orig=r2_orig,
        # 全体梯度
        r_all=r_all, p_all=p_all, r_fit=r_fit, p_fit=p_fit,
        C_all=C_all, g_all=g_all, r2_all=r2_all, r2_ns=r2_ns, g_ns=g_ns,
        # 奇偶
        r_fo=r_fo, p_fo=p_fo, C_odd=C_odd, g_odd=g_odd, r2_odd=r2_odd,
        r_fe=r_fe, p_fe=p_fe, C_even=C_even, g_even=g_even, r2_even=r2_even,
    )

    r2_lift = (r2_all - r2_orig) if not np.isnan(r2_all) else np.nan
    print(f'\n{city}  (L_mean={L_mean:.1f}m  K∈[{K_arr.min():.2f},{K_arr.max():.2f}]  梯度数={len(r_all)})')
    print(f'  原始度双幂律      R²={r2_orig:.4f}  (b={b:.3f}  参数退化={b>100})')
    print(f'  改进度梯度(含1st)  R²={r2_ns:.4f}  γ={g_ns:.4f}')
    print(f'  改进度梯度(去1st)  R²={r2_all:.4f}  γ={g_all:.4f}  提升Δ={r2_lift:+.4f}')
    print(f'  奇 γ={g_odd:.4f} R²={r2_odd:.4f}  |  偶 γ={g_even:.4f} R²={r2_even:.4f}')
    print(f'  (论文参考值: γ_all=1.69  γ_odd=1.68  γ_even=1.81)')

    results.append({
        '城市'             : city,
        '节点数'           : len(k_orig),
        'L_mean(m)'        : round(L_mean, 1),
        'K均值'            : round(K_arr.mean(), 4),
        'K最大值'          : round(K_arr.max(), 4),
        '梯度数(总)'       : len(r_all),
        '梯度数(拟合)'     : len(r_fit),
        'R²_原始度(对照)'  : round(r2_orig, 4),
        'b_双幂律'         : round(b, 2) if not np.isnan(b) else None,
        'R²_含1st梯度'     : round(r2_ns, 4),
        'γ_含1st梯度'      : round(g_ns, 4) if not np.isnan(g_ns) else None,
        'R²_去1st梯度'     : round(r2_all, 4),
        'γ_全体'           : round(g_all, 4) if not np.isnan(g_all) else None,
        'R²提升'           : round(r2_lift, 4) if not np.isnan(r2_lift) else None,
        'γ_奇数度'         : round(g_odd, 4) if not np.isnan(g_odd) else None,
        'R²_奇数度'        : round(r2_odd, 4) if not np.isnan(r2_odd) else None,
        'γ_偶数度'         : round(g_even, 4) if not np.isnan(g_even) else None,
        'R²_偶数度'        : round(r2_even, 4) if not np.isnan(r2_even) else None,
    })

df = pd.DataFrame(results)


# ════════════════════════════════════════════════════════════
# 图1：梯度分布 + 幂律拟合（线性坐标，复现论文 Fig.5b）
# ════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
fig1.suptitle(r'改进度梯度分布与幂律拟合  $p(r(K)) \sim [r(K)]^{-\gamma}$'
              '\n（线性坐标，灰色为去除的第1梯度，蓝色为参与拟合梯度）',
              fontsize=13, fontweight='bold', y=1.01)

for ax, city, color in zip(axes1.flat, CITIES, COLORS):
    d   = store[city]
    row = df[df['城市'] == city].iloc[0]

    # 第1梯度（灰色，不参与拟合）
    ax.bar(d['r_all'][:1], d['p_all'][:1], BIN_WIDTH * 0.8,
           color='#BDBDBD', alpha=0.85, label='第1梯度（去除）', zorder=2)
    # 拟合梯度
    ax.bar(d['r_fit'], d['p_fit'], BIN_WIDTH * 0.8,
           color=color, alpha=0.85, label='参与拟合', zorder=2)
    # 拟合曲线
    if not np.isnan(d['C_all']):
        rl = np.linspace(d['r_fit'].min(), d['r_fit'].max(), 300)
        ax.plot(rl, powerlaw_func(rl, d['C_all'], d['g_all']),
                'k-', lw=2.2, zorder=3,
                label=f"$\\gamma$={row['γ_全体']:.3f}\n$R^2$={row['R²_去1st梯度']:.4f}")

    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'梯度 $r(K)$', fontsize=9)
    ax.set_ylabel(r'$p(r(K))$', fontsize=9)
    ax.legend(fontsize=7.5, loc='upper right')
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

    ax.scatter(d['r_all'][:1], d['p_all'][:1], marker='x', s=80,
               color='#9E9E9E', zorder=3, label='第1梯度（去除）')
    ax.scatter(d['r_fit'], d['p_fit'], s=55, color=color,
               edgecolors='white', lw=0.5, zorder=4, label='拟合梯度')
    if not np.isnan(d['C_all']):
        rl = np.linspace(d['r_fit'].min(), d['r_fit'].max(), 300)
        ax.plot(rl, powerlaw_func(rl, d['C_all'], d['g_all']),
                'k--', lw=2, zorder=5,
                label=f"$\\gamma$={row['γ_全体']:.3f}  $R^2$={row['R²_去1st梯度']:.4f}")

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$r(K)$ (log)', fontsize=9)
    ax.set_ylabel(r'$p(r(K))$ (log)', fontsize=9)
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(True, which='both', alpha=0.2)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'hierarchy_fit_loglog.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 已保存: hierarchy_fit_loglog.png')


# ════════════════════════════════════════════════════════════
# 图3：奇偶分类拟合（论文 Fig.4b 对应）
# ════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
fig3.suptitle('奇偶度节点改进度梯度幂律\n'
              r'T形口（奇数度）vs 十字口（偶数度）  $p(r)\sim r^{-\gamma}$',
              fontsize=13, fontweight='bold', y=1.01)

for ax, city, color in zip(axes3.flat, CITIES, COLORS):
    d = store[city]

    for r_f, p_f, C, g, r2, label, lc, mk in [
        (d['r_fo'], d['p_fo'], d['C_odd'],  d['g_odd'],  d['r2_odd'],
         '奇数度（T形口）',  '#E91E63', 'o'),
        (d['r_fe'], d['p_fe'], d['C_even'], d['g_even'], d['r2_even'],
         '偶数度（十字口）', '#2196F3', 's'),
    ]:
        if len(r_f) < 3: continue
        ax.scatter(r_f, p_f, s=50, color=lc, marker=mk,
                   edgecolors='white', lw=0.5, zorder=3)
        if not np.isnan(C):
            rl = np.linspace(r_f.min(), r_f.max(), 200)
            ax.plot(rl, powerlaw_func(rl, C, g), color=lc, lw=2,
                    label=f'{label}\n$\\gamma$={g:.3f}  $R^2$={r2:.3f}')

    ax.axvline(1.69, color='gray', ls=':', lw=1, alpha=0.6)   # 论文参考线
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(city, fontsize=10, fontweight='bold')
    ax.set_xlabel(r'$r(K)$', fontsize=9); ax.set_ylabel(r'$p(r)$', fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.2)

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'odd_even_fit.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图3 已保存: odd_even_fit.png')


# ════════════════════════════════════════════════════════════
# 图4：R² 去/不去1st梯度对比 + 各城市 γ 汇总
# ════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle('改进度建模效果汇总', fontsize=14, fontweight='bold')

x = np.arange(len(CITIES)); w = 0.28

ax = axes4[0]
b1 = ax.bar(x - w,     df['R²_原始度(对照)'], w,
            color='#90A4AE', alpha=0.9, label='原始度双幂律（对照）')
b2 = ax.bar(x,         df['R²_含1st梯度'],    w,
            color='#FF9800', alpha=0.9, label='改进度含第1梯度')
b3 = ax.bar(x + w,     df['R²_去1st梯度'],    w,
            color='#2196F3', alpha=0.9, label='改进度去第1梯度（最终）')
for bar, v in zip(b1, df['R²_原始度(对照)']): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=7)
for bar, v in zip(b2, df['R²_含1st梯度']):   ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=7)
for bar, v in zip(b3, df['R²_去1st梯度']):   ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(CITIES, rotation=22, ha='right')
ax.set_ylim(0, 1.2); ax.set_ylabel('$R^2$', fontsize=11)
ax.set_title('拟合优度 $R^2$ 三方对比', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(axis='y', alpha=0.3)

ax2 = axes4[1]
ax2.plot(CITIES, df['γ_全体'],   'ko-',  lw=2, ms=8,  label='γ 全体')
ax2.plot(CITIES, df['γ_奇数度'], 'r^--', lw=1.8, ms=7, label='γ 奇数度（T形口）')
ax2.plot(CITIES, df['γ_偶数度'], 'bs--', lw=1.8, ms=7, label='γ 偶数度（十字口）')
ax2.axhline(1.69, color='#4CAF50', ls='--', lw=1.5, label='论文 γ_all=1.69')
ax2.axhline(1.68, color='#E91E63', ls=':',  lw=1.2, label='论文 γ_odd=1.68')
ax2.axhline(1.81, color='#2196F3', ls=':',  lw=1.2, label='论文 γ_even=1.81')
for i, v in enumerate(df['γ_全体']):
    if not np.isnan(v): ax2.text(i, v + 0.04, f'{v:.2f}', ha='center', fontsize=8)
ax2.set_ylabel('幂律指数 γ', fontsize=11)
ax2.set_title('各城市幂律指数 γ（论文参考线）', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3)
plt.setp(ax2.get_xticklabels(), rotation=22, ha='right')

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'r2_gamma_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图4 已保存: r2_gamma_summary.png')


# ════════════════════════════════════════════════════════════
# 图5：度-度相关性 K_nn(K)（论文 Section 4.2 / Fig.6）
# ════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(2, 4, figsize=(20, 10))
fig5.suptitle(r'度-度相关性  $K_{nn}$ vs 改进度 $K$'
              '\n（正相关 → 同类聚集；分段特征见论文 Fig.6）',
              fontsize=13, fontweight='bold', y=1.01)

for ax, city, color in zip(axes5.flat, CITIES, COLORS):
    d = store[city]
    K_dict = d['K_dict']
    Knn    = d['Knn']

    K_a   = np.array(list(K_dict.values()))
    Knn_a = np.array([Knn.get(u, 0) for u in K_dict])

    # 按 BIN_WIDTH 分级求 K_nn 均值
    rounded = np.round(K_a / BIN_WIDTH) * BIN_WIDTH
    bins_k  = sorted(set(rounded.tolist()))
    Knn_mean = [Knn_a[rounded == r].mean() for r in bins_k if (rounded == r).sum() > 0]
    bins2    = [r for r in bins_k if (rounded == r).sum() > 0]

    ax.scatter(bins2, Knn_mean, s=45, color=color,
               edgecolors='white', lw=0.5, zorder=3)
    coef = np.polyfit(bins2, Knn_mean, 1)
    xl   = np.linspace(min(bins2), max(bins2), 100)
    r_v, _ = pearsonr(bins2, Knn_mean)
    ax.plot(xl, np.polyval(coef, xl), 'k--', lw=1.5,
            label=f'斜率={coef[0]:.3f}  r={r_v:.3f}')
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.6)
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('改进度 $K$', fontsize=9)
    ax.set_ylabel('$K_{nn}$', fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'degree_degree_corr.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图5 已保存: degree_degree_corr.png')


# ════════════════════════════════════════════════════════════
# 图6：改进度 K 分布直方图（2×4）
# ════════════════════════════════════════════════════════════
fig6, axes6 = plt.subplots(2, 4, figsize=(20, 10))
fig6.suptitle('各城市改进度 $K$ 连续分布直方图',
              fontsize=13, fontweight='bold', y=1.01)

for ax, city, color in zip(axes6.flat, CITIES, COLORS):
    d = store[city]
    ax.hist(d['K_arr'], bins=40, color=color, alpha=0.8,
            edgecolor='white', linewidth=0.3)
    ax.axvline(d['K_arr'].mean(), color='black', linestyle='--', linewidth=1.3,
               label=f"均值 {d['K_arr'].mean():.3f}")
    ax.set_title(city, fontsize=11, fontweight='bold')
    ax.set_xlabel('改进度 $K$', fontsize=9); ax.set_ylabel('节点数', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, 'improved_degree_hist.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图6 已保存: improved_degree_hist.png')


# ════════════════════════════════════════════════════════════
# 保存 Excel（4个 Sheet）
# ════════════════════════════════════════════════════════════
excel_path = os.path.join(OUT_DIR, 'improved_degree_fit.xlsx')
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1：拟合结果汇总
    df.to_excel(writer, sheet_name='拟合结果汇总', index=False)

    # Sheet 2：节点改进度明细（每个节点的 k, K, 奇偶）
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
                '奇偶'    : '奇' if ko % 2 == 1 else '偶',
            })
    pd.DataFrame(detail_rows).to_excel(writer, sheet_name='节点改进度明细', index=False)

    # Sheet 3：梯度分布与拟合值（各城市合并）
    grad_rows = []
    for city in CITIES:
        d   = store[city]
        row = df[df['城市'] == city].iloc[0]
        for r, p, flag in zip(
            np.concatenate([d['r_all'][:1], d['r_fit']]),
            np.concatenate([d['p_all'][:1], d['p_fit']]),
            ['第1梯度(去除)'] + ['拟合'] * len(d['r_fit'])
        ):
            pred = powerlaw_func(r, d['C_all'], d['g_all']) \
                   if flag == '拟合' and not np.isnan(d['C_all']) else None
            grad_rows.append({
                '城市'      : city,
                '梯度r(K)'  : round(float(r), 2),
                'p_实测'    : round(float(p), 8),
                'p_拟合'    : round(float(pred), 8) if pred is not None else None,
                '残差'      : round(float(p) - float(pred), 8) if pred is not None else None,
                '类型'      : flag,
            })
    pd.DataFrame(grad_rows).to_excel(writer, sheet_name='梯度分布与拟合值', index=False)

    # Sheet 4：原始度分布（对照）
    orig_rows = []
    for city in CITIES:
        d = store[city]
        for k, p in zip(d['k_arr'], d['p_orig']):
            pred = double_powerlaw(k, d['A'], d['a'], d['b']) \
                   if not np.isnan(d['A']) else None
            orig_rows.append({
                '城市'    : city, '度k': int(k),
                'P(k)实测': round(float(p), 8),
                'P(k)拟合': round(float(pred), 8) if pred is not None else None,
            })
    pd.DataFrame(orig_rows).to_excel(writer, sheet_name='原始度分布(对照)', index=False)

print(f'\nExcel 已保存: improved_degree_fit.xlsx')

print('\n' + '=' * 72)
print('  拟合结果汇总（论文参考: γ_all=1.69  γ_odd=1.68  γ_even=1.81）')
print('=' * 72)
cols = ['城市', 'γ_全体', 'R²_去1st梯度', 'R²_原始度(对照)', 'R²提升',
        'γ_奇数度', 'R²_奇数度', 'γ_偶数度', 'R²_偶数度']
print(df[cols].to_string(index=False))
