"""
flow/get_weight.py
计算节点密度权重（m_i）和边流量（x_e），写回 JSON 文件。

新增字段
--------
节点层：  "node_weight": float          ← m_i
邻居层：  "flow":        float          ← x_e（两端共享同一值）

用法
----
python get_weight.py Chengdu_Network.json
python get_weight.py Chengdu_Network.json --sigma 1000 --beta 0.001
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── 引入同级 flow_simulation 和上级 utils_ig ──────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_simulation import (
    compute_node_density,
    compute_od_matrix,
    compute_edge_loads,
)
from utils_ig import load_graph


# =============================================================================
# 主流程
# =============================================================================

def process(
    json_path: str,
    sigma: float        = 1000.0,
    beta: float         = 0.001,
    cutoff_factor: float = 3.0,
    batch_size: int     = 256,
    verbose: bool       = True,
) -> None:
    """
    读取 JSON 路网，计算节点权重和边流量，原地写回同一文件。

    参数
    ----
    json_path     : *_Network.json 路径
    sigma         : 节点密度空间尺度，单位米
    beta          : OD 距离衰减系数
    cutoff_factor : OD 欧氏截断倍数（trunc = cutoff_factor × sigma）
    batch_size    : OD 矩阵批量 Dijkstra 大小
    """
    city = os.path.basename(json_path).split("_")[0]
    print(f"{'='*60}")
    print(f"城市: {city}   文件: {json_path}")
    print(f"参数: σ={sigma}m  β={beta}  截断={cutoff_factor}σ")
    print(f"{'='*60}")

    # ── 1. 加载图 ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    G, id2idx, _ = load_graph(json_path)
    N = G.vcount()
    M = G.ecount()
    print(f"[1] 加载完成  节点={N}  边={M}  ({time.perf_counter()-t0:.2f}s)")

    positions = np.array([v['pos'] for v in G.vs], dtype=np.float64)

    # ── 2. 节点密度 ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    m = compute_node_density(positions, sigma=sigma, trunc_factor=cutoff_factor)
    print(f"[2] 节点密度  范围=[{m.min():.2f}, {m.max():.2f}]"
          f"  ({time.perf_counter()-t0:.2f}s)")

    # ── 3. OD 矩阵 ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    D_keys, D_vals, D_total = compute_od_matrix(
        G, m,
        beta=beta,
        sigma=sigma,
        cutoff_factor=cutoff_factor,
        batch_size=batch_size,
        verbose=verbose,
    )
    print(f"[3] OD 矩阵  有效对={len(D_keys)}  D_total={D_total:.4e}"
          f"  ({time.perf_counter()-t0:.2f}s)")

    # ── 4. 边流量 ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    x0 = compute_edge_loads(G, id2idx, D_keys, D_vals, k=1, verbose=verbose)
    print(f"[4] 边流量  总={x0.sum():.4e}  非零={np.count_nonzero(x0)}/{M}"
          f"  ({time.perf_counter()-t0:.2f}s)")

    # ── 5. 构建查找表 ─────────────────────────────────────────────────────────
    # node_id → m_i
    node_weight: dict = {}
    for i in range(N):
        node_weight[G.vs[i]['node_id']] = float(m[i])

    # (min_json_id, max_json_id) → x_e
    flow_lookup: dict = {}
    node_ids = G.vs['node_id']
    for eid in range(M):
        e   = G.es[eid]
        u   = node_ids[e.source]
        v   = node_ids[e.target]
        flow_lookup[(min(u, v), max(u, v))] = float(x0[eid])

    # ── 6. 读取原始 JSON，注入新字段，保存到 flow/data/ ──────────────────────
    t0 = time.perf_counter()
    with open(json_path, 'r', encoding='utf-8') as f:
        data: dict = json.load(f)

    for key_str, node_data in data.items():
        nid = int(node_data['id'])
        node_data['node_weight'] = node_weight.get(nid, 0.0)
        for nb in node_data.get('neighbors', []):
            nb_id = int(nb['id'])
            edge_key = (min(nid, nb_id), max(nid, nb_id))
            nb['flow'] = flow_lookup.get(edge_key, 0.0)

    # 输出目录：<本脚本所在目录>/data/
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{city}_Network_weighted.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

    print(f"[5] 写出完成  ({time.perf_counter()-t0:.2f}s)")
    print(f"    → {out_path}")
    return out_path


# =============================================================================
# CLI 入口
# =============================================================================

def main() -> None:
    # 默认 JSON 目录：脚本上两级的 data/json_networks/
    default_json_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'json_networks',
    )

    parser = argparse.ArgumentParser(
        description=(
            '计算节点密度权重和边流量，保存到 flow/data/。'
            '不传 json_path 时自动处理默认目录下所有城市。'
        )
    )
    parser.add_argument(
        'json_path', nargs='?', default=None,
        help=f'*_Network.json 路径；省略则批量处理 data/json_networks/ 下全部城市',
    )
    parser.add_argument('--sigma',         type=float, default=1000.0)
    parser.add_argument('--beta',          type=float, default=0.001)
    parser.add_argument('--cutoff_factor', type=float, default=3.0)
    parser.add_argument('--batch_size',    type=int,   default=256)
    parser.add_argument('--quiet',         action='store_true',
                        help='关闭批次级进度输出')
    args = parser.parse_args()

    kwargs = dict(
        sigma         = args.sigma,
        beta          = args.beta,
        cutoff_factor = args.cutoff_factor,
        batch_size    = args.batch_size,
        verbose       = not args.quiet,
    )

    if args.json_path is not None:
        # 单文件模式
        process(args.json_path, **kwargs)
    else:
        # 批量模式：处理默认目录下所有 *_Network.json
        if not os.path.isdir(default_json_dir):
            print(f"找不到默认目录：{default_json_dir}")
            print("请手动指定 json_path，例如：")
            print("  python get_weight.py ../data/json_networks/Chengdu_Network.json")
            sys.exit(1)

        files = sorted(
            f for f in os.listdir(default_json_dir)
            if f.endswith('_Network.json')
        )
        if not files:
            print(f"目录下没有 *_Network.json：{default_json_dir}")
            sys.exit(1)

        print(f"批量模式：共 {len(files)} 个城市\n")
        for fname in files:
            process(os.path.join(default_json_dir, fname), **kwargs)
            print()


if __name__ == '__main__':
    main()
