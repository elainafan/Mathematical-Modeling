"""
Q5/select_candidates.py
从 Q5/data/{City}_candidates.csv 中筛选候选边，输出精选版本。

过滤规则：
  1. 欧氏距离 ≥ MIN_DIST（默认 300m），排除局部微环路
  2. 欧氏距离 ≤ MAX_DIST（默认 2000m），排除不现实的长边
  3. 可选：只保留特定来源

用法:
    python -m Q5.select_candidates
"""

import os
import csv

CITIES = [
    "Chengdu", "Dalian", "Dongguan", "Harbin",
    "Qingdao", "Quanzhou", "Shenyang", "Zhengzhou",
]

# ── 筛选参数 ──
MIN_DIST = 300.0    # 最小欧氏距离（米）
MAX_DIST = 5000.0   # 最大欧氏距离（米）


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "Q5", "data")

    print(f"  筛选参数: {MIN_DIST}m ≤ dist ≤ {MAX_DIST}m")
    print()

    for city in CITIES:
        in_path = os.path.join(data_dir, f"{city}_candidates.csv")
        out_path = os.path.join(data_dir, f"{city}_candidates_selected.csv")

        if not os.path.exists(in_path):
            print(f"  [跳过] {city}")
            continue

        total = 0
        kept = 0
        source_count = {}

        with open(in_path, "r", encoding="utf-8-sig") as fin, \
             open(out_path, "w", newline="", encoding="utf-8-sig") as fout:

            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                total += 1
                dist = float(row["euclid_dist"])

                if dist < MIN_DIST or dist > MAX_DIST:
                    continue

                writer.writerow(row)
                kept += 1
                src = row["source"]
                source_count[src] = source_count.get(src, 0) + 1

        print(f"  {city}: {total} → {kept} 条 ({kept/total*100:.1f}%)")
        for src, cnt in sorted(source_count.items(), key=lambda x: -x[1]):
            print(f"    {src:>15s}: {cnt}")

    print(f"\n  完成，输出文件后缀 _selected.csv")


if __name__ == "__main__":
    main()
