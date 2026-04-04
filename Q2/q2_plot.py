import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体，防止图表中文字符显示为方块
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def plot_and_calculate():
    base_dir = r"d:\Project\Model"
    output_dir = os.path.join(base_dir, "data", "B题数据", "Q2_Results")
    csv_raw_in = os.path.join(output_dir, "Q2_Raw_Simulation_Data.csv")

    if not os.path.exists(csv_raw_in):
        print(f"错误: 找不到计算数据文件 {csv_raw_in}")
        print("请确保先运行 q2_compute.py 完成图论网络仿真计算。")
        return

    print("已读取前期计算得出的城市 P(q) 基础数据表，准备独立渲染绘图和积分结算...")
    df_all_data = pd.read_csv(csv_raw_in)

    results = []

    # 按照城市分组进行处理
    for city_name, group in df_all_data.groupby("City"):
        print(f"正在处理城市: {city_name} 的绘图与评估结果...")

        # 按照 q 值排序以防乱序
        group = group.sort_values(by="q")
        q_points = group["q"].values
        P_q_avg = group["P_q_avg"].values

        step_size = q_points[1] - q_points[0] if len(q_points) > 1 else 0.01

        # 计算健壮性 R (梯形近似积分)
        R = np.sum((P_q_avg[:-1] + P_q_avg[1:]) / 2.0) * step_size

        # 计算显著崩溃点
        diffs = np.diff(P_q_avg)
        steepest_drop_idx = np.argmin(diffs)
        q_critical = q_points[steepest_drop_idx]

        results.append({"城市": city_name, "健壮性积分(R)": R, "网络显著崩溃比例(q)": q_critical})

        # -----------------------------
        # 独立的静态绘图逻辑
        # -----------------------------
        plt.figure(figsize=(8, 6))
        plt.plot(q_points, P_q_avg, marker=".", linestyle="-", color="b", linewidth=2)
        plt.axvline(x=q_critical, color="r", linestyle="--", label=f"显著变化点 (q={q_critical:.2f})")
        plt.xlabel("破坏节点比例 (q)")
        plt.ylabel("最大连通分量占比 P(q)")
        plt.title(f"{city_name} 城市交通网络随机故障下的性能变化曲线")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        plot_path = os.path.join(output_dir, f"{city_name}_Robustness_Random.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # 输出摘要表格
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="健壮性积分(R)", ascending=False)
    csv_out = os.path.join(output_dir, "Q2_Robustness_Summary.csv")
    df_results.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print(f"\n全部图表渲制完成！排行榜汇总数据已更新至: {csv_out}")
    print("各城市的单独健壮性曲线已生成在相同的对应目录下。")


if __name__ == "__main__":
    plot_and_calculate()
