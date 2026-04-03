import pandas as pd
import networkx as nx
import numpy as np
import copy


class CityNetwork:
    """
    城市交通网络基础数据结构类
    用于为 B题 1-5 问提供统一的数据加载、图构建、网络重置和基础指标计算接口。
    """

    def __init__(self, city_name, csv_path):
        self.city_name = city_name
        self.csv_path = csv_path

        # initial_graph 保存最原始、完整的网络状态，绝对不会被修改
        self.initial_graph = nx.Graph()

        # current_graph 用于问题2、3、4的模拟。在模拟过程中，节点会被不断删除
        self.current_graph = nx.Graph()

        # 独立存储节点坐标，方便在问题4(空间波及)和问题5(计算成本)中进行快速空间距离计算
        # 格式: {node_id: (XCoord, YCoord)}
        self.node_coords = {}

        # 记录初始网络规模（用于计算最大连通分量占比，必须除以最初的总节点数）
        self.initial_node_count = 0

    def build_network(self):
        """
        读取 CSV 数据并构建 NetworkX 图对象
        """
        print(f"正在加载 {self.city_name} 的路网数据...")
        df = pd.read_csv(self.csv_path)

        # 构建图
        for _, row in df.iterrows():
            u = int(row["START_NODE"])
            v = int(row["END_NODE"])
            length = row["LENGTH"]

            # 添加边，并将长度作为权重（问题5可能需要）
            self.initial_graph.add_edge(u, v, weight=length)

            # 存入坐标信息
            if u not in self.node_coords:
                self.node_coords[u] = (row["XCoord"], row["YCoord"])
            # 注意: End_Node的坐标在数据表里一般不直接在同一行体现为终点坐标，
            # 若源数据每行对应一条边且有起点XY，终点由于也会作为其他边的起点而录入。
            # 安全起见，如果在别的行作为START_NODE出现过，就会被收录。

        # 移除自环（自己连自己的边，交通网中通常无意义）
        self.initial_graph.remove_edges_from(nx.selfloop_edges(self.initial_graph))

        self.initial_node_count = self.initial_graph.number_of_nodes()
        self.reset_network()  # 初始化 current_graph
        print(
            f"{self.city_name} 网络构建完成! 节点数: {self.initial_node_count}, 边数: {self.initial_graph.number_of_edges()}"
        )

    def reset_network(self):
        """
        将工作网络重置为初始完整状态。
        在问题2的蒙特卡洛多次模拟，或者问题3重新评估策略时，必须调用此方法。
        """
        # 使用 copy.deepcopy 或 nx.Graph.copy()。后者对 nx 对象更高效
        self.current_graph = self.initial_graph.copy()

    # ==========================
    # 以下为第一问及基础评估通用接口
    # ==========================

    def get_max_connected_component_ratio(self):
        """
        计算当前网络的“最大连通分量节点数”占“原始网络总节点数”的比例
        这是贯穿问题 2、3、4 的核心性能评价指标 P(q)
        """
        if self.current_graph.number_of_nodes() == 0:
            return 0.0

        # 获取所有连通分量（按大小降序排列）
        connected_components = sorted(nx.connected_components(self.current_graph), key=len, reverse=True)
        # 获取最大的连通分量大小
        max_cc_size = len(connected_components[0])

        # 注意：分母永远是初始节点数 initial_node_count，不是当前图的节点数！
        return max_cc_size / self.initial_node_count

    def remove_nodes(self, nodes_to_remove):
        """
        从当前网络中瘫痪(删除)一批节点
        供问题2(随机删除)、问题3(按度删除)、问题4(空间范围删除)调用
        """
        # filter 掉本身就不在图里的节点，避免报错
        nodes_in_graph = [n for n in nodes_to_remove if n in self.current_graph]
        self.current_graph.remove_nodes_from(nodes_in_graph)

    # 第一问辅助分析接口
    def get_degrees(self):
        """返回当前所有节点的度数列表 (用于画度分布图)"""
        return [deg for _, deg in self.initial_graph.degree()]


if __name__ == "__main__":
    # 使用样例 (供你的队友参考跑通测试)
    import os

    # 构建成都网络测试
    chengdu_csv = os.path.join("data", "B题数据", "Chengdu_Edgelist.csv")
    if os.path.exists(chengdu_csv):
        cd_net = CityNetwork("Chengdu", chengdu_csv)
        cd_net.build_network()

        # 打印初始状态连通比例 (应该是接近 1.0)
        print("初始最大连通分量占比:", cd_net.get_max_connected_component_ratio())
