import pandas as pd
import json
import os


class Node:
    """
    城市交通网络中的节点类
    """

    def __init__(self, node_id, x, y):
        self.id = node_id
        # position 为二维坐标 [XCoord, YCoord]
        self.position = [float(x), float(y)]
        self.degree = 0
        # 邻居列表，存储形式为字典: [{"id": 邻居编号, "distance": 两点间真实路网距离}, ...]
        self.neighbors = []

    def add_neighbor(self, neighbor_id, distance):
        """
        添加邻居节点，并自动更新度数 (Degree)
        """
        # 检查是否重复添加同一邻居（应对原始数据中可能存在的多重边情况）
        if not any(n["id"] == neighbor_id for n in self.neighbors):
            self.neighbors.append({"id": int(neighbor_id), "distance": float(distance)})
            # 每次成功添加一个新的相连节点，度数 +1
            self.degree = len(self.neighbors)

    def to_dict(self):
        """
        序列化方法，方便转换为 JSON
        """
        return {"id": self.id, "position": self.position, "degree": self.degree, "neighbors": self.neighbors}


def convert_edgelist_to_json(csv_path, output_json_path):
    """
    读取官方 CSV 数据，构建全量 Node 对象，并输出为 JSON 格式
    """
    print(f"正在读取数据: {csv_path}")
    df = pd.read_csv(csv_path)

    nodes_dict = {}

    # 第一遍遍历：收集并实例化所有的节点坐标信息
    # (因为Edgelist里每行的坐标实际上是 START_NODE 的坐标)
    for _, row in df.iterrows():
        start_id = int(row["START_NODE"])
        if start_id not in nodes_dict:
            nodes_dict[start_id] = Node(start_id, row["XCoord"], row["YCoord"])

    # 第二遍遍历：构建邻居关系与边长信息
    for _, row in df.iterrows():
        start_id = int(row["START_NODE"])
        end_id = int(row["END_NODE"])
        length = row["LENGTH"]

        # 容错处理：如果某个 END_NODE 在整个数据集中从未作为 START_NODE 出现过（比如道路死胡同的终点）
        # 我们就先为它临时创建一个缺失坐标的节点，以免报错
        if end_id not in nodes_dict:
            nodes_dict[end_id] = Node(end_id, 0.0, 0.0)

        # 把 END_NODE 添加到 START_NODE 的邻居列表中
        nodes_dict[start_id].add_neighbor(end_id, length)

    # 将所有的 Node 对象转为字典格式，准备写入 JSON
    # 结构: {"1": {id: 1, position: [...], degree: ..., neighbors: [...]}, "2": ...}
    result_data = {str(node_id): node.to_dict() for node_id, node in nodes_dict.items()}

    # 导出为 JSON 文件
    print(f"正在保存为 JSON 文件: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！共解析了 {len(nodes_dict)} 个节点，已输出到 {output_json_path}\n")


if __name__ == "__main__":
    # 以成都的数据作为示例转换
    base_dir = r"d:\Project\Model"
    data_dir = os.path.join(base_dir, "data", "B题数据")
    output_dir = os.path.join(data_dir, "json_networks")

    # 创建存放 JSON 文件的输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 B题数据 文件夹下所有的 csv 文件
    for filename in os.listdir(data_dir):
        if filename.endswith("_Edgelist.csv"):
            city_name = filename.split("_")[0]
            csv_path = os.path.join(data_dir, filename)
            json_filename = f"{city_name}_Network.json"
            output_json_path = os.path.join(output_dir, json_filename)

            convert_edgelist_to_json(csv_path, output_json_path)

    print("所有城市的数据转换已完成！")
