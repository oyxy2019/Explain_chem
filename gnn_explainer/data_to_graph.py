import torch

import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from load_model import load_good_model_dataset



def data2graph(data):
    if not isinstance(data, Data):
        raise ValueError("Not PyG Data object")

    # 将PyG图数据转换为networkx图
    G = to_networkx(data, to_undirected=True)

    # 可视化
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.show()


if __name__ == '__main__':
    # 加载模型和数据
    config_path = 'final_configs/GOODE2SN2/size/concept/GSAT.yaml'
    model, dataset = load_good_model_dataset(config_path)
    train_dataset = dataset['train']
    data = train_dataset.data
    # Data(x=[7691, 165], edge_index=[2, 14374], edge_attr=[14374, 193], y=[504, 1], idx=[504], mol=[504], nodesize=[504], domain_id=[504], pyx=[504], env_id=[504])

    # 将模型和数据放到同一gpu上
    print(data.to(torch.device("cuda:1")))
    print(model.to(torch.device("cuda:1")))

    # 取出第一条数据data0
    data0 = train_dataset[0]
    print(data0)

    data2graph(data0)
