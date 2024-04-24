import os
from datetime import datetime

import torch

# from torch_geometric.explain import Explainer, GNNExplainer
from pyg252_explain import Explainer, GNNExplainer

from load_model import load_good_model_dataset


# 加载模型和数据
config_path = 'final_configs/GOODE2SN2/size/covariate/GSAT.yaml'
model, dataset = load_good_model_dataset(config_path)
train_dataset = dataset['train']
# data = train_dataset.data
# Data(x=[7691, 165], edge_index=[2, 14374], edge_attr=[14374, 193], y=[504, 1], idx=[504], mol=[504], nodesize=[504], domain_id=[504], pyx=[504], env_id=[504])

# 取出第一条数据data0
data0 = train_dataset[0]
# print(data0)
# Data(x=[17, 165], edge_index=[2, 32], edge_attr=[32, 193], y=[1, 1], idx=[1], mol='[Cl:1][C@:5]([C:4]([H:3])([C:31]([H:32])([H:33])[H:34])[C:41]([H:42])([H:43])[H:44])([N:11]([H:12])[H:13])[H:21].[F-:2]', nodesize=[1], domain_id=[1], pyx=[1], env_id=[1])

# 将模型和数据放到同一gpu上
print("data0 has been put on device: \n", data0.to(torch.device("cuda:1")))
print("model has been put on device: \n", model.to(torch.device("cuda:1")))

# 添加data0.batch属性，并打印真值data0.y
data0.batch = torch.zeros(data0.x.shape[0], dtype=torch.int64, device=torch.device('cuda:1'))
print(f'data0.mol={data0.mol}')
print(f'data0.y={data0.y}')

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',     # 可以取值None、'object'、'common_attributes'、'attributes'
    edge_mask_type='object',         # 可以取值None、'object'
    model_config=dict(
        mode='regression',
        task_level='graph',
    ),
)

explanation = explainer(data0.x, data0.edge_index, data=data0, batch_size=1)  # 模型需要{data, batch_size}
print(f'Generated explanations in {explanation.available_explanations}')

# 结果保存
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
os.makedirs('output', exist_ok=True)

# 特征图
# path = f'output/feature_importance_{current_time}.png'
# explanation.visualize_feature_importance(path, top_k=10)
# print(f"Feature importance plot has been saved to '{path}'")

# 子图
# path = f'output/subgraph_{current_time}.png'
# explanation.visualize_graph(path)
# print(f"Subgraph visualization plot has been saved to '{path}'")

# 其他函数
# path = f'output/explanation_subgraph_{current_time}.png'
# explanation.get_explanation_subgraph().visualize_graph(path)
# path = f'output/complement_subgraph_{current_time}.png'
# explanation.get_complement_subgraph().visualize_graph(path)


# gsat可视化
print()
print(f"edge_index: {explanation.edge_index.shape} \n{explanation.edge_index}")
print(f"edge_mask: {explanation.get('edge_mask').shape} \n{explanation.get('edge_mask')}")
from myUtils import visualize_a_graph
edge_index = explanation.edge_index
edge_mask = explanation.get('edge_mask')
node_label = torch.zeros(data0.x.shape[0])
visualize_a_graph(edge_index, edge_mask, node_label, dataset_name='mutag', coor=None, norm=True, mol_type=None, nodesize=300)