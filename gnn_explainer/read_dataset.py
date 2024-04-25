import torch
import numpy as np

from torch_geometric.data import Data

from ReactionOOD.chemprop.features import MolGraph, set_reaction, set_explicit_h
from gnn_explainer.load_model import load_good_model_dataset
from gnn_explainer.myUtils import *
from pyg252_explain import Explainer, GNNExplainer
from myUtils import visualize_a_graph

dataset = torch.load(f"/home/oyxy2019/projects/Explain_ws/ReactionOOD/data/ReactionOOD/barriers_e2.pt")[0].mol_graphs
print('Load data done!')

data_list = []
for i, data in enumerate(dataset):
    if data.mol.split(">")[0] == "[Cl-:2].[Cl:1][C@:5]([C:4]([H:3])([H:31])[N:41]([H:42])[H:43])([C:11]#[N:12])[N:21]([H:22])[H:23]":
        print(f'{data.mol}')
        print(f'data.targets:{data.targets}')

        # data可视化
        plot_reaction_with_smiles(data.mol)
        # visualize_chemprop_molgraph(data)

        # 运行chemprop.MolGraph
        # print("--- running_MolGraph ---")
        # set_reaction(True, 'reac_diff')
        # set_explicit_h(True)
        # molgraph = MolGraph(data.mol)
        # print('------------------------')

        ### gnnexplainer ###
        config_path = 'final_configs/GOODE2SN2/size/covariate/GSAT.yaml'
        model, _ = load_good_model_dataset(config_path)
        data0 = get_pygdata_from_Molgraph(data)
        print("data0 has been put on device: \n", data0.to(torch.device("cuda:1")))
        print("model has been put on device: \n", model.to(torch.device("cuda:1")))
        data0.batch = torch.zeros(data0.x.shape[0], dtype=torch.int64, device=torch.device('cuda:1'))
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',  # 可以取值None、'object'、'common_attributes'、'attributes'
            edge_mask_type='object',  # 可以取值None、'object'
            model_config=dict(
                mode='regression',
                task_level='graph',
            ),
        )
        explanation = explainer(data0.x, data0.edge_index, data=data0, batch_size=1)  # 模型需要{data, batch_size}
        print(f'Generated explanations in {explanation.available_explanations}')

        # gsat可视化
        edge_index = explanation.edge_index
        edge_mask = explanation.get('edge_mask')
        node_label = torch.zeros(data0.x.shape[0])
        mol_type = get_id_map_to_label(data)
        visualize_a_graph(edge_index, edge_mask, node_label, dataset_name='mutag', coor=None, norm=True, mol_type=mol_type, nodesize=600)
        break

print('Extract data done!')
