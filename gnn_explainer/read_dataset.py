import torch
import numpy as np

from torch_geometric.data import Data

from ReactionOOD.chemprop.features import MolGraph, set_reaction
from gnn_explainer.myUtils import plot_reaction_with_smiles, visualize_chemprop_molgraph, pygdata2graph

dataset = torch.load(f"/home/oyxy2019/projects/Explain_ws/ReactionOOD/data/ReactionOOD/barriers_e2.pt")[0].mol_graphs
print('Load data done!')

data_list = []
for i, data in enumerate(dataset):
    if data.mol.split(">")[0] == "[Cl-:2].[Cl:1][C@:5]([C:4]([H:3])([H:31])[N:41]([H:42])[H:43])([C:11]#[N:12])[N:21]([H:22])[H:23]":
        print(f'{data.mol}')
        print(f'data.targets:{data.targets}')

        # data可视化
        plot_reaction_with_smiles(data.mol)

        visualize_chemprop_molgraph(data)

        print("--- after process ---")
        print(f'data.edge_list:{data.edge_list}')
        print(f'data.f_atoms:{data.f_atoms}')
        print(f'data.edge_features:{data.edge_features}')

        # 运行chemprop.MolGraph
        print("--- running_MolGraph ---")
        set_reaction(True, data.reaction_mode)
        molgraph = MolGraph(data.mol)
        print('------------------------')
        break

print('Extract data done!')
