import torch
import numpy as np

from torch_geometric.data import Data


dataset = torch.load(f"/home/oyxy2019/projects/Explain_ws/ReactionOOD/data/ReactionOOD/barriers_e2.pt")[0].mol_graphs
print('Load data done!')

data_list = []
for i, data in enumerate(dataset):
    print(f'{data.mol}')
    print(f'data.targets:{data.targets}')
    mol = data.mol.split(">")[0]
    edge_index = np.array(data.edge_list, dtype=np.int64).T
    data = Data(edge_index=torch.LongTensor(edge_index),
                edge_attr=torch.tensor(data.edge_features),
                x=torch.tensor(data.f_atoms),
                y=torch.FloatTensor(data.targets).unsqueeze(1),
                idx=i,
                mol=mol)
    data_list.append(data)
num_data = data_list.__len__()
print(f'num_data={num_data}')
print('Extract data done!')


# [Cl-:2].[Cl:1][C@:5]([C:4]([H:3])([H:31])[N:41]([H:42])[H:43])([C:11]#[N:12])[N:21]([H:22])[H:23]>>[C:4](=[C:5](/[C:11]#[N:12])[N:21]([H:22])[H:23])(\[H:31])[N:41]([H:42])[H:43].[Cl-:1].[Cl:2][H:3]
# data.targets:[32.31539952947584]