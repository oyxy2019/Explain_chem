from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions
import networkx as nx
from chemprop.rdkit import make_mol

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, sort_edge_index


def plot_reaction_with_smiles(reaction_smiles):
    # 从SMILES解析化学反应
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles)

    # 生成反应的图像
    img = Draw.ReactionToImage(rxn, subImgSize=(500, 500))

    # 设置图形显示
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.imshow(img)
    ax.axis('off')  # 关闭坐标轴

    # 显示SMILES字符串
    # reaction_smiles = reaction_smiles.replace(">>", "\n>>")
    # plt.text(0.5, -0.1, f"SMILES: {reaction_smiles}", horizontalalignment='center',
    #          verticalalignment='center', transform=ax.transAxes, fontsize=12, color='blue')

    plt.show()


def get_id_map_to_label_4_reaction(mol_graph):
    '''
    根据反应式SMILE字符串，得到可视化图节点中的标签
    '''
    mol = mol_graph.mol
    mol_reac = make_mol(mol.split(">")[0], mol_graph.is_explicit_h, mol_graph.is_adding_hs,
                        mol_graph.is_keeping_atom_map)
    mol_prod = make_mol(mol.split(">")[-1], mol_graph.is_explicit_h, mol_graph.is_adding_hs,
                        mol_graph.is_keeping_atom_map)
    only_prod_ids = []
    prod_map_to_id = {}
    mapnos_reac = set([atom.GetAtomMapNum() for atom in mol_reac.GetAtoms()])
    for atom in mol_prod.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            prod_map_to_id[mapno] = atom.GetIdx()
            if mapno not in mapnos_reac:
                only_prod_ids.append(atom.GetIdx())
        else:
            only_prod_ids.append(atom.GetIdx())
    only_reac_ids = []
    reac_id_to_prod_id = {}
    for atom in mol_reac.GetAtoms():
        mapno = atom.GetAtomMapNum()
        if mapno > 0:
            try:
                reac_id_to_prod_id[atom.GetIdx()] = prod_map_to_id[mapno]
            except KeyError:
                only_reac_ids.append(atom.GetIdx())
        else:
            only_reac_ids.append(atom.GetIdx())

    id_map_to_symbol = {atom.GetIdx(): atom.GetSymbol() for atom in mol_prod.GetAtoms()}
    id_map_to_label = {v: str(id_map_to_symbol[v]) + ':' + str(k) for k, v in prod_map_to_id.items()}
    return id_map_to_label


def get_id_map_to_label_4_mol(mol_graph):
    '''
    根据反应物或者生成物的SMILE字符串，得到可视化图节点中的标签
    '''
    mol = make_mol(mol_graph.mol, mol_graph.is_explicit_h, mol_graph.is_adding_hs,
                        mol_graph.is_keeping_atom_map)

    # id_map_to_label = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
    id_map_to_label = {atom.GetIdx(): f'{atom.GetSymbol()} idx:{atom.GetIdx()}' for atom in mol.GetAtoms()}  # show idx
    return id_map_to_label


def visualize_chemprop_molgraph(mol_graph, is_reaction=True):
    # Create an empty NetworkX graph
    G = nx.Graph()

    # Add nodes (atoms)
    for atom_idx in range(mol_graph.n_atoms):
        G.add_node(atom_idx)

    # Add edges (bonds)
    for bond_idx in range(mol_graph.n_bonds):
        atom1_idx = mol_graph.b2a[bond_idx]
        atom2_idx = mol_graph.b2a[mol_graph.b2revb[bond_idx]]
        G.add_edge(atom1_idx, atom2_idx)

    # Draw the graph
    pos = nx.kamada_kawai_layout(G)
    if is_reaction:
        # 获取节点可视化标签
        id_map_to_label = get_id_map_to_label_4_reaction(mol_graph)
        nx.draw(G, pos, with_labels=True, labels={idx: id_map_to_label[idx] for idx in G.nodes()}, node_size=500, node_color='skyblue', font_size=10, font_color='black')
    else:
        id_map_to_label = get_id_map_to_label_4_mol(mol_graph)
        nx.draw(G, pos, with_labels=True, labels={idx: id_map_to_label[idx] for idx in G.nodes()}, node_size=500, node_color='skyblue', font_size=10, font_color='black')

    # Show the graph
    plt.title('Molecular Graph Visualization')
    plt.axis('off')
    plt.show()


def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, coor=None, norm=False, mol_type=None,
                      nodesize=300):
    plt.clf()
    if norm:
        edge_att = edge_att ** 10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00', 'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v.split(":")[0]) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax,
                               connectionstyle='arc3,rad=0.4')

    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    return fig, image


def get_pygdata_from_Molgraph(data):
    mol = data.mol.split(">")[0]
    edge_index = np.array(data.edge_list, dtype=np.int64).T
    pyg_data = Data(edge_index=torch.LongTensor(edge_index),
                    edge_attr=torch.tensor(data.edge_features),
                    x=torch.tensor(data.f_atoms),
                    y=torch.FloatTensor(data.targets).unsqueeze(1),
                    mol=mol)
    return pyg_data


def pygdata2graph(data):
    if not isinstance(data, Data):
        raise ValueError("Not PyG Data object")

    # 将PyG图数据转换为networkx图
    G = to_networkx(data, to_undirected=True)

    # 可视化
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.show()


if __name__ == '__main__':
    # test plot_reaction_with_smiles
    reaction_smiles = "[Cl:1][C@:5]([C@:4]([H:3])([C:31]#[N:32])[N:41]([H:42])[H:43])([C:11]#[N:12])[N+:21]([O-:22])[O:23].[H-:2]>>[C:4](=[C:5](/[C:11]#[N:12])[N+:21]([O-:22])[O:23])(\[C:31]#[N:32])[N:41]([H:42])[H:43].[Cl-:1].[H:2][H:3]"
    plot_reaction_with_smiles(reaction_smiles)

    # test gsat_visualize_a_graph
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 1, 4, 1, 5, 2, 6, 2, 7, 3, 8, 4, 9, 4, 10, 5, 11, 6, 12, 7, 13, 9, 14, 9, 15, 12, 16, 12, 17,
         13, 18, 15, 19, 15, 20, 18, 21, 18, 22, 6, 10, 8, 13, 11, 14, 3, 23, 5, 24, 7, 25, 8, 26, 10, 27, 11, 28, 14,
         29],
        [1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 2, 7, 2, 8, 3, 9, 4, 10, 4, 11, 5, 12, 6, 13, 7, 14, 9, 15, 9, 16, 12, 17, 12,
         18, 13, 19, 15, 20, 15, 21, 18, 22, 18, 10, 6, 13, 8, 14, 11, 23, 3, 24, 5, 25, 7, 26, 8, 27, 10, 28, 11, 29,
         14]
    ])
    edge_mask = torch.tensor([
        0.2114, 0.2114, 0.2061, 0.2061, 0.2019, 0.2019, 0.2061, 0.2061, 0.2019, 0.2019, 0.2169, 0.2169, 0.2051, 0.2051,
        0.1991, 0.1991, 0.2169, 0.2169, 0.2051, 0.2051, 0.1807, 0.1807, 0.1845, 0.1845, 0.2025, 0.2025, 0.2193, 0.2193,
        0.1845, 0.1845, 0.2063, 0.2063, 0.2063, 0.2063, 0.1688, 0.1688, 0.2063, 0.2063, 0.2063, 0.2063, 0.2063, 0.2063,
        0.2063, 0.2063, 0.2213, 0.2213, 0.2007, 0.2007, 0.1865, 0.1865, 0.2451, 0.2451, 0.2451, 0.2451, 0.2553, 0.2553,
        0.2530, 0.2530, 0.2553, 0.2553, 0.2296, 0.2296, 0.2530, 0.2530
    ])
    node_label = torch.zeros(30)
    visualize_a_graph(edge_index, edge_mask, node_label, dataset_name='mutag', coor=None, norm=True, mol_type=None,
                      nodesize=300)
