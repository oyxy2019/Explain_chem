"""
Atom (node) and bond (edge) feature encoding specified for molecule data.
"""
import torch
from torch import Tensor
from GOOD.utils.data import x_map, e_map
from GMN import global_setting


class AtomEncoder(torch.nn.Module):
    r"""
    atom (node) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):

        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        feat_dims = list(map(len, x_map.values()))

        for i, dim in enumerate(feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        #oyxy加，为了应对特征已经是one-hot编码后的情况，直接通过Linear层转换维度即可
        self.atom_feature_dim = global_setting.node_feature_dim
        self.atom_linear = torch.nn.Linear(self.atom_feature_dim, emb_dim)

    def forward(self, x):
        r"""
        atom (node) feature encoding specified for molecule data.

        Args:
            x (Tensor): node features

        Returns (Tensor):
            atom (node) embeddings
        """
        #oyxy加，为了应对特征已经是one-hot编码后的情况，直接通过Linear层转换维度即可
        if x.shape[1] == self.atom_feature_dim:
            return self.atom_linear(x)

        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    r"""
    bond (edge) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        edge_feat_dims = list(map(len, e_map.values()))

        for i, dim in enumerate(edge_feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

        #oyxy加，为了应对特征已经是one-hot编码后的情况，直接通过Linear层转换维度即可
        self.bond_feature_dim = global_setting.edge_feature_dim
        self.bond_linear = torch.nn.Linear(self.bond_feature_dim, emb_dim)

    def forward(self, edge_attr):
        r"""
        bond (edge) feature encoding specified for molecule data.

        Args:
            edge_attr (Tensor): edge attributes

        Returns (Tensor):
            bond (edge) embeddings

        """
        #oyxy加，为了应对特征已经是one-hot编码后的情况，直接通过Linear层转换维度即可
        if edge_attr.shape[1] == self.bond_feature_dim:
            return self.bond_linear(edge_attr)

        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
