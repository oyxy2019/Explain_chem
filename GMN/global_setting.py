from ReactionOOD.chemprop.features import get_atom_fdim, get_bond_fdim


node_feature_dim = get_atom_fdim()
edge_feature_dim = node_feature_dim * 2 + 14

# print(node_feature_dim)
# print(edge_feature_dim)