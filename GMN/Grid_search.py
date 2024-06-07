from sklearn.model_selection import ParameterGrid
import train_e2sn2
from GMN.configure import change_config

# param_grid = {
#     'node_state_dim': [16, 32],
#     'edge_state_dim': [32, 64],
#     'graph_rep_dim': [128, 256, 512],
# }

# param_grid = {
#     'share_prop_params': [True, False],
#     'use_reverse_direction': [True, False],
#     'reverse_dir_param_different': [True, False],
#     'layer_norm': [True, False],
#     'edge_update_type': ['gru', 'mlp', 'residual']
# }

# edge_state_dim = 32
# node_state_dim = 16
# param_grid = {
#     'edge_hidden_sizes': [[edge_state_dim * 2, edge_state_dim * 2, edge_state_dim * 2]],
#     'node_hidden_sizes': [[node_state_dim * 2, node_state_dim * 2, node_state_dim * 2]],
#     'n_prop_layers': [1,2,4,5,6,8,10],
#     'share_prop_params': [True, False],
# }

# edge_state_dim = 32
# node_state_dim = 16
# graph_rep_dim = 128
# param_grid = {
#     'edge_hidden_sizes': [
#         [edge_state_dim * 2],
#         [edge_state_dim * 2, edge_state_dim * 2],
#         [edge_state_dim * 2, edge_state_dim * 2, edge_state_dim * 2],
#     ],
#     'node_hidden_sizes': [
#         [node_state_dim * 2],
#         [node_state_dim * 2, node_state_dim * 2],
#         [node_state_dim * 2, node_state_dim * 2, node_state_dim * 2],
#     ],
#     'encoder_layers_node': [
#         [node_state_dim],
#         [node_state_dim*2, node_state_dim],
#         [node_state_dim*2, node_state_dim*2, node_state_dim],
#     ],
#     'encoder_layers_edge': [
#         [edge_state_dim],
#         [edge_state_dim * 2, edge_state_dim],
#         [edge_state_dim * 2, edge_state_dim * 2, edge_state_dim],
#     ],
#     'aggregator_layers': [
#         [graph_rep_dim],
#         [graph_rep_dim, graph_rep_dim],
#         [graph_rep_dim, graph_rep_dim, graph_rep_dim],
#     ]
# }

edge_state_dim = 32
node_state_dim = 16
graph_rep_dim = 128
param_grid = {
        'encoder_layers_edge': [
            [edge_state_dim],
            [edge_state_dim * 2, edge_state_dim],
        ],
}

results = []
for params in ParameterGrid(param_grid):
    train_e2sn2.config = change_config(**params)
    print("params changed ", params)

    try:
        val_metric, test_metric = train_e2sn2.main()
    except Exception as e:
        print(f"An error occurred: {e}")
        val_metric, test_metric = 0, 0

    result = {'params': params, 'val_score': val_metric, 'test_score': test_metric}
    results.append(result)

results.sort(key=lambda x: x['val_score'])

# 输出排序后的结果
print()
for idx, result in enumerate(results):
    print(f"Rank {idx+1}: Parameters:{result['params']} val_score:{result['val_score']:.5f} test_score:{result['test_score']:.5f}")
