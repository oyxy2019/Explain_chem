import traceback
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid
import train_e2sn2
from GMN.configure import change_config


# param_grid = [
#     {'node_state_dim': [32], 'edge_state_dim': [32], 'graph_rep_dim': [128]},
#     {'node_state_dim': [32], 'edge_state_dim': [64], 'graph_rep_dim': [256]},
#     {'node_state_dim': [64], 'edge_state_dim': [128], 'graph_rep_dim': [512]},
#     {'node_state_dim': [256], 'edge_state_dim': [256], 'graph_rep_dim': [512]},
#     {'node_state_dim': [512], 'edge_state_dim': [512], 'graph_rep_dim': [768]},
# ]

# param_grid = {
#     'share_prop_params': [True, False],
#     'use_reverse_direction': [True, False],
#     'reverse_dir_param_different': [True, False],
#     'layer_norm': [True, False],
#     'edge_update_type': ['gru', 'mlp', 'residual']
# }

# param_grid = {
#     'n_prop_layers': [1,2,4,5,6,8,10],
#     'share_prop_params': [True, False],
# }

edge_state_dim = 32
node_state_dim = 64
graph_rep_dim = 256
param_grid = [
    # {
    #     'edge_hidden_sizes': [[edge_state_dim * 2]],
    #     'node_hidden_sizes': [[node_state_dim * 2]],
    #     'encoder_node_hidden_sizes': [[node_state_dim]],
    #     'encoder_edge_hidden_sizes': [[edge_state_dim]],
    # },
    {
        'node_state_dim': [node_state_dim],
        'edge_state_dim': [edge_state_dim],
        'graph_rep_dim': [graph_rep_dim],
        'edge_hidden_sizes': [[edge_state_dim * 2, edge_state_dim * 2]],
        'node_hidden_sizes': [[node_state_dim * 2, node_state_dim * 2]],
        'encoder_node_hidden_sizes': [[node_state_dim * 2, node_state_dim]],
        'encoder_edge_hidden_sizes': [[edge_state_dim * 2, edge_state_dim]],
        'learning_rate':[1e-2, 1e-4]
    },
    {
        'node_state_dim': [node_state_dim],
        'edge_state_dim': [edge_state_dim],
        'graph_rep_dim': [graph_rep_dim],
        'edge_hidden_sizes': [[edge_state_dim * 2, edge_state_dim * 2, edge_state_dim * 2, edge_state_dim * 2, edge_state_dim * 2]],
        'node_hidden_sizes': [[node_state_dim * 2, node_state_dim * 2, node_state_dim * 2, node_state_dim * 2, node_state_dim * 2]],
        'encoder_node_hidden_sizes': [[node_state_dim * 2, node_state_dim]],
        'encoder_edge_hidden_sizes': [[edge_state_dim * 2, edge_state_dim]],
        'learning_rate':[1e-2, 1e-4]
    },
]

# param_grid = {
#     'aggregator_gated': [True, False],
#     'aggregator_graph_transform_sizes': [None, [128], [128, 128, 128]]
# }

results = []
for params in ParameterGrid(param_grid):
    train_e2sn2.config = change_config(**params)
    print("### params changed:", params)

    try:
        val_metric, test_metric = train_e2sn2.main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        val_metric, test_metric = 0, 0

    result = {'params': params, 'val_score': val_metric, 'test_score': test_metric}
    results.append(result)

results.sort(key=lambda x: x['val_score'])

# 输出排序后的结果
print()
for idx, result in enumerate(results):
    print(f"Rank {idx+1}: Parameters:{result['params']} val_score:{result['val_score']:.5f} test_score:{result['test_score']:.5f}")
