from sklearn.model_selection import ParameterGrid
import train_e2sn2
from GMN.configure import change_config

param_grid = {
    'node_state_dim': [8, 16, 32, 64, 128],
    'edge_state_dim': [8, 16, 32, 64, 128],
    'graph_rep_dim': [32, 64, 128, 256],
}

results = []
for params in ParameterGrid(param_grid):
    train_e2sn2.config = change_config(**params)
    print("params changed ", params)

    val_metric, test_metric = train_e2sn2.main()

    result = {'params': params, 'val_score': val_metric, 'test_score': test_metric}
    results.append(result)

results.sort(key=lambda x: x['val_score'])

# 输出排序后的结果
print()
for idx, result in enumerate(results):
    print(f"Rank {idx+1}: Parameters:{result['params']} val_score:{result['val_score']:.5f} test_score:{result['test_score']:.5f}")
