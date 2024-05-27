import os
import torch
import torch.nn as nn
from math import sqrt
from sklearn.metrics import mean_squared_error

from train_e2sn2 import get_new_batch, Predictor
from gnn_explainer.load_model import load_good_dataset_dataloader


def dataset_test_performance(test_loader, model, device):
    model.eval()

    test_loss = []
    test_metric = []

    for idx, data in enumerate(test_loader):
        training_n_graphs_in_batch = (data.batch[-1].item() + 1) * 2

        node_features, edge_features, from_idx, to_idx, graph_idx, graph_idx_4edge, labels = get_new_batch(data.mol, data.y)

        with torch.no_grad():
            pred = model(node_features.to(device),
                         edge_features.to(device),
                         from_idx.to(device),
                         to_idx.to(device),
                         graph_idx.to(device),
                         graph_idx_4edge.to(device),
                         training_n_graphs_in_batch)

        loss = nn.functional.l1_loss(pred, labels.to(device))

        # Compute the RMSE for current batch
        metric_rmse = sqrt(mean_squared_error(pred.detach().cpu(), labels.cpu()))

        # Record loss
        test_loss.append(loss.item())
        test_metric.append(metric_rmse)

    test_loss = sum(test_loss) / len(test_loss)
    test_metric = sum(test_metric) / len(test_metric)

    print(f"\n[ Test | test_loss = {test_loss:.5f}, test_RMSE = {test_metric:.5f}")


if __name__ == '__main__':
    model_checkpoint = 'best_model_2024-05-27_19-09-49_val_RMSE_12.24159.pt'

    # config
    config_path = 'final_configs/GOODE2SN2/size/no_shift/base_data.yaml'

    # get device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(torch.cuda.get_device_name(device) if device.type == 'cuda' else "CPU")

    # load dataloader
    dataset, dataloader = load_good_dataset_dataloader(config_path)
    print(f"#D#Dataset: {dataset}")
    print(f"#D#Dataloader: {dataloader}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])
    test_loader = dataloader['test']

    # load model
    model = torch.load(f"model_save/{model_checkpoint}")

    dataset_test_performance(test_loader, model, device)
