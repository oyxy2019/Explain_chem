import os
import torch

from train_e2sn2 import dataset_test_performance, Predictor
from gnn_explainer.load_model import load_good_dataset_dataloader


if __name__ == '__main__':
    model_checkpoint = 'best_model_2024-05-30_16-06-31_val_RMSE_3.33625.pt'

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
