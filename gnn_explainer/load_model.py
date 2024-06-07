import time
from typing import Tuple, Union
from munch import munchify

import torch.nn
from torch.utils.data import DataLoader

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.networks.models.GSATGNNs import GSATGIN
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.logger import load_logger


executed = False


def initialize_model_dataset(config: Union[CommonArgs, Munch]) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, dataset


def load_good_model_dataset(config_path, device=torch.device("cpu")):
    args = args_parser(['--config_path', config_path])
    config = config_summoner(args)
    config.device = device
    print(config)

    global executed
    if not executed:
        logger, writer = load_logger(config)    # 重复创建logger会导致递归错误，所以执行过一次后直接跳过
        executed = True

    model, dataset = initialize_model_dataset(config)

    ckpt = torch.load(config.test_ckpt, map_location=config.device)
    model.load_state_dict(ckpt['state_dict'])

    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    if isinstance(model, GSATGIN):
        return model.gnn, dataset


def load_good_dataset_dataloader(config_path, device=torch.device("cpu")):
    args = args_parser(['--config_path', config_path])
    config = config_summoner(args)
    config.device = device
    print(config)

    global executed
    if not executed:
        logger, writer = load_logger(config)    # 重复创建logger会导致递归错误，所以执行过一次后直接跳过
        executed = True

    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)

    return dataset, loader


if __name__ == '__main__':
    model, dataset = load_good_model_dataset('final_configs/GOODE2SN2/size/covariate/GSAT.yaml')
    print(model)
    train_dataset = dataset['train']
    print(dataset['train'][0].mol)
    print(dataset['train'][1].mol)
    print(dataset['train'][2].mol)

