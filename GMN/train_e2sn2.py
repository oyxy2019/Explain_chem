import torch
from datetime import datetime
from tqdm import tqdm
from rdkit import RDLogger

from gnn_explainer.load_model import load_good_dataset_dataloader
from ReactionOOD.chemprop.features import MolGraph, set_explicit_h
# from chemprop.features import MolGraph, set_explicit_h
from gnn_explainer.myUtils import visualize_chemprop_molgraph, plot_reaction_with_smiles

from math import sqrt
from sklearn.metrics import mean_squared_error

from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *
import numpy as np
import torch.nn as nn
import collections
import time
import os

# 当loss='margin'时，会报错梯度计算
torch.autograd.set_detect_anomaly(True)

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
print(torch.cuda.get_device_name(device) if device.type == 'cuda' else "CPU")

# Print configure
config = get_default_config()
for (k, v) in config.items():
    print("%s= %s" % (k, v))

# Set random seeds
seed = config['seed']
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# setting
set_explicit_h(True)
RDLogger.DisableLog('rdApp.warning')
pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                'dynamic_ncols': True, 'ascii': '░▒█'}


class Predictor(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()

        # gmn model
        config['encoder']['node_feature_dim'] = node_feature_dim
        config['encoder']['edge_feature_dim'] = edge_feature_dim
        encoder = GraphEncoder(**config['encoder'])
        aggregator = GraphAggregator(**config['aggregator'])
        self.gmn = GraphMatchingNet(encoder, aggregator, **config['graph_matching_net'])

        # predictor
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)  # input_dim * 2 because we concatenate x and y
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features, edge_features, from_idx, to_idx, graph_idx, training_n_graphs_in_batch):
        graph_vectors = self.gmn(node_features, edge_features, from_idx, to_idx, graph_idx, training_n_graphs_in_batch)
        x, y = reshape_and_split_tensor(graph_vectors, 2)
        combined = torch.cat((x, y), dim=1)  # Concatenate x and y
        out = torch.relu(self.fc1(combined))
        out = self.fc2(out)
        return out.squeeze()


def get_new_batch(batch_mol, batch_target):
    """
    将原来的CGR单图的batch转变为双分子图的batch
    :param batch_mol
    :param batch_target
    :return: node_features, edge_features, from_idx, to_idx, graph_idx, labels
    """
    # 得到所有molgraph，根据GMN，反应物分子和产物分子索引为i和i+1
    molgraph_list = []
    for mol in batch_mol:
        molgraph_reac = MolGraph(mol.split(">")[0])
        molgraph_prod = MolGraph(mol.split(">")[-1])
        molgraph_list.append(molgraph_reac)
        molgraph_list.append(molgraph_prod)
        # Debug
        # print(mol)
        # plot_reaction_with_smiles(mol)
        # break

    # 得到molgraph对应属性
    node_features = []
    edge_features = []
    from_idx = []
    to_idx = []
    graph_idx = []
    for i, molgraph in enumerate(molgraph_list):
        node_features.append(torch.tensor(molgraph.f_atoms))
        edge_features.append(torch.tensor(molgraph.f_bonds))
        atom1_idx = [molgraph.b2a[bond_idx] for bond_idx in range(molgraph.n_bonds)]
        atom2_idx = [molgraph.b2a[molgraph.b2revb[bond_idx]] for bond_idx in range(molgraph.n_bonds)]
        from_idx.append(torch.tensor(atom1_idx))
        to_idx.append(torch.tensor(atom2_idx))
        graph_idx.append(torch.zeros(molgraph.n_atoms, dtype=torch.long) + i)
        # Debug
        # print(f'n_atoms: {molgraph.n_atoms}')
        # print(f'n_bonds: {molgraph.n_bonds}')
        # visualize_chemprop_molgraph(molgraph, is_reaction=False)
        # print(f'atom1_idx:{atom1_idx}')
        # print(f'atom2_idx:{atom2_idx}')
        # break

    # 拼接张量
    node_features = torch.cat(node_features, dim=0)
    edge_features = torch.cat(edge_features, dim=0)
    from_idx = torch.cat(from_idx, dim=0)
    to_idx = torch.cat(to_idx, dim=0)
    graph_idx = torch.cat(graph_idx, dim=0)
    labels = batch_target.squeeze()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels


if __name__ == '__main__':
    # config
    config_path = 'final_configs/GOODE2SN2/size/no_shift/GSAT.yaml'

    # save dir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    os.makedirs('model_save', exist_ok=True)

    # load dataset
    dataset, dataloader = load_good_dataset_dataloader(config_path)
    print(f"#D#Dataset: {dataset}")
    print(f"#D#Dataloader: {dataloader}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])

    # dataloader
    train_loader = dataloader['train']
    valid_loader = dataloader['val']

    # model and optimizer
    node_feature_dim = 133
    edge_feature_dim = 147
    input_dim = 128
    hidden_dim = 64
    output_dim = 1
    model = Predictor(node_feature_dim, edge_feature_dim, input_dim, hidden_dim, output_dim).to(device)
    optimizer = torch.optim.Adam((model.parameters()), lr=config['training']['learning_rate'], weight_decay=1e-5)

    # train
    print(f'#IN#Training started')
    num_epoch = 200
    stale = 0
    best_epoch = 0
    patience = 20
    best_score = float('inf')
    for epoch in range(num_epoch):
        # ---------- Training ----------
        model.train()

        train_loss = []
        train_metric = []

        for idx, data in enumerate(tqdm(train_loader, **pbar_setting)):
            training_n_graphs_in_batch = (data.batch[-1].item() + 1) * 2

            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_new_batch(data.mol, data.y)

            pred = model(node_features.to(device),
                         edge_features.to(device),
                         from_idx.to(device),
                         to_idx.to(device),
                         graph_idx.to(device),
                         training_n_graphs_in_batch)

            loss = nn.functional.l1_loss(pred, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_value'])
            optimizer.step()

            # Compute the RMSE for current batch
            metric_rmse = sqrt(mean_squared_error(pred.detach().cpu(), labels.cpu()))

            # Record loss
            train_loss.append(loss.item())
            train_metric.append(metric_rmse)

        train_loss = sum(train_loss) / len(train_loss)
        train_metric = sum(train_metric) / len(train_metric)

        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, RMSE = {train_metric:.5f}", end='')

        # ---------- Validation ----------
        model.eval()

        valid_loss = []
        valid_metric = []

        for idx, data in enumerate(valid_loader):
            training_n_graphs_in_batch = (data.batch[-1].item() + 1) * 2

            node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_new_batch(data.mol, data.y)

            with torch.no_grad():
                pred = model(node_features.to(device),
                             edge_features.to(device),
                             from_idx.to(device),
                             to_idx.to(device),
                             graph_idx.to(device),
                             training_n_graphs_in_batch)

            loss = nn.functional.l1_loss(pred, labels.to(device))

            # Compute the RMSE for current batch
            metric_rmse = sqrt(mean_squared_error(pred.detach().cpu(), labels.cpu()))

            # Record loss
            valid_loss.append(loss.item())
            valid_metric.append(metric_rmse)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_metric = sum(valid_metric) / len(valid_metric)

        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, RMSE = {valid_metric:.5f}")

        # save model
        if valid_metric < best_score:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"model_save/best_epoch_{current_time}.ckpt")
            best_score = valid_metric
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
    print(f'\nTraining end, Best model found at epoch {best_epoch}, best_score={best_score:.5f}')