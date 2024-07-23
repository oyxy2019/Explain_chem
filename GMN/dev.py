import torch
import torch.nn as nn
from torch_geometric.nn import InstanceNorm


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size=300, dropout_p=0.5):
        super().__init__()
        self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, batch):
        att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                if batch.shape[0] == 0:
                    inputs = inputs
                else:
                    inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def concrete_sample(att_log_logit, temp, training):
    if training:
        random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
    else:
        att_bern = (att_log_logit).sigmoid()
    return att_bern


if __name__ == '__main__':
    extractor = ExtractorMLP()

    emb = torch.randn(10, 300)
    batch = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 4])
    att_log_logits = extractor(emb, batch)
    print(f"att_log_logits: {att_log_logits.shape} \n{att_log_logits}")

    att = concrete_sample(att_log_logits, temp=1, training=True)
    print(f"att: {att.shape} \n{att}")
