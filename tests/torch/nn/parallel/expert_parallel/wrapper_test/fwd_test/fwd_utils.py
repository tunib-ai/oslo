import random

import numpy as np

import torch
import torch.backends.cudnn as cudnn


class TestFFNBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features, out_features)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(out_features, in_features)

    def forward(self, inp):
        front_out = self.fc1(inp)
        inter = self.act(front_out)
        behind_out = self.fc2(inter)

        return behind_out


def fix_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def sequence_dataloader(
    batch_size, total_samples, hidden_dim, device, seq_len: int = 32, dtype=torch.half
):
    train_data = torch.randn(
        total_samples, seq_len, hidden_dim, device=device, dtype=dtype
    )
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(
        hidden_dim
    )

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    return train_loader
