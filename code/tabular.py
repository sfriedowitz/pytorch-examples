import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.data = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]
