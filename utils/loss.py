import torch
from torch import nn


class RSquare(nn.Module):
    def __init__(self, is_loss=True):
        super(RSquare, self).__init__()
        self.is_loss = is_loss

    def forward(self, y_pred, y_true):
        y_mean = torch.mean(y_true)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        if self.is_loss:
            return ss_res / ss_tot
        return 1 - ss_res / ss_tot
