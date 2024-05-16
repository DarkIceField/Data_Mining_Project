import torch
from torch import nn


class RSquare(nn.Module):
    def __init__(self):
        super(RSquare, self).__init__()

    def forward(self, y_pred, y_true):
        y_mean = torch.mean(y_true, dim=0)
        ss_total = torch.clamp(torch.sum((y_true - y_mean) ** 2, dim=0), min=1e-8)
        ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
        return 1 - torch.mean(ss_res / ss_total)


class STDLoss(nn.Module):
    def __init__(self):
        super(STDLoss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, y_pred, y_true):
        batch_loss = torch.mean(self.loss(y_pred, y_true), dim=1)
        mean_loss = torch.mean(batch_loss)
        std_loss = torch.std(batch_loss)
        return mean_loss,std_loss


if __name__ == "__main__":
    y_pred = torch.randn(32, 6)
    y_true = torch.randn(32, 6)
    r2 = RSquare()
    std = STDLoss()
    print(r2(y_pred, y_true))
    print(std(y_pred, y_true))
