import torch
from torch import  Tensor
from torch.nn.modules.module import Module
import torch.nn.functional as F


class SmoothLoss(Module):
    def __init__(self) -> None:
        super(SmoothLoss, self).__init__()

    def smooth_loss(self, pred, gold):
        eps = 0.2

        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1).long(), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()

        return loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.smooth_loss(input,target)