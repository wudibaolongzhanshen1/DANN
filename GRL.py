from typing import Optional

from torch import nn

from GRF import GRF


class GRL(nn.Module):
    def __init__(self, coeff: Optional[float] = 1.0):
        super(GRL, self).__init__()
        self.coeff = coeff

    def forward(self, x):
        # apply函数起到"运行 forward"的作用
        return GRF.apply(x, self.coeff)