import torch
from torch import nn, autograd
from typing import Any, Optional


class GRF(autograd.Function):

    @staticmethod
    def forward(ctx: Any, x, coeff: Optional[float] = 1.0):
        ctx.coeff = coeff
        return x*1.0

    # 返回的顺序与forward的参数顺序一致，即返回x的梯度，coeff的梯度（None）
    @staticmethod
    def backward(ctx:Any, grad_output:torch.Tensor):
        return grad_output.neg()* ctx.coeff, None

