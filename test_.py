import torch
from torch.autograd import Function
class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_x = output * (1 - output) * grad_output
        return grad_x


test_input = torch.randn(4, requires_grad=True)  # tensor([-0.4646, -0.4403,  1.2525, -0.5953], requires_grad=True)
print(torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-3))

def func(*a,**b):
    print(a)
    print(b)

func(1,1,1,1,1,1,1,1,1,1,1,a=1,b=2,c=3)