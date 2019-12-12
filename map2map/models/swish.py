import torch


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        result = input * torch.sigmoid(input)
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        sigmoid = torch.sigmoid(input)
        return grad_output * (sigmoid * (1 + input * (1 - sigmoid)))


class Swish(torch.nn.Module):
    def forward(self, input):
        return SwishFunction.apply(input)
