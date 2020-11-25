import torch
import numpy


class MonotonicNetwork(torch.nn.Module):
    """ Monotonic Network of One Dimension
    https://papers.nips.cc/paper/1358-monotonic-networks.pdf

    Example:
        >>> model = MonotonicNetwork()
        >>> X1 = torch.rand(50, 1)
        >>> X2 = model.inv(model(X1))
        >>> torch.isclose(X1, X2).all()
        tensor(True)
    """

    def __init__(self, n_group=9, n_each=10, const_sign=None):
        super().__init__()
        stdv = 1. / numpy.sqrt(n_group * n_each)
        self.n_group, self.n_each = n_group, n_each
        if const_sign is None:
            self.signer = torch.nn.Parameter(torch.FloatTensor(1))
            self.signer.data.uniform_(-0.1, 0.1)
        else:
            self.signer = torch.Tensor([const_sign])
        self.weight = torch.nn.Parameter(torch.Tensor(1, n_group * n_each))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias = torch.nn.Parameter(torch.Tensor(1, n_group * n_each))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        [n_batch, 1] -> [n_batch, 1]
        Example:
            >>> mn.forward(torch.rand(17, 1)).shape
            torch.Size([17, 1])
        """
        return monotonic_forward(x, self.weight, self.bias, self.signer, self.n_group, self.n_each)

    def inv(self, y):
        """
        [n_batch, 1] -> [n_batch, 1]
        Inverse of the learned monotonic function.
        Current strategy:
          The output value came from one of the outputs of the input layers. Therefore,
          1. Compute all possible candidates.
          2. Propagate the network again to find the activated input unit.
          3. Return the candidate corresponding to the activated input unit.
        Example:
            >>> mn.inv(torch.rand(17, 1)).shape
            torch.Size([17, 1])
        """
        return MonotonicInv.apply(y, self.weight, self.bias, self.signer, self.n_group, self.n_each)


def monotonic_forward(x, weight, bias, signer, n_group, n_each):
    W = signer * torch.exp(torch.clamp(weight, min=-20, max=20))
    dim_prefix = x.shape[:-1]
    mid = (x * W + bias).view(*dim_prefix, n_group, n_each)
    return mid.max(dim=-1)[0].min(dim=-1, keepdim=True)[0]


class MonotonicInv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, weight, bias, signer, n_group, n_each):
        dim_prefix = y.shape[:-1]
        W_inv = torch.exp(-torch.clamp(weight, min=-20, max=20)) / signer
        candidates = ((y - bias) * W_inv).view(*dim_prefix, n_group * n_each, 1)
        with torch.no_grad():
            candidates_y = monotonic_forward(candidates, weight, bias, signer, n_group, n_each)
        indices = torch.abs(candidates_y - y.view(*dim_prefix, 1, 1)).min(dim=-2)[1]
        ctx.save_for_backward(y, weight, bias, signer, indices)
        x = torch.cat([candidates[i, j] for i, j in enumerate(indices)])
        return x

    @staticmethod
    def backward(ctx, grad_output):
        y, weight, bias, signer, indices = ctx.saved_tensors
        ew = torch.exp(-torch.clamp(weight, min=-20, max=20))
        grad_y = torch.zeros_like(y)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_signer = torch.zeros_like(signer)
        for i, j in enumerate(indices):  # batch_index, linear_index
            grad_y[[i, 0]] += grad_output[i] * ew[0, j] / signer
            grad_weight[0, j] += grad_output[i] * (y[i, 0] - bias[0, j]) / signer * -ew[0, j]
            grad_bias[0, j] += -grad_output[i] * ew[0, j] / signer
            grad_signer += grad_output[i] * (y[i, 0] - bias[0, j]) * ew[0, j] * (-1) / (signer)**2

        return grad_y, grad_weight, grad_bias, grad_signer, None, None


if __name__ == '__main__':
    import doctest
    doctest.testmod(extraglobs={'mn': MonotonicNetwork()}, verbose=True)
