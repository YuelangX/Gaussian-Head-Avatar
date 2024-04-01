import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dims, last_op=None):
        super(MLP, self).__init__()

        self.dims = dims
        self.skip_layer = [int(len(dims) / 2)]
        self.last_op = last_op

        self.layers = []
        for l in range(0, len(dims) - 1):
            if l in self.skip_layer:
                self.layers.append(nn.Conv1d(dims[l] + dims[0], dims[l + 1], 1))
            else:
                self.layers.append(nn.Conv1d(dims[l], dims[l + 1], 1))
            self.add_module("conv%d" % l, self.layers[l])

    def forward(self, latet_code, return_all=False):
        y = latet_code
        tmpy = latet_code
        y_list = []
        for l, f in enumerate(self.layers):
            if l in self.skip_layer:
                y = self._modules['conv' + str(l)](torch.cat([y, tmpy], 1))
            else:
                y = self._modules['conv' + str(l)](y)
            if l != len(self.layers) - 1:
                y = F.leaky_relu(y)
        if self.last_op:
            y = self.last_op(y)
            y_list.append(y)
        if return_all:
            return y_list
        else:
            return y