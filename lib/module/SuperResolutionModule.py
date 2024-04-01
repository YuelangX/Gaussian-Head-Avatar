import torch
from torch import nn
from einops import rearrange

from lib.network.Upsampler import Upsampler

class SuperResolutionModule(nn.Module):
    def __init__(self, cfg):
        super(SuperResolutionModule, self).__init__()
        
        self.upsampler = Upsampler(cfg.input_dim, cfg.output_dim, cfg.network_capacity)

    def forward(self, input):
        output = self.upsampler(input)
        return output
