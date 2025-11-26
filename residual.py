import torch
import torch.nn as nn
from neuroscript_runtime.primitives.activations import GELU
from neuroscript_runtime.primitives.linear import Linear
from neuroscript_runtime.primitives.normalization import LayerNorm
from neuroscript_runtime.primitives.regularization import Dropout

class ResidualFFN(nn.Module):
    def __init__(self, dim, expansion=4, dropout_rate=0.1):
        super().__init__()
        self.add_6 = Add()
        self.dropout_4 = Dropout(p=dropout_rate)
        self.fork_0 = Fork()
        self.g_e_l_u_3 = GELU()
        self.layer_norm_1 = LayerNorm(dim)
        self.linear_2 = Linear(dim, dim * expansion)
        self.linear_5 = Linear(dim * expansion, dim)

    def forward(self, x):
        x0 = self.fork_0(x)
        x1, x2 = x0
        x3 = self.layer_norm_1(x1)
        x4 = self.linear_2(x3)
        x5 = self.g_e_l_u_3(x4)
        x6 = self.dropout_4(x5)
        x7 = self.linear_5(x6)
        x8 = self.add_6((x7, x2))
        return x8
