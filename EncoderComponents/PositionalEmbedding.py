import torch
import torch.nn as nn
from torch import Tensor as T

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(max_len, embed_dim))

    def forward(self, x: T) -> T:
        return x + self.pos_embed[:x.shape[1]]