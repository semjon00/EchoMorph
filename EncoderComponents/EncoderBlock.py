import torch.nn as nn
from torch import Tensor as T
from typing import Optional

from FeedForward import FeedForward
from SelfAttention import SelfAttention


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        max_len: int,
        attn_drop: float,
        drop: float,
    ):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads, max_len, attn_drop)
        self.self_attn_dropout = nn.Dropout(p=drop)
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ffn_dropout = nn.Dropout(p=drop)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: T) -> T:
        res = x
        out = self.self_attn(x)
        out = self.self_attn_dropout(out)
        out = self.self_attn_norm(out + res)

        res = out
        out = self.ffn(out)
        out = self.ffn_dropout(out)
        out = self.ffn_norm(out + res)

        return out