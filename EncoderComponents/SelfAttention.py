import torch
import torch.nn as nn
from torch import Tensor as T
import einops

from typing import Optional


class SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, max_len: int, attn_drop: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.project_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def head_partition(self, x: T) -> T:
        return einops.rearrange(x, '... s (h d) -> ... h s d', h=self.num_heads)

    def head_merging(self, x: T) -> T:
        return einops.rearrange(x, '... h s d -> ... s (h d)')

    def forward(self, x: T) -> T:
        q, k, v = self.project_qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = torch.einsum('...qc,...kc->...qk', q, k) * self.scale

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.einsum('...qv,...vc->...qc', attn_weights, v)
        out = self.head_merging(out)
        out = self.proj_out(out)
        return out