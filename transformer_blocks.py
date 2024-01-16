import torch
import torch.nn as nn
from torch import Tensor as T
import einops

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(max_len, embed_dim))

    def forward(self, x: T) -> T:
        return x + self.pos_embed[:x.shape[1]]


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.project_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def qkv_partition(self, x):
        q, k, v = self.project_qkv(x).chunk(3, dim=-1)
        return q, k, v

    def head_partition(self, x: T) -> T:
        return einops.rearrange(x, '... s (h d) -> ... h s d', h=self.num_heads)

    def head_merging(self, x: T) -> T:
        return einops.rearrange(x, '... h s d -> ... s (h d)')

    def forward(self, q, k, v: T) -> T:
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = torch.einsum('...qc,...kc->...qk', q, k) * self.scale

        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.einsum('...qv,...vc->...qc', attn_weights, v)
        out = self.head_merging(out)
        out = self.proj_out(out)
        return out

class FeedForward(nn.Sequential):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

class TransformerBlock(nn.Module):
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
        self.self_attn = AttentionBlock(embed_dim, num_heads, attn_drop)
        self.self_attn_dropout = nn.Dropout(p=drop)
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        # TODO: optional cross-attention should go here I think?
        # TODO: Encoder needs to cross-attend to source history, decoder to target history and speech embedding
        # TODO: So it is optional, and should support any number of stuff to cross-attend

        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ffn_dropout = nn.Dropout(p=drop)
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: T) -> T:
        res = x
        out = self.self_attn(self.self_attn.qkv_partition(x))
        out = self.self_attn_dropout(out)
        out = self.self_attn_norm(out + res)

        res = out
        out = self.ffn(out)
        out = self.ffn_dropout(out)
        out = self.ffn_norm(out + res)

        return out
