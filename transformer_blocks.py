import torch
import torch.nn as nn
from torch import Tensor as T
import einops

class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, embed_dim))

    def forward(self, x: T) -> T:
        return x + self.pos_embed  # seq_len is *always* constant in our case


class SelfAttention(nn.Module):
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

    def forward(self, x: T) -> T:
        q, k, v = self.qkv_partition(x)
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


class CrossAttention(nn.Module):
    """Cross attention and also the following normalization layer"""
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            drop_p: float
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_kv = nn.Linear(embed_dim, embed_dim * 2)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_p)
        self.drop = nn.Dropout(drop_p)
        self.norm = nn.LayerNorm(embed_dim)

    def head_partition(self, x: T) -> T:
        return einops.rearrange(x, '... s (h d) -> ... h s d', h=self.num_heads)

    def head_merging(self, x: T) -> T:
        return einops.rearrange(x, '... h s d -> ... s (h d)')

    def forward(self, layer_input: T, cross_attn_input: T) -> T:
        res = layer_input

        q = self.project_q(layer_input)
        k, v = self.project_kv(cross_attn_input).chunk(2, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = torch.einsum('...qc,...kc->...qk', q, k) * self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.einsum('...qv,...vc->...qc', attn_weights, v)
        out = self.head_merging(out)
        out = self.proj_out(out)
        out = self.drop(out)
        out = self.norm(out + res)

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        attn_drop: float,
        drop: float,
        n_cross_attn_blocks: int = 0,
        do_norm: bool = True
    ):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads, attn_drop)
        self.self_attn_dropout = nn.Dropout(p=drop)
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        self.cross_attn_blocks = []
        for _ in range(n_cross_attn_blocks):
            self.cross_attn_blocks.append(
                CrossAttention(embed_dim, num_heads, attn_drop)
            )

        self.ffn = FeedForward(embed_dim, hidden_dim)
        self.ffn_dropout = nn.Dropout(p=drop)
        self.ffn_norm = nn.LayerNorm(embed_dim) if do_norm else None

    def forward(self, layer_input: T, cross_attn_inputs: [T] = []) -> T:
        res = layer_input
        out = self.self_attn(layer_input)
        out = self.self_attn_dropout(out)
        out = self.self_attn_norm(out + res)

        for cross_attn_input, cross_attn_block in zip(
            cross_attn_inputs, self.cross_attn_blocks
        ):
            out = cross_attn_block(out, cross_attn_input)

        res = out
        out = self.ffn(out)
        out = self.ffn_dropout(out)
        if self.ffn_norm is not None:
            out = self.ffn_norm(out + res)

        return out
