import random
import torch
import torch.nn as nn
from torch import Tensor
import einops


class PlanePositionalEmbedding(nn.Module):
    def __init__(self, plane_length: int, plane_width: int, embed_dim: int):
        super().__init__()
        self.row_embed = nn.Parameter(torch.rand(plane_length, 1, embed_dim) * 0.01)
        self.column_embed = nn.Parameter(torch.rand(1, plane_width, embed_dim) * 0.01)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.row_embed + self.column_embed


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.project_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

    def head_partition(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, '... s (h d) -> ... h s d', h=self.num_heads)

    def head_merging(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, '... h s d -> ... s (h d)')

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.project_qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = torch.einsum('...qc,...kc->...qk', q, k) * self.scale
        attn_weights = self.softmax(attn_scores)

        out = torch.einsum('...qv,...vc->...qc', attn_weights, v)
        out = self.head_merging(out)
        out = self.proj_out(out)
        return out


class CrossAttention(nn.Module):
    """Cross attention and also the following normalization layer"""
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // self.num_heads) ** -0.5
        self.project_q = nn.Linear(embed_dim, embed_dim)
        self.project_kv = nn.Linear(embed_dim, embed_dim * 2)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def head_partition(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, '... s (h d) -> ... h s d', h=self.num_heads)

    def head_merging(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, '... h s d -> ... s (h d)')

    def forward(self, layer_input: Tensor, cross_attn_input: Tensor) -> Tensor:
        q = self.project_q(layer_input)
        k, v = self.project_kv(cross_attn_input).chunk(2, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = torch.einsum('...qc,...kc->...qk', q, k) * self.scale
        attn_weights = self.softmax(attn_scores)

        out = torch.einsum('...qv,...vc->...qc', attn_weights, v)
        out = self.head_merging(out)
        out = self.proj_out(out)

        return out


class MLP(nn.Sequential):
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
        mlp_hidden_dim: int,
        attn_drop: float,
        mlp_drop: float,
        n_cross_attn_blocks: int = 0
    ):
        super().__init__()
        self.self_attn = SelfAttention(embed_dim, num_heads)
        self.self_attn_dropout = nn.Dropout(p=attn_drop)
        self.self_attn_norm = nn.LayerNorm(embed_dim)

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttention(embed_dim, num_heads)
            for _ in range(n_cross_attn_blocks)
        ])
        # Dropout? Whatever...
        self.cross_attn_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(n_cross_attn_blocks)
        ])

        self.mlp = MLP(embed_dim, mlp_hidden_dim)
        self.mlp_dropout = nn.Dropout(p=mlp_drop)
        self.mlp_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, cross_attn_inputs: [Tensor] = []) -> Tensor:
        out = self.self_attn_norm(x)
        out = self.self_attn(out)
        out = self.self_attn_dropout(out)
        x = x + out

        for i in range(len(self.cross_attn_norm)):
            out = self.cross_attn_norm[i](x)
            # Normalize cross inputs?
            out = self.cross_attn_blocks[i](out, cross_attn_inputs[i])
            x = x + out

        out = self.mlp_norm(x)
        out = self.mlp(out)
        out = self.mlp_dropout(out)
        x = x + out

        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim, heads, drop, blocks_num, cross_n, mid_repeat_interval):
        super().__init__()

        blocks = []
        for i in range(sum(blocks_num)):
            this_cross_n = cross_n
            blocks += [TransformerBlock(
                embed_dim=embed_dim,
                num_heads=heads,
                mlp_hidden_dim=mlp_hidden_dim,
                attn_drop=drop,
                mlp_drop=drop,
                n_cross_attn_blocks=this_cross_n
            )]
        self.blocks_pre = nn.ModuleList(blocks[:blocks_num[0]])
        self.blocks_mid = nn.ModuleList(blocks[blocks_num[0]:blocks_num[0]+blocks_num[1]])
        self.blocks_post = nn.ModuleList(blocks[blocks_num[0]+blocks_num[1]:])
        self.mid_repeat_interval = mid_repeat_interval

    def set_mid_repeat_interval(self, new_val):
        self.mid_repeat_interval = (new_val, new_val + 1)

    def forward(self, x: Tensor, cross: list[Tensor], mid_rep=None):
        if mid_rep is None:  # TODO: this must always be passed from above
            mid_rep = random.randint(*self.mid_repeat_interval)
        for i, block in enumerate(self.blocks_pre):
            x = block(x, cross)
        for rep in range(mid_rep):
            for i, block in enumerate(self.blocks_mid):
                x = block(x, cross)
        for i, block in enumerate(self.blocks_post):
            x = block(x, cross)
        return x
