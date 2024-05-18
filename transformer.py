import string

import torch
import torch.nn as nn
from torch import Tensor
import einops


class MultidimPositionalEmbedding(nn.Module):
    def __init__(self, space_dims, embed_dim):
        super().__init__()
        if isinstance(space_dims, int): space_dims = (space_dims,)
        self.pars = nn.ParameterList()
        for i in range(len(space_dims)):
            size = [int(1)] * len(space_dims)
            size.append(embed_dim)
            size[i] = space_dims[i]
            self.pars.append(nn.Parameter(torch.empty(*size)))
            nn.init.trunc_normal_(self.pars[i])

    def forward(self, x):
        for par in self.pars:
            x = x + par
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.ql = nn.Linear(embed_dim, embed_dim)
        self.kl = nn.Linear(embed_dim, embed_dim)
        self.vl = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = (embed_dim // num_heads) ** -0.5
        self.num_heads = num_heads
        self.project = nn.Linear(embed_dim, embed_dim)

    def head_partition(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, '... n (nh ch) -> ... nh n ch', nh=self.num_heads)

    def head_merging(self, x: Tensor) -> Tensor:
        return einops.rearrange(x, '... nh n ch -> ... n (nh ch)')

    def forward(self, x: Tensor, cross: Tensor = None) -> Tensor:
        if cross == None: cross = x
        q, k, v = self.ql(x), self.kl(cross), self.vl(cross)
        q, k, v = map(self.head_partition, (q, k, v))

        attention = q @ k.transpose(-1, -2) * self.scale
        attention = self.softmax(attention)
        out = attention @ v
        out = self.head_merging(out)
        out = self.project(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_cross_attn_blocks: int = 0, num_heads: int = 4, mlp_hidden_dim: int = None):
        super().__init__()
        self.selfattn_norm = nn.LayerNorm(embed_dim)
        self.selfattn = MultiHeadAttention(embed_dim, num_heads)
        self.crossattn_norm = nn.ModuleList()
        self.crossattn = nn.ModuleList()
        for _ in range(n_cross_attn_blocks):
            self.crossattn_norm.append(nn.LayerNorm(embed_dim))
            self.crossattn.append(MultiHeadAttention(embed_dim, num_heads))
        self.mlp_norm = nn.LayerNorm(embed_dim)
        if not mlp_hidden_dim: mlp_hidden_dim = 4 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x: Tensor, cross: list[Tensor] = ()) -> Tensor:
        assert len(cross) == len(self.crossattn)
        out = x + self.selfattn(self.selfattn_norm(x))
        for i in range(len(cross)):
            out = out + self.crossattn[i](self.crossattn_norm[i](out), cross[i])
        out = out + self.mlp(self.mlp_norm(out))
        return out


class Transformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, input_size: tuple,
                 num_blocks: int, embed_dim: int, cross_n: int, num_heads: int = 4, mlp_hidden_dim: int = None,
                 rearrange_back=True):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.rearrange_back = rearrange_back
        self.input_size = input_size
        self.pos_embed = MultidimPositionalEmbedding(input_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, cross_n, num_heads, mlp_hidden_dim)
                                     for _ in range(num_blocks)])
        self.final_projection = nn.Linear(embed_dim, output_dim)

    def forward(self, x: Tensor, cross: list[Tensor] = ()) -> Tensor:
        out = self.embedding(x)
        out = self.pos_embed(out)
        rec = {string.ascii_lowercase[i]: self.input_size[i] for i in range(len(self.input_size))}
        reck = ' '.join(rec.keys())
        out = einops.rearrange(out, f'... {reck} ed -> ... ({reck}) ed', **rec)
        for block in self.blocks:
            out = block(out, cross)
        if self.rearrange_back:
            out = einops.rearrange(out, f'... ({reck}) ed -> ... {reck} ed', **rec)
        out = self.final_projection(out)
        return out


if __name__ == '__main__':
    print('Running transformer test')
    def task_a():
        def rep(x, t):
            x = einops.repeat(torch.Tensor(x), '... -> l ... a', a=1, l=1)
            return x
        def inp(ed, s):
            seq = torch.rand(s)
            cr_0 = torch.rand(s)
            cr_1 = torch.rand(s)
            ## A ##
            ans = cr_0 - cr_1
            seq *= 0
            ## B ##
            # ans = seq.flip(0)
            # for i in range(0, s, 3):
            #     ans[i] *= cr_0[i] - 0.5 * cr_1[(i + 1) % s]
            # End
            seq, cr_0, cr_1, ans = [rep(x, ed).clone() for x in [seq, cr_0, cr_1, ans]]
            return seq, cr_0, cr_1, ans
        def inpoup(b, ed, s):
            acm = [[], [], [], []]
            for _ in range(b):
                n = inp(ed, s)
                for i in range(len(acm)):
                    acm[i] += [n[i]]
            for i in range(len(acm)):
                acm[i] = torch.cat(acm[i])
            return acm
        return inpoup
    inpoup = task_a()

    b = 256
    ed = 64
    s = 16
    tr = Transformer(1, 1, (s,), 10, ed, 2)
    emp = Transformer(1, ed, (s,), 0, ed, 0)
    try:
        import matplotlib.pyplot as plt
        import torchinfo
        torchinfo.summary(tr)
    except:
        pass
    optimizer = torch.optim.Adam({*tr.parameters(), *emp.parameters()}, 0.0002)
    losses = []
    for its in range(100000 + 1):
        optimizer.zero_grad()
        x, cross_0, cross_1, ans = inpoup(b, ed, s)
        out = tr(x, (emp(cross_0), emp(cross_1)))
        loss: Tensor = torch.nn.functional.mse_loss(out, ans)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if len(losses) % 10 == 0:
            try:
                plt.title(f'Loss: {losses[-1]}')
                plt.plot(list(range(len(losses))), losses)
                plt.ylim(0, 0.25)
                plt.xlim(0, (len(losses) + 110) // 100 * 100)
                plt.show()
            except:
                pass
