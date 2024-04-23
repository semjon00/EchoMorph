from torch import Tensor, nn
import torch
import einops
import random


class RandMachine(nn.Module):
    def __init__(self, k_min, k_max, fun):
        super().__init__()
        eps = 0.0001
        assert k_min < k_max + eps / 2
        assert 0 <= k_min <= 1  # "p" for "portion"
        assert 0 <= k_max <= 1
        assert fun in ['lin', 'exp']
        self.mode = 'c' if abs(k_max - k_min) < eps else 'r'
        if self.mode == 'r':
            self.rng = random.Random()
        else:
            self.rng = None
        self.k_min = k_min
        self.k_max = k_max
        self.fun = fun

    def set_p(self, new_p):
        assert 0 <= new_p <= 1
        self.mode = 'c'
        self.k_min = self.k_max = new_p

    def deterministic(self, seed):
        self.rng.seed(random.randint(0, 2**32 - 1) if seed is None else seed)

    def get_val(self):
        pp = self.rng.random() if self.mode == 'r' else 0.0
        return pp * (self.k_max - self.k_min) + self.k_min


class RandoMask(RandMachine):
    def __init__(self, k_min, k_max, fun):
        super().__init__(k_min, k_max, fun)

    def forward(self, x: Tensor):
        els = x.shape[-2]
        pels = els * super().get_val() if self.fun == 'lin' else els ** super().get_val()
        x[..., round(pels):, :] = 0  # Masking rightmost columns, leaving pels leftmost columns intact
        return x


class PriorityNoise(RandMachine):
    def __init__(self, k_min, k_max, fun, embed_dim):
        super().__init__(k_min, k_max, fun)
        assert fun == 'lin', 'PriorityNoise only supports lin mode'
        self.embed_dim = embed_dim
        self.importance = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        val = super().get_val()
        val = val * (self.k_max - self.k_min) + self.k_min

        # Make quality levels strictly conform to the budget constraint
        unadjusted_quality = self.importance(x)
        p = unadjusted_quality.mean(dim=-1)
        s = torch.min((1 - val) / (1 - p), val / p)
        k = torch.max(torch.zeros_like(p), (val - p) / (1 - p))
        quality = k.unsqueeze(-1) + s.unsqueeze(-1) * unadjusted_quality

        noise_levels = -torch.log(quality)  # eps not needed sigmoid (as we scale it above) never touches 0 or 1
        noise_levels = einops.repeat(noise_levels, '... -> ... a', a=self.embed_dim)
        noise = torch.randn_like(noise_levels)

        return self.norm2(self.norm1(x) + noise_levels * noise)
