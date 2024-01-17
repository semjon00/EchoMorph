import numpy as np
import torch
from torch import Tensor, nn
import einops

from transformer_blocks import PositionalEmbedding, TransformerBlock

# TODO: We have a bottleneck in the middle of the model
# TODO: SpeakerEncoder also reduces input size 4-fold
# TODO: We totally should make block sizes different

# TODO: Consumes unholy amounts of memory.

# TODO: Maybe use intermediate representation for autoregressive feeding of history?

class VoicetronParameters:
    def __init__(self):
        one_sec_len = (32000 // 105) // 8 * 8  # sample_rate / hop_length; approximately

        self.target_sample_len = 8 * one_sec_len
        self.history_len = one_sec_len
        self.fragment_len = one_sec_len // 4
        self.spect_width = 512  # x_width

        self.sc_len = self.target_sample_len // 4  # Speaker characteristic

        self.batch_size = 256
        self.drop = 0.1

        self.se_blocks = 8
        self.se_heads = 8
        self.se_hidden_dim_m = 3

        self.mid_repeat_min = 2
        self.mid_repeat_max = 5

        self.ae_blocks = (6, 4, 4)
        self.ae_heads = 12
        self.ae_hidden_dim_m = 3

        self.rm_k_min = 1 / 8
        self.rm_k_max = 1
        self.rm_fun = 'exp'


class SpeakerEncoder(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()

        self.pos_embed = PositionalEmbedding(
            seq_len=pars.target_sample_len,
            embed_dim=pars.spect_width
        )
        blocks = []
        assert pars.se_blocks % 2 == 0, "Criss-crossing won't work"
        for i in range(pars.se_blocks):
            ed = pars.spect_width if i % 2 == 0 else pars.target_sample_len
            blocks += [TransformerBlock(
                embed_dim=ed,
                num_heads=pars.se_heads,
                hidden_dim=ed * pars.se_hidden_dim_m,
                attn_drop=pars.drop,
                drop=pars.drop
            )]
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(pars.drop)
        self.squeeze_k = self.target_sample_len // self.sc_len

    def forward(self, x: Tensor) -> Tensor:
        out = self.pos_embed(x)
        out = self.dropout(out)
        for block in self.blocks:
            out = block(out).T
        return einops.rearrange(x, '... (k l) w -> ... l k w', k=self.squeeze_k).sum(dim=-2)

class AudioEncoder(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()

        self.pos_embed = PositionalEmbedding(
            seq_len=pars.fragment_len,
            embed_dim=pars.spect_width
        )
        blocks = []
        assert all([x % 2 == 0 for x in pars.ae_blocks]), "Criss-crossing won't work"
        for i in range(sum(pars.ae_blocks)):
            ed = pars.spect_width if i % 2 == 0 else pars.fragment_len
            cross_attn = 1 if i % 2 == 0 else 0
            blocks += [TransformerBlock(
                embed_dim=ed,
                num_heads=pars.ae_heads,
                hidden_dim=ed * pars.ae_hidden_dim_m,
                attn_drop=pars.drop,
                drop=pars.drop,
                n_cross_attn_blocks=cross_attn
            )]
        self.blocks_pre = nn.ModuleList(blocks[:pars.ae_blocks[0]])
        self.blocks_mid = nn.ModuleList(blocks[pars.ae_blocks[0]:pars.ae_blocks[1]])
        self.mid_repeat_min = pars.mid_repeat_min
        self.mid_repeat_max = pars.mid_repeat_max
        self.blocks_post = nn.ModuleList(blocks[pars.ae_blocks[1]:])
        self.dropout = nn.Dropout(pars.drop)

    def forward(self, fragment: Tensor, history: Tensor) -> Tensor:
        mid_rep = np.random.randint(self.mid_repeat_min, self.mid_repeat_max + 1)

        out = self.pos_embed(fragment)
        out = self.dropout(out)
        for i, block in enumerate(self.blocks_pre):
            out = block(out, [history] if i % 2 == 0 else []).T
        for rep in range(mid_rep):
            for i, block in enumerate(self.blocks_pre):
                out = block(out, [history] if i % 2 == 0 else []).T
        for i, block in enumerate(self.blocks_pre):
            out = block(out, [history] if i % 2 == 0 else []).T
        return out

class RandoMask(nn.Module):
    """
    Masks some portion of the input.
    Allows us to use different settings for the quality/similarity tradeoff.
    """
    def __init__(self, k_min, k_max, fun):
        super().__init__()
        eps = 0.0001
        assert k_min < k_max + eps / 2
        assert 0 <= k_min <= 1  # "p" for "portion"
        assert 0 <= k_max <= 1
        self.mode = 'c' if abs(k_max - k_min) < eps else 'r'
        if self.mode == 'r':
            self.rng = np.random.default_rng(12345)
        else:
            self.rng = None
        self.k_min = k_min
        self.k_max = k_max
        self.fun = fun

    def set_p(self, new_p):
        assert 0 <= new_p <= 1
        self.mode = 'c'
        self.k_min = self.k_max = new_p

    def forward(self, x: Tensor):
        els = x.shape[1]
        pp = self.rng.random() if self.mode == 'r' else self.k_min
        if self.fun == 'lin':
            pels = els * (pp * (self.k_max - self.k_min) + self.k_min)
        elif self.fun == 'exp':
            pels = els ** (pp * (self.k_max - self.k_min) + self.k_min)
        else:
            raise "Wrong randomask fun"
        x[..., round(pels):] = 0
        return x


class AudioDecoder(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()


class Voicetron(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()
        self.pars = pars
        self.speaker_encoder = SpeakerEncoder(pars)
        self.audio_encoder = AudioEncoder(pars)
        self.rando_mask = RandoMask(pars.rm_k_min, pars.rm_k_max, pars.rm_fun)
        self.audio_decoder = AudioDecoder(pars)

    def forward(self, target_sample, source_history, source_fragment, target_history=None):
        """Used for training, use inference.py for usage"""
        if target_history is None:
            target_history = source_history

        speaker_characteristic = self.speaker_encoder(target_sample)
        intermediate = self.rando_mask(self.audio_encoder(source_fragment, source_history))
        output = self.audio_decoder(speaker_characteristic, target_history, intermediate)
        return output
