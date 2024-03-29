from itertools import chain
import numpy as np
import torch
from torch import Tensor, nn
import einops
import random
import pickle
import os

from transformer_blocks import PositionalEmbedding, TransformerBlock, FeedForward

# TODO: We have a bottleneck in the middle of the model
#       SpeakerEncoder also reduces input size 4-fold
#       We totally should make block sizes different
#       Currently it consumes too much memory.

# TODO: Maybe use intermediate representation for autoregressive feeding of history?

class EchoMorphParameters:
    """Training parameters"""
    def __init__(self, **kwargs):
        """By default, contains large model specs"""
        one_sec_len = round(32000 / 105 / 64) * 64  # sample_rate / hop_length; approximately

        self.target_sample_len = 8 * one_sec_len
        self.history_len = one_sec_len // 2
        self.fragment_len = one_sec_len // 4
        self.spect_width = 256  # x_width

        self.sc_len = one_sec_len // 2  # Speaker characteristic shrink
        self.ir_width = self.spect_width  # Intermediate representation allowance

        self.se_blocks = 4
        self.se_heads = 8
        self.se_hidden_dim_m = 3

        self.ae_blocks = (2, 2, 0)
        self.ae_heads = 4
        self.ae_hidden_dim_m = 2

        self.ad_blocks = (0, 2, 2)
        self.ad_heads = 8
        self.ad_hidden_dim_m = 2

        self.drop = 0.01
        self.rm_k_min = 5 / 8
        self.rm_k_max = 8 / 8
        self.rm_fun = 'exp'
        self.mid_repeat_interval = (2, 5)  # (inclusive, exclusive)

        for key, value in kwargs.items():
            setattr(self, key, value)


class AudioCoder(nn.Module):
    def __init__(self, spect_width, hidden_dim_m, heads, spect_len, drop, blocks_num, cross_n, mid_repeat_interval):
        super().__init__()
        assert all([x % 2 == 0 for x in blocks_num]), "Criss-crossing won't work"

        blocks = []
        for i in range(sum(blocks_num)):
            ed = spect_width if i % 2 == 0 else spect_len
            this_cross_n = cross_n if i % 2 == 0 else 0
            blocks += [TransformerBlock(
                embed_dim=ed,
                num_heads=heads,
                hidden_dim=ed * hidden_dim_m,
                attn_drop=drop,
                drop=drop,
                n_cross_attn_blocks=this_cross_n
            )]
        self.blocks_pre = nn.ModuleList(blocks[:blocks_num[0]])
        self.blocks_mid = nn.ModuleList(blocks[blocks_num[0]:blocks_num[0]+blocks_num[1]])
        self.blocks_post = nn.ModuleList(blocks[blocks_num[0]+blocks_num[1]:])
        self.mid_repeat_interval = mid_repeat_interval

        self.dropout = nn.Dropout(drop)

    def set_mid_repeat_interval(self, new_val):
        self.mid_repeat_interval = (new_val, new_val + 1)

    def forward(self, x: Tensor, cross: list[Tensor], mid_rep=None):
        if mid_rep is None:
            mid_rep = random.randint(*self.mid_repeat_interval)
        for i, block in enumerate(self.blocks_pre):
            x = torch.transpose(block(x, cross if i % 2 == 0 else []), -1, -2)
        for rep in range(mid_rep):
            for i, block in enumerate(self.blocks_mid):
                x = torch.transpose(block(x, cross if i % 2 == 0 else []), -1, -2)
        for i, block in enumerate(self.blocks_post):
            x = torch.transpose(block(x, cross if i % 2 == 0 else []), -1, -2)
        return x

class SpeakerEncoder(AudioCoder):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(pars.spect_width, pars.se_hidden_dim_m, pars.se_heads, pars.target_sample_len,
                         pars.drop, (pars.se_blocks, 0, 0), 0, (0, 1))
        self.pos_embed = PositionalEmbedding(
            seq_len=pars.target_sample_len,
            embed_dim=pars.spect_width
        )
        self.dropout = nn.Dropout(pars.drop)
        self.squeeze_k = pars.target_sample_len // pars.sc_len

    def forward(self, x: Tensor) -> Tensor:
        x = self.pos_embed(x)
        x = self.dropout(x)
        x = super().forward(x, [])
        x = einops.rearrange(x, '... (k l) w -> ... l k w', k=self.squeeze_k).sum(dim=-2)
        return x

class AudioEncoder(AudioCoder):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(pars.spect_width, pars.ae_hidden_dim_m, pars.ae_heads, pars.fragment_len,
                         pars.drop, pars.ae_blocks, 1, pars.mid_repeat_interval)
        self.pos_embed = PositionalEmbedding(
            seq_len=pars.fragment_len,
            embed_dim=pars.spect_width
        )
        self.out_w = pars.ir_width

    def forward(self, x: Tensor, history: Tensor, mid_rep=None) -> Tensor:
        x = self.pos_embed(x)
        x = self.dropout(x)
        x = super().forward(x, [history], mid_rep=mid_rep)
        x[..., self.out_w:] = 0
        return x


class AudioDecoder(AudioCoder):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(pars.spect_width, pars.ad_hidden_dim_m, pars.ad_heads, pars.fragment_len,
                         pars.drop, pars.ad_blocks, 3, pars.mid_repeat_interval)
        self.finish = nn.ModuleList([FeedForward(pars.spect_width, pars.spect_width) for _ in range(2)])

    def forward(self, x: Tensor, speaker_characteristic: Tensor, history: Tensor, mid_rep=None) -> Tensor:
        x = super().forward(x, [speaker_characteristic, history, x], mid_rep=mid_rep)
        for layer in self.finish:
            x = layer(x)
        return x


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
        els = x.shape[-2]
        pp = self.rng.random() if self.mode == 'r' else self.k_min
        if self.fun == 'lin':
            pels = els * (pp * (self.k_max - self.k_min) + self.k_min)
        elif self.fun == 'exp':
            pels = els ** (pp * (self.k_max - self.k_min) + self.k_min)
        else:
            raise "Wrong randomask fun"
        pels = round(pels)
        x[..., pels:, :] = 0  # Masking rightmost columns, leaving pels leftmost columns intact
        return x


class EchoMorph(nn.Module):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__()
        self.pars = pars
        self.speaker_encoder = SpeakerEncoder(pars)
        self.audio_encoder = AudioEncoder(pars)
        self.rando_mask = RandoMask(pars.rm_k_min, pars.rm_k_max, pars.rm_fun)
        self.audio_decoder = AudioDecoder(pars)

    def forward(self, target_sample, source_history, source_fragment, target_history=None, middle_repeats=None):
        """Used for training, use inference.py for inference"""
        if target_history is None:
            target_history = source_history

        speaker_characteristic = self.speaker_encoder(target_sample)
        intermediate = self.rando_mask(self.audio_encoder(source_fragment, source_history, middle_repeats))
        output = self.audio_decoder(intermediate, speaker_characteristic, target_history, middle_repeats)
        return output

    def get_multiplicating_parameters(self):
        return chain.from_iterable([
            m.blocks_mid.parameters() for m in [self.speaker_encoder, self.audio_encoder, self.audio_decoder]
        ])

    def get_base_parameters(self):
        mult_params = set(self.get_multiplicating_parameters())
        all_params = set(self.parameters())
        base_params = list(all_params - mult_params)
        return base_params


def load_model(directory, device, dtype, verbose=False):
    # TODO: load safer
    try:
        pars = pickle.load(open(directory / 'parameters.bin', 'rb'))
    except:
        pars = EchoMorphParameters()
    model = EchoMorph(pars).to(device=device, dtype=dtype)
    model.load_state_dict(torch.load(directory / 'model.bin', map_location=device))
    if verbose:
        print(f'Model parameters: {dict(model.pars.__dict__.items())}')
    return model

def save_model(directory, model: EchoMorph):
    os.makedirs(directory, exist_ok=True)
    pickle.dump(model.pars, open(directory / 'parameters.bin', 'wb'))
    torch.save(model.state_dict(), directory / 'model.bin')
