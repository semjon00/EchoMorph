from itertools import chain
import numpy as np
import torch
from torch import Tensor, nn
import einops
import random
import pickle
import os

from components import PriorityNoise
from transformer_blocks import TransformerBlock, PlanePositionalEmbedding

# TODO: Re-introduce the history
# TODO: Use intermediate representation for autoregressive feeding of history.

# TODO: Explore better compression methods
# TODO: Conv encoders

# TODO: Refactor training parameters into a separate class (don't forget kl_loss!)
# TODO: Sub-quadratic speaker encoder

class EchoMorphParameters:
    """Training parameters"""
    def __init__(self, **kwargs):
        """By default, contains large model specs"""
        one_sec_len = round(24000 / 84 / 64) * 64  # sample_rate / hop_length; approximately
        self.target_sample_len = one_sec_len // 4
        self.history_len = one_sec_len // 4
        self.fragment_len = one_sec_len // 8

        self.spect_width = 128  # x_width
        self.length_of_patch = 8

        self.embed_dim = 64

        self.se_blocks = (1, 2, 1)
        self.se_heads = 8
        self.se_hidden_dim_m = 3
        self.se_output_tokens = 256

        self.ae_blocks = (1, 2, 1)
        self.ae_heads = 4
        self.ae_hidden_dim_m = 2

        self.ad_blocks = (3, 2, 1)
        self.ad_heads = 8
        self.ad_hidden_dim_m = 2

        self.drop = 0.00
        self.rm_k_min = 0.0
        self.rm_k_max = 1.0
        self.rm_fun = 'lin'
        self.mid_repeat_interval = (2, 5)  # (inclusive, exclusive)

        for key, value in kwargs.items():
            setattr(self, key, value)


class AudioCoder(nn.Module):
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

        self.dropout = nn.Dropout(drop)

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


class SpeakerVAE(AudioCoder):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(embed_dim=pars.embed_dim, mlp_hidden_dim=pars.se_hidden_dim_m, heads=pars.se_heads,
                         drop=pars.drop, blocks_num=pars.se_blocks, cross_n=0, mid_repeat_interval=(0, 1))
        self.patch_len = pars.length_of_patch
        self.entok = nn.Linear(2 * pars.length_of_patch, pars.embed_dim)
        self.pos_embed = PlanePositionalEmbedding(
            pars.target_sample_len // self.patch_len, pars.spect_width, pars.embed_dim
        )
        self.out_tokens = pars.se_output_tokens
        self.mean_linear = nn.Linear(pars.embed_dim, pars.embed_dim)
        self.log_var_linear = nn.Linear(pars.embed_dim, pars.embed_dim)

    def forward_shared(self, x: Tensor) -> (Tensor, Tensor):
        # This should do: tokenization, pos embed, coding
        x = einops.rearrange(x, '... (l ld) w c -> ... l w (c ld)', ld=self.patch_len)
        x = self.entok(x)
        x = self.pos_embed(x)
        x = einops.rearrange(x, '... l w d -> ... (l w) d')
        x = super().forward(x, [])

        means = self.mean_linear(x[..., :self.out_tokens, :])
        return means, x[..., self.out_tokens:2 * self.out_tokens, :]

    def forward_train(self, x):
        means, log_vars_t = self.forward_shared(x)
        log_vars = self.log_var_linear(log_vars_t)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_vars) + means ** 2 - log_vars - 1, dim=-1))

        epsilon = torch.randn_like(means)
        std = torch.exp(0.5 * log_vars)
        z = std * epsilon + means
        return z, kl_loss

    def forward_use(self, x):
        means, log_vars_t = self.forward_shared(x)
        return means


class AudioEncoder(AudioCoder):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(embed_dim=pars.embed_dim, mlp_hidden_dim=pars.ae_hidden_dim_m, heads=pars.ae_heads,
                         drop=pars.drop, blocks_num=pars.ae_blocks, cross_n=0,
                         mid_repeat_interval=pars.mid_repeat_interval)
        self.patch_len = pars.length_of_patch
        self.entok = nn.Linear(2 * pars.length_of_patch, pars.embed_dim)
        self.pos_embed = PlanePositionalEmbedding(
            pars.fragment_len // self.patch_len, pars.spect_width, pars.embed_dim
        )

    def forward(self, x: Tensor, mid_rep=None) -> Tensor:
        # This should do: tokenization, pos embed, coding
        x = einops.rearrange(x, '... (l ld) w c -> ... l w (c ld)', ld=self.patch_len)
        x = self.entok(x)
        x = self.pos_embed(x)
        x = einops.rearrange(x, '... l w d -> ... (l w) d')
        x = super().forward(x, [], mid_rep=mid_rep)
        return x


class AudioDecoder(AudioCoder):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(embed_dim=pars.embed_dim, mlp_hidden_dim=pars.ad_hidden_dim_m, heads=pars.ad_hidden_dim_m,
                         drop=pars.drop, blocks_num=pars.ad_blocks, cross_n=2,
                         mid_repeat_interval=pars.mid_repeat_interval)
        self.detok = nn.Linear(pars.embed_dim, 2 * pars.length_of_patch)
        self.spect_width = pars.spect_width
        self.length_of_patch = pars.length_of_patch

    def forward(self, im: Tensor, sc: Tensor, mid_rep=None) -> Tensor:
        # This should do: coding, de-tokenization
        x = super().forward(im, [im, sc], mid_rep=mid_rep)

        x = einops.rearrange(x, '... (l w) d -> ... l w d', w=self.spect_width)
        x = self.detok(x)
        x = einops.rearrange(x, ' ... l w (c ld) -> ... (l ld) w c', ld=self.length_of_patch)
        return x


class EchoMorph(nn.Module):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__()
        self.pars = pars
        self.speaker_encoder = SpeakerVAE(pars)
        self.audio_encoder = AudioEncoder(pars)
        self.bottleneck = PriorityNoise(
            pars.rm_k_min, pars.rm_k_max, pars.rm_fun, pars.embed_dim
        )
        self.audio_decoder = AudioDecoder(pars)

    def forward(self, target_sample, source_fragment, middle_repeats=None):
        """Used for training, use inference.py for inference"""
        speaker_characteristic, se_loss = self.speaker_encoder.forward_train(target_sample)
        intermediate = self.audio_encoder(source_fragment, middle_repeats)
        intermediate = self.bottleneck(intermediate)
        output = self.audio_decoder(intermediate, speaker_characteristic, middle_repeats)
        extra_loss = 0.003 * se_loss
        return output, extra_loss

    def get_multiplication_parameters(self):
        return chain.from_iterable([
            m.blocks_mid.parameters() for m in [self.speaker_encoder, self.audio_encoder, self.audio_decoder]
        ])

    def get_base_parameters(self):
        mult_params = set(self.get_multiplication_parameters())
        all_params = set(self.parameters())
        base_params = list(all_params - mult_params)
        return base_params


def load_model(directory, device, dtype, verbose=False):
    fp = directory / 'model.bin'
    if not fp.is_file():
        raise FileNotFoundError('Model not found.')
    pars = EchoMorphParameters()
    model = EchoMorph(pars).to(device=device, dtype=dtype)
    model.load_state_dict(torch.load(directory / 'model.bin', map_location=device))
    if verbose:
        print(f'Model parameters: {dict(model.pars.__dict__.items())}')
    return model


def save_model(directory, model: EchoMorph):
    os.makedirs(directory, exist_ok=True)
    # TODO: parameters aren't actually loaded
    pickle.dump(model.pars, open(directory / 'parameters.bin', 'wb'))
    torch.save(model.state_dict(), directory / 'model.bin')
