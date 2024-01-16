import numpy as np
import torch
from torch import Tensor, nn

from transformer_blocks import PositionalEmbedding, EncoderBlock


class VoicetronParameters:
    sample_rate: int

    randomask_p_min: float
    randomask_p_max: float

    target_sample_len: int
    history_tlen: int
    fragment_tlen: int
    batch_size: int


class SpeakerEncoder(nn.Module):
    def __init__(self, dims: VoicetronParameters):
        super().__init__()

        num_blocks = 6
        max_len = dims.target_sample_len
        embed_dim = max_len
        num_heads = 8
        hidden_dim = max_len
        drop = 0.1
        
        self.pos_embed = PositionalEmbedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, hidden_dim, max_len, attn_drop=drop, drop=drop)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pos_embed(x)
        out = self.dropout(out)
        for block in self.blocks:
            out = block(out)
        return out

class AudioEncoder(nn.Module):
    def __init__(self, dims: VoicetronParameters):
        super().__init__()


class RandoMask(nn.Module):
    """
    Masks some portion of the input.
    Allows us to use different settings for the quality/similarity tradeoff.
    """
    def __init__(self, p_min, p_max):
        super().__init__()
        eps = 0.0001
        assert p_min < p_max + eps / 2
        assert 0 <= p_min <= 1  # "p" for "portion"
        assert 0 <= p_max <= 1
        self.mode = 'c' if abs(p_max - p_min) < eps else 'r'
        if self.mode == 'r':
            self.rng = np.random.default_rng(12345)
        else:
            self.rng = None
        self.p_min = p_min
        self.p_max = p_max

    def set_p(self, new_p):
        assert 0 <= new_p <= 1
        self.mode = 'c'
        self.p_min = self.p_max = new_p

    def forward(self, x: Tensor):
        els = x.shape[1]
        p = self.rng.random() * (self.p_max - self.p_min) + self.p_min
        els_passthrough = round(els * p)
        x[:, els_passthrough:] = 0
        return x


class AudioDecoder(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()


class Voicetron(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()
        self.speaker_encoder = SpeakerEncoder(pars)
        self.audio_encoder = AudioEncoder(pars)
        self.rando_mask = RandoMask(pars.randomask_p_min, pars.randomask_p_max)
        self.audio_decoder = AudioDecoder(pars)

    def forward(self, target_sample, source_history, source_fragment, target_history=None):
        if target_history is None:
            target_history = source_history

        embedding = self.speaker_encoder(target_sample)
        intermediate = self.rando_mask(self.audio_encoder(source_fragment, target_history))
        output = self.audio_decoder(embedding, source_history, intermediate)
        return output
