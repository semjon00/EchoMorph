import numpy as np
import torch
from torch import Tensor, nn

from transformer_blocks import PositionalEmbedding, TransformerBlock


class VoicetronParameters:
    def __init__(self):
        self.target_sample_len = (32000 // 128) * 8 // 1  # 8sec * sample_rate / hop_length
        self.history_len = (32000 // 128)  # 1 sec
        self.fragment_len = (32000 // 128) // 4  # 0.25sec
        self.spect_width = 514  # x_width

        self.batch_size = 256
        self.drop = 0.1

        self.randomask_p_min = 1 / 8
        self.randomask_p_max = 1


class SpeakerEncoder(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()

        num_blocks = 6
        num_heads = 8

        # TODO: should cross-attend source_history

        self.pos_embed = PositionalEmbedding(
            seq_len=pars.target_sample_len, 
            embed_dim=pars.target_sample_len
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=pars.spect_width, 
                num_heads=num_heads, 
                hidden_dim=pars.spect_width,
                attn_drop=pars.drop, 
                drop=pars.drop
            )
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(pars.drop)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pos_embed(x)
        out = self.dropout(out)
        for block in self.blocks:
            out = block(out)
        return torch.mean(out, dim=-1)

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
        self.pars = pars
        self.speaker_encoder = SpeakerEncoder(pars)
        self.audio_encoder = AudioEncoder(pars)
        self.rando_mask = RandoMask(pars.randomask_p_min, pars.randomask_p_max)
        self.audio_decoder = AudioDecoder(pars)

    def forward(self, target_sample, source_history, source_fragment, target_history=None):
        """Used for training, use inference.py for usage"""
        if target_history is None:
            target_history = source_history

        embedding = self.speaker_encoder(target_sample)
        intermediate = self.rando_mask(self.audio_encoder(source_fragment, source_history))
        output = self.audio_decoder(embedding, target_history, intermediate)
        return output
