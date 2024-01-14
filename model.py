import numpy as np
import torch
from torch import Tensor, nn

class VoicetronParameters:
    sample_rate: int
    randomask_p_min: float
    randomask_p_max: float
    sample_len: int
    history_len: int
    fragment_len: int
    batch_size: int


class SpeakerEncoder(nn.Module):
    def __init__(self, dims: VoicetronParameters):
        super().__init__()


class AudioEncoder(nn.Module):
    def __init__(self, dims: VoicetronParameters):
        super().__init__()


class RandoMask:
    """
    Masks some portion of the input.
    Allows us to use different settings for the quality/similarity tradeoff.
    """
    def __init__(self, p_min, p_max):
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
    def __init__(self, dims: VoicetronParameters):
        super().__init__()


class Voicetron(nn.Module):
    def __init__(self, pars: VoicetronParameters):
        super().__init__()
        self.speaker_encoder = SpeakerEncoder(pars)
        self.audio_encoder = AudioEncoder(pars)
        self.rando_mask = RandoMask(pars.randomask_p_min, pars.randomask_p_max)
        self.audio_decoder = AudioDecoder(pars)

    def forward(self, sample, history, fragment):
        embedding = self.speaker_embedder(sample)
        intermediate = self.audio_encoder(fragment)
        intermediate = self.randomask(intermediate)
        output = self.audio_decoder(embedding, history, intermediate)
        return output
