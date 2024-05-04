from itertools import chain
import torch
from torch import Tensor, nn
import einops
import pickle
import os

from components import PriorityNoise
from cnn import CNN
from transformer import PlanePositionalEmbedding, Transformer

# TODO: Re-introduce the history

# TODO: Swin
# TODO: Better loss function
# TODO: in CNN, different convolutions should be used for different pitches (use groups parameter)

# TODO: Refactor training parameters into a separate class (don't forget kl_loss!)

# TODO: Do a code self-review, there can be some "fun" surprises!


class EchoMorphParameters:
    """Training parameters"""
    def __init__(self, **kwargs):
        """By default, contains large model specs"""
        one_sec_len = round(24000 / 84 / 64) * 64  # sample_rate / hop_length; approximately
        self.target_sample_len = one_sec_len // 2
        self.history_len = one_sec_len // 2
        self.fragment_len = one_sec_len // 8
        assert self.target_sample_len == self.history_len, "oh no! - speaker encoding is TODO"

        self.spect_width = 128  # x_width
        self.length_of_patch = 8

        self.embed_dim = 128

        self.se_convrec = (2, 8, 32, 64)
        self.se_convrepeat = 6
        self.se_blocks = (10, 0, 0)
        self.se_heads = 4
        self.se_hidden_dim = 4 * self.embed_dim
        self.se_output_tokens = 256
        self.se_kl_loss_k = 0.003

        self.ae_convrec = (2, 8, 16, 32)
        self.ae_convrepeat = 4
        self.ae_blocks = (6, 0, 0)
        self.ae_heads = 4
        self.ae_hidden_dim = 3 * self.embed_dim

        self.ad_blocks = (16, 0, 0)
        self.ad_heads = 8
        self.ad_hidden_dim = 6 * self.embed_dim

        self.drop = 0.00
        self.rm_k_min = 0.0
        self.rm_k_max = 1.0
        self.rm_fun = 'lin'
        self.mid_repeat_interval = (2, 5)  # (inclusive, exclusive)

        for key, value in kwargs.items():
            setattr(self, key, value)


class SpeakerVAE(nn.Module):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__()

        self.cnn = CNN(pars.se_convrec, pars.se_convrepeat)
        self.transformer = Transformer(embed_dim=pars.embed_dim, mlp_hidden_dim=pars.se_hidden_dim, heads=pars.se_heads,
                                       drop=pars.drop, blocks_num=pars.se_blocks, cross_n=0, mid_repeat_interval=(0, 1))
        self.entok = nn.Linear(pars.se_convrec[-1], pars.embed_dim)
        reduction = 2 ** (len(pars.se_convrec) - 1)
        self.pos_embed = PlanePositionalEmbedding(
            pars.history_len // reduction, pars.spect_width // reduction, pars.embed_dim
        )
        self.out_tokens = pars.se_output_tokens
        self.mean_linear = nn.Linear(pars.embed_dim, pars.embed_dim)
        self.log_var_linear = nn.Linear(pars.embed_dim, pars.embed_dim)

    def forward_shared(self, x: Tensor) -> (Tensor, Tensor):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        x = self.entok(x)
        x = self.pos_embed(x)
        x = einops.rearrange(x, '... l w d -> ... (l w) d')
        x = self.transformer(x, [])

        ret = x[..., :self.out_tokens, :]
        assert ret.size(-2) == self.out_tokens
        return ret

    def forward_train(self, x):
        ret_tok = self.forward_shared(x)
        means = self.mean_linear(ret_tok)
        log_vars = self.log_var_linear(ret_tok)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_vars) + means ** 2 - log_vars - 1, dim=-1))

        epsilon = torch.randn_like(means)
        std = torch.exp(0.5 * log_vars)
        z = std * epsilon + means
        return z, kl_loss

    def forward_use(self, x):
        return self.mean_linear(self.forward_shared(x))


class AudioEncoder(nn.Module):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__()

        self.cnn = CNN(pars.ae_convrec, pars.ae_convrepeat)
        self.transformer = Transformer(embed_dim=pars.embed_dim, mlp_hidden_dim=pars.ae_hidden_dim, heads=pars.ae_heads,
                                       drop=pars.drop, blocks_num=pars.ae_blocks, cross_n=0,
                                       mid_repeat_interval=pars.mid_repeat_interval)
        self.entok = nn.Linear(pars.ae_convrec[-1], pars.embed_dim)
        reduction = 2 ** (len(pars.ae_convrec) - 1)
        self.pos_embed = PlanePositionalEmbedding(
            pars.fragment_len // reduction, pars.spect_width // reduction, pars.embed_dim
        )

    def forward(self, x: Tensor, mid_rep=None) -> Tensor:
        x = self.cnn(x)
        x = self.entok(x)
        x = self.pos_embed(x)
        x = einops.rearrange(x, '... l w d -> ... (l w) d')
        x = self.transformer.forward(x, [], mid_rep=mid_rep)
        return x


class AudioDecoder(Transformer):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(embed_dim=pars.embed_dim, mlp_hidden_dim=pars.ad_hidden_dim, heads=pars.ad_heads,
                         drop=pars.drop, blocks_num=pars.ad_blocks, cross_n=2,
                         mid_repeat_interval=pars.mid_repeat_interval)

        self.spect_width = pars.spect_width
        self.fragment_len = pars.fragment_len
        self.length_of_patch = pars.length_of_patch
        self.embed_dim = pars.embed_dim

        self.detok = nn.Linear(pars.embed_dim, 2 * pars.length_of_patch)
        self.pos_embed = PlanePositionalEmbedding(
            self.fragment_len // self.length_of_patch, self.spect_width, self.embed_dim
        )

    def forward(self, im: Tensor, sc: Tensor, mid_rep=None) -> Tensor:
        dims = [self.fragment_len // self.length_of_patch, self.spect_width, self.embed_dim]
        if len(im.size()) > 2:
            dims = [im.size(0)] + dims
        feed = self.pos_embed(torch.zeros(dims, dtype=im.dtype, device=im.device))
        feed = einops.rearrange(feed, '... l w d -> ... (l w) d')
        x = super().forward(feed, [im, sc], mid_rep=mid_rep)

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
        intermediate = self.audio_encoder(source_fragment, mid_rep=middle_repeats)
        intermediate = self.bottleneck(intermediate)
        output = self.audio_decoder(intermediate, speaker_characteristic, middle_repeats)
        extra_loss = self.pars.se_kl_loss_k * se_loss
        return output, extra_loss

    def get_multiplication_parameters(self):
        return chain.from_iterable([
            m.blocks_mid.parameters() for m in [self.speaker_encoder.transformer,
                                                self.audio_encoder.transformer,
                                                self.audio_decoder]
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
