import torch
from torch import Tensor, nn
import einops
import pickle
import os

from components import PriorityNoise
from cnn import CNN
from transformer import Transformer

# TODO: Re-introduce the history
# TODO: Better loss function
# TODO: in CNN, different convolutions should be used for different pitches (use groups parameter)

# TODO: Refactor training parameters into a separate class (don't forget kl_loss!)


class EchoMorphParameters:
    """Training parameters"""
    def __init__(self, **kwargs):
        """By default, contains large model specs"""
        one_sec_len = round(24000 / 84 / 64) * 64  # sample_rate / hop_length; approximately
        self.target_sample_len = one_sec_len // 32
        self.history_len = one_sec_len // 32
        self.fragment_len = one_sec_len // 32
        assert self.target_sample_len == self.history_len, "oh no! - speaker encoding is TODO"

        self.spect_width = 128
        self.length_of_patch = 8

        self.embed_dim = 128
        self.bottleneck_dim = 32

        self.se_convrec = (2, 8)
        self.se_convrepeat = 6
        self.se_blocks = 4
        self.se_output_tokens = 1024

        self.ae_convrec = (2, 8)
        self.ae_convrepeat = 4
        self.ae_blocks = 6

        self.rs_blocks = 6
        self.ad_blocks = 12

        self.rm_k_min = 1.0
        self.rm_k_max = 1.0
        self.rm_fun = 'lin'
        self.se_kl_loss_k = 0.000

        for key, value in kwargs.items():
            setattr(self, key, value)


class SpeakerVAE(nn.Module):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__()

        self.cnn = CNN(pars.se_convrec, pars.se_convrepeat)
        reduction = self.cnn.res_reduction_factor()
        self.transformer = Transformer(
            input_dim=self.cnn.out_channels, output_dim=pars.embed_dim,
            input_size=(pars.history_len // reduction, pars.spect_width // reduction),
            num_blocks=pars.se_blocks, embed_dim=pars.embed_dim, cross_n=0,
            rearrange_back=False
        )

        self.out_tokens = pars.se_output_tokens
        self.mean_linear = nn.Linear(pars.embed_dim, pars.embed_dim)
        self.log_var_linear = nn.Linear(pars.embed_dim, pars.embed_dim)

    def forward_shared(self, x: Tensor) -> (Tensor, Tensor):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        x = self.transformer(x, [])

        ret = x[..., :self.out_tokens, :]
        assert ret.size(-2) == self.out_tokens, f"{ret.size(-2) = } {self.out_tokens = }"
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
        reduction = self.cnn.res_reduction_factor()
        self.transformer = Transformer(
            input_dim=self.cnn.out_channels, output_dim=pars.bottleneck_dim,
            input_size=(pars.fragment_len // reduction, pars.spect_width // reduction),
            num_blocks=pars.ae_blocks, embed_dim=pars.embed_dim, cross_n=0,
            rearrange_back=False
        )
        self.num_output_tokens = self.transformer.input_size[0] * self.transformer.input_size[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.transformer(x, [])
        return x


class AudioDecoder(Transformer):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__(input_dim=1, output_dim=2 * pars.length_of_patch,
                         input_size=(pars.fragment_len // pars.length_of_patch, pars.spect_width),
                         num_blocks=pars.ad_blocks, embed_dim=pars.embed_dim, cross_n=2)

        self.spect_width = pars.spect_width
        self.fragment_len = pars.fragment_len
        self.length_of_patch = pars.length_of_patch
        self.embed_dim = pars.embed_dim

    def forward(self, im: Tensor, sc: Tensor) -> Tensor:
        dims = [self.fragment_len // self.length_of_patch, self.spect_width, 1]
        if len(im.size()) > 2:
            dims = [im.size(0)] + dims
        feed = torch.zeros(dims, dtype=im.dtype, device=im.device)
        x = super().forward(feed, [im, sc])
        x = einops.rearrange(x, ' ... l w (c ld) -> ... (l ld) w c', ld=self.length_of_patch)
        return x


class EchoMorph(nn.Module):
    def __init__(self, pars: EchoMorphParameters):
        super().__init__()
        self.pars = pars
        self.speaker_encoder = SpeakerVAE(pars)
        self.audio_encoder = AudioEncoder(pars)
        self.bottleneck = PriorityNoise(
            pars.rm_k_min, pars.rm_k_max, pars.rm_fun, pars.bottleneck_dim
        )
        self.restorer = Transformer(pars.bottleneck_dim, pars.embed_dim,
                                    (self.audio_encoder.num_output_tokens, ),
                                    pars.rs_blocks, pars.embed_dim, 0)
        self.audio_decoder = AudioDecoder(pars)

    def forward(self, target_sample, source_fragment):
        """Used for training, use inference.py for inference"""
        speaker_characteristic, se_loss = self.speaker_encoder.forward_train(target_sample)
        intermediate = self.audio_encoder(source_fragment)
        intermediate = self.restorer(self.bottleneck(intermediate))
        output = self.audio_decoder(intermediate, speaker_characteristic)
        extra_loss = self.pars.se_kl_loss_k * se_loss
        return output, extra_loss


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
