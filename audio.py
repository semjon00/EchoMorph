import torch
import torchaudio
import torchaudio.transforms as transforms

class AudioConventer:
    def __init__(self, x_device):
        self.sample_rate = 32000
        self.n_fft = 510
        self.hop_length = self.n_fft // 5
        self.device = x_device
        self.dtype = torch.float32
        self.transform_to = transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)
        self.transform_from = transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length)

    def total_frames(self, path):
        # Total frames number is the same for stereo and corresponding mono
        return torchaudio.info(path).num_frames

    def load_audio(self, *args, **kwargs):
        wv, sr = torchaudio.load(*args, **kwargs)
        wv = wv.to(self.device, self.dtype).mean(dim=0)  # To correct device, type, and to mono
        r = transforms.Resample(sr, self.sample_rate, dtype=self.dtype)
        return r(wv)

    def convert_from_wave(self, wv):
        sg = self.transform_to(wv).T
        return torch.cat([sg.real, sg.imag], dim=1).to(self.device, self.dtype)

    def convert_to_wave(self, x):
        split_size = x.size(1) // 2
        sg = torch.complex(x[..., :split_size], x[..., split_size:]).T
        wv = self.transform_from(sg)
        return wv

    def save_audio(self, wv, path):
        torchaudio.save(path, wv.unsqueeze(0), self.sample_rate)

    def x_width(self):
        return (self.n_fft // 2 + 1) * 2
