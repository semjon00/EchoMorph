import torch
import torchaudio
import torchaudio.transforms as transforms

class AudioConventer:
    def __init__(self, x_device, sample_rate=32000, width=512, stretch=5):
        self.sample_rate = sample_rate
        self.n_fft = width - 2
        self.hop_length = self.n_fft // stretch
        self.device = x_device
        self.dtype = torch.float32
        self.transform_to = transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None)
        self.transform_from = transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        self.log10 = torch.log(torch.tensor(10))

    def total_frames(self, path):
        # Total frames number is the same for stereo and corresponding mono
        return torchaudio.info(path).num_frames

    def load_audio(self, *args, **kwargs):
        wv, sr = torchaudio.load(*args, **kwargs)
        wv = wv.to(self.device, self.dtype).mean(dim=0)  # To correct device, type, and to mono
        r = transforms.Resample(sr, self.sample_rate, dtype=self.dtype)(wv)
        r = r / max(r.max(), -r.min())
        return r

    def convert_from_wave(self, wv):
        """
        Obtains the spectrogram of the provided waveform.
        Then, re-encodes the spectrogram as a stack
        of normalized log-amplitudes and phases in interval (from -1 to +1)
        """
        sg = self.transform_to(wv).T
        logamp = torch.clamp(torch.abs(sg), min=1e-10, max=1e2).log10()
        logamp = (logamp + 10) / 12
        phase = torch.angle(sg) / torch.pi
        sg = torch.cat([logamp, phase], dim=1).to(self.device, self.dtype)
        return sg

    def convert_to_wave(self, x):
        """Reverses convert_from_wave"""
        split_size = x.size(1) // 2
        magnitude = x[..., :split_size] * 12 - 10
        magnitude = (magnitude * self.log10).exp()
        phase = x[..., split_size:] * torch.pi

        real_part = magnitude * torch.cos(phase)
        imag_part = magnitude * torch.sin(phase)
        sg = torch.complex(real_part, imag_part).T
        wv = self.transform_from(sg)
        return wv

    def save_audio(self, wv, path):
        torchaudio.save(path, wv.unsqueeze(0), self.sample_rate)

    def x_width(self):
        return (self.n_fft // 2 + 1) * 2

if __name__ == '__main__':
    print('Audio conversion test.')
    ac = AudioConventer('cpu')
    wv = ac.convert_to_wave(ac.convert_from_wave(ac.load_audio('./dataset/tests/example4.mp3')))
    ac.save_audio(wv, './dataset/tests/back.wav')
    print('Please test that the audio has no distortions.')
