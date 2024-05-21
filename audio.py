import torch
import torchaudio
import torchaudio.transforms as transforms

AUDIO_FORMATS = ['aac', 'mp3', 'flac', 'wav']


class AudioConventer:
    def __init__(self, target_device, precision=torch.float32, sample_rate=24000, width=128, stretch=3):
        self.sample_rate = sample_rate
        self.n_fft = 2 * width - 2
        self.hop_length = self.n_fft // stretch
        # For some reason, resampling on GPU is unbelievably slow,
        # therefore we actually perform the computations on the CPU and send the result to the needed device.
        self.target_device = target_device
        self.target_dtype = precision
        self.transform_to = transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=None).to(target_device)
        self.transform_from = transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length).to(target_device)
        self.log10 = torch.log(torch.tensor(10))

    def total_frames(self, path):
        # Total frames number is the same for stereo and corresponding mono
        return torchaudio.info(path).num_frames

    def load_audio(self, *args, **kwargs):
        degrade_keep = None
        if 'degrade_keep' in kwargs:
            degrade_keep = kwargs['degrade_keep']
            del kwargs['degrade_keep']

        wv, sr = torchaudio.load(*args, **kwargs)
        wv = wv.mean(dim=0)  # To correct device, type, and to mono
        if degrade_keep is not None and degrade_keep < 1.0:
            # Re-sampling frequencies with a small gcd is a pain. 300 is a divider of both 44100 and 48000.
            im_sample_frequency = round(self.sample_rate * degrade_keep) // 300 * 300
            wv = transforms.Resample(sr, im_sample_frequency)(wv)
            wv = transforms.Resample(im_sample_frequency, self.sample_rate)(wv)
        else:
            wv = transforms.Resample(sr, self.sample_rate)(wv)

        wv = wv / max(wv.max(), -wv.min())
        return wv.to(self.target_device, self.target_dtype)

    def convert_from_wave(self, wv):
        """Obtains the spectrogram of the provided waveform."""
        sg = self.transform_to(wv.to(self.target_dtype)).T
        logamp = torch.clamp(torch.abs(sg), min=1e-10, max=1e2).log10()
        logamp = (logamp + 10.0) / 12.0  # Into [0;1]
        logsqamp = logamp ** 2.0
        sg = torch.cat([
            logsqamp.unsqueeze(-1),
            torch.sin(torch.angle(sg)).unsqueeze(-1),
            torch.cos(torch.angle(sg)).unsqueeze(-1)
        ], dim=-1)
        return sg.to(self.target_device, self.target_dtype)

    def convert_to_wave(self, t):
        """Reverses convert_from_wave, output precision is float32"""
        logamp = torch.clamp(t[..., 0], 0.0, 1.0) ** 0.5
        logamp = logamp * 12 - 10.0
        magnitude = torch.clamp((logamp * self.log10).exp(), max=1e2)
        phase = torch.atan2(t[..., 1], t[..., 2])  # Try (t[..., 2], t[..., 1]) for a fun bug

        real_part = magnitude * torch.cos(phase)  # Maybe these two are mixed up. Whatever.
        imag_part = magnitude * torch.sin(phase)
        sg = torch.complex(real_part.to(torch.float32), imag_part.to(torch.float32)).T
        wv = self.transform_from(sg)
        wv /= max(wv.max(), -wv.min())
        wv *= 0.5
        return wv

    def save_audio(self, wv, path):
        torchaudio.save(path, wv.unsqueeze(0).to('cpu'), self.sample_rate)


if __name__ == '__main__':
    print('Audio conversion test.')
    ac = AudioConventer('cpu')
    wv = ac.convert_to_wave(ac.convert_from_wave(ac.load_audio('./dataset/tests/example4.mp3', degrade_keep=1.0)))
    ac.save_audio(wv, './dataset/tests/back.wav')
    print('Please test that the audio has no distortions.')
