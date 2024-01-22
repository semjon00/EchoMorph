import numpy
import torch
import torchaudio
import torchaudio.transforms as T

AUDIO_FORMATS = ['aac', 'mp3', 'flac', 'wav']


class AudioConventer:
    def __init__(self, target_device, precision=torch.float32, sample_rate=32000, width=80, stretch=2):
        self.sample_rate = sample_rate
        self.n_fft = width
        self.hop_length = self.n_fft // stretch
        # For some reason, resampling on GPU is unbeliveably slow,
        # therefore we actually perform the computations on the CPU and send the result to the needed device.
        self.target_device = target_device
        self.target_dtype = precision
        self.mel_transform = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=4096, n_mels=width, hop_length=self.hop_length)
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
            wv = T.Resample(sr, im_sample_frequency)(wv)
            wv = T.Resample(im_sample_frequency, self.sample_rate)(wv)
        else:
            wv = T.Resample(sr, self.sample_rate)(wv)

        wv = wv / max(wv.max(), -wv.min())
        return wv

    def convert_from_wave(self, wv):
        import torchaudio.transforms as T
        # v = T.GriffinLim(n_fft=2048)(T.Spectrogram(n_fft=2048, power=2)(wv))
        mel_spec = self.mel_transform(wv)
        logmel = numpy.log10(numpy.clip(mel_spec, a_min=1.01e-10, a_max=1e2))
        logmel = (logmel + 10) / 12
        return torch.Tensor(logmel).to(self.target_device, self.target_dtype)

    def convert_to_wave(self, x):
        """Reverses convert_from_wave, output precision is float32"""
        x = x * 12 - 10
        sg = torch.clamp((x * self.log10).exp(), max=100.0).cuda()

        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        vocoder = bundle.get_vocoder().cpu()

        with torch.inference_mode():
            wv, lengths = vocoder(sg.unsqueeze(0).float().cpu(), torch.Tensor([sg.size(1)]))
            wv /= 1.5 * max(wv.max(), -wv.min())
            return torch.Tensor(wv).squeeze(0).to(self.target_device, self.target_dtype)

    def save_audio(self, wv, path):
        torchaudio.save(path, wv.unsqueeze(0).to('cpu'), self.sample_rate)


if __name__ == '__main__':
    print('Audio conversion test.')
    ac = AudioConventer('cuda', torch.float16)
    wv = ac.convert_to_wave(ac.convert_from_wave(ac.load_audio('./dataset/tests/example4.mp3')))
    ac.save_audio(wv, './dataset/tests/back.wav')
    print('Please test that the audio has no distortions.')
