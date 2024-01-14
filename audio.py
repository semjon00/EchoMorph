import torch
import torchaudio
import torchaudio.transforms as transforms

sample_rate = 32000

if __name__ == '__main__':
    wv, sr = torchaudio.load('./dataset/example.aac')

    resampler = transforms.Resample(sr, sample_rate, dtype=torch.float32)
    transform_to = transforms.Spectrogram(n_fft=512, power=None)
    transform_from = transforms.InverseSpectrogram(n_fft=512)

    wv = resampler(wv)
    sg = transform_to(wv)
    wv_back = transform_from(sg)

    torchaudio.save('./dataset/temp2.wav', wv_back, sample_rate)
