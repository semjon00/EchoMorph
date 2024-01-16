import torch

from audio import AudioConventer
from model import Voicetron, VoicetronParameters

if __name__ == '__main__':
    # Example of inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ac = AudioConventer(device)
    audio = ac.load_audio('./dataset/example.aac')
    x = ac.convert_from_wave(audio)

    pars = VoicetronParameters()
    model = Voicetron(pars)

    # Just for testing purposes, we select sample, history and fragment from the same file, consecutively,
    # from the start of the file. In the real-world, sample will be in the separate file,
    # model should be called many times, for all the fragments.
    t = [0]
    for n in (pars.target_sample_len, pars.history_len, pars.fragment_len):
        t += [t[-1] + n]
    out = model(x[t[0]:t[1], :], x[t[1]:t[2], :], x[t[2]:t[3], :])

    audio = ac.convert_to_wave(out)
    ac.save_audio(audio, './dataset/temp2.wav')
