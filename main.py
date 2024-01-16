import torch

from audio import AudioConventer
from model import Voicetron, VoicetronParameters
from inference import standard_inference

if __name__ == '__main__':
    # Example of inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ac = AudioConventer(device)
    target_sample = ac.load_audio('./dataset/example.aac')
    source = ac.load_audio('./dataset/example2.aac')
    ts_x = ac.convert_from_wave(target_sample)
    src_x = ac.convert_from_wave(source)

    pars = VoicetronParameters()
    model = Voicetron(pars)

    output = standard_inference(model, ts_x, src_x)
    #output = ts_x

    audio = ac.convert_to_wave(output)
    ac.save_audio(audio, './dataset/result_temp.wav')
