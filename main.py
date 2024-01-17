import torch

from audio import AudioConventer
from model import EchoMorph, EchoMorphParameters
from inference import standard_inference

if __name__ == '__main__':
    # Example of inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ac = AudioConventer(device)
    target_sample = ac.load_audio('./dataset/tests/example1.mp3')
    source = ac.load_audio('./dataset/tests/example2.mp3')
    ts_x = ac.convert_from_wave(target_sample)
    src_x = ac.convert_from_wave(source)

    pars = EchoMorphParameters()
    model = EchoMorph(pars)
    model.eval()

    output = standard_inference(model, ts_x, src_x)
    #output = ts_x

    audio = ac.convert_to_wave(output)
    ac.save_audio(audio, './dataset/result_temp.wav')
