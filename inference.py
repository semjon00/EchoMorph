from audio import AudioConventer
from model import EchoMorph
import torch
import os
import pathlib

def crop_from_middle(x, length):
    if x.size(0) < length:
        padding = torch.zeros([(length + 1) // 2, x.size(1)])
        x = torch.cat((padding, x, padding), dim=0)
    start = (x.size(0) - length) // 2
    x = x[start:start + length, ...]
    return x

def standard_inference(model: EchoMorph, target_sample, source):
    """Trivial inference procedure. No interpolating or using the entire target sample"""
    speaker_characteristic = model.speaker_encoder(crop_from_middle(target_sample, model.pars.target_sample_len))

    hl = model.pars.history_len
    fl = model.pars.fragment_len
    # source_length = len(source)

    # Padding the source so we have empty history beforehand and some space after the end
    source = torch.cat((torch.zeros([hl, source.size(1)]), source, torch.zeros([fl, source.size(1)])), dim=0)
    target = torch.zeros_like(source)

    print('Inferencing: [', end='')
    for cur in range(hl, target.size(0), fl):
        intermediate = model.rando_mask(model.audio_encoder(source[cur:cur+fl, :], source[cur-hl:cur, :]))
        target[cur:cur+fl, :] = model.audio_decoder(intermediate, speaker_characteristic, target[cur-hl:cur, :])
        print('.', end='')
    print(']')
    print('Done!')

    # Maybe don't actually truncate the last fragment, what if there is useful signal there?
    return target[hl:, ...]

# TODO: set eval parameters before launching the model (mid blocks repeat, randomask values)

# TODO: allow replacing the target_sample with some nonsense:
#        - Mix two (or more) speaker representations
#        - Allow interpolation across different speaker representations (time-axis)
#        - Add some noise to the representation
#        - Different types of randomized representations
#       it screams class and inheritance (SpeakerRepresentationProvider -style)

# TODO: feeding fake history

def inference_base(sample_path, source_path, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ac = AudioConventer(device)
    target_sample = ac.load_audio(sample_path)
    source = ac.load_audio(source_path)
    ts_x = ac.convert_from_wave(target_sample)
    src_x = ac.convert_from_wave(source)

    root_snapshots = pathlib.Path("snapshots")
    snapshots = sorted(os.listdir(root_snapshots))
    if len(snapshots) < 1:
        print('No model snapshot means no inference is possible.')
        exit(1)

    directory = root_snapshots / snapshots[-1]
    print(f'  Loading an EchoMorph model stored in {directory}...')

    with torch.no_grad():
        model = torch.load(directory / 'model.bin')
        model.eval()
        output = standard_inference(model, ts_x, src_x)
        # output = ts_x

        audio = ac.convert_to_wave(output)
        ac.save_audio(audio, './dataset/result_temp.wav')


if __name__ == '__main__':
    print('=== EchoMorph inference demo ===')

    src = input('Speech file path: ')
    if len(src) < 1:
        src = './dataset/tests/example1.mp3'
        tgt_s = './dataset/tests/example2.mp3'
        save = './dataset/result_temp.wav'
    else:
        tgt_s = input('Speaker file path: ')
        save = input('Save into: ')

    inference_base(src, tgt_s, save)
