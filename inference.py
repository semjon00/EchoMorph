import torch

from model import EchoMorph

# TODO: training might need stuff like this, maybe extract to a separate file
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
#        - Allow interpolation across different speaker representations
#        - Different types of randomized representations
