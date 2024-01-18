import builtins
import datetime
import os
import pathlib
import torch
import random
import pickle
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import einops

from model import EchoMorph, EchoMorphParameters
from audio import AudioConventer

# TODO: not optimized at all

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
ac = AudioConventer(device)

batch_size = 256  # Applies to AudioEncoder and AudioDecoder, does not apply to SpeakerEncoder


def print(*args):
    builtins.print(datetime.datetime.now().replace(microsecond=0).isoformat(), *args)


def report(model, consume, avg_loss, avg_loss_origin: pathlib.Path):
    # TODO: Isn't it too slow?
    sum_consume = consume([el[1] for el in consume])
    tot_consume = consume([el[2] for el in consume])
    percent_consumed = 100 * sum_consume / tot_consume
    fn_string = f'{avg_loss_origin.parts[-2]}/{avg_loss_origin.parts[-1]}'
    print(f'Report | {percent_consumed:2.3f}% | {avg_loss:3.3f} loss on {fn_string}')


def verify_compatibility():
    f = 'NONE'
    try:
        tests_dir = pathlib.Path('./dataset/tests')
        for f in os.listdir(tests_dir):
            ac.convert_to_wave(ac.convert_from_wave(ac.load_audio(tests_dir / f)))
    except:
        print(f'Compatibility check FAILED on file {f}')
        print('Please ensure that ffmpeg (and maybe sox) are installed - '
              'these are necessary to read audio files')
        exit(1)


def load_progress():
    # TODO: initialize or load a saved model
    p_snapshots = pathlib.Path("snapshots")
    os.makedirs(p_snapshots, exist_ok=True)
    if len(os.listdir(p_snapshots)) == 0:
        # Initialize new model and fresh dataset consuming progress
        print('  Initializing a new EchoMorph model...')
        pars = EchoMorphParameters()
        model = EchoMorph(pars)

        print('  Fetching dataset info...')
        dfiles = list(pathlib.Path("./dataset").rglob("*.*"))
        allowed_extensions = ['.aac', '.mp3', '.flac']
        dfiles = [x for x in dfiles
                  if any([x.parts[-1].endswith(ext) for ext in allowed_extensions])
                  and x.parts[1] not in ['tests', 'disabled']]
        consume = [[x, 0, ac.total_frames(x)] for x in dfiles]
        print('  Saving zero progress...')
        save_progress(model, consume)
        print('Training initialized!')
        return model, consume
    else:
        # TODO: does not allow new files in consume
        directory = p_snapshots / sorted(os.listdir(p_snapshots))[-1]
        print(f'  Loading an EchoMorph model stored in {directory}...')
        model = torch.load(directory / 'model.bin')
        consume = pickle.load(open(directory / 'consume.bin', 'rb'))
        print('Loading progress done!')
        return model, consume


def save_progress(model, consume):
    p_snapshots = pathlib.Path("snapshots")
    directory = p_snapshots / datetime.datetime.now().replace(microsecond=0).isoformat().strip(':').strip('-')
    os.makedirs(directory, exist_ok=True)
    torch.save(model, directory / 'model.bin')
    pickle.dump(consume, open(directory / 'consume.bin', 'wb'))
    print('Saved progress.')


def take_a_bite(consume):
    """Randomly selects a file from dataset and takes a bite.
    This thing is slow, but it gets the job done"""
    # TODO: Use DataLoader
    # TODO: Custom priority

    load_opt = 45678 * 300  # About 5 minutes, don't care about the bitrate and the exact value

    tot_rem = sum([el[2] - el[1] for el in consume])
    if tot_rem == 0:
        return None, None

    drop = random.randint(0, tot_rem - 1)
    sel = 0
    for i, el in enumerate(consume):
        if drop < el[2] - el[1]:
            sel = i
            break
        drop -= el[2] - el[1]
    load_now = load_opt if consume[sel][2] - consume[sel][1] > 2 * load_opt else consume[sel][2]
    loaded = ac.load_audio(consume[sel][0], frame_offset=consume[sel][1], num_frames=load_now)
    consume[sel][1] += load_now
    return ac.convert_from_wave(loaded), consume[sel]


class CustomAudioDataset(Dataset):
    def __init__(self, train_spect, hl, fl):
        assert hl % fl == 0, 'Not implemented'
        chunks_n = train_spect.size(0) // fl
        train_spect = train_spect[:chunks_n * fl, :]

        chunks = einops.rearrange(train_spect, '(s x) w -> s x w', x=fl)
        fragments_i = torch.arange(hl // fl, chunks.size(0))
        self.fragments = chunks[fragments_i, ...]
        history_i = torch.arange(0, chunks.size(0) - hl // fl).view(-1, 1) + torch.arange(hl // fl)
        self.history = einops.rearrange(chunks[history_i, ...], '... x s w -> ... (x s) w')

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        return self.history[idx], self.fragments[idx]


def loss_function(pred, truth):  # TODO: now this is the hard part
    return 0


def train_on_bite(model: EchoMorph, optimizer: torch.optim.Optimizer, train_spect: Tensor):
    tsl = model.pars.target_sample_len
    target_sample = train_spect[0:tsl, :]

    hl = model.pars.history_len
    fl = model.pars.fragment_len
    dataloader = DataLoader(CustomAudioDataset(train_spect[tsl:, ...], hl=hl, fl=fl),
                                      batch_size=batch_size, shuffle=True)

    total_loss = 0
    model.train()
    for history, fragments in iter(dataloader):
        optimizer.zero_grad()
        pred = model(target_sample, history, fragments)
        loss = loss_function(pred, fragments)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def training():
    print('Training initiated!')

    verify_compatibility()
    print('Compatibility verified.')

    model, consume = load_progress()

    # TODO: Mess with the gradient application here
    # TODO: Adjust learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00017)

    while True:
        train_spect, origin = take_a_bite(consume)
        if origin is None:
            break

        avg_loss = train_on_bite(model, optimizer, train_spect)
        report(model, consume, avg_loss, origin[0])

        # TODO: Save occasionally
    save_progress(model, consume)
    print('Training finished! ')


if __name__ == '__main__':
    training()
