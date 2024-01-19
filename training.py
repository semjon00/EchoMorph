import builtins
import datetime
import os
import pathlib
import time

import torch
import random
import pickle
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import einops

from model import EchoMorph, EchoMorphParameters
from audio import AudioConventer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
ac = AudioConventer(device)

batch_size = 128  # Applies to AudioEncoder and AudioDecoder, does not apply to SpeakerEncoder
# TODO: Adjust learning rate
learning_rate = 0.00017  # Universal
save_time = 60 * 60

def print(*args, **kwargs):
    builtins.print(datetime.datetime.now().replace(microsecond=0).isoformat(), *args, **kwargs)


def report(model, consume, avg_loss, avg_loss_origin: pathlib.Path):
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


def get_dataset_files():
    dfiles = list(pathlib.Path("./dataset").rglob("*.*"))
    allowed_extensions = ['.aac', '.mp3', '.flac']
    dfiles = [x for x in dfiles
              if any([x.parts[-1].endswith(ext) for ext in allowed_extensions])
              and x.parts[1] not in ['tests', 'disabled']]
    return dfiles

def load_progress():
    p_snapshots = pathlib.Path("snapshots")
    os.makedirs(p_snapshots, exist_ok=True)
    if len(os.listdir(p_snapshots)) == 0:
        # Initialize new model and fresh dataset consuming progress
        print('  Initializing a new EchoMorph model...')
        pars = EchoMorphParameters()
        model = EchoMorph(pars).to(device)

        print('  Fetching dataset info... ', end='')
        dfiles = get_dataset_files()
        print(f'for {len(dfiles)} new files...')
        consume = [[x, 0, ac.total_frames(x)] for x in dfiles]

        print('  Saving zero progress...')
        save_progress(model, consume)
        print('Training initialized!')
        return model, consume
    else:
        directory = p_snapshots / sorted(os.listdir(p_snapshots))[-1]
        print(f'  Loading an EchoMorph model stored in {directory}...')
        training_parameters = EchoMorphParameters()
        model = EchoMorph(training_parameters)
        model.load_state_dict(torch.load(directory / 'model.bin'))

        consume = pickle.load(open(directory / 'consume.bin', 'rb'))
        print('  Fetching extra info... ', end='')
        new_dfiles = [x for x in get_dataset_files() if x not in [y[0] for y in consume]]
        if len(new_dfiles) > 0:
            print(f'for {len(new_dfiles)} new files...')
            new_dfiles = [[x, 0, ac.total_frames(x)] for x in new_dfiles]
            consume.extend(new_dfiles)
        else:
            print('')
        print('Loading progress done!')
        return model, consume


def save_progress(model, consume):
    p_snapshots = pathlib.Path("snapshots")
    directory = p_snapshots / datetime.datetime.now().replace(microsecond=0).isoformat().strip(':').strip('-')
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), directory / 'model.bin')
    pickle.dump(consume, open(directory / 'consume.bin', 'wb'))
    print('Saved progress.')


def take_a_bite(consume):
    """Randomly selects a file from dataset and takes a bite.
    This thing is slow, but it gets the job done"""
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


def loss_function(pred, truth):
    """Custom loss function, for comparing two spectrograms. Not the best one, but it should work."""
    width = pred.size(-1) // 2

    # This amp code is no more sane than the person who wrote it was when they wrote it
    amp_distance = truth[..., :width] - pred[..., :width]
    # Undershoot = bad; overshoot = veeeery baaaad
    amp_distance = torch.max(amp_distance, 3 * (-amp_distance)) * 12
    if amp_distance.max() > 10:
        amp_distance = amp_distance / (amp_distance.max() / 10.0)

    # Phase part of the spectrogram works like a circle.
    phase_distance = torch.abs(pred[..., width:] - truth[..., width:]) % 2.0
    phase_distance = torch.min(phase_distance, phase_distance * (-1.0) + 2.0)  # Clamp to [0;1], where 1 is the opposite phase
    # Correct phase is not as important as correct amplitude

    # We want to minimize distance squared.
    # Phase is not as important as amplitude
    loss = torch.mean(torch.cat([amp_distance, phase_distance]) ** 2)
    return loss


def train_on_bite(model: EchoMorph, optimizer: torch.optim.Optimizer, train_spect: Tensor):
    tsl = model.pars.target_sample_len
    target_sample = train_spect[0:tsl, :]

    hl = model.pars.history_len
    fl = model.pars.fragment_len
    dataloader = DataLoader(CustomAudioDataset(train_spect[tsl:, ...], hl=hl, fl=fl),
                                      batch_size=batch_size, shuffle=True)

    # TODO: add training in .half() mode
    total_loss = 0
    model.train()
    for history, fragments in iter(dataloader):
        optimizer.zero_grad()
        pred = model(target_sample, history, fragments)
        loss = loss_function(pred, fragments)
        loss.backward()
        # TODO: repeating blocks will have way bigger effective learning rate
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def training():
    print('Training initiated!')

    verify_compatibility()
    print('Compatibility verified.')

    model, consume = load_progress()
    last_save = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while True:
        train_spect, origin = take_a_bite(consume)
        if origin is None:
            break

        avg_loss = train_on_bite(model, optimizer, train_spect)
        report(model, consume, avg_loss, origin[0])
        if last_save < time.time() - last_save:
            last_save = time.time()
            save_progress(model, consume)
    save_progress(model, consume)
    print('Training finished!')


if __name__ == '__main__':
    training()
