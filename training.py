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
from audio import AudioConventer, AUDIO_FORMATS

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
precision = torch.float32 if device == "cpu" else torch.float16
print(f"Using {device} device with {precision} precision")
ac = AudioConventer(device, precision)

batch_size = 32  # Applies to AudioEncoder and AudioDecoder, does not apply to SpeakerEncoder
# TODO: Adjust learning rate
learning_rate = 0.0002  # Universal
save_time = 60 * 60


def print(*args, **kwargs):
    builtins.print(datetime.datetime.now().replace(microsecond=0).isoformat(), *args, **kwargs)


class ConsumeProgress:
    def __init__(self, names_and_durations, starting_epoch=0):
        self.starting_epoch = starting_epoch
        self.paths, self.consumed, self.durations = [], [], []

        self.consumed_duration = 0
        self.total_duration = 0
        self.add_files(names_and_durations)

    def consumed_prop(self):
        return self.consumed_duration / self.total_duration

    def add_files(self, names_and_durations):
        self.paths += [x[0] for x in names_and_durations]

        new_durations = [x[1] for x in names_and_durations]
        self.durations.extend(new_durations)
        self.total_duration += sum(new_durations)

        new_consumed = [13 * self.starting_epoch for _ in range(len(names_and_durations))]
        self.consumed.extend(new_consumed)
        self.consumed_duration = sum(new_consumed)

    def lottery_idx(self):
        if self.consumed_duration == self.total_duration:
            return None

        tot_rem = self.total_duration - self.consumed_duration
        drop = random.randint(0, tot_rem - 1)
        sel = None
        for i in range(len(self.paths)):
            rem = self.durations[i] - self.consumed[i]
            if drop < rem:
                sel = i
                break
            drop -= rem
        return sel

    def bite(self, idx, max_duration):
        start = self.consumed[idx]
        rem = (self.durations[idx] - start) / max_duration
        if rem > 4:
            end = start + max_duration
        elif rem < 1:
            end = self.durations[idx]
        else:
            times = int(rem + 1)  # How many bites left to do, does ceiling(rem)
            end = start + (self.durations[idx] - start) // times
        self.consumed[idx] = end
        self.consumed_duration += end - start
        return self.paths[idx], start, end

    def forget(self):
        self.consumed_duration = 0
        self.consumed = [0 for _ in range(len(self.consumed))]


def report(optimizer, consume, avg_loss, avg_loss_origin: pathlib.Path):
    percent_consumed = 100 * consume.consumed_prop()
    current_lr = optimizer.param_groups[0]['lr']
    fn_string = f'{avg_loss_origin.parts[-2]}/{avg_loss_origin.parts[-1]}'
    print(f'Report | {percent_consumed:2.3f}% | lr {1e6*current_lr:3.2f}q | {avg_loss:3.5f} loss on "{fn_string}"')


def upd_timings(timings, name, start_time):
    if name not in timings:
        timings[name] = 0
    timings[name] += time.time() - start_time


def verify_compatibility():
    tests_dir = pathlib.Path('./dataset/tests')
    if not tests_dir.is_dir():
        print('!! Tests directory does not exist, compatibility testing was not performed')
        return

    f = 'NONE'
    try:
        for f in os.listdir(tests_dir):
            ac.convert_to_wave(ac.convert_from_wave(ac.load_audio(tests_dir / f)))
    except:
        print(f'Compatibility check FAILED on file {f}')
        print('Please ensure that ffmpeg (and maybe sox) are installed - '
              'these are necessary to read audio files')
        exit(1)


def get_dataset_paths():
    dfiles = list(pathlib.Path("./dataset").rglob("*.*"))
    allowed_extensions = [f'.{x}' for x in AUDIO_FORMATS]
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
        model = EchoMorph(pars).to(device=device, dtype=precision)

        print('  Fetching dataset info... ', end='')
        dpaths = get_dataset_paths()
        print(f'for {len(dpaths)} new files...')

        consume = ConsumeProgress([[x, ac.total_frames(x)] for x in dpaths])

        print('  Saving zero progress...')
        save_progress(model, consume)
        print('Training initialized!')
        return model, consume
    else:
        directory = p_snapshots / sorted(os.listdir(p_snapshots))[-1]
        print(f'  Loading an EchoMorph model stored in {directory}...')
        training_parameters = EchoMorphParameters()
        model = EchoMorph(training_parameters).to(device=device, dtype=precision)
        model.load_state_dict(torch.load(directory / 'model.bin'))

        consume: ConsumeProgress = pickle.load(open(directory / 'consume.bin', 'rb'))
        print('  Fetching extra info... ', end='')

        new_dpaths = [x for x in get_dataset_paths() if x not in consume.paths]
        if len(new_dpaths) > 0:
            print(f'for {len(new_dpaths)} new files...')
            consume.add_files([[x, ac.total_frames(x)] for x in new_dpaths])
        else:
            print('')
        print('Loading progress done!')
        return model, consume


def save_progress(model, consume):
    p_snapshots = pathlib.Path("snapshots")
    directory = p_snapshots / datetime.datetime.now().replace(microsecond=0).isoformat().replace(':', '.')
    os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), directory / 'model.bin')
    pickle.dump(consume, open(directory / 'consume.bin', 'wb'))
    print('Saved progress.')


def random_degradation_value():
    # Eyeballed
    r = random.random()
    return min((r ** 1.5) + 0.2, 1.000001)


def take_a_bite(consume: ConsumeProgress):
    """Randomly selects a file from dataset and takes a bite."""
    sel = consume.lottery_idx()
    if sel is None:
        return None, None

    # About 5 minutes, don't care about the bitrate and the exact value
    cap = 45678 * 300
    path, start, end = consume.bite(sel, cap)
    loaded = ac.load_audio(path, frame_offset=start, num_frames=end - start,
                           degrade_keep=random_degradation_value())
    sg = ac.convert_from_wave(loaded)
    return sg, path


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


loss_function_freq_significance_cache = None
def loss_function_freq_significance(width, device):
    global loss_function_freq_significance_cache
    if loss_function_freq_significance_cache is None or loss_function_freq_significance_cache[0] != width:
        vals = torch.arange(start=2.0, end=0, step=-2.0 / width, device=device).exp()
        vals = vals / torch.sum(vals) * width
        loss_function_freq_significance_cache = width, vals
    if loss_function_freq_significance_cache[1].device != device:
        loss_function_freq_significance_cache = width, loss_function_freq_significance_cache[1].to(device)
    return loss_function_freq_significance_cache[1]


def loss_function(pred, truth):
    """Custom loss function, for comparing two spectrograms. Not the best one, but it should work."""
    # TODO: this can be infinitely improved:
    #  * equal-loudness contour
    #  * auditory masking
    #  * jitter in neighbor values across time-domain
    #  * slightly different pitch is not too bad
    width = pred.size(-1) // 2

    # This amp code is no more sane than the person who wrote it was when they wrote it
    # Pretty much all the things are eye-balled and not rigorously determined
    amp_distance = truth[..., :width] - pred[..., :width]
    # Undershoot = bad; overshoot = veeeery baaaad
    # Large overshoot = "masked"
    amp_distance = torch.max(torch.clamp(amp_distance, max=0.25), 3.0 * (-amp_distance)) * 12
    # Frequency ranges are not created equal
    amp_distance = torch.clamp(amp_distance, max=10)
    amp_distance *= loss_function_freq_significance(width, amp_distance.device)

    # Phase part of the spectrogram works like a circle.
    phase_distance = torch.abs(pred[..., width:] - truth[..., width:]) % 2.0
    # Clamp to [0;1], where 1 is the opposite phase
    phase_distance = torch.min(phase_distance, phase_distance * (-1.0) + 2.0)
    phase_distance *= loss_function_freq_significance(width, amp_distance.device)
    # Correct phase is not as important as correct amplitude

    # We want to minimize distance squared.
    # Phase is not as important as amplitude
    loss = torch.mean(torch.cat([amp_distance, phase_distance]) ** 2)
    return loss


def train_on_bite(model: EchoMorph, optimizer: torch.optim.Optimizer, scheduler, train_spect: Tensor, timings):
    tsl = model.pars.target_sample_len
    target_sample = train_spect[0:tsl, :]

    hl = model.pars.history_len
    fl = model.pars.fragment_len
    bt = time.time()
    dataloader = DataLoader(CustomAudioDataset(train_spect[tsl:, ...], hl=hl, fl=fl),
                                      batch_size=batch_size, shuffle=True)
    upd_timings(timings, 'dataloading', bt)

    bt = time.time()
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
    mean_loss = total_loss / len(dataloader)
    scheduler.step(mean_loss)
    upd_timings(timings, 'training', bt)
    return mean_loss


def training():
    print('Training initiated!')

    verify_compatibility()
    print('Compatibility verified.')

    model, consume = load_progress()
    last_save = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=10e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75)
    timings = {}
    while True:
        bt = time.time()
        train_spect, origin = take_a_bite(consume)
        upd_timings(timings, 'loading', bt)
        if origin is None:
            break

        avg_loss = train_on_bite(model, optimizer, scheduler, train_spect, timings)
        report(optimizer, consume, avg_loss, origin)
        if last_save < time.time() - last_save:
            last_save = time.time()
            save_progress(model, consume)
            print(f'Timings: {timings}')
    print(f'Timings: {timings}')
    save_progress(model, consume)
    print('Training finished!')


if __name__ == '__main__':
    training()
