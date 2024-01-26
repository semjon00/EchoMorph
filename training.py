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
import sys

from model import EchoMorph, EchoMorphParameters, save_model, load_model
from audio import AudioConventer, AUDIO_FORMATS

import argparse
parser = argparse.ArgumentParser(description='Training routine')
parser.add_argument('--total_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--save_time', type=int, default=60 * 60)
parser.add_argument('--baby_parameters', action='store_const', const=True, default=False)
parser.add_argument('--use_dumb_loss_function', action='store_const', const=True, default=False)
parser.add_argument('--no_random_degradation', action='store_const', const=True, default=False)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
precision = torch.float32  # fp16 is a scam, it does not converge! I repeat, do not try fp16!
print(f"Using {device} device with {precision} precision")
ac = AudioConventer(device, precision)


def print(*args, **kwargs):
    kwargs['flush'] = True
    builtins.print(datetime.datetime.now().replace(microsecond=0).isoformat(), *args, **kwargs)
    sys.stdout.flush()


def print_cuda_stats():
    if str(device) == "cpu":
        return
    try:
        vals = torch.cuda.mem_get_info()
        print(f'Cuda memory availability: {str(vals)}')
    except:
        print('Failed to display cuda memory availability.')

class ConsumeProgress:
    def __init__(self, names_and_durations):
        self.epoch = 1
        self.total_epochs = args.total_epochs
        self.paths, self.consumed, self.durations = [], [], []

        self.total_consumed = 0
        self.total_durations = 0
        self.add_files(names_and_durations)

    def consumed_prop(self):
        return self.epoch + self.total_consumed / self.total_durations

    def add_files(self, names_and_durations):
        self.paths += [x[0] for x in names_and_durations]

        new_durations = [x[1] for x in names_and_durations]
        self.durations.extend(new_durations)
        self.total_durations = sum(self.durations)

        new_consumed = [(13 * self.epoch) % 256 for _ in range(len(names_and_durations))]
        self.consumed.extend(new_consumed)
        self.total_consumed = sum(self.consumed)

    def check_presence(self):
        """Checks presence of all the dataset files. Excludes the files that are not present."""
        remd = []
        for i, path in enumerate(self.paths):
            if not path.is_file():
                remd += [i]
        remd = set(remd)

        if len(remd):
            self.paths = [ele for idx, ele in enumerate(self.paths) if idx not in remd]
            self.consumed = [ele for idx, ele in enumerate(self.consumed) if idx not in remd]
            self.durations = [ele for idx, ele in enumerate(self.durations) if idx not in remd]
            print(f'Removed {len(remd)} files from the dataset')


    def lottery_idx(self):
        if self.total_consumed == self.total_durations:
            if self.epoch_rollover():
                return self.lottery_idx()
            else:
                return None

        tot_rem = self.total_durations - self.total_consumed
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
        self.total_consumed += end - start
        return self.paths[idx], start, end

    def epoch_rollover(self):
        if self.epoch == self.total_epochs:
            return False
        self.epoch += 1
        self.consumed = [13 * self.epoch for _ in range(len(self.consumed))]
        self.total_consumed = sum(self.consumed)
        print(f'As the epoch took a bow, the stage was set for a grand sequel! '
              f'"Epoch {self.epoch}, the return of epoch {self.epoch - 1}"')
        return True


def report(optimizer, consume, avg_loss, avg_loss_origin: pathlib.Path):
    percent_consumed = 100 * consume.consumed_prop()
    current_lr = optimizer.param_groups[0]['lr']
    fn_string = f'{avg_loss_origin.parts[-2]}/{avg_loss_origin.parts[-1]}'
    avg_loss = "\u221E" if avg_loss is None else f'{avg_loss:03.5f}'
    print(f'Report | {percent_consumed:02.3f}% | lr {1e6*current_lr:03.3f}q | {avg_loss} loss on "{fn_string}"')


def upd_timings(timings, name, start_time):
    if name not in timings:
        timings[name] = 0
    timings[name] += time.time() - start_time


def verify_compatibility():
    tests_dir = pathlib.Path('./dataset/tests')
    if not tests_dir.is_dir():
        print('! Tests directory does not exist, compatibility testing was not performed')
        return

    f = 'NONE'
    try:
        for f in os.listdir(tests_dir):
            if f.endswith('.gitkeep'):
                continue
            ac.convert_to_wave(ac.convert_from_wave(ac.load_audio(tests_dir / f)))
    except:
        print(f'Compatibility check FAILED on file {f}')
        print('Please ensure that ffmpeg (and maybe sox) are installed - these are necessary for reading audio files.')
        print('No training can be done if audio files can not be read.')
        exit(1)
    print('Compatibility verified.')


def get_dataset_paths(for_eval=False):
    p = f"./dataset/eval" if for_eval else "./dataset"
    dfiles = list(pathlib.Path(p).rglob("*.*"))
    allowed_extensions = [f'.{x}' for x in AUDIO_FORMATS]
    banned_dirs = ['tests', 'disabled']
    if not for_eval:
        banned_dirs += ['eval']
    dfiles = [x for x in dfiles
              if any([x.parts[-1].endswith(ext) for ext in allowed_extensions])
              and x.parts[1] not in banned_dirs]
    return dfiles


def load_progress():
    if args.baby_parameters:
        one_sec_len = round(24000 / 84 / 64) * 64
        overrided_pars = {
            'target_sample_len': 4 * one_sec_len, 'history_len': one_sec_len // 4, 'fragment_len': one_sec_len // 4,
            'sc_len': one_sec_len // 4,
            'spect_width': 256, 'ir_width': 256,
            'se_blocks': 2,
            'ae_blocks': (2, 0, 0), 'ae_heads': 4, 'ae_hidden_dim_m': 1,
            'ad_blocks': (0, 0, 2), 'ad_heads': 4, 'ad_hidden_dim_m': 1,
            'rm_k_min': 0.9, 'rm_k_max': 1.0, 'mid_repeat_interval': (2, 4)
        }
    else:
        overrided_pars = {}

    p_snapshots = pathlib.Path("snapshots")
    os.makedirs(p_snapshots, exist_ok=True)
    directory = None
    try:
        directory = p_snapshots / sorted([x for x in os.listdir(p_snapshots) if 'disable' not in x])[-1]
    except:
        pass
    print(f'  Snapshot directory {directory}')

    try:
        model = load_model(directory, device, precision, verbose=True)
        print(f'  Loaded an EchoMorph model.')
    except:
        pars = EchoMorphParameters(**overrided_pars)
        model = EchoMorph(pars).to(device=device, dtype=precision)
        print('  Initialized a new EchoMorph model...')

    try:
        consume: ConsumeProgress = pickle.load(open(directory / 'consume.bin', 'rb'))
    except:
        consume = ConsumeProgress([])
    print(f'  Consume has {len(consume.paths)} training paths before refresh... ', end='')
    consume.check_presence()
    consume.total_epochs = args.total_epochs
    new_dpaths = [x for x in get_dataset_paths() if x not in consume.paths]
    consume.add_files([[x, ac.total_frames(x)] for x in new_dpaths])
    print(f'and {len(consume.paths)} files after refresh... ')

    try:
        training_params = pickle.load(open(directory / 'training_params.bin', 'rb'))
        print('  Loaded training params.')
    except:
        training_params = [1.0]
        print('  Initialized training params.')
    training_params[0] = min(args.learning_rate, training_params[0])

    if not directory:
        save_progress(model, consume, training_params)
    return model, consume, training_params


def save_progress(model, consume, training_params):
    time.sleep(0.5)
    p_snapshots = pathlib.Path("snapshots")
    directory = p_snapshots / datetime.datetime.now().replace(microsecond=0).isoformat().replace(':', '.')
    os.makedirs(directory, exist_ok=True)
    pickle.dump(consume, open(directory / 'consume.bin', 'wb'))
    pickle.dump(training_params, open(directory / 'training_params.bin', 'wb'))
    save_model(directory, model)
    print('Saved progress.')


def random_degradation_value():
    if args.no_random_degradation:
        return 1.000001
    # Eyeballed
    r = random.random()
    return min((r ** 1.5) + 0.2, 1.000001)


def take_a_bite(consume: ConsumeProgress):
    """Randomly selects a file from dataset and takes a bite."""
    sel = consume.lottery_idx()
    if sel is None:
        return None, None

    # About 10 minutes, don't care about the bitrate and the exact value
    cap = 45678 * 600
    path, start, end = consume.bite(sel, cap)
    loaded = None
    try:
        loaded = ac.load_audio(path, frame_offset=start, num_frames=end - start,
                               degrade_keep=random_degradation_value())
    except:
        print(f"Pain... could not load audio file {str(path)}!")
        return take_a_bite(consume)
    sg = ac.convert_from_wave(loaded)
    return sg, path


class CustomAudioDataset(Dataset):
    """Provides history and fragments from given spectrogram, in an alligned way."""
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


def create_eval_datasets(model_pars: EchoMorphParameters):
    tsl = model_pars.target_sample_len
    hl = model_pars.history_len
    fl = model_pars.fragment_len

    eval_datasets = []
    for dfile in get_dataset_paths(for_eval=True):
        loaded = ac.load_audio(dfile)
        eval_spect = ac.convert_from_wave(loaded)
        eval_datasets += [(eval_spect[:tsl, ...], DataLoader(CustomAudioDataset(eval_spect[tsl:, ...], hl=hl, fl=fl),
                                                            batch_size=args.batch_size, shuffle=False))]
    return eval_datasets


class LossNaNException(Exception):
    pass


def eval_model(model, eval_datasets):
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        model.eval()
        m_def_reps = model.pars.mid_repeat_interval
        for middle_repeats in range(m_def_reps[0], m_def_reps[1], max(1, (m_def_reps[1] - m_def_reps[1]) // 4)):
            for target_sample, dataloader in eval_datasets:
                for history, fragments in iter(dataloader):
                    pred = model(target_sample, history, fragments, middle_repeats=middle_repeats)
                    lf = trivial_loss_function if args.use_dumb_loss_function else loss_function
                    loss: Tensor = lf(pred.float(), fragments.float()).to(dtype=precision)
                    if loss.isnan():
                        raise LossNaNException()
                    total_loss += loss.item()
                total_items += len(dataloader)
    if total_items == 0:
        return None
    return total_loss / total_items


loss_function_freq_significance_cache = None
def loss_function_freq_significance(width, device):
    global loss_function_freq_significance_cache
    if loss_function_freq_significance_cache is None or loss_function_freq_significance_cache[0] != width:
        vals = torch.arange(start=20, end=16000 * (width + 1) / width, step=16000 / 128).log2()
        vals = vals[1:] - vals[:-1]
        vals = vals / torch.sum(vals) * width
        loss_function_freq_significance_cache = width, vals
    if loss_function_freq_significance_cache[1].device != device:
        loss_function_freq_significance_cache = width, loss_function_freq_significance_cache[1].to(device)
    return loss_function_freq_significance_cache[1]


def trivial_loss_function(pred, truth):
    """Very stupid, but 100% bug-free loss function"""
    return torch.mean((truth - pred) ** 2)


def loss_function(pred, truth):
    """Custom loss function, for comparing two spectrograms. Not the best one, but it should work."""
    # TODO: this can be infinitely improved:
    #  * equal-loudness contour
    #  * auditory masking
    #  * jitter in neighbor values across time-domain
    #  * slightly different pitch is not too bad
    #  * large undershoot = "masked"
    width = pred.size(-1) // 2

    amp_to_phase_significance = 4  # Phase is not as important as amplitude

    # This amp code is no more sane than the person who wrote it was when they wrote it
    # Pretty much all the things are eye-balled and not rigorously determined
    truth_amp = truth[..., :width]
    pred_amp = pred[..., :width]
    amp_distance = truth_amp - pred_amp
    # Undershoot = bad; overshoot = veeeery baaaad
    amp_distance = torch.max(amp_distance, 2.0 * (-amp_distance))
    # Frequency ranges are not created equal
    amp_distance *= loss_function_freq_significance(width, amp_distance.device)

    # Phase part of the spectrogram works like a circle.
    phase_distance = torch.abs(pred[..., width:] - truth[..., width:]) % 2.0
    # Clamp to [0;1], where 1 is the opposite phase
    phase_distance = torch.min(phase_distance, phase_distance * (-1.0) + 2.0)
    # Phase is more important for more prominent frequencies
    phase_distance *= (truth_amp + truth_amp.min()) / (0.00001 + truth_amp.max() - truth_amp.min())
    # Frequency ranges are not created equal
    phase_distance *= loss_function_freq_significance(width, amp_distance.device)

    # We want to minimize distance squared.
    loss = torch.mean(torch.cat([amp_distance * amp_to_phase_significance, phase_distance]) ** 2) * 2
    return loss


def train_on_bite(model: EchoMorph, optimizer: torch.optim.Optimizer, train_spect: Tensor, timings):
    """Train the model on the prettified spectrogram."""
    tsl = model.pars.target_sample_len
    target_sample = train_spect[0:tsl, :]

    hl = model.pars.history_len
    fl = model.pars.fragment_len
    bt = time.time()
    # batch_size applies to AudioEncoder and AudioDecoder, does not apply to SpeakerEncoder
    dataloader = DataLoader(CustomAudioDataset(train_spect[tsl:, ...], hl=hl, fl=fl),
                            batch_size=args.batch_size, shuffle=True)
    upd_timings(timings, 'dataloading', bt)

    bt = time.time()
    total_loss = 0
    model.train()
    for history, fragments in iter(dataloader):
        optimizer.zero_grad()
        pred = model(target_sample, history, fragments)
        lf = trivial_loss_function if args.use_dumb_loss_function else loss_function
        loss: Tensor = lf(pred.float(), fragments.float()).to(dtype=precision)
        if loss.isnan():
            raise LossNaNException()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(dataloader)
    upd_timings(timings, 'training', bt)
    return train_loss


def training():
    """Main training routine, start to train the model."""
    verify_compatibility()

    print(f'Loading... Args: {args}')
    model, consume, training_params = load_progress()

    lr, = training_params
    eval_datasets = create_eval_datasets(model.pars)
    last_save = time.time()
    optimizer = torch.optim.Adam([
        {'params': model.get_base_parameters(),
         'lr': lr},
        {'params': model.get_multiplicating_parameters(),
         'lr': (lr / (sum(model.pars.mid_repeat_interval) - 1))}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15, min_lr=1e-8,
                                                           threshold=0.001, threshold_mode='rel')
    print_cuda_stats()
    print(f'Training initiated!')
    timings = {}
    try:
        bite_i = 0
        eval_loss, cumm_train_loss = None, None
        while True:
            if bite_i % 4 == 0:
                bt = time.time()
                eval_loss = eval_model(model, eval_datasets)
                if eval_loss:
                    print('Eval loss: {}'.format(eval_loss))
                    scheduler.step(eval_loss)
                elif cumm_train_loss:
                    scheduler.step(eval_loss)
                cumm_train_loss = 0
                upd_timings(timings, 'eval', bt)

            bt = time.time()
            train_spect, origin = take_a_bite(consume)
            upd_timings(timings, 'loading', bt)
            if origin is None:
                break  # Dataset is over

            cur_train_loss = train_on_bite(model, optimizer, train_spect, timings)
            cumm_train_loss += cur_train_loss

            report(optimizer, consume, cur_train_loss, origin)
            if last_save + args.save_time < time.time():
                last_save = time.time()
                save_progress(model, consume, [optimizer.param_groups[0]['lr']])
                print(f'Timings: {timings}')
            bite_i += 1
            if bite_i == 1:
                print_cuda_stats()
    except KeyboardInterrupt:
        print('Exiting gracefully...')
    except LossNaNException:
        print('!!! BUSTED! Something exploded! This is super bad!')
        print(f'Timings: {timings}')
        exit(1)
    print(f'Timings: {timings}')
    save_progress(model, consume, [optimizer.param_groups[0]['lr']])
    print('Training finished!')


if __name__ == '__main__':
    training()
    # Use like: python training.py --save_time=90 --batch_size=16
