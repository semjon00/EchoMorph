import builtins
import datetime
import os
import pathlib
import torch
import random
import pickle

from model import EchoMorph, EchoMorphParameters
from audio import AudioConventer

allowed_extensions = ['.aac', '.mp3', '.flac']


def print(*args):
    builtins.print(datetime.datetime.now().replace(microsecond=0).isoformat(), *args)


def verify_compatibility():
    f = 'NONE'
    try:
        ac = AudioConventer('cpu')
        tests_dir = pathlib.Path('./dataset/tests')
        for f in os.listdir(tests_dir):
            ac.convert_to_wave(ac.convert_from_wave(ac.load_audio(tests_dir / f)))
    except:
        print(f'Compatibility check FAILED on file {f}')
        print('Please ensure that ffmpeg (and maybe sox) are installed - these are necessary to read audio files')
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
        ac = AudioConventer('cpu')
        consume = [(x, 0, ac.total_frames(x)) for x in dfiles
                   if any([x.parts[-1].endswith(ext) for ext in allowed_extensions]) and
                   x.parts[-2] != 'tests']
        return model, consume
    else:
        directory = p_snapshots / sorted(os.listdir(p_snapshots))[-1]
        model = torch.load(directory / 'model.bin')
        consume = pickle.load(open(directory / 'consume.bin', 'rb'))
        return model, consume


def save_progress(model, consume):
    p_snapshots = pathlib.Path("snapshots")
    directory = p_snapshots / datetime.datetime.now().replace(microsecond=0).isoformat()
    torch.save(model, directory / 'model.bin')
    pickle.dump(consume, open(directory / 'consume.bin', 'wb'))
    print('Saved progress.')


def take_a_bite(ac, consume):
    """Randomly selects a file from dataset and takes a bite.
    This thing is slow, but it gets the job done"""
    # TODO: Use DataLoader
    # TODO: Custom priority

    load_opt = 45678 * 300  # About 5 minutes, don't care about the bitrate and the exact value

    tot_rem = sum([el[2] - el[1] for el in consume])
    drop = random.randint(0, tot_rem - 1)
    sel = 0
    for i, el in enumerate(consume):
        if drop < el[2] - el[2]:
            sel = i
            break
        drop -= el[2] - el[2]
    load_now = load_opt if consume[sel][2] - consume[sel][1] > 2 * load_opt else consume[sel][2]
    ac.load_audio(consume[sel][0], frame_offset=consume[sel][1], num_frames=load_now)
    consume[sel][1] += load_now
    return None


def training():
    print('Training initiated')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    verify_compatibility()
    print('Compatibility verified.')

    model, consume = load_progress()
    print('Loading progress done!')
    ac = AudioConventer(device)

    while len(consume):
        train_wav = take_a_bite(ac, consume)

        pass
        # TODO: Load some of the file
        # TODO: Train on it a bit (like, 10 minutes maybe?)
        # TODO: Save consume progress

        # TODO: Save occasionally

    batch_size = 256


if __name__ == '__main__':
    training()
