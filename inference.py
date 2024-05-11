#!/usr/bin/python -u
import torch
import os
import pathlib
import sys
import time

from audio import AudioConventer, AUDIO_FORMATS
from model import load_model


def play_audio(filename):
    if sys.platform.startswith('linux'):
        # Install alsa-utils / alsa-tools if this does not work
        import subprocess
        subprocess.call(['aplay', filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    elif sys.platform.startswith('win'):
        # Install pygame
        import pygame
        pygame.mixer.init(frequency=44100)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        try:
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)
        except:
            pass  # Somebody's ears bled
        pygame.mixer.music.unload()
    else:
        print('Playback not implemented...')


class InferenceFreestyle:
    """InferenceFreestyle is a way to interact with the EchoMorph model. The class provides
    methods to perform actions on the objects from the internal bank. Objects are of two types:
    sounds and characteristics. Actions that result in creation of new objects save these objects with unique,
    predictable and non-configurable names."""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = torch.float32 if str(self.device) == "cpu" else torch.float16
        self.ac = AudioConventer(self.device, self.precision)

        root_snapshots = pathlib.Path("snapshots")
        available_snapshots = sorted([x for x in os.listdir(root_snapshots) if 'disable' not in x])

        if len(available_snapshots) == 0:
            print('No model snapshot means no inference is possible.')
            print('Put a snapshot into the snapshots folder.')
            exit(1)
        directory = root_snapshots / available_snapshots[-1]
        print(f'Loading an EchoMorph model stored in {directory}... ', end='')
        self.model = load_model(directory, self.device, self.precision)
        self.model.eval()
        print('Done!')

        self.bank = {}
        self.next_num = 10

    def to_bank(self, letter, obj, recipie):
        """Stores an object that can later be used with Freestyle"""
        name = f'{letter}{self.next_num}'
        self.next_num += 1
        self.bank[name] = (obj, recipie)
        print(f'{name}: {recipie}')
        return name

    def load(self, path):
        """Loads an object - sound or characteristic."""
        if not pathlib.Path(path).is_file():
            for cand in ['demo', pathlib.Path('dataset') / 'tests']:
                if (pathlib.Path(cand) / path).is_file():
                    path = pathlib.Path(cand) / path
                    break
        path = str(path)
        if path.split('.')[-1] in AUDIO_FORMATS:
            wv = self.ac.load_audio(path)
            sg = self.ac.convert_from_wave(wv)
            return self.to_bank('s', sg, f'Loaded from {path}')
        elif path.split('.')[-1] in ['emc']:
            sc = torch.load(path, weights_only=True)
            return self.to_bank('c', sc, f'Loaded from {path}')
        else:
            raise NotImplementedError()

    def save(self, name, filename):
        """Saves an object - sound or characteristic."""
        if filename == '':
            filename = name
        if name[0].startswith('s'):
            if filename.split('.')[-1] not in AUDIO_FORMATS:
                filename += '.wav'
            audio = self.ac.convert_to_wave(self.bank[name][0])
            if not any([x in filename for x in '/\\']):
                filename = 'demo/' + filename
            self.ac.save_audio(audio, filename)
        elif name[0].startswith('c'):
            if filename.split('.')[-1] not in ['emc']:
                filename += '.emc'
            if not any([x in filename for x in '/\\']):
                filename = 'demo/' + filename
            torch.save(self.bank[name][0], pathlib.Path('demo') / filename)
        else:
            raise NotImplementedError()

    def cap_freq(self, name, percent):
        percent = float(percent)
        obj: torch.Tensor = self.bank[name][0]
        bor = obj.size(-1) // 2
        preserve = round(bor * percent / 100)
        capped_obj = torch.clone(obj)
        capped_obj[..., preserve:bor] = 0
        capped_obj[..., bor+preserve:bor+bor] = 0
        self.to_bank('s', capped_obj, f'Capped from {name} (passed: {percent}%)')

    def _play_object(self, object):
        import datetime
        time = datetime.datetime.now().replace(microsecond=0).strftime("%H:%M:%S")
        code = self.to_bank('sd', object, f'Debug object spawned at {time}')
        self.play_sample(code)

    def play_sample(self, name):
        """Tries its best to play a sound on your system."""
        import tempfile
        tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmpfile.close()
        self.save(name, tmpfile.name)
        play_audio(tmpfile.name)
        os.remove(tmpfile.name)

    def derive_sc(self, name, repeats=1, block_repeats=1):
        """Derives speaker characteristic from waveform. Can derive multiple and average them out."""
        assert repeats > 0
        with torch.inference_mode():
            tsl = self.model.pars.target_sample_len
            sg = self.bank[name][0]

            if sg.size(0) < tsl:
                padding = torch.zeros([(tsl - sg.size(0) + 2) // 2, sg.size(1)], dtype=sg.dtype, device=sg.device)
                sg = torch.cat((padding, sg, padding), dim=0)

            print('Deriving: [', end='')
            out = self.model.speaker_encoder.forward_use(sg[0:0+tsl, :], block_repeats)
            print('.', end='')
            start_end = sg.size(0) - tsl
            start = 0
            for i in range(1, repeats):
                # Deterministic, but uniform-like distribution of starting points
                start = (start + 2147483647) % start_end
                out += self.model.speaker_encoder.forward_use(sg[start:start+tsl, :], block_repeats)
                print('.', end='')
            print('] Done!')
            out /= repeats
            return self.to_bank('c', out, f'Derived directly from {name} (r={repeats})')

    def merge_sc(self, name1, name2, proportion=0.5):
        """Merge two speaker characteristics."""
        proportion = float(proportion)
        assert 0.0 <= proportion
        obj = self.bank[name1][0] * (1.0 - proportion) + self.bank[name2][0] * proportion
        if proportion <= 1.0:
            recipie = f'Merged from {name1} ({(1.0 - proportion):.3f}) and {name2} ({proportion:.3f})'
        else:
            recipie = f'Overcharged from {name2} (+{proportion:.3f}) by ' \
                      f'un-tainting {name1} ({(1.0 - proportion):.3f}) from it'
        self.to_bank('c', obj, recipie)

    # TODO: king queen man woman

    def randomize_sc(self, name, p):
        obj = self.bank[name][0]
        o_max, o_min = obj.max(), obj.min()
        obj += torch.randn_like(obj) * p
        torch.clamp(obj, min=o_min, max=o_max)
        self.to_bank('c', obj, f'Forged from {name} by applying {p:.3f} grams of pure chaos')

    def infer(self, sc_name, source_name, tradeoff: float = 0.95, quality: int = 4, radiation: float = 0):
        # TODO: multi-merge (averaging multiple infer-s with slightly different windowing)
        # Updating model settings
        assert 0 <= quality
        self.model.bottleneck.set_p(tradeoff)

        do_lerp = '->' in sc_name
        if do_lerp:
            sc = [self.bank[x][0] for x in sc_name.split('->')]
        else:
            sc = self.bank[sc_name][0]
        source = self.bank[source_name][0]

        hl = self.model.pars.history_len
        fl = self.model.pars.fragment_len
        source = torch.nn.functional.pad(source, (0, 0, 0, 0, hl, fl))
        target = torch.zeros_like(source)
        with torch.inference_mode():
            print('Inferencing: [', end='')
            for cur in range(hl, target.size(0) - fl, fl):
                if do_lerp:
                    lerp_c = (cur - hl) / (target.size(0) - hl)
                    cur_sc = sc[0] * (1 - lerp_c) + lerp_c * sc[1]
                else:
                    cur_sc = sc
                intermediate = self.model.audio_encoder(
                    source[cur:cur + fl, :].unsqueeze(0), quality
                )
                if radiation > 1e-9:
                    intermediate += torch.where(intermediate == 0, torch.tensor(0),
                                                torch.randn_like(intermediate) * radiation)
                intermediate = self.model.bottleneck(intermediate)
                target[cur:cur + fl, :] = self.model.audio_decoder(
                    intermediate, cur_sc, quality
                )
                print('.', end='')
            print('] Done!')

        target = target[hl:, ...]
        return self.to_bank('s', target, f'Inferenced from {source_name} by {sc_name} with '
                                         f't={tradeoff:.3f}, q={quality}, r={radiation:.3f}')

    def list(self):
        for f in self.bank.keys():
            print(f'{f}: {self.bank[f][1]}')

    def clear_bank(self):
        # TODO: memory cleanup?
        self.bank = {}

    # Future freestyling ideas:
    #     time-domain sc lerping - replace values gradually
    #     feeding fake history


def demo(freestyle: InferenceFreestyle):
    src = input('Speech file path: ')
    if len(src) < 1:
        src = './dataset/tests/example1.mp3'
        tgt_s = './dataset/tests/example1.mp3'
        save = './demo/result_temp.wav'
    else:
        tgt_s = input('Speaker file path: ')
        save = input('Save into: ')

    tgt_s_n = freestyle.load(tgt_s)
    sc = freestyle.derive_sc(tgt_s_n)
    src = freestyle.load(src)
    out = freestyle.infer(sc, src)
    freestyle.save(out, save)
    freestyle.play_sample(out)


if __name__ == '__main__':
    print('=== EchoMorph inference demo ===')
    freestyle = InferenceFreestyle()

    while True:
        try:
            cmd = input('> ').split(' ')
            mods = {}
            match cmd[0]:
                case 'demo':
                    demo(freestyle)
                case 'playtest':
                    s = freestyle.load('./dataset/tests/example1.mp3')
                    freestyle.play_sample(s)
                case 'load':
                    freestyle.load(' '.join(cmd[1:]))
                case 'save':
                    freestyle.save(cmd[1], ' '.join(cmd[2:]))
                case 'derive':
                    for mod in cmd[2:]:
                        if mod.startswith('r='):
                            mods['repeats'] = int(mod[2:])
                    freestyle.derive_sc(cmd[1], **mods)
                case 'merge':
                    freestyle.merge_sc(*cmd[1:])
                case 'randomize':
                    freestyle.randomize_sc(cmd[1], float(cmd[2]))
                case 'infer':
                    for mod in cmd[3:]:
                        if mod.startswith('q='):
                            mods['quality'] = int(mod[2:])
                        if mod.startswith('t='):
                            mods['tradeoff'] = float(mod[2:])
                        if mod.startswith('r='):
                            mods['radiation'] = float(mod[2:])
                    if cmd[1][0] == 's' and cmd[2][0] == 'c':
                        cmd[1], cmd[2] = cmd[2], cmd[1]
                    freestyle.infer(cmd[1], cmd[2], **mods)
                case 'cap':
                    freestyle.cap_freq(cmd[1], cmd[2])
                case 'play':
                    freestyle.play_sample(cmd[1])
                case 'list':
                    freestyle.list()
                case 'clear':
                    freestyle.clear_bank()
                case _:
                    print('Commands: load save derive\n'
                          '          merge randomize infer\n'
                          '          play list clear')
        except KeyError as e:
            print(f'No such object {e.args[0]}')
        except:
            import traceback
            traceback.print_exc()
