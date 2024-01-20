import pickle
import torch
import os
import pathlib
import sys

from audio import AudioConventer, AUDIO_FORMATS
from model import EchoMorph, EchoMorphParameters


def play_audio(filename):
    import subprocess
    if sys.platform.startswith('linux'):
        subprocess.call(['aplay', filename], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        print('Playback not implemented...')


def crop_from_middle(x, length):
    if x.size(0) < length:
        padding = torch.zeros([(length + 1) // 2, x.size(1)])
        x = torch.cat((padding, x, padding), dim=0)
    start = (x.size(0) - length) // 2
    x = x[start:start + length, ...]
    return x


class InferenceFreestyle:
    def __init__(self):
        # TODO: half mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ac = AudioConventer(device)

        root_snapshots = pathlib.Path("snapshots")
        if not root_snapshots.is_dir() or len(os.listdir(root_snapshots)) == 0:
            print('No model snapshot means no inference is possible.')
            print('Put a snapshot into the snapshots folder.')
            exit(1)

        directory = root_snapshots / sorted(os.listdir(root_snapshots))[-1]
        print(f'Loading an EchoMorph model stored in {directory}... ', end='')
        training_parameters = EchoMorphParameters()  #TODO: inference parameters
        self.model = EchoMorph(training_parameters)
        self.model.load_state_dict(torch.load(directory / 'model.bin'))
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
        if path.split('.')[-1] in AUDIO_FORMATS:
            wv = self.ac.load_audio(path)
            sg = self.ac.convert_from_wave(wv)
            return self.to_bank('s', sg, f'Loaded from {path}')
        elif path.split('.')[-1] in ['emc']:
            sc = pickle.load(open(path, 'rb'))
            return self.to_bank('c', sc, f'Loaded from {path}')
        else:
            raise NotImplementedError()

    def save(self, name, path):
        if name[0] == 's':
            if path.split('.')[-1] not in AUDIO_FORMATS:
                path += '.wav'
            audio = self.ac.convert_to_wave(self.bank[name][0])
            self.ac.save_audio(audio, path)
        elif name[0] == 'c':
            if path.split('.')[-1] not in ['emc']:
                path += '.emc'
            pickle.dump(self.bank[name][0], open(path, 'wb'))
        else:
            raise NotImplementedError()

    def play_sample(self, name):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
            self.save(name, tmpfile.name)
            play_audio(tmpfile.name)

    def derive_sc(self, name):
        with torch.no_grad():
            print('Deriving: [', end='')
            speaker_characteristic = self.model.speaker_encoder(
                crop_from_middle(self.bank[name][0], self.model.pars.target_sample_len)
            )
            print('.] Done!')
            return self.to_bank('c', speaker_characteristic, f'Derived directly from {name}')

    def merge_sc(self, name1, name2, proportion=None):
        if proportion is None:
            proportion = 0.5
        else:
            proportion = int(proportion)
        obj = self.bank[name1][0] * (1.0 - proportion) + self.bank[name2][0]
        recipie = f'Merged from {name1} ({(1.0 - proportion):.3f}) and {name2} {proportion:.3f}'
        self.to_bank('c', obj, recipie)

    def randomize_sc(self, name, p):
        obj = self.bank[name][0]
        o_max, o_min = obj.max(), obj.min()
        obj += torch.randn_like(obj) * p
        torch.clamp(obj, min=o_min, max=o_max)
        self.to_bank('c', obj, f'Forged from {name} by applying {p:.3f} grams of pure chaos')

    def infer(self, sc_name, source_name, tradeoff: float = 0.5, quality: int = 10):
        # Updating model settings
        assert 0 <= quality
        for mp in [self.model.audio_encoder, self.model.audio_decoder]:
            mp.set_mid_repeat_interval(quality)
        assert 0.0 <= tradeoff <= 1.0
        self.model.rando_mask.set_p(tradeoff)

        hl = self.model.pars.history_len
        fl = self.model.pars.fragment_len

        sc = self.bank[sc_name][0]

        source = self.bank[source_name][0]
        source = torch.cat((torch.zeros([hl, source.size(1)]), source, torch.zeros([fl, source.size(1)])), dim=0)
        target = torch.zeros_like(source)

        with torch.no_grad():
            print('Inferencing: [', end='')
            for cur in range(hl, target.size(0), fl):
                intermediate = self.model.audio_encoder(source[cur:cur + fl, :], source[cur - hl:cur, :])
                intermediate = self.model.rando_mask(intermediate)
                target[cur:cur + fl, :] = self.model.audio_decoder(intermediate, sc, target[cur - hl:cur, :])
                print('.', end='')
            print('] Done!')

        target = target[hl:, ...]
        return self.to_bank('s', target, f'Inferenced from {source_name} by {sc_name}')

# TODO: set eval parameters before launching the model (mid blocks repeat, randomask values)

# TODO: Allow interpolation across different speaker representations (time-axis)
#       More types of randomized representations

# TODO: feeding fake history

def demo(freestyle: InferenceFreestyle):
    src = input('Speech file path: ')
    if len(src) < 1:
        src = './dataset/tests/example1.mp3'
        tgt_s = './dataset/tests/example2.mp3'
        save = './dataset/result_temp.wav'
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
            match cmd[0]:
                case 'demo':
                    demo(freestyle)
                case 'playtest':
                    s = freestyle.load('./dataset/tests/example1.mp3')
                    freestyle.play_sample(s)
                case 'load':
                    freestyle.load(cmd[1])
                case 'save':
                    freestyle.save(cmd[1], cmd[2])
                case 'derive':
                    freestyle.derive_sc(cmd[1])
                case 'merge':
                    freestyle.merge_sc(*cmd[1:])
                case 'randomize':
                    freestyle.randomize_sc(cmd[1], float(cmd[2]))
                case 'infer':
                    # TODO: sloppy syntax - use sample as characteristic (derive beforehand)
                    # TODO: sloppy syntax - swap arguments, figure out by first letter
                    mods = {}
                    for mod in cmd[3:]:
                        if mod.startswith('q='):
                            mods['quality'] = int(mod[2:])
                        if mod.startswith('t='):
                            mods['tradeoff'] = float(mod[2:])
                    freestyle.infer(cmd[1], cmd[2], **mods)
                case 'list':
                    raise NotImplementedError()
                case 'clear':
                    raise NotImplementedError()
                case _:
                    print('Commands: load save derive'
                          '          merge randomize infer'
                          '          list clear')
        except KeyError as e:
            print(f'No such object {e.args[0]}')
        except:
            import traceback
            traceback.print_exc()
