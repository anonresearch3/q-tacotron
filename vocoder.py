import os
import numpy as np
from scipy.io.wavfile import read
import pyreaper
import importlib.util
from matplotlib import pyplot as plt


class Vocoder:
    def __init__(self, config):
        self.path_to_vocoders = "./"
        self.voice = "CHE"
        self.lpcnet_bin_name  = "mel_mac_119_king.so"
        self.replace_pitch_by_reaper  = False#config['vocoder']['replace_pitch_by_reaper']
        self.rate = 24000#config['global']['rate']
        self.backend = 'lpcnet'
        self.load()

    def load(self):
        if self.backend == 'lpcnet':
            lpcnet_src = os.path.join(os.path.join(self.path_to_vocoders, self.voice, self.lpcnet_bin_name))
            print(lpcnet_src)
            module_spec = importlib.util.spec_from_file_location("pylpcnet", lpcnet_src)
            lpcnet_bin_lib = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(lpcnet_bin_lib)
            self._lpcnet = lpcnet_bin_lib

    def compute_features(self, audio):
        if self.backend == 'lpcnet':
            spec = self._lpcnet.compute_features(audio)
            if self.replace_pitch_by_reaper:
                pitch_reaper = pyreaper.reaper(audio, self.rate, frame_period=0.01, unvoiced_cost=0.9)[3]
                pitch_reaper = [(24000 / x - 100) / 50 if x > 0 else 0 for x in pitch_reaper]
                pitch_reaper = [pitch_reaper[0]] + pitch_reaper  # left shift by 1 frame
                if len(pitch_reaper) > len(spec):
                    pitch_reaper = pitch_reaper[:len(spec)]
                else:
                    while len(pitch_reaper) != len(spec):
                        pitch_reaper += [pitch_reaper[-1]]

                pitch_reaper = np.array(pitch_reaper, dtype=float)
                spec[:, 20] = pitch_reaper

            return spec

    def synthesize(self, mel):
        if self.backend == 'lpcnet':
            return self._lpcnet.synthesize(mel)


if __name__ == '__main__':
    config = load_config()
    config['vocoder']['lpcnet_bin_name'] = 'pylpcnet_macos.cpython-35m-x86_64-linux-gnu.so'

    vocoder = Vocoder(config)

    r, audio = read('/Users/sterling-gg/Downloads/she_for_pqmos/wavs/020851_000000.wav')

    features = vocoder.compute_features(audio)
    restored_audio = vocoder.synthesize(features)

    plt.plot(audio)
    plt.plot(restored_audio)
    plt.show()


