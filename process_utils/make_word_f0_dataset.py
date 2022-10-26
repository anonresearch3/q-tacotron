from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy.io.wavfile import read
import os
import json

def read_f0(f):
    table     = open(f, 'r').read().split('\n')[7:-1]
    times = np.array([t.split(' ')[0] for t in table], dtype=float)
    f0 = np.array([t.split(' ')[2] for t in table], dtype=float)
    return times, f0

def read_textgrid(path):
    grid = open(path).read().split('\n')

    i = 0
    intervals = []
    while i < len(grid):
        if grid[i].find('intervals [') >= 0:
            xmin = float(grid[i + 1].split('=')[1].replace(' ', ''))
            xmax = float(grid[i + 2].split('=')[1].replace(' ', ''))
            word = grid[i + 3].split('=')[1].replace(' ', '').replace('"', '')
            if len(word) > 0:
                intervals.append([word, xmin, xmax])
            i += 4
        else:
            i += 1
        if i < len(grid):
            if grid[i].find('item [2]') >= 0:
                i = 123123
    return intervals

def process_data(data, hparams):

    tmp_dir = hparams.path_to_tmp
    path_to_pitch = '{}/pitch'.format(tmp_dir)
    path_to_alignment = '{}/alignment'.format(tmp_dir)
    path_word_f0_pickle = '{}/word_f0_dataset'.format(tmp_dir)

    assert os.path.isdir(path_to_pitch)
    assert os.path.isdir(path_to_alignment)

    dataset = []
    wavs = [d['filename'].split('/')[-1].split('.')[0] for d in data]

    for f in tqdm(wavs):
        if not os.path.exists(os.path.join(path_to_alignment, f + '.TextGrid')):
            continue

        if not os.path.exists(os.path.join(path_to_pitch, f + '.txt')):
            continue


        _, f0 = read_f0(os.path.join(path_to_pitch, f + '.txt'))
        times = np.arange(len(f0)) * 0.01
        intervals = read_textgrid(os.path.join(path_to_alignment, f + '.TextGrid'))

        for index, i in enumerate(intervals):
            word, xmin, xmax = i
            if xmax - xmin > 0.01:
                wf0 = f0[(times >= xmin) * (times <= xmax)]
                dataset.append([f, intervals, index, word, wf0])
    print(dataset[:3])
    np.save(path_word_f0_pickle, dataset)

