import os
import multiprocessing as mp
from tqdm import tqdm
import pyreaper
from scipy.io.wavfile import read

num_proc = 8

def process(wavs, hparams):
    dataset_dir = hparams.path_to_dataset
    tmp_dir = hparams.path_to_tmp

    os.makedirs(os.path.join(dataset_dir, 'pitch'), exist_ok=True)
    for w in tqdm(wavs):
        r, a = read(os.path.join(dataset_dir, 'wavs', w))
        pitch_reaper = pyreaper.reaper(a, r, frame_period=0.01, unvoiced_cost=0.9)[3]

        with open(os.path.join(tmp_dir, 'pitch', w.split('.')[0] + '.txt'), 'w') as fil:
            fil.write('\n'*7)
            fil.write('\n'.join(list(map(lambda x: '1 1 ' + str(x), pitch_reaper))))




def process_data(data, hparams):
    wavs = [d['filename'].split('/')[-1].split('.')[0] + '.wav' for d in data if 'filename' in d]

    os.makedirs('{}'.format(hparams.path_to_tmp), exist_ok=True)
    os.makedirs('{}/pitch'.format(hparams.path_to_tmp), exist_ok=True)

    ps = []
    mp.set_start_method('fork')
    for n in range(num_proc):
        p = mp.Process(target=process, args=(wavs[n::num_proc], hparams))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

