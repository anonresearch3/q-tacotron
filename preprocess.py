import os


from hparams import Hparams
from process_utils import compute_f0, make_word_f0_dataset,quantization,make_features,cleaners
from taco_utils import audio_to_mel

def load_LJSpeech_data(data_file='metadata.csv'):
    path = os.path.join(hparams.path_to_dataset, data_file)
    data = []
    with open(path, 'r') as f:
        for line in f:
            path_file, _, text_file = line.strip().split("|")
            d = {}
            d['filename'] = path_file
            d["text"] = cleaners.english_cleaners(text_file)
            data.append(d)
    return data

hparams = Hparams()
data = load_LJSpeech_data()

print(data[:3])
compute_f0.process_data(data, hparams)

make_word_f0_dataset.process_data(data, hparams)

quantization.process(data, hparams)

make_features.process(hparams)

audio_to_mel(data, hparams)