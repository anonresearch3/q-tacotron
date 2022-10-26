
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy.io.wavfile import read
import os
import json
import warnings
from scipy import stats



def quantization_dim_sizes(quantization_dim_param):
    sizes = quantization_dim_param
    if isinstance(quantization_dim_param[0], list):
        sizes = [len(f) for f in quantization_dim_param]
    return sizes


def read_durations_json(src):
    durations_dict = json.load(open(src))
    durations = {}
    durations_counts = {}
    for ci, c in enumerate(durations_dict['char']):
        durations_counts[c] = [durations_dict['duration_mean'][ci], durations_dict['count'][ci]]

    for c in durations_counts:
        if c.lower() in durations_counts and c.upper() in durations_counts:
            c1 = durations_counts[c.lower()][1]
            c2 = durations_counts[c.upper()][1]
            durations[c.lower()] = (c1 * durations_counts[c.lower()][0] + c2 * durations_counts[c.upper()][0]) / float(c1 + c2)
        else:
            durations[c.lower()] = durations_counts[c][0]

    return durations


def compute_f0_slope(f0):
    x = np.arange(len(f0))
    if len(f0[f0 > 0]) == 0:
        return 0
    else:
        return stats.linregress(x[f0 > 0], f0[f0 > 0])[0]


def compute_speed(start, end, text, durations):
    expected_length = 0
    for l in text:
        if l in durations:
            expected_length += durations[l]
    return (end - start) / expected_length


def compute_percentiles(features_dict, path_to_tmp):
    features = []
    for f in features_dict:
        for u in features_dict[f]:
            features.append(u[4:])
    features = np.array(features)
    np.save(os.path.join(path_to_tmp,'features'), features)

    feature_percentiles = [
                                [10, 25, 30, 25, 10],
                                [10, 25, 30, 25, 10],
                                [10, 25, 30, 25, 10],
                                [10, 25, 30, 25, 10],
                                [20, 20, 20, 20, 20]
                        ]
    percentiles = {}
    for i in range(features.shape[1]):  # iterate over feature number
        idx = np.array([j for j in range(len(features)) if not np.isnan(features[j, i])])
        bounds = [np.min(features[idx, i])]

        if not isinstance(feature_percentiles[0], list):
            ps = [100 // feature_percentiles[i] * j for j in range(1, feature_percentiles[i]+1)]
        else:
            ps = np.cumsum(feature_percentiles[i])

        for b in ps:
            perc = np.percentile(features[idx, i], b)
            bounds.append(perc)
        percentiles[i] = bounds

    return percentiles


def quantize_features_dict(features_dict, percentiles):
    quantized_dict = {}
    feature_dims = [5,5,5,5,5]

    for key in features_dict:
        quantized_dict[key] = []
        for u in features_dict[key]:
            word, index, start, end = u[:4]
            features = u[4:]

            labels = []
            for i in range(len(features)):  # feature number
                if np.isnan(features[i]):
                    label = feature_dims[i] // 2
                else:
                    label = 0
                    for j in range(len(percentiles[i]) - 1):
                        if features[i] >= percentiles[i][1 + j]:
                            label = 1 + j

                labels.append(label)

            quantized_dict[key].append([word, index, start, end] + labels)

    return quantized_dict  # word-level set of quantized features


def compute_quantized_features(word_f0, hparams):
    wavs_dict = {}
    letter_durations = read_durations_json(hparams.path_to_letter_durations)

    features_dict = {}

    for w in tqdm(word_f0):
        key, timing, index, word, f0 = w

        if not key in features_dict:
            features_dict[key] = []

        start, end = timing[index][1], timing[index][2]

        if not key in wavs_dict:
            r, a = read(os.path.join(hparams.path_to_dataset, 'wavs', key + '.wav'))
            wavs_dict[key] = a

        audio = wavs_dict[key][int(r * start):int(r * end)]

        loudness = np.std(audio)
        mean_f0 = np.mean(f0[f0 > 0])
        left_slope = compute_f0_slope(f0[:len(f0) // 2])
        right_slope = compute_f0_slope(f0[len(f0) // 2:])
        if word != '<unk>':
            speed = compute_speed(start, end, word, letter_durations)
        else:
            speed = np.nan

        features_dict[key].append([word, index, start, end, loudness, mean_f0, left_slope, right_slope, speed])

    percentiles = compute_percentiles(features_dict, hparams.path_to_tmp)

    labels_dict = quantize_features_dict(features_dict, percentiles)

    return labels_dict


def get_paused_text(text, word_labels_dict):
    empty_labels = [5,5,5,5,5]
    text = "="+text+"~"
    ltext = text.lower()

    cw = 0
    i = 0
    paused_text = ''
    letter_labels = []

    while i < len(text):
        if ltext[i:].startswith(word_labels_dict[cw][0]):
            paused_text += text[i:i + len(word_labels_dict[cw][0])]
            letter_labels += [word_labels_dict[cw][4:]] * len(word_labels_dict[cw][0])
            i += len(word_labels_dict[cw][0])

            if cw < len(word_labels_dict) - 1:
                cw += 1
        if i<len(text):
            if ltext[i] in 'abcdefghijklmnopqrstuvwxyz':
                print("BAD",text)
                return None,None
            paused_text += text[i]
            letter_labels += [empty_labels]
            i += 1
    return paused_text, letter_labels


def process(data, hparams):
    word_f0 = np.load(os.path.join(hparams.path_to_tmp, 'word_f0_dataset.npy'), allow_pickle=True)
    word_labels_dict = compute_quantized_features(word_f0, hparams)

    new_data = []

    for d in data:
        if 'filename' in d:
            key = d['filename'].split('/')[-1].split('.')[0]

            if key in word_labels_dict:
                text = d['text']
                paused_text, letter_labels = get_paused_text(text, word_labels_dict[key])
                if paused_text is not None:
                    if all([l in hparams.symbols for l in paused_text]):
                        d['paused_text'] = paused_text
                        d['quantized_features'] = letter_labels
                        new_data.append(d)
                    else:
                        print(d)
                else:
                    print(d)

    json.dump(new_data, open(os.path.join(hparams.path_to_dataset, 'data_quantized.json'), 'w'), indent=4, ensure_ascii=False)

    return new_data








