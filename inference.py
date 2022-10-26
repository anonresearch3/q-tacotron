import sys
sys.path.append('hifigan/')
import numpy as np
import torch
import re
from hparams import Hparams
import os
from model import Tacotron2
from process_utils import cleaners
from process_utils.make_features import prepare_bert
from scipy.special import softmax
from scipy.io.wavfile import read, write
from hifigan.models import Generator
import matplotlib.pylab as plt
from torch import nn
from vocoder import Vocoder
import json

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       interpolation='none')
    plt.savefig("mygraph.png")
    plt.show()

def compute_bert_embeddings(text,tokenizer,embedder,device):
    tokens = tokenizer.tokenize(text.lower())
    ids = tokenizer.convert_tokens_to_ids(tokens)
    embeddings = embedder(torch.tensor([ids]).long().to(device))[0]

    return tokens, embeddings


def predict_f0_clusters(model, text, tokens, embeddings, question_cluster_ids, dims,device):
    inds = []
    cur_token = 0
    i = 0
    #text = text.lower()

    while i < len(text):
        if text.lower()[i:].startswith(tokens[cur_token].replace('#', '')):
            inds.append(i + len(tokens[cur_token]) - tokens[cur_token].count('#'))
            i += len(tokens[cur_token]) - tokens[cur_token].count('#')
            if cur_token < len(tokens) - 1:
                cur_token += 1
        else:
            i += 1



    indices = [letter2index[l] for l in text]

    text_encoder_out = model.encode((torch.tensor([indices]).long().to(device))).detach().cpu().numpy()[0]

    f0_out = model.label_predictor(embeddings.to(device), torch.tensor([text_encoder_out]).float().to(device),
                          torch.tensor([inds]).long().to(device))
    if len(f0_out.shape) == 3:
        f0_out = f0_out.unsqueeze(3)
    f0_out = f0_out.detach().cpu().numpy()[0]

    f0_word_labels = {i: [] for i in range(len(dims))}
    f0_labels = {}

    for j in range(len(dims)):
        f0_outj = softmax(f0_out[:, :, j], axis=1)
        empty_label = dims[j]

        for i in range(len(tokens)):
            if re.match('\w+', tokens[i].replace('#', '')):
                f0_word_labels[j].append(np.argmax(f0_outj[i][:empty_label]))
            else:
                f0_word_labels[j].append(empty_label)

        f0_labels[j] = []

        cur_token = 0
        i = 0
        while i < len(text):
            if text[i:].lower().startswith(tokens[cur_token].replace('#', '')):
                i += len(tokens[cur_token].replace('#', ''))
                f0_labels[j] += [f0_word_labels[j][cur_token]] * len(tokens[cur_token].replace('#', ''))
                if cur_token < len(tokens) - 1:
                    cur_token += 1
            else:
                f0_labels[j].append(empty_label)
                i += 1
        print(''.join([str(l) if l != 10 else '*' for l in f0_labels[j]]))

    f0_labels = np.array([f0_labels[j] for j in range(len(dims))]).T
    return text_encoder_out, f0_labels


def split(line):
    if len(line) < 100:
        return [[line, 10]]
    texts = []
    # line = line.replace('.', '.|').replace('?', '?|').replace(';', '.|').replace('!', '!|').split('|')
    line = line.replace('.', '.|').replace('?', '?|').replace(';', '.|').replace('!', '.|').split('|')

    for l in line:
        l = l.strip()
        if len(l) > 1:
            #if not l[-1] in '.?!;':
            #    l = l + '.'
            if len(l)>150 and "," in l[100:]:
                _l = l[100:].split(",")
                l1 = l[:100]+_l[0]
                l2 = ",".join(_l[1:])
                texts.append([l1.strip(),4])
                texts.append([l2.strip(),7])
            elif len(l)>200 and " " in l[150:]:
                _l = l[150:].split(" ")
                l1 = l[:150]+_l[0]
                l2 = " ".join(_l[1:])
                texts.append([l1.strip(),1])
                texts.append([l2.strip(),7])
            else:
                texts.append([l,7])
    new_line = [texts[0]]
    if len(texts)>1:
        for i in range(1,len(texts)):
            if len(texts[i][0])<10:
                new_line[-1] = [new_line[-1][0]+" "+texts[i][0],texts[i][1]]
            else:
                new_line.append(texts[i])
    return new_line

def load_hifigan():
    config_file = os.path.join("hifigan/" 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)

    state_dict_g = checkpoint_dict = torch.load("hifigan/generator_v1", map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    elu = nn.ELU()
    generator.eval()
    generator.remove_weight_norm()
    return generator

def load_tacotron():
    checkpoint_path = "outdir/190_all"
    model = Tacotron2(hparams).to(device)
    chk = torch.load(checkpoint_path,map_location=device)
    if 'state_dict' in chk:
        model.load_state_dict(chk['state_dict'])
    else:
        model.load_state_dict(chk)
    model.eval()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hparams = Hparams()
letter2index = {hparams.symbols[i]: i for i in range(len(hparams.symbols))}
if __name__ == "__main__":
    embedder, tokenizer = prepare_bert()
    model = load_tacotron()
    generator = load_hifigan()

    os.makedirs('gen', exist_ok=True)

    texts = open('filelists/ljs_audio_text_test_filelist.txt', 'r').read().lower().split('\n')[:2]
    if len(texts[-1]) == 0:
        texts = texts[:-1]

    all_a = []
    bad = 0
    for iline, line in enumerate(texts):
        #path = line.split("|")[0].upper()
        #r, a = read(os.path.join(hparams.path_to_dataset, 'wavs', path))
        #write(os.path.join("orig", str(iline + 1).zfill(6) + '_000000.wav'),22050, a)
        line = line.split("|")[1]
        line = cleaners.english_cleaners(line)
        audio = []
        line = split(line)


        for i, (text,after_pause) in enumerate(line):#[[line,10]]):

            text = text.strip()
            if text[-1] not in ".!?":
                text += "."
            text = "=" + text + "~"
            print(text)
            tokens, embeddings = compute_bert_embeddings(text,tokenizer,embedder,device)
            print(tokens)
            text_encoder_out, f0_labels = predict_f0_clusters(model, text, tokens, embeddings, [], [5,5,5,5,5],device)
            print(text_encoder_out)
            batch = [torch.tensor([text_encoder_out]).float(), torch.tensor([f0_labels]).long(),
                     torch.tensor(embeddings).float()]

            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(batch)
            plot_data((mel_outputs.float().data.cpu().numpy()[0],
                       mel_outputs_postnet.float().data.cpu().numpy()[0],
                       alignments.float().data.cpu().numpy()[0].T))
            with torch.no_grad():
                y = generator(mel_outputs_postnet).squeeze().cpu().numpy().tolist() + [0] * 1200 * after_pause

            audio += y
        audio = np.array(audio) * 32768

        output_file = os.path.join("gen", str(iline + 1).zfill(6) + '_000000.wav')
        write(output_file, 24000, audio.astype('int16'))
        print(output_file)
    print(bad)