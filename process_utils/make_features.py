import os
import numpy as np
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

def load_data(hparams, data_file='data.json'):
    path = os.path.join(hparams.path_to_dataset, data_file)
    all_data = json.load(open(path, 'r'))
    data = []
    for d in all_data:
        data.append(d)
    return data

def prepare_bert():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")

    embedder = AutoModel.from_pretrained('cointegrated/rubert-tiny')

    embedder.eval()
    try:
        embedder.cuda()
    except:
        pass
    return embedder, tokenizer

def make_sample_for_f0_prediction(text, tokens):
    text = text.strip()

    inds = []
    i = 0
    cur_word = 0

    while i < len(text):
        if text.lower()[i:].startswith(tokens[cur_word].replace('#', '')):
            inds.append(i + len(tokens[cur_word]) - tokens[cur_word].count('#'))
            i += len(tokens[cur_word]) - tokens[cur_word].count('#')
            cur_word += 1
        else:
            i += 1

    return inds



def get_embedding(text, tokenizer, embedder):
    tokens = tokenizer.tokenize(text.lower())
    ids = tokenizer.convert_tokens_to_ids(tokens)
    embedding = embedder(torch.tensor([ids]).long())[0].detach().cpu().numpy()
    return embedding,tokens


def compute_features(data, hparams, embedder, tokenizer, path_to_dataset):


    min_text_len = hparams.min_text_len
    max_text_len = hparams.max_text_len

    for f in tqdm(data):
        try:
            fn = f['filename'].split('/')[-1].split('.')[0]
            text = f['paused_text']
            labels = f['quantized_features']

            embedding,tokens = get_embedding(text, tokenizer, embedder)
            ids = make_sample_for_f0_prediction(text, tokens)

            ids = np.array(ids, dtype=int)
            labels = np.array(labels, dtype=float)
            #print(len(text),len(labels))
            if len(text) == len(labels) and min_text_len <= len(text) <= max_text_len:
                #print(str(os.path.join(hparams.path_to_tmp, "labels", fn)))
                np.save(os.path.join(hparams.path_to_tmp, "labels", fn),labels)
                np.save(os.path.join(hparams.path_to_tmp, "ids", fn), ids)
                np.save(os.path.join(hparams.path_to_tmp, "embedding", fn), embedding)
                np.save(os.path.join(hparams.path_to_tmp, "text", fn), text)

        except Exception as e:
            print(e)
            print('skip {}'.format(f))


def process(hparams):
    path_to_dataset = hparams.path_to_dataset
    data_labels_json = hparams.data_for_train
    data = load_data(hparams, data_labels_json)

    embedder, tokenizer = prepare_bert()
    print(len(data))
    os.makedirs('{}/labels'.format(hparams.path_to_tmp), exist_ok=True)
    os.makedirs('{}/ids'.format(hparams.path_to_tmp), exist_ok=True)
    os.makedirs('{}/text'.format(hparams.path_to_tmp), exist_ok=True)
    os.makedirs('{}/embedding'.format(hparams.path_to_tmp), exist_ok=True)

    compute_features(data, hparams, embedder, tokenizer, path_to_dataset)


