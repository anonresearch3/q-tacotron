import random
import numpy as np
import torch
import torch.utils.data
import os
from taco_utils import  load_filepaths_and_text
import glob
from tqdm import tqdm
class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)#[:1000]
        good_files = glob.glob(os.path.join(hparams.path_to_tmp, "text") + "/*")
        good_files = [os.path.basename(path) for path in good_files]
        print("load train data")
        self.all_data = []
        for i in tqdm(range(len(self.audiopaths_and_text))):
            filename = self.audiopaths_and_text[i][0].replace(".wav",".npy")
            if filename in good_files:
                labels = np.load(os.path.join(hparams.path_to_tmp, 'labels',filename))
                text = np.load(os.path.join(hparams.path_to_tmp, 'text', filename)).tolist()
                ids = np.load(os.path.join(hparams.path_to_tmp, 'ids', filename))
                embedding = np.load(os.path.join(hparams.path_to_tmp, 'embedding', filename))[0]
                mel = np.load(os.path.join(hparams.path_to_tmp, 'mel', filename))
                mel = torch.from_numpy(mel)
                self.all_data.append([self.audiopaths_and_text[i][0],text,labels,ids,embedding,mel])
        print("done")
        self.letter2index = {hparams.symbols[i]: i for i in range(len(hparams.symbols))}
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.path_to_dataset = hparams.path_to_dataset
        random.seed(hparams.seed)
        random.shuffle(self.all_data)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text,labels,ids,embedding,mel  = audiopath_and_text
        #text = "="+text+"~"
        #labels = np.concatenate([[[5] * 5],labels,[[5] * 5]])
        #ids = np.concatenate([[ids[0]],ids,[ids[-1]]])
        #embedding = np.concatenate([[embedding[0]], embedding,[embedding[-1]]])
        text = torch.tensor([self.letter2index[j] for j in text])
        return (text, mel,labels,ids,embedding)


    def __getitem__(self, index):
        return self.get_mel_text_pair(self.all_data[index])

    def __len__(self):
        return len(self.all_data)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        labels = torch.LongTensor(len(batch), max_input_len, 5)
        labels.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
            labels[i, :text.size(0)] = torch.from_numpy(batch[ids_sorted_decreasing[i]][2])


        # Right zero-pad all one-hot bert sequences to max input length

        max_input_len = max([len(x[4]) for x in batch])


        bert_embeddings = torch.FloatTensor(len(batch), max_input_len,312)
        bert_embeddings.zero_()
        ind = torch.LongTensor(len(batch), max_input_len)
        ind.zero_()
        for i in range(len(ids_sorted_decreasing)):
            bert_embedding = torch.from_numpy(batch[ids_sorted_decreasing[i]][4])
            bert_embeddings[i, :bert_embedding.size(0)] = bert_embedding
            ind[i, :bert_embedding.size(0)] = torch.from_numpy(batch[ids_sorted_decreasing[i]][3])



        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, labels, bert_embeddings, ind
