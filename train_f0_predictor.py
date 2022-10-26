import os
import numpy as np
import torch
import json
import argparse
from tqdm import tqdm, trange
import re
import pickle
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from utils import *
from hparams import Hparams
import os
from model import Tacotron2,QuantizedLabelPredictor
from distutils.version import LooseVersion
from process_utils.make_features import prepare_bert
is_pytorch_16plus = LooseVersion(torch.__version__) >= LooseVersion("1.6")
from train import prepare_dataloaders
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader




parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', type=str, default=60000)

args = parser.parse_args()
checkpoint = args.checkpoint

hparams = Hparams()
hparams.sampling_rate = 22050
device = torch.device('cpu')
embedder, tokenizer = prepare_bert()
letter2index = {hparams.symbols[i]: i for i in range(len(hparams.symbols))}
dims = [5,5,5,5,5]
model_name = "1"
checkpoint_path = "outdir/checkpoint_190000_reg_nogst"
tacotron = Tacotron2(hparams)#.cuda()

tacotron.load_state_dict(torch.load(checkpoint_path,map_location=device)['state_dict'])
tacotron.eval()
for layer in tacotron.label_predictor.children():
   if hasattr(layer, 'reset_parameters'):
       layer.reset_parameters()

model_path = 'outdir'
feature_set = [f.replace(' ', '') for f in "loudness, mean_f0, left_slope, right_slope, speed".split(',')]


destination_path = os.path.join(model_path, 'f0_predictor')
if not os.path.exists(destination_path):
    os.mkdir(destination_path)
import glob
audiopaths_and_text = load_filepaths_and_text(hparams.training_files)  # [:100]
good_files = glob.glob(os.path.join(hparams.path_to_tmp, "text") + "/*")
good_files = [os.path.basename(path) for path in good_files]
print("load train data")
all_data = []

batch_size = 64
num_epochs = 10
train_loader, valset, collate_fn = prepare_dataloaders(hparams)
val_loader = DataLoader(valset, sampler=None, num_workers=0,
                        shuffle=False, batch_size=batch_size,
                        pin_memory=False, collate_fn=collate_fn)

loss_function = CrossEntropyLoss()
lr = 1.e-3
optimizer = torch.optim.Adam(tacotron.label_predictor.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min= 1.e-5)
best_score = 0
best_epoch = 0
for epoch in range(num_epochs):

    print(f'Epoch: {epoch} lr {scheduler.get_last_lr()}')
    tacotron.label_predictor.train()

    train_losses = []
    for i,batch in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        f0_out = tacotron.f0(batch)
        btargets,ind, bmask =  batch[-3], batch[-1], batch[-4]
        #print(btargets.shape, ind.shape, bmask.shape, f0_out.shape)
        f0_target = torch.zeros((btargets.shape[0], ind.shape[1], btargets.shape[2])).long().to(device)
        for i in range(f0_target.shape[0]):
            f0_target[i] = torch.index_select(btargets[i], 0, torch.clamp(ind[i]-1, 0, len(btargets[i])-1))
        f0_loss = 0
        for j in range(f0_out.shape[3]):
            for i in range(f0_out.shape[0]):
                outj = f0_out[i, :bmask[i], :, j]
                tarj = f0_target[i, :bmask[i], j]

                f0_loss += loss_function(outj,tarj)
        f0_loss = f0_loss/f0_out.shape[0]
        f0_loss.backward()
        optimizer.step()
        print(f0_loss.item())
        train_losses.append(f0_loss.item())
    scheduler.step()

    tacotron.label_predictor.eval()
    eval_losses = []
    preds = {j: [] for j in range(len(dims))}
    gts = {j: [] for j in range(len(dims))}
    with torch.no_grad():
        for i,batch in tqdm(enumerate(val_loader)):
            f0_out = tacotron.f0(batch)
            btargets,ind, bmask =  batch[-3], batch[-1], batch[-4]

            f0_target = torch.zeros((btargets.shape[0], ind.shape[1], btargets.shape[2])).long().to(device)
            for i in range(f0_target.shape[0]):
                f0_target[i] = torch.index_select(btargets[i], 0, torch.clamp(ind[i] - 1, 0, len(btargets[i]) - 1))

            if len(f0_out.shape) == 3:
                f0_out = f0_out.unsqueeze(3)
                f0_target = f0_target.unsqueeze(2)
            f0_loss = 0
            f0_out = f0_out
            for j in range(f0_out.shape[3]):
                for i in range(f0_out.shape[0]):
                    outj = f0_out[i, :bmask[i], :, j]
                    tarj = f0_target[i, :bmask[i], j]
                    gts[j] += tarj.tolist()
                    preds[j] += np.argmax(outj.detach().cpu().numpy(), axis=1).tolist()
                    f0_loss += loss_function(outj,tarj)
        f0_loss = f0_loss/f0_out.shape[0]
        print(f0_loss.item())
    for j in range(len(dims)):
        print('feature: {}'.format(feature_set[j]))
        gts[j] = np.array(gts[j])
        preds[j] = np.array(preds[j])

        idx = gts[j] != dims[j] + 1
        gts[j] = gts[j][idx]
        preds[j] = preds[j][idx]
        print('accuracy: {}'.format(np.mean(gts[j] == preds[j])))
        for c in range(dims[j] + 1):
            print('acc {}: {:.3f}'.format(c, np.mean(preds[j][gts[j] == c] == c)))

    score = np.mean([f1_score(gts[j], preds[j], average='macro') for j in range(len(dims))])
    print('f1 macro: {}'.format(score))
    print('f1 micro: {}'.format(np.mean([f1_score(gts[j], preds[j], average='micro') for j in range(len(dims))])))
    if score > best_score:
        print('Model updated')
        best_score = score
        best_epoch = epoch

        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        if not os.path.exists(os.path.join(destination_path, 'for_ckpt_{}'.format(checkpoint))):
            os.mkdir(os.path.join(destination_path, 'for_ckpt_{}'.format(checkpoint)))

        if is_pytorch_16plus:
            torch.save(tacotron.state_dict(),
                       os.path.join(destination_path, 'for_ckpt_{}'.format(checkpoint), 'f0_predictor'),
                       _use_new_zipfile_serialization=False)

        else:
            torch.save(tacotron.state_dict(),
                       os.path.join(destination_path, 'for_ckpt_{}'.format(checkpoint), 'f0_predictor'))

    else:
        print('Model not updated from epoch {}.'.format(best_epoch))