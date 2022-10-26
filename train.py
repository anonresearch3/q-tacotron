import os
import time
import argparse

import torch

from torch.utils.data import DataLoader
import numpy as np
from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import TPGSTTacotron2Loss
from hparams import Hparams
from tqdm import tqdm

def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(trainset, num_workers=0, shuffle=True,
                              sampler=None,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn




def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, collate_fn):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(valset, sampler=None, num_workers=0,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        i = 0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            losses = criterion(y_pred, y)
            loss = losses[0] + losses[1] + losses[2] + losses[3] + losses[4]
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))

def step2lr(step, max_lr):
    # linearly increase lr from `1.e-5` to `max_lr` from 0-th to 30000-th step
    # exponential decay lr from `max_lr` to `1.e-5` from 30000-th step
    # alpha in lr decay is set to decrease lr in 1.2 times every 10000 steps

    min_lr = 1.e-5

    if step < 30000:
        return min_lr + (max_lr - min_lr) * step / 30000
    else:
        return min_lr + (max_lr - min_lr) / 1.2 ** ((step - 30000) / 10000)

def train(output_directory, checkpoint_path, warm_start, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    checkpoint_path(string): checkpoint path
    hparams (object): comma separated list of "name=value" pairs.
    """

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = Tacotron2(hparams).to(device)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    criterion = TPGSTTacotron2Loss()

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
    history = {}
    loss_name = ["mel_loss",'postnet_mel_loss','gate_loss','tp_loss','f0_loss']
    for key in loss_name:
        history[key] = []
    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for step, batch in enumerate(tqdm(train_loader)):
            start = time.perf_counter()
            for u in range(len(optimizer.param_groups)):
                optimizer.param_groups[u]['lr'] = step2lr(step + epoch * hparams.iters_per_checkpoint, learning_rate)

            model.zero_grad()

            x, y = model.parse_batch(batch)

            y_pred = model(x)

            losses = criterion(y_pred, y)
            loss = losses[0] + losses[1] + losses[2] + losses[3] + losses[4]
            for i,key in enumerate(loss_name):
                history[key].append(losses[i].item())


            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                duration = time.perf_counter() - start
                print('Epoch {} ended {} step {} grad {}'.format(epoch,iteration,duration,grad_norm))
                for i,key in enumerate(loss_name):
                    print('{} loss: {}'.format(key,np.mean(history[key][-100:])))

                validate(model, criterion, valset, iteration,
                         hparams.batch_size, collate_fn)

                checkpoint_path = os.path.join(
                    output_directory, "checkpoint_{}".format(iteration))
                save_checkpoint(model, optimizer, learning_rate, iteration,
                                checkpoint_path)

            iteration += 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')

    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = Hparams()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("device", device)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)
    
    os.makedirs(args.output_directory, exist_ok=True)
    
    train(args.output_directory, args.checkpoint_path, args.warm_start, hparams)
