from torch import nn
import torch

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss


class TPGSTTacotron2Loss(nn.Module):
    def __init__(self):
        super(TPGSTTacotron2Loss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss()
    def forward(self, model_output, targets):
        mel_target, gate_target, btargets, ind, input_lengths = targets

        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_out, mel_out_postnet, gate_out, _, gst, tp, f0_out = model_output

        try:
            f0_target = torch.zeros((btargets.shape[0], ind.shape[1], btargets.shape[2])).long().cuda()
        except:
            f0_target = torch.zeros((btargets.shape[0], ind.shape[1], btargets.shape[2])).long()
        for i in range(f0_target.shape[0]):
            f0_target[i] = torch.index_select(btargets[i], 0, torch.clamp(ind[i]-1, 0, len(btargets[i])-1))

        if len(f0_out.shape) == 3:
            f0_out = f0_out.unsqueeze(3)
            f0_target = f0_target.unsqueeze(2)

        f0_loss = 0
        for j in range(f0_out.shape[3]):
            for i in range(f0_out.shape[0]):
                outj = f0_out[i, :input_lengths[i], :, j]
                tarj = f0_target[i, :input_lengths[i], j]

                f0_loss += self.loss_function(outj,tarj)
        f0_loss = f0_loss/f0_out.shape[0]



        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target)
        postnet_mel_loss = nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        tp_loss = 10 * nn.MSELoss()(gst.squeeze(1), tp)
        return mel_loss, postnet_mel_loss, gate_loss, tp_loss, f0_loss
