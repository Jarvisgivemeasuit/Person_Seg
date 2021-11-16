import os
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.autograd import Variable


NUM_CLASSES = 2


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def split_params(net):
    decay, no_decay = [], []
    
    for stage in net.modules():
        for m in stage.modules():
            if isinstance(m, nn.BatchNorm2d):
                no_decay.append(m.bias)
                decay.append(m.weight)
            else:
                decay.extend([*m.parameters()])
    return decay, no_decay


class ResetLR(_LRScheduler):
    def __init__(self, optimizer, lr_init, lr_min, warm_up_epoch, reset_times, epochs, iterations):
        super().__init__()
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.warm_up_epoch = warm_up_epoch
        self.reset_times = reset_times
        self.epochs = epochs
        self.iterations = iterations
    
    def get_lr(self, iters):
        warm_step, lr_gap = self.iterations * self.warm_up_epoch, self.lr_init - self.lr_min
        if iters < warm_step:
            lr = lr_gap / warm_step * iters
        else:
            lr_lessen = int((self.epochs - self.warm_up_epoch) / self.reset_times * self.iterations)
            lr = 0.5 * ((math.cos((iters - warm_step) % lr_lessen / lr_lessen * math.pi)) + 1) * lr_gap  + self.lr_min
        
        return [lr]
    
    def _get_closed_form_lr(self):
        return [0.5 * ((math.cos((self.last_epoch - self.warm_up_epoch) / 
                                 (self.epochs - self.warm_up_epoch) * math.pi)) + 1) * self.lr_gap  + self.lr_min]


class SoftCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-1, times=1, eps=1e-7, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.times = times
        self.eps = eps
        if weight is None:
            self.weight = weight
        else:
            self.weight = weight.cuda()

    def forward(self, pred, target):
        mask = target != self.ignore_index
        pred = F.log_softmax(pred, dim=-1)
        loss = -pred * target
        loss = loss * mask.float()
        # print(loss, pred, target, mask)
        if self.weight is None:
            return self.times * loss.sum() / (mask.sum() + self.eps)
        else:
            return self.times * (self.weight * loss).sum() / (mask.sum() + self.eps)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, eps=1e-7, reducation='mean'):
        super().__init__()
        self.alpha = Variable(torch.tensor(alpha))
        self.gamma = gamma
        self.eps = eps
        self.reducation = reducation

    def forward(self, pred, target):
        N = pred.shape[0]
        C = pred.shape[1]
        num_pixels = pred.shape[2] * pred.shape[3]

        target_index = target.view(target.shape[0], target.shape[1], target.shape[2], 1)
        class_mask = torch.zeros([N, pred.shape[2], pred.shape[3], C]).cuda()
        class_mask = class_mask.scatter_(3, target_index, 1.)
        class_mask = class_mask.transpose(1, 3)
        class_mask = class_mask.view(pred.shape)

        logsoft_pred = F.log_softmax(pred, dim=1)
        soft_pred = F.softmax(pred, dim=1)

        loss = -self.alpha * ((1 - soft_pred)) ** self.gamma * logsoft_pred
        loss = loss * class_mask
        loss = loss.sum(1)

        if self.reducation == 'mean':
            return loss.sum() / (class_mask.sum() + self.eps)
        else:
            return loss.sum()


if __name__ == '__main__':
    pass
    