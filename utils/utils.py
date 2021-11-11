import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
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


class Circumference(nn.Module):
    '''计算mask中所有种类的平均周长'''
    def __init__(self):
        super().__init__()

    def cal_circle(self, pred):
        segmap = self.label_indices(pred.transpose(2, 0, 1))
        print(segmap.shape)
        scale = segmap.shape
        ans_rows = np.zeros((scale[0] - 1, scale[1]))
        ans_cols = np.zeros((scale[1] - 1, scale[0]))
        for i in range(scale[0] - 1):
            ans_rows[i] = segmap[i] != segmap[i + 1]
        for i in range(scale[1] - 1):
            ans_cols[i] = segmap[:, i] != segmap[:, i + 1]
        return (ans_rows.sum() + ans_cols.sum()) / NUM_CLASSES

    def label_indices(self, mask):
        # # colormap2label
        colormap2label = np.zeros(256**3)
        mask_colormap = np.array([[0, 0, 0], [255, 255, 255]])
        for i, colormap in enumerate(mask_colormap):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

        # colormap2mask
        mask = mask.astype('int32')
        idx = (mask[0, :, :] * 256 + mask[1, :, :]) * 256 + mask[2, :, :]
        return colormap2label[idx].astype('int32')


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
    # path = '/home/arron/dataset/rssrai_grey/rssrai/train/img'
    # path = '/home/arron/dataset/rssrai_grey/increase/rssrai/test'
    # save_path = '/home/arron/dataset/rssrai_grey/results/dt_resunet-resnet50' 
    # res_path = '/home/arron/dataset/rssrai_grey/results/tmp_output/dt_resunet-resnet50'

    path = '/home/mist/rssrai/ori_img/val/img'
    save_path = '/home/mist/results/unet-resnet50' 
    res_path = '/home/mist/results/tmp'
    supermerger = SuperMerger(path, res_path, save_path)
    supermerger.merge_image()


    # lists = os.listdir(res_path)
    # i = 0
    # for files in lists:
    #     if '_'.join(files.split('_')[:-2]) == 'GF2_PMS1__20160623_L1A0001660727-MSS1':
    #         print(files)
    #         i += 1
    # print(i)
    # print(len(lists))
    