import torch
import numpy as np
from sklearn.metrics import f1_score as fs


class AverageMeter:
    '''
    A generic class for averaging.
    '''
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


class PixelAccuracy:
    def __init__(self, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.eps = eps

    def update(self, pred, target):
        pred = torch.argmax(pred, dim=1)

        self.num_correct += (1 - pred.long() ^ target.long()).sum().item()
        self.num_instance += target.numel()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


class IoU:
    def __init__(self, eps=1e-7):
        self.num_intersection = 0
        self.num_union = 0
        self.eps = eps

    def update(self, pred, target):
        pred = torch.argmax(pred, dim=1)

        pred_mask = (pred == 1).byte()
        target_mask = (target == 1).byte()

        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()

        self.num_intersection += intersection.item()
        self.num_union += union.item()

    def get(self):
        iou_list = self.num_intersection / (self.num_union + self.eps)
        return iou_list

    def reset(self):
        self.num_intersection = 0
        self.num_union = 0

    def get_all(self):
        return (self.num_intersection / (self.num_union + self.eps))
