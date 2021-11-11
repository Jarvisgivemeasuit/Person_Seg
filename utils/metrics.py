import torch
import numpy as np
from sklearn.metrics import f1_score as fs


class PixelAccuracy:
    def __init__(self, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.eps = eps

    def update(self, pred, target):
        pred = torch.argmax(pred, dim=1)

        self.num_correct += (pred.long() & target.long()).sum().item()
        # print('\t', (pred.long() & target.long()).sum().item(), pred.sum(), target.sum())
        self.num_instance += target.sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


class MeanIoU:
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


class Kappa:
    def __init__(self, num_classes):
        self.pre_vec = np.zeros(num_classes)
        self.cor_vec = np.zeros(num_classes)
        self.tar_vec = np.zeros(num_classes)
        self.num = num_classes

    def update(self, output, target):
        pre_array = torch.argmax(output, dim=1)

        pre_mask = (pre_array == 1).byte()
        tar_mask = (target == 1).byte()
        self.cor_vec[0] = (pre_mask & tar_mask).sum().item()
        self.pre_vec[0] = pre_mask.sum().item()
        self.tar_vec[0] = tar_mask.sum().item()

    def get(self):
        assert len(self.pre_vec) == len(self.tar_vec) == len(self.pre_vec)
        tmp = 0.0
        for i in range(len(self.tar_vec)):
            tmp += self.pre_vec[i] * self.tar_vec[i]
        pe = tmp / (sum(self.tar_vec) ** 2 + 1e-8)
        p0 = sum(self.cor_vec) / (sum(self.tar_vec) + 1e-8)
        cohens_coefficient = (p0 - pe) / (1 - pe)
        return cohens_coefficient

    def reset(self):
        self.pre_vec = np.zeros(self.num)
        self.cor_vec = np.zeros(self.num)
        self.tar_vec = np.zeros(self.num)


class F1:
    def __init__(self):
        self.score = 0
        self.num = 0
        self.all = np.zeros(16)
        self.map = None
        self.map_tar = None

    def update(self, output, target):
        output = torch.argmax(output, dim=1).reshape(-1).cpu() 
        target = target.reshape(-1).cpu()

        self.score += fs(output, target, average='macro')
        self.num += 1
        if self.map == None:
            self.map = output
        else:
            self.map = torch.cat([self.map, output])
        if self.map_tar == None:
            self.map_tar = target
        else:
            self.map_tar = torch.cat([self.map_tar, target])

    def get(self):
        return self.score / self.num

    def reset(self):
        self.score = 0
        self.num = 0

    def get_all(self):
        return fs(self.map, self.map_tar, average=None)
