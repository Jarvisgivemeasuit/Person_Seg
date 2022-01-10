import os
import cv2
import time
import numpy as np

from collections import namedtuple
from progress.bar import Bar

import utils.metrics as metrics
from utils.args import Args
from utils.utils import *
from model import get_model
from dataset.person_dataset import PersonSeg
from dataset.path import Path

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim


class Validator:
    def __init__(self, Args):
        self.num_classes = PersonSeg.NUM_CLASSES
        self.args = Args

        data_set = PersonSeg('', Path.db_root_dir('medi'))

        self.data_loader = DataLoader(data_set, batch_size=self.args.test_batch_size,
                                       shuffle=False, num_workers=self.args.num_workers)
        self.mean, self.std = data_set.mean, data_set.std

        self.net = get_model(self.args.backbone, self.args.model_name, self.num_classes)
        if self.args.model_pretrain:
            self.net.load_state_dict(torch.load(self.args.param_path))

        self.criterion = nn.CrossEntropyLoss()

        if self.args.cuda:
            self.net, self.criterion = self.net.cuda(), self.criterion.cuda()

        if len(self.args.gpu_id) > 1:
            self.net = nn.DataParallel(self.net, self.args.gpu_ids)

        self.Metric = namedtuple('Metric', 'pixacc iou')
        self.metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        iou=metrics.IoU(self.num_classes))

    def validation(self):

        self.metric.pixacc.reset()
        self.metric.iou.reset()

        batch_time = metrics.AverageMeter()
        losses = metrics.AverageMeter()
        starttime = time.time()

        num_val = len(self.data_loader)
        bar = Bar('Validation', max=num_val)

        self.net.eval()

        for idx, sample in enumerate(self.data_loader):
            img, tar, img_file = sample['image'], sample['label'], sample['file']
            if self.args.cuda:
                img, tar = img.cuda(), tar.cuda()

            with torch.no_grad():
                output = self.net(img)
                loss = self.criterion(output, tar.long())
            losses.update(loss)

            self.metric.pixacc.update(output, tar)
            self.metric.iou.update(output, tar)

            self.save_image(output, img_file)
            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f} | Acc:{Acc:.4f} | IoU:{IoU:.4f}'.format(
                batch=idx + 1,
                size=len(self.data_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                IoU=self.metric.iou.get(),
                Acc=self.metric.pixacc.get(),
            )
            bar.next()
        bar.finish()

    def save_image(self, output, img_file):
        output = torch.argmax(output, dim=1).cpu().detach().permute(1, 2, 0).numpy()

        output_tmp = np.zeros(output.shape)
        output_tmp[output == 1] = 255

        make_sure_path_exists(self.args.test_results_path)
        cv2.imwrite(os.path.join(self.args.test_results_path, img_file[0]), output_tmp)


def train():
    args = Args()
    validator = Validator(args)
    print("==> Start validation")
    validator.validation()

train()
