import os
import time
import numpy as np
from matplotlib import pyplot as plt

from collections import namedtuple
from progress.bar import Bar
from PIL import Image

import utils.metrics as metrics
from utils.args import Args
from utils.utils import *
from model import get_model, save_model
from dataset.person_dataset import PersonSeg

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim


class Tester:
    def __init__(self, Args):
        self.num_classes = PersonSeg.NUM_CLASSES
        self.args = Args
        self.start_epoch, self.epochs = 1, self.args.epochs
        self.best_pred, self.best_iou = 0, 0

        test_set = PersonSeg('test')
        self.train_loader = DataLoader(test_set, batch_size=self.args.test_batch_size,
                                       shuffle=True, num_workers=self.args.num_workers)
        self.mean, self.std = test_set.mean, test_set.std

        self.net = get_model(self.args.model_name, self.num_classes)
        self.net.load_state_dict(self.args.param_path)

        if self.args.cuda:
            self.net = self.net.cuda()

        if len(self.args.gpu_id) > 1:
            self.net = nn.DataParallel(self.net, self.args.gpu_ids)

    def testing(self):
        batch_time = AverageMeter()
        starttime = time.time()

        num_test = len(self.val_loader)
        bar = Bar('Infering', max=num_test)

        self.net.eval()

        for idx, sample in enumerate(self.val_loader):
            img, img_name = sample['image'], sample['file']
            if self.args.cuda:
                img = img.cuda()

            with torch.no_grad():
                output = self.net(img)

            self.save_image(output, img_name)
            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()
        bar.finish()
        print(f'numImages: {num_test * self.args.test_batch_size}]')

    def save_image(self, output, img_file):
        output_tmp = np.zeros(output.shape)
        output_tmp[output == 1] = 255
        make_sure_path_exists(self.args.test_results_path)
        output_tmp.save(os.path.join(self.args.test_results_path, img_file[0]))


def test():
    tester = Tester(Args)
    print("==> Start testing")
    tester.testing()

test()