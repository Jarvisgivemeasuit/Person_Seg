import os
import time
import sys
import numpy as np

from collections import namedtuple
from progress.bar import Bar
from apex import amp
from PIL import Image
from tensorboardX import SummaryWriter

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
import numpy as np 
import math


class Trainer:
    def __init__(self, Args):
        self.num_classes = PersonSeg.NUM_CLASSES
        self.args = Args
        self.start_epoch, self.epochs = 1, self.args.epochs
        self.best_pred, self.best_iou = 0, 0

        train_set, val_set = PersonSeg('train'), PersonSeg('val')
        self.train_loader = DataLoader(train_set, batch_size=self.args.tr_batch_size,
                                       shuffle=True, num_workers=self.args.num_workers)
        self.val_loader = DataLoader(val_set, batch_size=self.args.vd_batch_size,
                                     shuffle=False, num_workers=self.args.num_workers)
        self.mean, self.std = train_set.mean, train_set.std

        self.net = get_model(self.args.model_name, self.num_classes)
        # params, _ = split_params(self.net)
        # self.optimizer = torch.optim.SGD(params, lr=self.args.lr,
        #                                  momentum=0.9, weight_decay=self.args.weight_decay)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr,
                                         momentum=0.9, weight_decay=self.args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        if self.args.cuda:
            self.net, self.criterion = self.net.cuda(), self.criterion.cuda()

        if self.args.apex:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')

        if len(self.args.gpu_id) > 1:
            self.net = nn.DataParallel(self.net, self.args.gpu_ids)

        if self.args.reset_times:
            self.lr = lambda iters: self.args.get_lr(reset_times=self.args.reset_times, 
                                                     epochs=self.args.epochs, 
                                                     iterations=len(self.train_loader), 
                                                     iters=iters, 
                                                     lr_init=self.args.lr,
                                                     lr_min=self.args.lr_min, 
                                                     warm_up_epoch=self.args.warm_up_epoch)

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs * len(self.train_loader), eta_min=1e-5)

        self.Metric = namedtuple('Metric', 'pixacc iou')
        self.train_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        iou=metrics.IoU(self.num_classes))
        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        iou=metrics.IoU(self.num_classes))

    def training(self, epoch):

        self.train_metric.pixacc.reset()
        self.train_metric.iou.reset()

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_train = len(self.train_loader)
        bar = Bar('Training', max=num_train)

        self.net.train()
        for idx, sample in enumerate(self.train_loader):
            img, tar = sample['image'], sample['label']
            if self.args.cuda:
                img, tar = img.cuda(), tar.cuda()

            self.optimizer.zero_grad()
            output = self.net(img)
            loss = self.criterion(output, tar.long())
            losses.update(loss)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss = scale_loss.half()
                    scale_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            self.train_metric.pixacc.update(output, tar)
            self.train_metric.iou.update(output, tar)

            batch_time.update(time.time() - starttime)
            starttime = time.time()
            # print([group['lr'] for group in self.optimizer.param_groups])

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f} | Acc:{Acc:.4f} | IoU:{IoU:.4f} | LR:{lr:.5f}'.format(
                batch=idx + 1,
                size=len(self.train_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                IoU=self.train_metric.iou.get(),
                Acc=self.train_metric.pixacc.get(),
                lr=self.get_lr()
            )
            bar.next()
        bar.finish()
        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)

    def validation(self, epoch):

        self.train_metric.pixacc.reset()
        self.val_metric.iou.reset()

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_val = len(self.val_loader)
        bar = Bar('Validation', max=num_val)

        self.net.eval()

        for idx, sample in enumerate(self.val_loader):
            img, tar = sample['image'], sample['label']
            if self.args.cuda:
                img, tar = img.cuda(), tar.cuda()

            with torch.no_grad():
                output = self.net(img)
                loss = self.criterion(output, tar.long())
            losses.update(loss)

            if idx < 5:
                self.visualize_batch_image(img, tar, output, epoch, idx)

            self.val_metric.pixacc.update(output, tar)
            self.val_metric.iou.update(output, tar)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f} | Acc:{Acc:.4f} | IoU:{IoU:.4f}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                IoU=self.val_metric.iou.get(),
                Acc=self.val_metric.pixacc.get(),
            )
            bar.next()
        bar.finish()

        print(f'Validation:[Epoch: {epoch}, numImages: {num_val * self.args.vd_batch_size}]')
        print(f'Valid Loss: {losses.avg:.4f}')
        if self.val_metric.iou.get() > self.best_iou:
            if  self.val_metric.pixacc.get() > self.best_pred:
                self.best_pred = self.val_metric.pixacc.get()
            self.best_iou = self.val_metric.iou.get()

            save_model(self.net, epoch, self.best_pred, self.best_iou)
        print("-----best acc:{:.4f}, best iou:{:.4f}-----".format(self.best_pred, self.best_iou))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def visualize_batch_image(self, image, target, output, epoch, batch_index):
        # image (B,C,H,W) To (B,H,W,C)
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, axes=[0, 2, 3, 1])
        image_np *= self.std
        image_np += self.mean
        image_np *= 255
        image_np = image_np.astype(np.uint8)
        image_np = image_np[:, :, :, 1:]

        # target (B,H,W)
        target = target.cpu().numpy()

        # output (B,C,H,W) to (B,H,W)
        output = torch.argmax(output, dim=1).cpu().detach().numpy()

        for i in range(min(3, image_np.shape[0])):
            img_tmp = image_np[i]
            img_rgb_tmp = np.array(Image.fromarray(img_tmp).convert("RGB")).astype(np.uint8)

            target_tmp = np.zeros(target[i].shape)
            target_tmp[target[i] == 1] = 255

            output_tmp = np.zeros(output[i].shape)
            output_tmp[output[i] == 1] = 255

            plt.figure()
            plt.title('display')
            plt.subplot(131)
            plt.imshow(img_rgb_tmp, vmin=0, vmax=255)
            plt.subplot(132)
            plt.imshow(target_tmp, vmin=0, vmax=255)
            plt.subplot(133)
            plt.imshow(output_tmp, vmin=0, vmax=255)

            save_path = os.path.join(self.args.vis_image_dir, f'epoch_{epoch}')
            make_sure_path_exists(save_path)
            plt.savefig(f"{save_path}/{batch_index}-{i}.jpg")
            plt.close('all')


def train():
    args = Args()
    trainer = Trainer(args)
    print("==> Start training")
    print('Total Epoches:', trainer.epochs)
    print('Starting Epoch:', trainer.start_epoch)
    for epoch in range(trainer.start_epoch, trainer.epochs + 1):
        trainer.training(epoch)

        if not args.no_val:
            new_pred = trainer.validation(epoch)

train()
