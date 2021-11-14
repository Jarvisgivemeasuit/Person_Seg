import torch
import os
import math


class Args:
    def __init__(self):
        self.tr_batch_size = 64
        self.vd_batch_size = 64

        self.num_workers = 4
        self.inplanes = 3

        self.model_name = 'pspnet'
        self.epochs = 100

        self.lr = 0.01
        self.lr_min = 1e-5
        self.reset_times = 5
        
        self.warm_up_epoch = 0
        self.weight_decay = 5e-5
        self.no_val = False

        self.gpu_ids = [0, 1, 2, 3]
        self.gpu_id = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        self.vis_image_dir = '/home/lijl/Documents/Person_Seg/vis_image/'
        # self.board_dir = 'pspnet'

    def get_lr(self, reset_times, epochs, iterations, iters, lr_init, lr_min, warm_up_epoch=0):
        warm_step, lr_gap = iterations * warm_up_epoch, lr_init - lr_min
        if iters < warm_step:
            lr = lr_gap / warm_step * iters
        else:
            lr_lessen = int((epochs - warm_up_epoch) / reset_times * iterations)
            lr = 0.5 * ((math.cos((iters - warm_step) % lr_lessen / lr_lessen * math.pi)) + 1) * lr_gap  + lr_min
        
        return lr