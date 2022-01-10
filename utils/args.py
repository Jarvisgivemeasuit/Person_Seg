import torch
import os
import math


class Args:
    def __init__(self):
        self.tr_batch_size = 32
        self.vd_batch_size = 64
        self.test_batch_size = 1

        self.num_workers = 4
        self.inplanes = 3

        self.backbone = 'mobilenetv3'
        self.model_name = 'pspnet'
        self.epochs = 50

        self.lr = 1e-3
        self.lr_min = 1e-5
        self.reset_times = 0
        
        self.warm_up_epoch = 0
        self.weight_decay = 5e-5
        self.no_val = False

        self.gpu_ids = [0, 1, 2, 3]
        self.gpu_id = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        self.model_pretrain = True
        self.vis_image_path = '/home/lijl/Documents/Person_Seg/vis_image/'
        self.model_save_path = '/home/lijl/Documents/Person_Seg/model_saving/'
        self.param_path = '/home/lijl/Documents/Person_Seg/model_saving/2022-01-10/50-0.9907-0.7359.pt'
        # self.param_path = '/home/lijl/Documents/Person_Seg/pspnet_params/98-0.9363-0.7183.pt'
        # self.test_results_path = '/home/lijl/Datasets/segmentation/coco_person/test/Results'
        self.test_results_path = '/home/lijl/Datasets/segmentation/medical_room/Results'

        self.thres = 4000

    def get_lr(self, reset_times, epochs, iterations, iters, lr_init, lr_min, warm_up_epoch=0):
        warm_step, lr_gap = iterations * warm_up_epoch, lr_init - lr_min
        if iters < warm_step:
            lr = lr_gap / warm_step * iters
        else:
            lr_lessen = int((epochs - warm_up_epoch) / reset_times * iterations)
            lr = 0.5 * ((math.cos((iters + 1 - warm_step) % lr_lessen / lr_lessen * math.pi)) + 1) * lr_gap  + lr_min
        
        return lr / lr_init
