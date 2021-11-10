import torch
import os
class Args:

    def __init__(self):
        self.tr_batch_size = 64
        self.vd_batch_size = 64

        self.num_workers = 4
        self.inplanes = 3

        self.model_name = 'pspnet'
        self.epochs = 100

        self.lr = 0.01
        self.warm_up_epoch = 5
        self.no_val = False

        self.gpu_ids = [0, 1, 2, 3]
        self.gpu_id = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        self.vis_image_dir = '/home/lijl/Documents/Person_Seg/vis_image/'
        # self.board_dir = 'pspnet_dpa'
