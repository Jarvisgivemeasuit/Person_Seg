import os
import math

from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


def make_sure_path_exists(path):
    '''
    Determines if a path exists; if it does not exist, creates this path.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def split_params(net):
    '''
    This function separate out the bias of BatchNorm2d from the parameters of the neural network.
    The bias of BatchNorm2d can be done without weight decay.

    Args: 
        net: A neural network.

    Returns: 
        decay: A list of parameters that need to be penalty.
        no_decay: A list of parameters that need not to be penalty.
    '''
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
    '''
    A pytorch LRScheduler with warm up strategy and reset LR stratgy.
    '''
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
