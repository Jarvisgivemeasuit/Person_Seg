import os
import torch
from model.pspnet import PSPNet
from model.unet import UNet


def get_model(backbone, model_name, num_classes):
    if model_name == 'pspnet':
        return PSPNet(num_classes)
    elif model_name == 'unet':
        return UNet(backbone, num_classes)


def save_model(model, epoch, pred, miou, save_path, today):
    save_path = os.path.join(save_path, today)
    make_sure_path_exists(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "{}-{:.4f}-{:.4f}.pt".format(epoch, pred, miou)))

    print('saved model successful.')


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
