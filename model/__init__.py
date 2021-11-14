import os
import torch
from model.Unet.unet import UNet
from model.PSPnet.pspnet import PSPNet
import datetime


def get_model(model_name, num_classes):
    if model_name == 'pspnet':
        return PSPNet(num_classes)


def save_model(model, epoch, pred, miou):
    save_path = f'/home/lijl/Documents/Person_Seg/model_saving/{datetime.date.today}'
    make_sure_path_exists(save_path)
    torch.save(model, os.path.join(save_path, "{}-{:.4f}-{:.4f}.pth".format(epoch, pred, miou)))

    print('saved model successful.')


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
