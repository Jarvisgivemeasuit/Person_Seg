import os
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from thop import profile
from thop import clever_format

from torchvision.models.mobilenetv3 import *


NUM_CLASSES = 2
mobilenet_downsample_stage = [0, 1, 2, 4, 9]


class MobileNetV3(nn.Module):
    def __init__(self, depth='small'):
        super().__init__()

        assert depth in ['large', 'small']
        self.model = mobilenet_v3_small(pretrained=True) if depth == 'small' \
                     else mobilenet_v3_large(pretrained=False)
        self.mobilenet_concat_stage = [0, 1, 3, 12]

        for m in self.model.features[9:].modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1

    def forward(self, x):
        out = [x]
        for i, stage in enumerate(self.model.features):
            x = stage(x)
            if i in self.mobilenet_concat_stage:
                out.append(x)
        return out


class Up(nn.Module):
    def __init__(self, d_inplanes, d_planes, last_cat=False):
        super(Up, self).__init__()
        self.last_cat = last_cat
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = Double_conv(d_inplanes, d_planes)

    def forward(self, x1, x2):
        if not self.last_cat:
            x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetV3()

        self.up1 = Up(576+24, 16)
        self.up2 = Up(16+16, 16)
        self.up3 = Up(16+16, 16)
        self.up4 = Up(16+3, num_classes)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3 = self.up1(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up3(x2, x1)
        x = self.up4(x1, x0)
        return x

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# net = UNet(2).cuda()

# summary(net.cuda(), (3, 256, 256))
# inputs = torch.randn(1, 3, 256, 256).cuda()
# flops, params = profile(net, inputs=(inputs, ))
# flops, params = clever_format([flops, params], "%.3f")
# print('flops=', flops, 'params=', params)