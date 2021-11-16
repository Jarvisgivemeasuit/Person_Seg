import os
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from thop import profile

from model.PSPnet.mobilenet_v3 import *
# from mobilenet_v3 import *


NUM_CLASSES = 2


class MobileNetV3(nn.Module):
    def __init__(self, depth='small'):
        super().__init__()

        assert depth in ['large', 'small']
        self.model = mobilenet_v3_small(pretrained=True) if depth == 'small' \
                     else mobilenet_v3_large(pretrained=False)

        for m in self.model.features[4:].modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1

    def forward(self, x):
        out = self.model(x)
        return out

class PPM(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        planes = int(inplanes / 4)
        self.ppm1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
        )
        self.ppm2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
        )
        self.ppm3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
        )
        self.ppm4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        ppm1 = F.interpolate(self.ppm1(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm2 = F.interpolate(self.ppm2(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm3 = F.interpolate(self.ppm3(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm4 = F.interpolate(self.ppm4(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm = torch.cat([x, ppm1, ppm2, ppm3, ppm4], dim=1)
        return ppm


class Double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, inplanes, planes):
        super(Double_conv, self).__init__()
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


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = mobilenet_v3_small(pretrained=True).features
        for m in self.backbone[4:].modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1

        out_channels = self.backbone[-1].out_channels
        self.ppm = PPM(out_channels)
        self.ppm_conv = nn.Sequential(
            Double_conv(out_channels * 2, num_classes),
            nn.Dropout(p=0.1),
        )

        self.out_conv = nn.Conv2d(num_classes, num_classes, 3, padding=1, bias=True)
        self.out_conv.bias.data.zero_() - 20

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.backbone(x)

        x = self.ppm(x)
        x = self.ppm_conv(x)

        x = F.interpolate(x,
                            size=size,
                            mode='bilinear',
                            align_corners=True)

        out = self.out_conv(x)
        return out

    def freeze_backbone(self):
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
        for param in self.backbone.layer3.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = False

    def train_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# net = PSPNet(2).cuda()
# summary(net.cuda(), (3, 256, 256))