import torch.nn as nn
from torchsummary import summary

from torchvision.models.mobilenetv3 import *
from torchvision.models.mobilenetv2 import * 
from torchvision.models.resnet import *


mobilenetv3_downsample_stage = [0, 1, 2, 4, 9]
mobilenetv2_downsample_stage = [0, 2, 4, 7, 14]


class MobileNetV3(nn.Module):
    def __init__(self, depth='small', get_all=False):
        super().__init__()
        assert depth in ['large', 'small']

        self.get_all = get_all
        self.model = mobilenet_v3_small(pretrained=True) if depth == 'small' \
                     else mobilenet_v3_large(pretrained=False)
        self.mobilenet_concat_stage = [0, 1, 3, 12]
        self.out_channels = []
        for i, stage in enumerate(self.model.features):
            if i in self.mobilenet_concat_stage:
                self.out_channels.append(stage.out_channels)

        for m in self.model.features[9:].modules():
            if isinstance(m, nn.Conv2d):
                m.stride = 1

    def forward(self, x):
        out = [x]
        for i, stage in enumerate(self.model.features):
            x = stage(x)
            if i in self.mobilenet_concat_stage:
                out.append(x)
        if self.get_all:
            return out
        else:
            return out[-1]


class MobileNetV2(nn.Module):
    def __init__(self, get_all=False):
        super().__init__()
        self.get_all = get_all
        self.model = mobilenet_v2(pretrained=True).features[:-1]
        self.mobilenet_concat_stage = [1, 3, 6, 13]
        self.out_channels = []
        for i, stage in enumerate(self.model):
            if i in self.mobilenet_concat_stage:
                self.out_channels.append(stage.out_channels)

    def forward(self, x):
        out = [x]
        for i, stage in enumerate(self.model):
            x = stage(x)
            if i in self.mobilenet_concat_stage:
                out.append(x)
                self.out_channels.append(stage.out_channels)
        if self.get_all:
            return out
        else:
            return out[-1]


class ResNet(nn.Module):
    def __init__(self, get_all=False):
        super().__init__()
        self.get_all = get_all
        self.model = resnet18(pretrained=True)
        self.out_channels = []
        self.first_layer = nn.Sequential(*list(self.model.children())[:2])
        self.layer0 = nn.Sequential(*list(self.model.children())[2:4])

        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        # self.layer4 = self.model.layer4
        # for m in self.layer4.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.stride = 1

        self.out_channels.append(self.first_layer[0].out_channels)
        self.out_channels.append(self.layer1[0].conv2.out_channels)
        self.out_channels.append(self.layer2[0].conv2.out_channels)
        self.out_channels.append(self.layer3[0].conv2.out_channels)
        

    def forward(self, x):
        out = [x]
        x = self.first_layer(x)
        out.append(x)
        x = self.layer0(x)
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        out.append(x)

        if self.get_all:
            return out
        else:
            return out[-1]


def get_backbone(model_name, get_all):
    assert model_name in ['mobilenetv2', 'mobilenetv3', 'resnet']

    if model_name == 'mobilenetv2':
        model = MobileNetV2(get_all)
    elif model_name == 'mobilenetv3':
        model = MobileNetV3(get_all=get_all)
    elif model_name == 'resnet':
        model = ResNet(get_all)

    return model


# net = ResNet()

# summary(net.cuda(), (3, 256, 256))
# print(net.out_channels)