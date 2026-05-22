__author__ = "Pinkas Matěj"
__maintainer__ = "Pinkas Matěj"
__email__ = "pinkas.matej@gmail.com"
__created__ = "22/05/2026"
__date__ = "22/05/2026"
__status__ = "Prototype"
__version__ = "0.1.0"
__credits__ = []

"""
Project: BBAP20
Filename: model.py
Directory: src/localizer/
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.net(x)

def replace_bn_with_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features

            groups = num_groups
            if num_channels % groups != 0:
                groups = num_channels // 2

            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels)

            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)

class ResNetEncoder(nn.Module):
    def __init__(self, in_ch, out_ch=128, dropout=0.2):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if in_ch == 3:
            self.conv1.weight.data = resnet.conv1.weight.data
        else:
            self.conv1.weight.data = resnet.conv1.weight.data.sum(dim=1, keepdim=True)

        #replace_bn_with_gn(resnet)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        #self.maxpool = nn.Identity()

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer2[0].conv1.stride = (1, 1)
        self.layer2[0].downsample[0].stride = (1, 1)

        self.layer3 = resnet.layer3

        for module in self.layer3.modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)

        if self.layer3[0].downsample is not None:
            self.layer3[0].downsample[0].stride = (1, 1)

        self.final_conv = nn.Conv2d(256, out_ch, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.dropout(x)

        x = self.final_conv(x)
        return x

class GlobalLocalizationNet(nn.Module):
    def __init__(self, embed_dim=128, temperature=0.1):
        super().__init__()
        #self.enc_lidar = Encoder(1, embed_dim)
        #self.enc_sat = Encoder(3, embed_dim)
        self.enc_lidar = ResNetEncoder(1, embed_dim)
        self.enc_sat = ResNetEncoder(3, embed_dim)

        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        #self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.match = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        self.output_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, lidar, sat):
        B = lidar.shape[0]

        f_l = self.enc_lidar(lidar)
        f_s = self.enc_sat(sat)

        f_l = F.normalize(f_l, dim=1)
        f_s = F.normalize(f_s, dim=1)

        sat_input = f_s.view(1, B * f_s.shape[1], f_s.shape[2], f_s.shape[3])

        lidar_kernels = f_l

        k_h, k_w = f_l.shape[2], f_l.shape[3]

        pad_h = k_h // 2
        pad_w = k_w // 2

        heatmap = F.conv2d(sat_input, lidar_kernels, groups=B, padding=(pad_h, pad_w))

        heatmap = heatmap.view(B, 1, heatmap.shape[2], heatmap.shape[3])

        heatmap = heatmap.squeeze(1)

        scale = self.logit_scale.exp().clamp(max=10)
        #scale = self.logit_scale.exp()
        heatmap = heatmap * scale

        """heatmap = heatmap + self.output_bias
        heatmap = torch.sigmoid(heatmap)"""

        return heatmap


if __name__ == '__main__':
    pass

