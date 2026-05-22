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
Directory: src/embedder/
"""


import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class CrossModalNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(CrossModalNetwork, self).__init__()

        self.sat_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)     # rgb for satellite encoder
        self.sat_encoder.fc = nn.Linear(self.sat_encoder.fc.in_features, embedding_dim) # replace to return embedding

        self.lidar_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # grayscale for lidar encoder
        original_conv = self.lidar_encoder.conv1
        self.lidar_encoder.conv1 = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )   # replace for single channel on input

        self.lidar_encoder.fc = nn.Linear(self.lidar_encoder.fc.in_features, embedding_dim) # replace to return embedding

    def forward(self, lidar_img, sat_img):
        sat_emb = self.sat_encoder(sat_img)
        lidar_emb = self.lidar_encoder(lidar_img)

        sat_emb = F.normalize(sat_emb, p=2, dim=1)      # L2 norm
        lidar_emb = F.normalize(lidar_emb, p=2, dim=1)  # L2 norm

        return sat_emb, lidar_emb


class CrossModalNetwork2(nn.Module):
    def __init__(self, embedding_dim=256):
        super(CrossModalNetwork2, self).__init__()

        self.sat_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        sat_in_features = self.sat_encoder.fc.in_features

        self.sat_encoder.fc = nn.Sequential(
            nn.Linear(sat_in_features, sat_in_features),
            nn.BatchNorm1d(sat_in_features),
            nn.ReLU(inplace=True),
            nn.Linear(sat_in_features, embedding_dim)
        )

        self.lidar_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        original_conv = self.lidar_encoder.conv1

        self.lidar_encoder.conv1 = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            self.lidar_encoder.conv1.weight[:] = torch.sum(original_conv.weight, dim=1, keepdim=True)

        lidar_in_features = self.lidar_encoder.fc.in_features

        self.lidar_encoder.fc = nn.Sequential(
            nn.Linear(lidar_in_features, lidar_in_features),
            nn.BatchNorm1d(lidar_in_features),
            nn.ReLU(inplace=True),
            nn.Linear(lidar_in_features, embedding_dim)
        )

    def forward(self, lidar_img, sat_img):
        sat_emb = self.sat_encoder(sat_img)
        lidar_emb = self.lidar_encoder(lidar_img)

        sat_emb = F.normalize(sat_emb, p=2, dim=1)  # L2 norm
        lidar_emb = F.normalize(lidar_emb, p=2, dim=1)  # L2 norm

        return sat_emb, lidar_emb

class CrossModalNetwork3(nn.Module):
    def __init__(self, embedding_dim=256):
        super(CrossModalNetwork3, self).__init__()

        self.sat_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        sat_in_features = self.sat_encoder.fc.in_features

        self.sat_encoder.fc = nn.Sequential(
            nn.Linear(sat_in_features, sat_in_features),
            nn.BatchNorm1d(sat_in_features),
            nn.ReLU(inplace=True),
            nn.Linear(sat_in_features, embedding_dim)
        )

        self.lidar_encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        original_conv = self.lidar_encoder.conv1

        self.lidar_encoder.conv1 = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            self.lidar_encoder.conv1.weight[:] = torch.sum(original_conv.weight, dim=1, keepdim=True)

        lidar_in_features = self.lidar_encoder.fc.in_features

        self.lidar_encoder.fc = nn.Sequential(
            nn.Linear(lidar_in_features, lidar_in_features),
            nn.BatchNorm1d(lidar_in_features),
            nn.ReLU(inplace=True),
            nn.Linear(lidar_in_features, embedding_dim)
        )

    def forward(self, lidar_img, sat_img):
        sat_emb = self.sat_encoder(sat_img)
        lidar_emb = self.lidar_encoder(lidar_img)

        sat_emb = F.normalize(sat_emb, p=2, dim=1)  # L2 norm
        lidar_emb = F.normalize(lidar_emb, p=2, dim=1)  # L2 norm

        return sat_emb, lidar_emb


class MarginContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, sat_emb, lidar_emb, label, is_positive_pair):

        euclidean_distance = F.pairwise_distance(sat_emb, lidar_emb)    # euclid distance between vectors

        if is_positive_pair:
            pass

        loss_positive = label * torch.pow(euclidean_distance, 2)    # loss of positive pairs (min)

        loss_negative = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # loss of negative pairs (at leat margin)

        loss = torch.mean(loss_positive + loss_negative)
        return loss


class TransformerCrossModalNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(TransformerCrossModalNetwork, self).__init__()

        self.sat_encoder = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        self.sat_encoder.head = nn.Linear(self.sat_encoder.head.in_features, embedding_dim)

        self.lidar_encoder = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

        original_conv = self.lidar_encoder.features[0][0]
        self.lidar_encoder.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        self.lidar_encoder.head = nn.Linear(self.lidar_encoder.head.in_features, embedding_dim)

    def forward(self, lidar_img, sat_img):
        sat_emb = self.sat_encoder(sat_img)
        lidar_emb = self.lidar_encoder(lidar_img)

        sat_emb = F.normalize(sat_emb, p=2, dim=1)
        lidar_emb = F.normalize(lidar_emb, p=2, dim=1)

        return sat_emb, lidar_emb


if __name__ == '__main__':
    pass

