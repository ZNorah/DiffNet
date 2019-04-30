#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 20:42 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Find difference of two similar image through learning in pytorch.
"""

import torch.nn as nn
import torchvision
import torch
from torchvision import datasets, models, transforms
import pdb

class ChannelAttention(nn.Module):
    def __init__(self, in_plane, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_plane, in_plane//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_plane//ratio, in_plane, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DiffNetwork(nn.Module):
    def __init__(self):
        super(DiffNetwork, self).__init__()
        # 16x16 and 512 channels
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(512)
        self.resnet18 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        self.regression = nn.Sequential(
                            # To 14x14
                            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 35, kernel_size=3, stride=1, padding=1)) # (5x7)x14x14

    def forward(self, inputa, inputb):
        outputa = self.resnet18(inputa)
        outputb = self.resnet18(inputb)
        sub_fea = outputa - outputb

        ###
        sub_fea = self.ca(sub_fea) * sub_fea
        sub_fea = self.sa(sub_fea) * sub_fea
        output = self.regression(sub_fea)
        return output
