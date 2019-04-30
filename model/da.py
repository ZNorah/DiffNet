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

class PAM(nn.Module):
    def __init__(self, in_plane):
        super(PAM, self).__init__()

        self.query_conv = nn.Conv2d(in_plane, in_plane//8, 1, bias=False)
        self.key_conv = nn.Conv2d(in_plane, in_plane//8, 1, bias=False)
        self.value_conv = nn.Conv2d(in_plane, in_plane, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, C, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, w*h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, w*h)
        energy = torch.bmm(proj_query, proj_key)
        att = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, -1, w*h)

        out = torch.bmm(proj_value, att.permute(0, 2, 1))
        out = out.view(b, C, h, w)
        out = self.gamma*out + x
        return out


class CAM(nn.Module):
    def __init__(self, in_plane):
        super(CAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        b, C, h, w = x.size()
        proj_query = x.view(b, C, -1)
        proj_key = x.view(b, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        att = self.softmax(energy_new)
        proj_value = x.view(b, C, -1)

        out = torch.bmm(att, proj_value)
        out = out.view(b, C, h, w)
        out = self.gamma*out + x
        return out


class DiffNetwork(nn.Module):
    def __init__(self):
        super(DiffNetwork, self).__init__()
        # 16x16 and 512 channels
        self.ca = CAM(512)
        self.sa = PAM(512)
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
        sub_ca = self.ca(sub_fea)
        sub_sa = self.sa(sub_fea)
        output = self.regression(sub_ca+sub_sa)
        return output
