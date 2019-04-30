#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from torch.autograd import Variable
import pdb


__all__ = ['criterion']

"""
How to get value from GPU RAM
    You first need to get tensor out of the variable using .data 
    Then you need to move the tensor to cpu using .cpu()
    After that you can convert tensor to numpy using .numpy()
    And you probably know the rest... So basically a.data.cpu().numpy()[0] will give you just the value
"""

def criterion(label, pred, object_scale=1, noobject_scale=0.1, class_scale=0.3, coord_scale=5, num_class=2):
    ### INIT ###
    pred = pred.type('torch.cuda.DoubleTensor') 
    loss = Variable(torch.zeros((1)), requires_grad=True).cuda().type('torch.cuda.DoubleTensor')
    BS, ROW, COL = pred.size()[0], pred.size()[2], pred.size()[3]
    avg_iou, avg_cat, avg_allcat, avg_obj, avg_anyobj, count = 0, 0, 0, 0, 0, 0 
    delta = pred[:, 0, :, :]
    loss += noobject_scale * torch.sum(torch.mul(delta, delta))

    for i_pair in range(BS):
        count_obj = 0
        for row in range(ROW):
            for col in range(COL):
                ### Compute match bounding box of groundtruth
                if label[i_pair, 0, row, col].data[0]:
                    """
                    Label semantic:
                        _ __ ____
                        0 12 3456
                        + +  +> bbox
                        + ++> Which obj
                        ++> Object
                    """

                    # Forward and backward of category(2) 
                    delta = pred[i_pair, 1:1+num_class, row, col] - label[i_pair, 1:1+num_class, row, col]
                    loss += class_scale * torch.sum(torch.mul(delta, delta))

#                    print ("Loss stage 1, class: {:.5f}".format(loss.data[0]))

                    # Forward and backward of prob of obj
                    delta = pred[i_pair, 0, row, col] - 1.0
                    loss -= noobject_scale * (torch.mul(pred[i_pair, 0, row, col], pred[i_pair, 0, row, col]))
                    loss += object_scale * (delta ** 2)

#                    print ("Loss stage 2, obj: {:.5f}".format(loss.data[0]))

                    # Forward and backward of coordinates
                    tx, ty, tw, th = label[i_pair, 1+num_class:, row, col].data
                    #tx = tx / ROW
                    #ty = ty / psz[1]
                    ox, oy, ow, oh = pred[i_pair, 1+num_class:num_class+1+4, row, col].data
                   # ox /= psz[2]
                   # oy /= psz[1]
                    iou = box_iou([(ox+col)/COL, (oy+row)/ROW, ow, oh], [(tx+col)/COL, (ty+row)/ROW, tw, th])
#                    print('iou: {:.5f}'.format(iou))
                    loss += (iou - 1.0) *(iou - 1.0)
#                    print ("Loss stage 3, iou: {:.5f}".format(loss.data[0]))

                    delta = pred[i_pair, 1+num_class:, row, col] - label[i_pair, 1+num_class:, row, col]
                    loss += coord_scale * torch.sum(torch.mul(delta, delta))
#                    print ("Loss stage 4, coord: {:.5f}\n".format(loss.data[0]))
    loss = loss / BS

    return loss


### UTILS FOR IOU ###

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def box_intersection(a, b):
    def overlap(mida, wa, midb, wb):
        l1, l2 = mida - wa/2, midb - wb/2
        left = max(l1, l2)
        r1, r2 = mida + wa/2, midb + wb/2
        right = min(r1, r2)
        return right - left
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if (w <= 0 or h <= 0):
        return 0
    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a[2] * a[3] + b[2] * b[3] - i
    return u

# 均方根误差 root-mean-square error
def box_rmse(a, b):
    res = (a[0] - b[0]) ** 2
    res += (a[1] - b[1]) ** 2
    res += (a[2] - b[2]) ** 2
    res += (a[3] - b[3]) ** 2
    return sqrt(res)



