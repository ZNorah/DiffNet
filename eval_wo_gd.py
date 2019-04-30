#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017-10-25 21:09 zq <zq@mclab>
#
# Distributed under terms of the MIT license.

"""
Evaluate our best model without groundtruth.
"""

import argparse
import torch
import numpy as np
import time
import importlib

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from data.dataset import Pair_Dataset
from loss.v2loss import v2loss
import os
import os.path as osp
from utils import render_v2wo_gd


def evaluate(args):
    ### DATA ###
    dataclass = Pair_Dataset(args.test_dir, test=True)
    imkey_list = dataclass.imkey_list
    dataloader = {}
    dataloader["test"] = DataLoader(dataclass,
                                  1, 
                                  shuffle=False, 
                                  num_workers=args.num_workers)
    if not osp.exists(args.desdir):
        os.makedirs(args.desdir)
    fa, fb = open(args.deta_fn, "w"), open(args.detb_fn, "w")
    anchors = [(41.4514,52.2819), (63.2610,59.1658), (52.3392,76.1706), (94.5413,77.2516), (71.5646,108.9415)]
    criterion = v2loss(anchors=anchors, coord_scale=2)
    ### LOAD MODEL ###
    if osp.exists(args.model_fn):
        model_weights = torch.load(args.model_fn)
        model = importlib.import_module("model." + args.model).DiffNetwork()
        model.load_state_dict(model_weights)
        if args.cuda:
            model = model.cuda()
    else:
        raise IOError

    ### START TO EUALUATE ###
    tic = time.time()
    running_loss = 0

    for ii, (index, im_a, im_b, label) in enumerate(dataloader["test"]):
        inp_a, inp_b = Variable(im_a), Variable(im_b)
        if args.cuda:
            inp_a, inp_b = inp_a.cuda(), inp_b.cuda()
        pred = model(inp_a, inp_b)
        imkey = imkey_list[index[0]]
        
        deta_str, detb_str = render_v2wo_gd(args, imkey, pred, anchors=anchors)
        fa.write(deta_str), fb.write(detb_str)
        
    fa.close(), fb.close()

    print (" | Time consuming: {:.2f}s".format(time.time()-tic))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="./data/cdata/test")
    parser.add_argument('--num_workers', type=int, default=1,
                            help="Number of data loading threads.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help="Disable CUDA training.")
    parser.add_argument('--model', type=str, default="subv2", 
                            help="A model name to generate network.")
    parser.add_argument('--model_fn', type=str, default="./default.pth.tar",
                            help="A model tar file to test.")
    parser.add_argument('--deta_fn', type=str, default="./result_wo/det_a.txt", 
                            help="Detection result filename of image a.")
    parser.add_argument('--detb_fn', type=str, default="./result_wo/det_b.txt", 
                            help="Detection result filename of image b.")
    parser.add_argument('--desdir', type=str, default="./result_wo",
                            help="Rendered image directory.")
    parser.add_argument('--fontfn', type=str, default="./srcs/droid-sans-mono.ttf",
                            help="Font filename when rendering.")
    parser.add_argument('--render', type=int, default=0,
                            help="Output rendered files to result directory")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

if __name__ == "__main__":
    args = parse()
    evaluate(args)
