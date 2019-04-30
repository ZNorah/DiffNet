#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Evaluate our best model.
"""

import argparse
import torch
import time
import os
import os.path as osp
import importlib

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 

from data.dataset import Pair_Dataset
from loss.loss import criterion
import os.path as osp
from utils import render_orim

tensor2PIL = lambda x: transforms.ToPILImage()(x.view(-1, 512, 512))

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

    ### LOAD MODEL ###
    if osp.exists(args.model_fn):
        model_weights = torch.load(args.model_fn)
        model = importlib.import_module("model." + args.model).DiffNetwork()
        model.load_state_dict(model_weights)
        if args.cuda:
            model = model.eval().cuda()
    else:
        raise IOError

    ### START TO EUALUATE ###
    tic = time.time()
    running_loss = 0

    cnt = 0
    allt = 0
    for ii, (index, im_a, im_b, label) in enumerate(dataloader["test"]):
        cnt += 1
        inp_a, inp_b = Variable(im_a), Variable(im_b)
        label = Variable(label)
        if args.cuda:
            inp_a, inp_b = inp_a.cuda(), inp_b.cuda()
            label = label.cuda()
        ta = time.time()
        pred = model(inp_a, inp_b)
        tb = time.time()
        dt = tb - ta
        allt += dt
        loss = criterion(label, pred)
        running_loss += loss.data[0]
        imkey = imkey_list[index[0]]
        
        deta_str, detb_str = render_orim(args, imkey, label, pred)
        fa.write(deta_str), fb.write(detb_str)
        
    fa.close(), fb.close()

    print (" | -- Eval Ave Loss {:.2f}".format(running_loss/(ii+1))) 
    print (" | Time consuming: {:.2f}s".format(time.time()-tic))
    print('count: %d' % cnt)
    print('all infer time: {:.2f}s, avg infer time: {:.2f}ms'.format(allt, 1000*allt/cnt))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="./data/cdata/test")
    parser.add_argument('--num_workers', type=int, default=4,
                            help="Number of data loading threads.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help="Disable CUDA training.")
    parser.add_argument('--model', type=str, default="base", 
                            help="A model name to generate network.")
    parser.add_argument('--model_fn', type=str, default="./default.pth.tar",
                            help="A model tar file to test.")
    parser.add_argument('--deta_fn', type=str, default="./result/v120/det_a.txt", 
                            help="Detection result filename of image a.")
    parser.add_argument('--detb_fn', type=str, default="./result/v120/det_b.txt", 
                            help="Detection result filename of image b.")
    parser.add_argument('--desdir', type=str, default="./result/v120",
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
