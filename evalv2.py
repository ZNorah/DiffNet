import argparse
import torch
import numpy as np
import time
import importlib

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

import xml.etree.ElementTree as ET
from data.d14 import Pair_Dataset
from loss.v2loss import v2loss
import os
import os.path as osp
from utils import render_v2orim
import pdb

def evaluate(args):
    dataclass = Pair_Dataset(args.test_dir, test=True)
    imkey_list = dataclass.imkey_list
    dataloader = DataLoader(dataclass, 1, shuffle=False, num_workers=args.num_workers)
    if osp.exists(args.model_fn):
        model_weights = torch.load(args.model_fn)
        model = importlib.import_module('model.' + args.model).DiffNetwork()
        model.load_state_dict(model_weights)
        model = model.cuda().eval()
    else:
        raise IOError

    if not osp.exists(args.desdir):
        os.makedirs(args.desdir)
    fa, fb = open(args.deta_fn, "w"), open(args.detb_fn, "w")

    anchors = [(41.4514,52.2819), (63.2610,59.1658), (52.3392,76.1706), (94.5413,77.2516), (71.5646,108.9415)]

    print(' | Start evaluating...')
    tic = time.time()
    criterion = v2loss(anchors=anchors, coord_scale=2)
    running_loss= 0

    cnt = 0
    allt = 0
    for ii, (index, im_a, im_b, label) in enumerate(dataloader):
        cnt += 1
        inp_a, inp_b = Variable(im_a).cuda(), Variable(im_b).cuda()
        label = Variable(label).cuda()
        ta = time.time()
        pred = model(inp_a, inp_b)
        tb = time.time()
        dt = tb - ta
        allt += dt
        loss = criterion(pred.float(), label.float(), ii)
        running_loss += loss.data[0]
        imkey = imkey_list[index[0]]

        deta_str, detb_str = render_v2orim(args, imkey, label, pred, anchors=anchors)
        fa.write(deta_str), fb.write(detb_str)
    fa.close(), fb.close()

    print (" | -- Eval Ave Loss {:.4f}".format(running_loss/(ii+1))) 
    print (" | Time consuming: {:.2f}s".format(time.time()-tic))
    print('count: %d' % cnt)
    print('all infer time: {:.2f}s, avg infer time: {:.2f}ms'.format(allt, 1000*allt/cnt))

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='./data/cdata/test')
    parser.add_argument('--num_workers', type=int, default=1,
                            help="Number of data loading threads.")
    parser.add_argument('--model', type=str, default="base", 
                            help="A model name to generate network.")
    parser.add_argument('--model_fn', type=str, default="./base_2019-01-07.pth.tar",
                            help="A model tar file to test.")
    parser.add_argument('--deta_fn', type=str, default="./result/v2/det_a.txt", 
                            help="Detection result filename of image a.")
    parser.add_argument('--detb_fn', type=str, default="./result/v2/det_b.txt", 
                            help="Detection result filename of image b.")
    parser.add_argument('--desdir', type=str, default="./result/v2",
                            help="Rendered image directory.")
    parser.add_argument('--fontfn', type=str, default="./srcs/droid-sans-mono.ttf",
                            help="Font filename when rendering.")
    parser.add_argument('--render', type=int, default=0,
                            help="Output rendered files to result directory")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse()
    evaluate(args)
