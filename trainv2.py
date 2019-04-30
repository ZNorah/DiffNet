#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

""" Training learning difference of two similar images network. """

import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 

import time
import importlib

from data.dataset import Pair_Dataset
from loss.v2loss import v2loss

def train(args):
    ### DATA ###
    traindata = Pair_Dataset(args.trainval_dir, train=True)
    train_list = traindata.imkey_list
    validdata = Pair_Dataset(args.trainval_dir, train=False)
    valid_list = validdata.imkey_list
    dataloader = {}
    dataloader["train"] = DataLoader(traindata,
                                     args.batch_size, 
                                     shuffle=True, 
                                     num_workers=args.num_workers)
    dataloader["valid"] = DataLoader(validdata,
                                     args.batch_size, 
                                     shuffle=False, 
                                     num_workers=args.num_workers)

    ### MODEL and METHOD ###
    model = importlib.import_module("model." + args.model).DiffNetwork()
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr = args.lr,
                                momentum=0.9,
                                weight_decay = args.weight_decay)
    if args.lr_policy == "exp":
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize, gamma=0.9)

    anchors = [(41.4514,52.2819), (63.2610,59.1658), (52.3392,76.1706), (94.5413,77.2516), (71.5646,108.9415)]
    ### START TO MACHINE LEARNING ###
    criterion = v2loss(anchors=anchors, coord_scale=2)
    tic = time.time()
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    iters = 0
    for epoch in range(args.nepochs):
        print (" | Seen %d" %iters)
        print (" | Epoch {}/{}".format(epoch, args.nepochs-1))
        print (" | " + "-" * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                if args.lr_policy == "exp":
                    exp_lr_scheduler.step()
                elif args.lr_policy == "custom":
                    custom_lr_scheduler(optimizer, epoch)
                model.train(True)   # Set model in training mode
            else:
                model.train(False)

            running_loss = 0
            for ii, (index, im_a, im_b, label) in enumerate(dataloader[phase]):
                inp_a, inp_b = Variable(im_a), Variable(im_b)
                label = Variable(label)
                if args.cuda:
                    inp_a, inp_b = inp_a.cuda(), inp_b.cuda()
                    label = label.cuda()
                optimizer.zero_grad()
                #pdb.set_trace()
                pred = model(inp_a, inp_b)
                
                loss = criterion(pred.float(), label.float(), iters)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                iters += args.batch_size

                if ii % args.log_freq == 0 and phase == "train":
                    print (" | Epoch{}: {}, Loss {:.4f}".format(epoch, ii, loss.data[0]))
                running_loss += loss.data[0]
            epoch_loss = running_loss / (ii+1)
            print (" | Epoch {} {} Loss {:.4f}".format(epoch, phase, epoch_loss)) 

            # Deep copy of the model
            if phase == 'valid' and best_loss >= epoch_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, args.model_fn)
                print (" | Epoch {} state saved, now loss reaches {}...".format(epoch, best_loss))
        print (" | Time consuming: {:.4f}s".format(time.time()-tic))
        print (" | ")
   
def custom_lr_scheduler(optimizer, epoch):
    if 0 <= epoch < 20:
        lr = (epoch / 5.0)*0.001 + 0.0001
    elif epoch < 75:
        lr = 1e-3
    elif epoch < 105:
        lr = 1e-4
    elif 105 <= epoch:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse():
    parser = argparse.ArgumentParser()
    date = time.strftime("%Y-%m-%d", time.localtime())
    ### DATA ###
    parser.add_argument('--trainval_dir', type=str, default="./data/cdata/train")
    parser.add_argument('--nepochs', type=int, default=800,
                            help="Number of sweeps over the dataset to train.")
    parser.add_argument('--batch_size', type=int, default=8,
                            help="Number of images in each mini-batch.")
    parser.add_argument('--num_workers', type=int, default=4,
                            help="Number of data loading threads.")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                            help="Disable CUDA training.")
    parser.add_argument('--model', type=str, default="sub", 
                            help="Model module name in model dir and I will save best model the same name.")
    parser.add_argument('--model_fn', type=str, default="", 
                            help="Model filename to save.")
    parser.add_argument('--lr_policy', type=str, default="exp", 
                            help="Policy of learning rate change.")
    parser.add_argument('--lr_stepsize', type=int, default=100, 
                            help="Control exponent learning rate decay.")
    parser.add_argument('--log_freq', type=int, default=10)
    # As a rule of thumb, the more training examples you have, the weaker this term should be. 
    # The more parameters you have the higher this term should be.
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help="Goven the regularization term of the neural net.")
    parser.add_argument('--lr', type=float, default=1e-3,
                            help="Goven the learning rate of the neural net.")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.model_fn == "":
        args.model_fn = args.model + "_" + date + ".pth.tar"
    return args


if __name__ == "__main__":
    args = parse()
    print (args)
    train(args)
    print (args)

