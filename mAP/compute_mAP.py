#!/usr/bin/env python3
# This file was modified from
# https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import gflags
import xml.etree.ElementTree as ET
import os, sys
import pickle
import time
import numpy as np
import os.path as osp
import math
import matplotlib.pyplot as plt
import pdb

floor = lambda x: math.floor(float(x))


CLASSES = ["a", "b"]

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    if osp.exists(filename):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            #obj_struct['name'] = tree.findall('filename')[0].text[-5:-4]
            obj_struct['name'] = filename[-5:-4]
            obj_struct['difficult'] = 0
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [floor(bbox.find('xmin').text),
                   floor(bbox.find('ymin').text),
                   floor(bbox.find('xmax').text),
                   floor(bbox.find('ymax').text)]
            objects.append(obj_struct)
    else:
        objects = []

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots == records
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    if confidence.shape[0] == 0:    # None this class
        return 0, 0, 0

    # sort by confidence
    sorted_ind = np.argsort(-confidence)   # gen a sort index
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def assistor(detpath, annopath, imagesetfile, classname):
    rec, prec, ap = voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False)
    print (' | {}--------------'.format(classname))
    print (' | --ap: {}'.format(ap))
    print (' | --max prec: {}'.format(max(prec)))
    print (' | --max rec: {}'.format(max(rec)))
    print (" | ")
    return rec, prec, ap

def pr(prec, rec, ap, classname, use_07_metric=False):
    xs = np.arange(0, 1.1, 0.1)
    name = classname
    plt.figure()
    plt.title('AP(%s)=%.4f' %(name, ap))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1.05])
    plt.xticks(xs)
    plt.yticks(xs)
    if use_07_metric:
        ps = []
        for x in xs:
            if np.sum(rec>=x) == 0:
                ps.append(0)
            else:
                np.append(np.max(np.nan_to_num(prec)[rec>=x]))
        plt.plot(xs, ps)
        plt.savefig('%s.png' % (name))
        plt.close()
    else:
        mpre = np.concatenate(([0], np.nan_to_num(prec), [0]))
        mrec = np.concatenate(([0], rec, [1]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ts = mrec[i+1]
        ps = mpre[i+1]
        plt.plot(ts, ps)
        plt.savefig('%s.png' % (name))
        plt.close()

if __name__ == '__main__':
    """
    det_a_path = "./cache/det_a.txt"
    det_b_path = "./cache/det_b.txt"
    annopath = "/home/zq/diff_resnet/data/test/{}.xml"
    im_a_txt = "./cache/testa.txt"
    im_b_txt = "./cache/testb.txt"
    """
    det_a_path, det_b_path = sys.argv[1], sys.argv[2]
    annopath = sys.argv[3]
    im_a_txt, im_b_txt = sys.argv[4], sys.argv[5]

    mAP = 0.0
    tic = time.time()
    rec_a, prec_a, ap_a = assistor(det_a_path, annopath, im_a_txt, "a") 
    rec_b, prec_b, ap_b = assistor(det_b_path, annopath, im_b_txt, "b") 
    #pr(prec_a, rec_a, ap_a, 'a')
    #pr(prec_b, rec_b, ap_b, 'b')
    mAP += ap_a/2.0
    mAP += ap_b/2.0
    print (' | mAP: {}'.format(mAP))
    print (" | This evalution consumes {0:.1f}s".format(time.time()-tic))
    print (" | Done!")

