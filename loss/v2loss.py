# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import itertools
import pdb

class v2loss(nn.Module):

    def __init__(self, anchors,
                 obj_scale=5, noobj_scale=0.5, class_scale=1, coord_scale=1, num_class=2):
        super(v2loss, self).__init__()
        self.num_class   = num_class
        self.num_anchors = len(anchors)
        self.anchors     = torch.Tensor(anchors) / (512/14.0)
        #anchor化为相对于grid的w和h

        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.obj_scale   = obj_scale
        self.class_scale = class_scale

        self.size = 1 + num_class + 4

    def forward(self, pred, label, seen):
        '''
        pred:  bsx35x14x14
        label: bsx 7x14x14
        '''
        BS = pred.data.size(0)
        H = pred.data.size(2)
        W = pred.data.size(3)

        wh = Variable(torch.from_numpy(np.reshape([W, H], [1, 1, 1, 1, 2]))).float().cuda()
        anchor_bias = Variable(self.anchors.view(1, 1, 1, self.num_anchors, 2)).cuda() / wh
        w_list = np.array(list(range(W)), np.float32)
        wh_ids = Variable(torch.from_numpy(np.array(list(map(lambda x: np.array(list(itertools.product(w_list, [x]))), range(H)))).reshape(1, H, W, 1, 2))).float().cuda()

        loss = Variable(torch.zeros((1)), requires_grad=True).cuda()
        class_zeros = Variable(torch.zeros(self.num_class)).cuda()
        mask_loss = Variable(torch.zeros(H*W*self.num_anchors*5).contiguous().view(H, W, self.num_anchors, 5)).cuda()
        zero_coords_loss = Variable(torch.zeros(H*W*self.num_anchors*4).contiguous().view(H, W, self.num_anchors, 4)).cuda()
        zero_coords_obj_loss = Variable(torch.zeros(H*W*self.num_anchors*5).contiguous().view(H, W, self.num_anchors, 5)).cuda()

        zero_pad = Variable(torch.zeros(2).contiguous().view(1, 2)).cuda()
        pad_var = Variable(torch.zeros(2*self.num_anchors).contiguous().view(self.num_anchors, 2)).cuda()
        anchor_padded = torch.cat([pad_var, anchor_bias.contiguous().view(self.num_anchors, 2)], 1)

        pred = pred.permute(0, 2, 3, 1)
        pred = pred.contiguous().view(-1, H, W, self.num_anchors, self.size)

        adjust_xy = pred[:, :, :, :, 1+self.num_class:1+self.num_class+2].sigmoid()
        adjust_coords = (adjust_xy + wh_ids) / wh
        adjust_wh = torch.exp(pred[:, :, :, :, 1+self.num_class+2:1+self.num_class+4]) * anchor_bias

        for b in range(BS):
            gtb = parse_label(label[b])
            if not len(gtb):
                continue
            gt = Variable(torch.zeros((len(gtb), 4))).cuda()
            for i, anno in enumerate(gtb):
                gt[i,0] = anno[0] / W
                gt[i,1] = anno[1] / H
                gt[i,2] = anno[2]
                gt[i,3] = anno[3]
            pred_outputs = torch.cat([adjust_coords[b], adjust_wh[b]], 3)
            bboxes_iou = bbox_ious(pred_outputs, gt, False)

            boxes_max_iou = torch.max(bboxes_iou, -1)[0]
            all_obj_mask = boxes_max_iou.le(0.6)
            all_obj_loss = all_obj_mask.float() *(self.noobj_scale * (pred[b, :, :, :, 0]))
            #所有iou<0.6的计为noobj，计算noobj loss

            all_coords_loss = zero_coords_loss.clone()
            if seen < 12800:
                all_coords_loss = 0.01 * torch.cat([(adjust_xy[b]-0.5), (pred[b, :, :, :, 1+self.num_class+2:1+self.num_class+4]-0)], -1)
                #训练图片<12800张时，计算框的形状loss
            coord_obj_loss = torch.cat([all_coords_loss, all_obj_loss.unsqueeze(-1)], -1)

            batch_mask = mask_loss.clone()
            truth_coord_obj_loss = zero_coords_obj_loss.clone()
            for truth_iter in torch.arange(len(gtb)):
                truth_iter = int(truth_iter)
                truth_box = gt[truth_iter]
                anchor_select = bbox_ious(torch.cat([zero_pad.t(), truth_box[2:].view(1,2).t()], 0).t(), anchor_padded, True)
                anchor_id = torch.max(anchor_select, 1)[1]

                truth_i = truth_box[0] * W
                w_i = truth_i.int()
                truth_x = truth_i - w_i.float()
                truth_j = truth_box[1] * H
                h_j = truth_j.int()
                truth_y = truth_j - h_j.float()
                truth_wh = (truth_box[2:] / anchor_bias.contiguous().view(self.num_anchors, 2).index_select(0, anchor_id.long())).log()
                if (truth_wh[0] == Variable( - torch.cuda.FloatTensor([float('inf')]))).data[0] == 1:
                    pdb.set_trace()
                truth_coords = torch.cat([truth_x.unsqueeze(0), truth_y.unsqueeze(0), truth_wh], 1)
                pred_output = pred[b].index_select(0, h_j.long()).index_select(1, w_i.long()).index_select(2, anchor_id.long())[0][0][0]
                pred_xy = adjust_xy[b].index_select(0, h_j.long()).index_select(1, w_i.long()).index_select(2, anchor_id.long())[0][0][0]
                pred_wh = pred_output[1+self.num_class+2:1+self.num_class+4]
                pred_coords = torch.cat([pred_xy, pred_wh], 0)
                coords_loss = self.coord_scale * (2 - truth_x.data[0]*truth_y.data[0]) *(pred_coords.unsqueeze(0) - truth_coords)
                #坐标回归损失，选中的那个anchor的预测结果和真实坐标的损失
                iou = bboxes_iou.index_select(0, h_j.long()).index_select(1, w_i.long()).index_select(2, anchor_id.long())[0][0][0][truth_iter]

                #print(iou, h_j.data[0], w_i.data[0], label[b, :, h_j.data[0], w_i.data[0]])

                obj_loss = self.obj_scale * (pred_output[0] - iou)
                #obj loss 选中的anchor的iou loss
                truth_co_obj = torch.cat([coords_loss, obj_loss.view(1, 1)], 1)

                class_vec = class_zeros.clone()
                class_vec[0] = label[b, 1, int(h_j), int(w_i)] 
                class_vec[1] = label[b, 2, int(h_j), int(w_i)]
                #class_loss = self.class_scale * (class_vec - pred_output[1:1+self.num_class])
                class_loss = self.class_scale * (pred_output[1:1+self.num_class] - class_vec)
                #分类损失，选中的那个anchor的分类损失

                mask_ones = Variable(torch.ones(5)).cuda()
                batch_mask[h_j.long(), w_i.long(), anchor_id.long()] = mask_ones
                truth_coord_obj_loss[h_j.long(), w_i.long(), anchor_id.long()] = truth_co_obj
                loss += class_loss.pow(2).sum()
            batch_coord_obj_loss = batch_mask * truth_coord_obj_loss + (1 - batch_mask) * coord_obj_loss
            #只计算选中anchor对应的坐标和obj loss，别的框的置0忽略
            loss += batch_coord_obj_loss.pow(2).sum()
            if loss.data[0] > 10000:
                print('loss(%d) is too high'%(loss.data[0]))
                print(pred[b,:,:,:,0].pow(2).sum())
                pdb.set_trace()

        return loss/BS


def bbox_ious(bboxes1, bboxes2, is_anchor):
    x1, y1, w1, h1 = bboxes1.chunk(4, dim=-1)
    x2, y2, w2, h2 = bboxes2.chunk(4, dim=-1)
    x11 = x1 - 0.5*w1
    y11 = y1 - 0.5*h1
    x12 = x1 + 0.5*w1
    y12 = y1 + 0.5*h1
    x21 = x2 - 0.5*w2
    y21 = y2 - 0.5*h2
    x22 = x2 + 0.5*w2
    y22 = y2 + 0.5*h2
    xI1 = torch.max(x11, x21.transpose(1, 0))
    yI1 = torch.max(y11, y21.transpose(1, 0))
            
    xI2 = torch.min(x12, x22.transpose(1, 0))
    yI2 = torch.min(y12, y22.transpose(1, 0))
    inner_box_w = torch.clamp((xI2 - xI1), min=0)
    inner_box_h = torch.clamp((yI2 - yI1), min=0)
    inter_area = inner_box_w * inner_box_h
    bboxes1_area = (x12 - x11) * (y12 - y11)
    bboxes2_area = (x22 - x21) * (y22 - y21)
    union = (bboxes1_area + bboxes2_area.transpose(1, 0)) - inter_area
    return torch.clamp(inter_area / union, min=0)

def parse_label(label):
    _, h, w = label.size()
    gts = []
    for r in range(h):
        for c in range(w):
            if label[0, r, c].data[0]:
                tx, ty, tw, th = label[1+2:, r, c].data
                tx, ty = tx+c, ty+r
                gts.append([tx,ty,tw,th])
    return gts

