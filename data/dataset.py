#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Distributed under terms of the MIT license.

"""
Dataset preprocessing class for dataloader.
"""
from torch.utils import data
import torch
import os
from PIL import Image
from PIL import ImageChops, ImageEnhance
from random import shuffle
import os.path as osp
from torchvision import transforms as T
from torch.utils.data import DataLoader
import numpy as np
import random
import xml.etree.ElementTree as ET
import pdb

def constrain(min_val, max_val, val):
    return min(max_val, max(min_val, val))

class Pair_Dataset(data.Dataset):

    def __init__(self, im_root, scale_size=512, label_shape=(7, 14, 14), transforms=None, train=False, test=False):
        """Get all images and spearate dataset to training and testing set."""
        self.test, self.train = test, train
        self.im_root = im_root
        self.imkey_list = self.get_imkeylist()
        self.label_shape = label_shape
        self.scale_size = scale_size
        self.tensor2PIL = T.Compose([T.ToPILImage()])
        # Separate dataset
        if self.test:
            self.imkey_list = self.imkey_list
        elif self.train:
            self.imkey_list = self.imkey_list[:int(0.8*len(self.imkey_list))]
        else:
            self.imkey_list = self.imkey_list[int(0.8*len(self.imkey_list)):]
        self.imnum = len(self.imkey_list)

        # Transform
        if transforms is None:
            # 在ToTensor这个类中已经实现了0~1的映射
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
            # No enhancement on training and validating set
            self.transforms = T.Compose([T.ToTensor()])
                    #, normalize])

    def __getitem__(self, index):
        """ Return a pair of images. """
        ima_path = osp.join(self.im_root, self.imkey_list[index]+"_a.jpg")
        imb_path = osp.join(self.im_root, self.imkey_list[index]+"_b.jpg")
        im_a, im_b, flip, dx, dy, sx, sy = self.load_pair_im(ima_path, imb_path)


        """ Return normalized groundtruth bboxes space. """
        labela_path = osp.join(self.im_root, self.imkey_list[index]+"_a.xml")
        labelb_path = osp.join(self.im_root, self.imkey_list[index]+"_b.xml")
        if random.uniform(0, 1) > 0.5 and self.train == True:
            im_a, im_b = im_b, im_a
            labela_path, labelb_path = labelb_path, labela_path
        label = self.load_pair_label(labela_path, labelb_path, flip, dx, dy, sx, sy)
        return index, im_a, im_b, label

    def load_pair_im(self, ima_path, imb_path):
        """ Modify PAIR tagged code to make it to load single image """
        im_ori, impair_ori = Image.open(ima_path), Image.open(imb_path) #PAIR
        ow, oh = im_ori.size[0], im_ori.size[1]
        if self.train == True and self.test == False:
            #随机crop和随机翻转
            jitter = 0.4
            dw, dh = int(ow*jitter), int(oh*jitter)
            pleft, pright = random.randint(-dw, dw), random.randint(-dw, dw)
            ptop,  pbot   = random.randint(-dh, dh), random.randint(-dh, dh)
            swidth, sheight = ow-pleft-pright, oh-ptop-pbot     # image size after random
            sx, sy = float(swidth) / ow, float(sheight) / oh  #random后相对于原图的比例
            im_cropped = im_ori.crop((pleft, ptop, ow-pright, oh-pbot))    # (left, upper, right, lower)
            impair_cropped = impair_ori.crop((pleft, ptop, ow-pright, oh-pbot)) #PAIR
            dx, dy = (float(pleft)/ow) / sx, (float(ptop)/oh) / sy   #偏移量相对于random后的图像尺寸
            im_sized = im_cropped.resize((self.scale_size, self.scale_size))   #将crop后的图像resize成512
            impair_sized = impair_cropped.resize((self.scale_size, self.scale_size)) #PAIR
            flip   = (random.uniform(0, 1) > 0.5)
            if flip:
                im_sized = im_sized.transpose(Image.FLIP_LEFT_RIGHT)
                impair_sized = impair_sized.transpose(Image.FLIP_LEFT_RIGHT) #PAIR

            #亮度
            bright_factor = random.uniform(0.8, 1.2)
            im_sized = self.RandomBrightness(im_sized, bright_factor)
            impair_sized = self.RandomBrightness(impair_sized, bright_factor)

            flag = (random.uniform(0, 1) > 0.5)
            if flag:
                #对比度
                contra_factor = random.uniform(0.5, 1.5)
                im_sized = self.RandomContrast(im_sized, contra_factor)
                impair_sized = self.RandomContrast(impair_sized, contra_factor)
            else:
                #对比度
                contra_factor = random.uniform(0.5, 1.5)
                im_sized = self.RandomContrast(im_sized, contra_factor)
                impair_sized = self.RandomContrast(impair_sized, contra_factor)
                #饱和度
                satu_factor = random.uniform(0.5, 1.5)
                im_sized = self.RandomSaturation(im_sized, satu_factor)
                impair_sized = self.RandomSaturation(impair_sized, satu_factor)
                #锐度
                shap_factor = random.uniform(0, 2.0)
                im_sized = self.RandomSharp(im_sized, shap_factor)
                impair_sized = self.RandomSharp(impair_sized, shap_factor)
                #通道变换
                swap_flag = (random.uniform(0, 1) > 0.5)
                if swap_flag:
                    swap_factor = random.randint(0, 5)
                    im_sized = self.RandomLightingNoise(im_sized, swap_factor)
                    impair_sized = self.RandomLightingNoise(impair_sized, swap_factor)
        else:
            dx = dy = 0
            flip = False
            im_sized = im_ori.resize((self.scale_size, self.scale_size))
            impair_sized = impair_ori.resize((self.scale_size, self.scale_size)) #PAIR
            sx, sy = 1, 1           # NOTE here this BUG
        im  = self.transforms(im_sized) # Normalize and adjust the mean and var
        impair = self.transforms(impair_sized) #PAIR

        return im, impair, flip, dx, dy, sx, sy
    

    def load_pair_label(self, labela_path, labelb_path, flip, dx, dy, sx, sy):
        labela = self.get_label(labela_path, flip, dx, dy, sx, sy)
        labelb = self.get_label(labelb_path, flip, dx, dy, sx, sy)
        return self.mergelabel(labela, labelb)

    # GET LABEL TO STANDARD FORMAT, 5x14x14
    def get_label(self, label_path, flip, dx, dy, sx, sy):
        ROW, COL = self.label_shape[1], self.label_shape[2]
        label = np.zeros((5, self.label_shape[1], self.label_shape[2]))
        if osp.exists(label_path):
            tree = ET.parse(label_path)
            im_size = tree.findall("size")[0]
            ow, oh = int(im_size.find("width").text), int(im_size.find("height").text)  #得到原图尺寸
            boxes = []
            for obj in tree.findall('object'):
                bbox = obj.find('bndbox')
                t_boxes = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                           int(bbox.find('xmax').text), int(bbox.find('ymax').text)] #左上，右下，真实坐标
                boxes.append([1,
                            (t_boxes[0] + t_boxes[2])/(2.0*ow), # center x 归一化后
                            (t_boxes[1] + t_boxes[3])/(2.0*oh), # center y
                            (t_boxes[2] - t_boxes[0])*1.0/ow,  # w
                            (t_boxes[3] - t_boxes[1])*1.0/oh]) # h
            ### Correct boxes ###
            for i in range(len(boxes)):
                left = (boxes[i][1] - boxes[i][3] / 2) * (1.0 / sx) - dx  #crop后的左上 相对于crop后图片归一化后的坐标
                right = (boxes[i][1] + boxes[i][3] / 2) * (1.0 / sx) - dx
                top = (boxes[i][2] - boxes[i][4] / 2) * (1.0 / sy) - dy
                bottom = (boxes[i][2] + boxes[i][4] / 2) * (1.0 / sy) - dy
                if flip:
                    swap = left
                    left = 1.0 - right
                    right = 1.0 - swap
                
                left = constrain(0, 1, left)
                right = constrain(0, 1, right)
                top = constrain(0, 1, top)
                bottom = constrain(0, 1, bottom)
                
                boxes[i][1] = (left + right) / 2
                boxes[i][2] = (top + bottom) / 2
                boxes[i][3] = constrain(0, 1, right - left) 
                boxes[i][4] = constrain(0, 1, bottom - top)

            lst = list(range(len(boxes)))       
            shuffle(lst)
            for i in lst:
                x, y, w, h = boxes[i][1:]
                x, y, w, h = constrain(0, 1, x), constrain(0, 1, y), constrain(0, 1, w), constrain(0, 1, h)
                if (w < 0.01 or h < 0.01):
                    continue
                col, row = int(x * self.label_shape[2]), int(y * self.label_shape[1])
                x, y = x * self.label_shape[2] - col, y * self.label_shape[1] - row  #xy为相对于grid左上角的偏移量， wh为相对全图的归一化值
                if label[0, row, col] != 0:
                    continue
                label[:, row, col] = 1, x, y, w, h
        return label
    
    def mergelabel(self, labela, labelb):
        label = np.zeros(self.label_shape)
        for row in range(self.label_shape[1]):
            for col in range(self.label_shape[2]):
                """
                    You should be aware of these code:
                        We can set _ 11 ____, the object are in the same position on image a and image b
                """
                if labela[0, row, col] == 1 and label[1, row, col] == 0:
                    label[0, row, col] = 1
                    label[1, row, col] = 1
                    label[3:7, row, col] = labela[1:, row, col]
                if labelb[0, row, col] == 1 and label[2, row, col] == 0:
                    label[0, row, col] = 1
                    label[2, row, col] = 1
                    label[3:7, row, col] = labelb[1:, row, col]
        return label

    def get_imkeylist(self):
        imkey_list = []
        for i in os.listdir(self.im_root):
            if i[-3:] == "jpg" and (i[:-6] not in imkey_list):
                imkey_list.append(i[:-6])
        return imkey_list
    
    def __len__(self):
        """ Return image pair number of the dataset. """
        return self.imnum

    def RandomBrightness(self, img, brightness):
        enh_bri = ImageEnhance.Brightness(img)
        img_bri = enh_bri.enhance(brightness)
        return img_bri

    def RandomContrast(self, img, contrast):
        enh_con = ImageEnhance.Contrast(img)
        img_con = enh_con.enhance(contrast)
        return img_con

    def RandomSaturation(self, img, saturation):
        enh_col = ImageEnhance.Color(img)
        img_col = enh_col.enhance(saturation)
        return img_col

    def RandomSharp(self, img, sharpness):
        enh_sha = ImageEnhance.Sharpness(img)
        img_sha = enh_sha.enhance(sharpness)
        return img_sha

    def RandomLightingNoise(self, img, swap):
        r, g, b = img.split()
        perms = ((0, 1, 2), (0, 2, 1),
                 (1, 0, 2), (1, 2, 0), 
                 (2, 0, 1), (2, 1, 0))
        if swap == 0:
            return Image.merge('RGB', (r, g, b))
        elif swap == 1:
            return Image.merge('RGB', (r, b, g))
        elif swap == 2:
            return Image.merge('RGB', (g, r, b))
        elif swap == 3:
            return Image.merge('RGB', (g, b, r))
        elif swap == 4:
            return Image.merge('RGB', (b, r, g))
        elif swap == 5:
            return Image.merge('RGB', (b, g, r))



if __name__ == "__main__":
    train_dataset = Pair_Dataset("./diff_vending/test", test=True)
    trainloader = DataLoader(train_dataset, 
                             batch_size=1,
                             shuffle=True,
                             num_workers=0)
    #print (np.sort(train_dataset.imkey_list))
    #print (len(train_dataset.imkey_list))
    #for ii, (im, label) in enumerate(trainloader):
    imkeys = train_dataset.imkey_list
    for ii, (index, im_a, im_b, label) in enumerate(trainloader):
        #print (ii, im_a.size(), labela.shape, im_b.size(), labelb.shape)
        #print (type(im_a), type(labela))
        #print (labela.shape[2]*labela.shape[3])
        #print (index[:], "-----", ii)
        #print (imkeys[index[0]])
        #exit()
        print (label)
        exit()
        pass

        """
        diff_ab = self.transforms(ImageChops.subtract(im_a, im_b))
        diff_ba = self.transforms(ImageChops.subtract(im_b, im_a))
        ### TEST ###
        dab = diff_ab.numpy()
        dba = diff_ba.numpy()
        #dab[dab < 0.5] = 0
        dab = self.tensor2PIL(torch.from_numpy(dab))
        #draw_bbox(dab, )
        dab.save("/tmp/test/" + "{:05d}".format(index) + "_render_a.jpg")
        t = ImageChops.subtract(im_a, im_b)
        t.save("/tmp/ori/" + "{:05d}".format(index) + "_render_a.jpg")
        #dba[dba < 0.5] = 0
        dba = self.tensor2PIL(torch.from_numpy(dba))
        dba.save("/tmp/test/" + "{:05d}".format(index) + "_render_b.jpg")
        t = ImageChops.subtract(im_b, im_a)
        t.save("/tmp/ori/" + "{:05d}".format(index) + "_render_b.jpg")
        posa = np.where(labela[0] == 1)
        posb = np.where(labelb[0] == 1)
        bboxa = labela[1:, posa[0], posa[1]].transpose()
        bboxb = labelb[1:, posb[0], posb[1]].transpose()
        def r(bbox, im_size):
            bbox[:, 0], bbox[:, 2] = bbox[:, 0]*im_size[0], bbox[:, 2]*im_size[0]
            bbox[:, 1], bbox[:, 3] = bbox[:, 1]*im_size[1], bbox[:, 3]*im_size[1]
            return bbox
        bboxa = r(bboxa, im_a.size)
        bboxb = r(bboxb, im_a.size)
        labelrender("/tmp/ori", index, bboxa, bboxb)
        return index, im_a, im_b, labela, labelb
        """
