import torch.utils.data as data
import torch
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import os
import os.path
from os import listdir
from os.path import splitext
from pathlib import Path
import random
import numpy as np
from skimage.io import imread
import cv2
from glob import glob
import imageio
from skimage.measure import label, regionprops
import skimage.feature
import skimage.segmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv



class fzjDataset(data.Dataset):
    def __init__(self, state, transform=None, target_transform=None):
        self.val_mask_path = None
        self.train_mask_path = None
        self.val_img_path = None
        self.train_img_path = None
        self.train_imgs, self.train_masks, self.val_imgs, self.val_masks, self.test_imgs, self.test_masks = [], [], [], [], [], []
        self.state = state
        self.aug = True
        self.root = r'/data/mrs/fzj_data4/'
        self.img_paths = []
        self.mask_paths = []
        self.train_img_paths, self.val_img_paths = None, None
        self.train_mask_paths, self.val_mask_paths = None, None
        self.pics, self.masks = self.getDataPath()
        self.transform = transform
        self.target_transform = target_transform
        self.max_width = 0
        self.max_height = 0
        self.pix_size_dic = self.get_pix_size_dic()


    def getDataPath(self):
        self.img_path = os.path.join(self.root, 'img')
        self.mask_path = os.path.join(self.root, 'mask')

        for folder in listdir(self.img_path):
            if not folder.startswith('.'):
                self.img_paths.extend([(self.img_path + '/' + folder)])
                self.mask_paths.extend([(self.mask_path + '/' + folder)])
        self.trainval_img_path, self.test_img_path, self.trainval_mask_path, self.test_mask_path = \
            train_test_split(self.img_paths, self.mask_paths, test_size=0.2, random_state=1)
        self.train_img_path, self.val_img_path, self.train_mask_path, self.val_mask_path = \
                train_test_split(self.trainval_img_path, self.trainval_mask_path, test_size=0.15, random_state=1)
        assert self.state == 'train' or self.state == 'val' or self.state == 'test'
        if self.state == 'train':
            for i, folder in enumerate(self.train_img_path):
                if not folder.startswith('.'):
                    self.train_imgs.extend(
                        [(folder + '/' + file) for file in listdir(folder) if
                         not file.startswith('.')])
                    self.train_masks.extend([(self.train_mask_path[i] + '/' + file) for file in listdir(folder) if
                                             not file.startswith('.')])
            return self.train_imgs, self.train_masks
        if self.state == 'val':
            for i, folder in enumerate(self.val_img_path):
                if not folder.startswith('.'):
                    self.val_imgs.extend(
                        [(folder + '/' + file) for file in listdir(folder) if
                         not file.startswith('.') and len(file)<10])
                    self.val_masks.extend([(self.val_mask_path[i] + '/' + file) for file in listdir(folder) if
                                           not file.startswith('.') and len(file)<10])
            return self.val_imgs, self.val_masks
        if self.state == 'test':
            for i, folder in enumerate(self.test_img_path):
                if not folder.startswith('.'):
                    self.test_imgs.extend(
                        [(folder + '/' + file) for file in listdir(folder) if
                         not file.startswith('.')and len(file)<10])
                    self.test_masks.extend([(self.test_mask_path[i] + '/' + file) for file in listdir(folder) if
                                           not file.startswith('.')and len(file)<10])
            return self.test_imgs, self.test_masks


    def get_pix_size_dic(self):
        spacings_dic = {}
        with open("/data/mrs/fzj_data4/spacing.txt", 'r') as f:
            spacings = f.readlines()
            for i in range(len(spacings)):
                spacings_dic[spacings[i].split('|')[0]] = float(
                    spacings[i].split(' ')[1].strip('\n').strip('(').strip(','))
        return spacings_dic


    def largestConnectComponent(self, bw_img):
        labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)
        max_label = 0
        max_num = 0
        for i in range(1, num + 1):  # 注意这里的范围，为了与连通域的数值相对应
            # 计算面积，保留最大面积对应的索引标签，然后返回二值化最大连通域
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        mcr = (labeled_img == max_label)
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        # ax.imshow(mcr)

        minr, minc, maxr, maxc = regionprops(labeled_img)[max_label - 1].bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        # ax.add_patch(rect)
        return mcr, rect, minr, minc, maxr, maxc

    def draw_min_rect_rectangle(self, path, image):
        distan=0
        out = [0, 0, 0, 0]
        w0, h0 = image.shape[1], image.shape[0]
        out[0] = 0
        out[2] = w0

        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = np.copy(image)
        out1 = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h>50:
            # 绘制矩形
            # cv2.rectangle(img, (x, y + h), (x + w, y), (0, 255, 255))

                out1.append([x, y, w, h])
        if len(out1) == 2:
            if out1[0][0]<out1[1][0]:
                distan=out1[1][0]-(out1[0][0]+out1[0][2])
            else:
                distan=out1[0][0]-(out1[1][0]+out1[1][2])

            if out1[0][1] < out1[1][1]:
                out[1] = out1[0][1]
            else:
                out[1] = out1[1][1]
            if (out1[0][1] + out1[0][3]) < (out1[1][1] + out1[1][3]):
                out[3] = out1[1][1] + out1[1][3] - out[1]
            else:
                out[3] = out1[0][1] + out1[0][3] - out[1]
        elif len(out1) == 1:
            distance=0
            out[1] = out1[0][1]
            out[3] = out1[0][3]
            print(path,'counters =1')
        elif len(out1) == 0:
            print(path, 'can not find count')
        else:
            print(path)
        return out,distan



    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]
        fold = mask_path.split('/')[-2]
        pixel_size=self.pix_size_dic[fold]

        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        pic = cv2.imread(pic_path)
        pic_gray = cv2.imread(pic_path, 0)
        ret, binary = cv2.threshold(pic_gray, 0, 255, cv2.THRESH_BINARY)
        mcr, rect, miny, minx, maxy, maxx = self.largestConnectComponent(binary)
        pic = pic[miny:maxy, minx:maxx, :]  #pic->(h,w,c)
        mask = mask[miny:maxy, minx:maxx]
        rect0, dist = self.draw_min_rect_rectangle(mask_path, mask) # rect0=[x,y,w,h]
        pic_gray=pic_gray[miny:maxy, minx:maxx]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        w_pic = pic.shape[1]
        w_mask = mask.shape[1]
        h_pic = pic.shape[0]
        h_mask = mask.shape[0]
        assert w_pic == w_mask
        assert h_pic == h_mask
        set_w_size = 512
        set_h_size = 512
        proportion=set_w_size/w_pic
        new_dist=dist*proportion
        ground_truth_dist=dist*pixel_size

        if ground_truth_dist>=25:
            class_label = 1  # 分离
        else:
            class_label = 0  # 正

        with open(r'1.csv', mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data2 = [fold,pic_path,ground_truth_dist,class_label,pixel_size]
            wf.writerow(data2)
        cfa.close()
        pic_gray2=cv2.resize(pic_gray2,(set_w_size, set_h_size))
        pic = cv2.resize(pic, (set_w_size, set_h_size))
        mask = cv2.resize(mask, (set_w_size, set_h_size))
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        border = [minx, miny , maxx, maxy ]
        class_label=torch.from_numpy(np.array(int(class_label)))
        ground_truth_dist = torch.from_numpy(np.array(ground_truth_dist))
        if self.transform is not None:
            img_x = self.transform(pic)

        if self.target_transform is not None:
            img_y = self.target_transform(mask)
            img_y=torch.squeeze(img_y)
        return img_x, img_y, pic_path, mask_path, border, mask,new_dist,proportion,ground_truth_dist,class_label,pixel_size

    def __len__(self):
        return len(self.pics)
