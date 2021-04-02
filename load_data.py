#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:09:47 2020

@author: wenminggong

load data set
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import sys 


# 构建数据提取器，利用dataloader
# 利用torchvision中的transforms进行图像预处理
class  SelfCustomDataset(Dataset):
    def __init__(self, label_file):
        # label_file:存储图像路径及标签的txt文件
        # 读取文件的路径及标签
        with open(label_file, 'r') as f:
            #self.img的格式， （img_path image_label)
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        # print(img_path)
        # 读取img，并转化为RGB格式
        img = Image.open(img_path).convert('RGB')
        # img.show()
        # print(img.size)
        # 统一将输入图片设置为(4608, 3456)
        img_crop = transforms.CenterCrop((4608, 3456))
        img = img_crop(img)
        # img.show()
        # print(img.size)
        
        loader = transforms.Compose([transforms.ToTensor()])
        img = loader(img)
        return img, torch.from_numpy(np.array(int(label)))
 
    def __len__(self):
        return len(self.imgs)


##进行数据提取函数的测试
if __name__ =="__main__":
    train_img_path = 'hand_gestures_data_set_nju/val.txt'
    train_datasets = SelfCustomDataset(train_img_path)
    print(train_datasets.__len__())
    batch_size = 1
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False, num_workers=7)
    for step, (images, labels) in enumerate(train_dataloader):
        print(step)