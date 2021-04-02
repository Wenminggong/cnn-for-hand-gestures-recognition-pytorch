#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:47:07 2020

@author: wenminggong

labeling the image data
"""

import os
import glob
import sys 
import random



def data_preprocess(traindata_path, labels, txt_path):
    # valdata_path = cfg.BASE + 'test'
    for index, label in enumerate(labels):
        sub_paths = os.listdir(os.path.join(traindata_path, label))
        # print('sub path:{}'.format(sub_paths))
        
        imglist = []
        for sub_path in sub_paths:
            imglist += glob.glob(os.path.join(traindata_path,label, sub_path, '*'))
        
        # 将当前类别文件夹下的图像列表随机排序
        random.shuffle(imglist)
        print(len(imglist))
        trainlist = imglist[:int(0.9*len(imglist))]
        vallist = imglist[(int(0.9*len(imglist))+1):]
        print("the number of training data is {}".format(len(trainlist)))
        print("the number of testing data is {}".format(len(vallist)))
        with open (os.path.join(txt_path, 'train.txt'), 'a') as f:
            for img in trainlist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

        with open (os.path.join(txt_path, 'val.txt'), 'a') as f:
            for img in vallist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

    # imglist = glob.glob(os.path.join(valdata_path, '*.jpg'))
    # with open(txtpath + 'test.txt', 'a')as f:
    #     for img in imglist:
    #         f.write(img)
    #         f.write('\n')
            
            
if __name__ == "__main__":
    print(sys.version)
    # print(sys.path)
    traindata_path = 'hand_gestures_data_set_nju/train'  #相对路径
    sub_data_path = os.listdir(traindata_path)
    print("sub classes: {}".format(sub_data_path))
    labels = []
    for sub_path in sub_data_path:
        sub_labels = os.listdir(os.path.join(traindata_path, sub_path))
        for sub_label in sub_labels:
            labels.append(os.path.join(sub_path, sub_label))
    print("labels:{}".format(labels))
    
    txt_path = 'hand_gestures_data_set_nju'
    data_preprocess(traindata_path, labels, txt_path)