#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 16:21:26 2020

@author: wenminggong

load pre-trained cnn model and prediction
"""


import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from cnn_model import cnn_model
from load_data import SelfCustomDataset
import torch.utils.data as Data


if __name__ == "__main__":
    print(sys.version)
    
    #加载训练好的网络模型
    model_name = os.path.join('saves', 'cnn_model.pkl')
    cnn = torch.load(model_name, map_location='cpu')
    cnn = cnn.module
    
    # 读取待预测的图片
    test_img_path = 'hand_gestures_data_set_nju/test.txt'
    test_datasets = SelfCustomDataset(test_img_path)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4)
    
    accuracy = 0.0
    for test_step, (test_x, test_y) in enumerate(test_dataloader):
        with torch.no_grad():
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            print('img:', test_step, '预测类别是:{}'.format(pred_y))
        print('img:', test_step, "实际类别是:{}".format(test_y.data.numpy()))
        accuracy += float((pred_y == test_y.data.numpy()).astype(int).sum())
    
    accuracy = accuracy / float(test_datasets.__len__())
    print('测试准确率为: % .4f' % accuracy)
