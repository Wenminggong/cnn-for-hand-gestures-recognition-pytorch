#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:47:07 2020

@author: wenminggong

cnn model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import math


# spatial pyramid pooling
class SPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        # num_levels: 池化的金字塔层数，保存特征
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size_h = h // (2 ** i)
            kernel_size_w = w // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=(kernel_size_h, kernel_size_w),stride=(kernel_size_h, kernel_size_w)).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=(kernel_size_h, kernel_size_w),stride=(kernel_size_h, kernel_size_w)).view(bs, -1)
            pooling_layers.append(tensor)
        # 对金字塔各层进行拼接
        x = torch.cat(pooling_layers, dim=-1)
        return x


# initialize the parameters of neural network (how?)
def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain = math.sqrt(2))
        # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        module.bias.data.zero_()


class cnn_model(nn.Module):
    def __init__(self, spp_level):
        super(cnn_model, self).__init__()
        # self.spp_level = spp_level
        # self.num_grids = 0
        # for i in range(spp_level):
        #     self.num_grids += 2**(i*2)
        # print("spatial pyramid pooling output features: {}".format(self.num_grids))
        
        self.conv1 = nn.Sequential(         # input shape (3, 3456, 4608)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=9,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (9, 3456, 4608)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=4),    # choose max value in 4x4 area, output shape (9, 864, 1152)
        )
        self.conv2 = nn.Sequential(         # input shape (9, 864, 1152)
            nn.Conv2d(9, 18, 5, 1, 2),     # output shape (18, 864, 1152)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(4),                # output shape (18, 216, 288)
        )
        self.conv3 = nn.Sequential(         # input shape (18, 216, 288)
            nn.Conv2d(18, 36, 5, 1, 2),     # output shape (36, 216, 288)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (36, 108, 144)
        )
        self.conv4 = nn.Sequential(         # input shape (36, 108, 144)
            nn.Conv2d(36, 72, 5, 1, 2),     # output shape (72, 108, 144)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (72, 54, 72)
        )
        self.conv5 = nn.Sequential(         # input shape (72, 54, 72)
            nn.Conv2d(72, 144, 5, 1, 2),     # output shape (144, 54, 72)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (144, 27, 36)
        )
        self.conv6 = nn.Sequential(         # input shape (144, 27, 36)
            nn.Conv2d(144, 288, 5, 1, 2),     # output shape (288, 27, 36)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (288, 13, 18)
        )
        
        # self.spp_layer = SPPLayer(spp_level)
        
        self.linear1 = nn.Linear(288 * 13 * 18, 1024) # fully connected layer1
        self.linear2 = nn.Linear(1024, 512) # fully connected layer2
        self.linear3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 20)   # fully connected layer, output 20 classes

	# initilize the parameters of linear layers
        self.apply(weight_init)


    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.view(output.size(0), -1)           # flatten the output of conv2 to (batch_size, 288 * 13 * 18)
        # output = self.spp_layer(output)
        feature_map = output
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.relu(output)
        output = self.linear3(output)
        output = F.relu(output)
        output = self.out(output)
        #output = F.softmax(output, 1)
        return output, feature_map    # return x for visualization
