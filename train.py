#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:40:18 2020

@author: wenminggong

training cnn model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from load_data import SelfCustomDataset
from cnn_model import cnn_model
from PIL import Image
from torchvision import transforms
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
from tqdm import tqdm
import time


def train(train_img_path, test_img_path, save_folder):
    # train_img_path:训练集txt
    
    # Hyper Parameters
    EPOCH = 100               # train the training data n times
    BATCH_SIZE = 8
    LR = 0.001              # learning rate
    spp_level = 3
    
    # 加载训练数据
    train_datasets = SelfCustomDataset(train_img_path)
    print("There are {} images in training data".format(train_datasets.__len__()))
    
    # show the image
    # unloader = transforms.ToPILImage()
    # image = unloader(train_datasets.__getitem__(0)[0])
    # image.show()
    
    # Data Loader for easy mini-batch return in training, the image batch shape will be (1, c, h, w)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    # 加载测试数据
    test_datasets = SelfCustomDataset(test_img_path)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    # test_x = None
    # test_y = None
    # for step, (t_x, t_y) in enumerate(test_dataloader):
    #     test_x = t_x
    #     test_y = t_y
    
    cnn = cnn_model(spp_level)
    # print(cnn)  # net architecture
    
    device = 'cuda'
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs".format(torch.cuda.device_count()))
        cnn = nn.DataParallel(cnn) 
    
    cnn.to(device)
    
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().to(device)
    
    test_acc_list = []
    train_loss_list = []
    # training
    for epoch in tqdm(range(EPOCH)):
        start_time = time.time()
        for step, (b_x, b_y) in enumerate(train_dataloader):   # gives batch data
            print("EPOCH: ", epoch, "| step: ", step)
            train_x = b_x.to(device)
            train_y = b_y.to(device)
            output = cnn(train_x)[0]               # cnn output
            loss = loss_func(output, train_y)   # cross entropy loss
            optimizer.zero_grad()
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()
            train_loss_list.append(loss.to('cpu').data.numpy())
            print('loss is: % .8f' %loss.to('cpu').data.numpy())
            # if (step % BATCH_SIZE == 0) or (step == len(train_dataloader) - 1):
            #     print('update')
            #     optimizer.step()                # apply gradients
            #     optimizer.zero_grad()           # clear gradients for this training step
            # print("training a net using a iamge need time: % .2f" % (time.time() - start_time))

        if epoch % 1 == 0:
            accuracy = 0.0
            for test_step, (test_x, test_y) in enumerate(test_dataloader):
                with torch.no_grad():
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)
                    test_output, last_layer = cnn(test_x)
                    # print(test_output)
                    pred_y = torch.max(test_output.to('cpu'), 1)[1].data.numpy()
                    # print(pred_y)
                    accuracy += float((pred_y == test_y.to('cpu').data.numpy()).astype(int).sum())
                    # accuracy += float((pred_y == test_y.data.numpy()).astype(int))
            accuracy = accuracy / float(test_datasets.__len__())
            test_acc_list.append(accuracy)
            print('Epoch: ', epoch, '| train loss: %.8f' % loss.to('cpu').data.numpy(), '| test accuracy: %.4f' % accuracy)
        end_time = time.time()
        print("training an epoch need time: % .2f s." %(end_time - start_time))
    
    # save the cnn model
    name_model = os.path.join(save_folder, 'cnn_model.pkl')
    print('Save the model to %s' %name_model)
    torch.save(cnn.to('cpu'), name_model)
    
    name_test_acc = os.path.join(save_folder, 'test_acc.npy')
    print('Save the accuracies to %s' %name_test_acc)
    np.save(name_test_acc, np.array(test_acc_list))
    
    name_train_loss = os.path.join(save_folder, 'train_loss.npy')
    print('Save the rewards to %s' %name_train_loss)
    np.save(name_train_loss, np.array(train_loss_list))


if __name__ == "__main__":
    print(sys.version)
    # 将训练模型保存于save_folder中
    save_folder = 'saves'
    os.makedirs(save_folder, exist_ok=True)
    train_img_path = 'hand_gestures_data_set_nju/train.txt'
    test_img_path = 'hand_gestures_data_set_nju/val.txt'
    train(train_img_path, test_img_path, save_folder)
