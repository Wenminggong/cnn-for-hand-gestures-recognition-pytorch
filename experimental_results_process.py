#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:58:09 2020

@author: wenminggong

plot the experimental results
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    print(sys.version)
    
    train_loss = np.load(os.path.join('saves', 'train_loss.npy'))
    test_acc = np.load(os.path.join('saves', 'test_acc.npy'))
    
    # x = np.arange(0, 24200, 1210)
    x = np.arange(0, 100, 5)
    print(x)
    
    plt.figure(figsize=(6,4))
    plt.plot(x, test_acc[x], lw=2, c = 'r', marker = 's')
    # plt.plot(x, train_loss[x], lw=2, c = 'purple', marker = 'o')
    
    plt.legend()
    plt.ylabel("Test Accuracy")
    # plt.ylabel("Training Loss")
    plt.xlabel("Epoches")
    # plt.xlabel("Iterations")
    # plt.title("")
    plt.grid(axis = 'y', ls = '--')
    plt.savefig('saves/test_acc', format='eps')
    # plt.savefig('saves/training_loss', format='eps')
    plt.show()
