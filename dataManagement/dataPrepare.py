# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage

def get_training_set(file_path):
    train_imgs = []
    train_gt = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            img = np.array(ndimage.imread(eachline[0]))
            train_imgs.append(img)
            gt = int(eachline[1])
            train_gt.append(gt)
            line = f.readline()
           
    return train_imgs, train_gt