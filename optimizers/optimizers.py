# -*- coding: utf-8 -*-

from tensorflow.keras.optimizers import Adam, SGD

def get_Adam(cfg):
    return Adam(lr=cfg['learningRate'], decay=cfg['decay'], clipvalue=cfg['clipvalue'], epsilon=cfg['epsilon'])

def get_SGD(cfg):
    return SGD(lr=cfg['learningRate'], momentum=cfg['momentum'], decay=cfg['decay'], nesterov=cfg['nesterov'])