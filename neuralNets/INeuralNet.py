# -*- coding: utf-8 -*-

# 本平台所有网络的接口类，虚拟函数必须由继承该类的网络实现
class INeuralNet(object):
    
    def __init__(self, cfg):
        self._cfg = cfg
        print("Neural Network Name: " + self._cfg['modelName'])
    
    # 虚拟函数，由不同网络实现
    def build_network(self):
        raise NotImplementedError("Not implemented virtual function.")