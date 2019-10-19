# -*- coding: utf-8 -*-

from neuralNets.VDSR.VDSRNet import VDSRNet
from neuralNets.backbones.VGGNets import VGGNet16
        
class NetFactory():
    
    def __init__(self, cfg):
        self._cfg = cfg
    
    # ========= 本平台注册的网络类型 ==========    
    def _create_VDSRNet(self):
        return VDSRNet(self._cfg)
    
    def _create_VGGNet16(self):
        return VGGNet16(self._cfg)
    
    # 利用反射机制创建神经网络实例
    def get_net_instance(self):
        instance = getattr(self, "_create_" + self._cfg['modelName'])
        return instance