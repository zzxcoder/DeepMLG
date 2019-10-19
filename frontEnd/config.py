# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

class Config(object):
    
    def __init__(self, abs_path_to_cfg):
        self._configStruct = {}
        self._abs_path_to_cfg = abs_path_to_cfg 
        print("Given configuration file: ", self._abs_path_to_cfg)
        exec(open(self._abs_path_to_cfg).read(), self._configStruct)  # 打开cfg文件读取键值对，保存到字典变量Configs中
        self._check_for_deprecated_cfg()
        
    def __getitem__(self, key): 
        return self.get(key)
    
    def get(self, key): # 重写[]
        return self._configStruct[key] if key in self._configStruct else None
    
    def get_abs_path_to_cfg(self):
        return self._abs_path_to_cfg
    
    def _check_for_deprecated_cfg(self):
        pass
    
    def override_file_cfg_with_cmd_line_cfg(self, log, args):
        pass
    