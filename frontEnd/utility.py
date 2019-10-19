# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import os

def get_file_absolutely_path(path_from_cmd, absolute_path_cmd):
    if os.path.isabs(path_from_cmd) : 
        return os.path.normpath(path_from_cmd)
    else : #relative path given. Need to make absolute path
        if os.path.isdir(absolute_path_cmd) :
            relative_path = absolute_path_cmd
        elif os.path.isfile(absolute_path_cmd) :
            relative_path = os.path.dirname(absolute_path_cmd)
        else : #not file, not dir, exit.
            print("ERROR: ", absolute_path_cmd, " does not correspond to neither an existing file nor a directory. Exiting!"); exit(1)
        return os.path.normpath(relative_path + "/" + path_from_cmd)
    

    