#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path
import argparse

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

# =========== 本平台自定义的包 ==============
from neuralNets.netFactory import NetFactory
from optimizers import optimizers
from frontEnd.utility import get_file_absolutely_path
from frontEnd.config import Config
import objFunctions.metrics as metrics
from dataManagement.dataPrepare import get_training_set


def setup_arg_parser():
    #OPT_MODEL = "-model"
    OPT_TRAIN = "-train"
    OPT_TEST = "-test"
    OPT_LOAD = "-load"
    OPT_TASK = "-task"
    OPT_CFG = "-cfg"
    
    parser = argparse.ArgumentParser(prog = "DeepMLG", formatter_class=argparse.RawTextHelpFormatter, 
                                     description="\nThis platform is built by Machine Learning Group of Chongqing University. "+\
                                     "It intergrates deep learning techniques about Natural Language Processing and Computer Vision")
    # parser.add_argument(OPT_MODEL, dest="model_cfg", type=str, help="Specify the architecture of the model to be used, by providing a config file [MODEL_CFG].")
    # parser.add_argument(OPT_TRAIN, dest="train_cfg", type=str, help="Train a model with training parameters given by specifying config file [TRAINING_CFG].\n"+\
    #                                                                "Additionally, an existing checkpoint of the model can be specified in the [TRAIN_CFG] file or by the additional option ["+OPT_LOAD+"], to continue training it.")
    # parser.add_argument(OPT_TEST, dest="test_cfg", type=str, help="Test with an existing model. The testing session's parameters should be given in config file [TEST_CFG].\n"+\
    #                                                                "Existing pretrained model can be specified in the given [TEST_CFG] file or by the additional option ["+OPT_LOAD+"].\n"+\
    #                                                                "This option cannot be used in combination with ["+OPT_TRAIN+"].")
    parser.add_argument(OPT_LOAD, dest='saved_model', type=str, help="The path to a saved existing checkpoint with learnt weights of the model, to train or test with.\n"+\
                                                                    "This option must follow a ["+OPT_TRAIN+"] or ["+OPT_TEST+"] option.\n"+\
                                                                    "If given, this option will override any \"model\" parameters given in the [TRAIN_CFG] or [TEST_CFG] files.")
    parser.add_argument(OPT_TASK, dest="task", type=str, help="The type of the task: train or test")
    parser.add_argument(OPT_CFG, dest="config", type=str, help="The path of a configure file.")
    
    return parser
    

if __name__ == "__main__":
    # 终端命令行参数
    cwd = os.getcwd()
    parser = setup_arg_parser()
    args = parser.parse_args()
#
#    # 判断输入的命令参数是否正确
#    if len(sys.argv) == 1:
#        print("For help on the usage of this program, please use the option -h."); exit(1)
#    if not args.model_cfg:
#        print("ERROR: Option ["+OPT_MODEL+"] must be specified, pointing to a [MODEL_CFG] file that describes the architecture.\n"+\
#              "Please try [-h] for more information. Exiting."); exit(1)
#    if not (args.train_cfg or args.test_cfg):
#        print("ERROR: One of the options must be specified:\n"+\
#              "\t["+OPT_TRAIN+"] to start a training session on a model.\n"+\
#              "\t["+OPT_TEST+"] to test with an existing model.\n"+\
#              "Please try [-h] for more information. Exiting."); exit(1)
#    if args.test_cfg and args.train_cfg:
#        print("ERROR:\t["+OPT_TEST+"] cannot be used in conjuction with ["+OPT_TRAIN+"].\n"+\
#              "\tTo test with an existing network, please just specify a configuration file for the testing process, "+\
#              "which will include a path to a trained model, or specify a model with ["+OPT_LOAD+"].. Exiting."); exit(1)
    
    args.config = "./neuralNets/UNetLevelSet/config.cfg"
    args.task = "train"
    
    # 从配置文件中获取网络结构配置
    cfg_file = get_file_absolutely_path(args.config, cwd)
    cfg = Config(cfg_file)
    
    # 通过抽象工厂模式创建网络模型实例
    net_factory = NetFactory(cfg)
    create_net = net_factory.get_net_instance()
    net = create_net()
    model = net.build_network()
    
    if args.task == "train":
        # 从配置文件中读取并设置优化函数
        optimizer = getattr(optimizers, str("get_") + cfg['optimizer'])
        optimizer = optimizer(cfg)
        
        # 从配置文件中读取并设置度量函数
        metric_names = cfg['metrics']
        metric_list = []
        for metric in metric_names:
            func = getattr(metrics, metric)
            metric_list.append(func)
        
        # 每个epoch保存一次模型
        save_path = cfg['folderOutput'] + "/" + cfg['taskName'] + "/" + cfg['modelName'] + '{epoch:02d}-{' + "+".join(cfg['metrics']) + ':2f}.hdf5'
        save_path = get_file_absolutely_path(save_path, cwd)
        checkpoint = ModelCheckpoint(save_path, monitor=metric_list, verbose=1, mode='max')
        callbacks_list = [checkpoint]
        
        model.compile(optimizer, loss=cfg['loss'], metrics=metric_list)
        
        # 记录每个函数的loss，acc等
        log_file = cfg['folderOutput'] + "/" + cfg['logFile']
        log_file = get_file_absolutely_path(log_file, cwd)
        train_log = CSVLogger(filename=log_file)
    
        # 通过自己编写的数据处理将训练集和标签导出
        train_file_list = get_file_absolutely_path(cfg['trainList'], cwd)
        train_input, train_label = get_training_set(train_file_list)
        model.fit(x=train_input, y=train_label, batch_size=cfg['batchSizeTrain'], epochs=cfg['epochs'], callbacks=callbacks_list, shuffle=cfg['shuffle'])
        
        print("Training is finished.")
        
    elif args.task == "test":
        
        print("Testing is finished.")

    
        


    
    
    
    
    
    
    
    

    




    
    
