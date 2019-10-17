# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import sys
import os
import os.path
import argparse
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from metrics.PSNR import PSNR
# 将自己定义的网络模型写入该文件下并导出，名字自拟
from neuralNets.netInterface import model
# 导出数据处理，返回数据以及标签，自己定义
from dataManagement import dataPrepare

BATCH_SIZE = 64
EPOCHS = 200

if __name__ == "main":
    ## 终端命令行参数
    parser = argparse.ArgumentParser(description='')
    parser.add_argument()

    args = parser.parse_args()

    ## 判断输入的命令参数是否正确
    if len(sys.argv) == 1:
        print("For help on the usage of this program, please use the option -h.")
        exit(1)

    # 定义一些优化函数，参数可调 可补充，具体参数可以查询keras文档
    adam = Adam(lr=0.001, decay=1e-5, clipvalue=0.1, epsilon=1e-8)
    sgd = SGD(lr=1e-2, momentum=0.9, decay=1e-4, nesterov=False)

    # 回调函数 可以保存一些需要的东西（以VDSR举例，后面可以更换）
    # 每个epoch保存一次模型
    filepath = "./checkpoints/vdsr-{epoch:02d}-{PSNR:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max')
    callbacks_list = [checkpoint]

    # 记录每个函数的loss，acc等
    train_log = CSVLogger(filename="train.log")

    #开始训练
    my_model = model()
    my_model.compile(adam, loss='mse', metrics=[PSNR, "accuracy"])
    # 通过自己编写的数据处理将训练集和标签导出
    train_input, train_label = dataPrepare()
    my_model.fit(x=train_input, y=train_label, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list, shuffle=True)

    print("Training is done")




    
    
