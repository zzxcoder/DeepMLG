# -*- coding: utf-8 -*-
# 
# 从配置文件读取的度量函数名，利用反射机制获得该回调函数
# 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics

#直接给出GT和预测后的值进入即可（需要归一化，即除以最大值）
def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def accuracy(y_true, y_pred):
    return metrics.categorical_accuracy(y_true, y_pred)