# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Activation
import tensorflow as tf

class RLSLayer(object):
    
    def __init__(self, img, phi):
        self.__img = img
        self.__phi = phi
        self.__C1 = 0
        self.__C2 = 0
        self.__UzDim = 1
        self.__WzDim = 1
        self.__UrDim = 1
        self.__WrDim = 1
        self.__UhDim = 1
        self.__WhDim = 1
        self.__Vdim = 1
        
    def generate_sequence_data(self, phi):
        '''
        生成当前序列数据：由输入图像和当前水平集函数phi得到的delta phi公式中的中括号部分
        '''
        eps = 1e-6
        height = phi.shape[0]
        width = phi.shape[1]
        seq_data = np.array( (height, width), dtype=np.float64 )
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                C1 = 1 / np.sqrt( eps + np.power((phi[y + 1, x] - phi[y, x]), 2) + np.power((phi[y, x + 1] - phi[y, x - 1]), 2) / 4 )
                C2 = 1 / np.sqrt( eps + np.power((phi[y, x] - phi[y - 1, x]), 2) + np.power((phi[y - 1, x + 1] - phi[y - 1, x - 1]), 2) / 4 )
                C3 = 1 / np.sqrt( eps + np.power((phi[y + 1, x] - phi[y - 1, x]), 2) / 4 + np.power((phi[y, x + 1] - phi[y, x]), 2) )
                C4 = 1 / np.sqrt( eps + np.power((phi[y + 1, x - 1] - phi[y - 1, x - 1]), 2) / 4 + np.power((phi[y, x] - phi[y, x - 1]), 2) )
                C = C1 + C2 + C3 + C4
                
                deltaPhi = self._Dirac(phi[y, x])
                factor = self.__dt * deltaPhi * self.__mu
                F1 = factor * C1 / (self.__h + factor * C)
                F2 = factor * C2 / (self.__h + factor * C)
                F3 = factor * C3 / (self.__h + factor * C)
                F4 = factor * C4 / (self.__h + factor * C)
                F = self.__h / (self.__h + factor * C)
                pij = phi[y, x] - self.__dt * deltaPhi * ( self.__upsilon + self.__lambda1 * np.power(self.__img.getpixel((x, y)) - self.__C1, 2) - self.__lambda2 * np.power(self.__img.getpixel((x, y)) - self.__C2, 2) )
                seq_data[y, x] = F1 * phi[y + 1, x] + F2 * phi[y - 1, x] + F3 * phi[y, x + 1] + F4 * phi[y, x - 1] + F * pij
                
        for y in range(height):
            seq_data[y, 0] = seq_data[y, 1]
            seq_data[y, width - 1] = seq_data[y, width - 2]
        for x in range(width):
            seq_data[0, x] = seq_data[1, x]
            seq_data[height - 1, x] = seq_data[height - 2, x]
        
        return seq_data
    
    
    def compute_update_gate(self, seq, phi):
        assert( (seq.shape[0] == phi.shape[0]) and (seq.shape[1] == phi.shape[1]) )
        assert(self.__UzDim == self.__WzDim)
        U = tf.Variable(shape=[self.__HerperParam_WDim, seq.shape[0]], initial_value=tf.norm, name="Uz")
        W = tf.Variable(shape=[self.__HerperParam_WDim, phi.shape[0]], initial_value=tf.norm, name="Wz")
        b = tf.Variable(shape=[self.__HerperParam_WDim, seq.shape[1]], initial_value=tf.norm, name="bz")
        output = tf.matmul(U, seq) + tf.matmul(W, phi) + b
        return output
    

    def compute_reset_gate(self, seq, phi):
        assert( (seq.shape[0] == phi.shape[0]) and (seq.shape[1] == phi.shape[1]) )
        assert(self.__UrDim == self.__WrDim)
        U = tf.Variable(shape=[self.__UrDim, seq.shape[0]], initial_value=tf.norm, name="Ur")
        W = tf.Variable(shape=[self.__WrDim, phi.shape[0]], initial_value=tf.norm, name="Wr")
        b = tf.Variable(shape=[self.__WrDim, phi.shape[1]], initial_value=tf.norm, name="br")
        output = tf.matmul(U, seq) + tf.matmul(W, phi) + b
        return output
    
    
    def compute_hidden_state(self, seq, phi, resetGate):
        assert( (seq.shape[0] == phi.shape[0]) and (seq.shape[1] == phi.shape[1]) )
        assert(self.__UhDim == self.__WhDim)
        U = tf.Variable(shape=[self.__UhDim, seq.shape[0]], initial_value=tf.norm, name="Uh")
        dot = phi * resetGate
        W = tf.Variable(shape=[self.__WhDim, dot.shape[0]], initial_value=tf.norm, name="Wh")
        b = tf.Variable(shape=[self.__WhDim, dot.shape[1]], initial_value=tf.norm, name="bh")
        output = tf.tanh(tf.matmul(U, seq) + tf.matmul(W, dot) + b)
        return output
        
    
    def compute_output(self, phi):
        V = tf.Variable(shape=[self.__Vdim, phi.shape[0]], initial_value=tf.norm, name="V")
        b = tf.Variable(shape=[self._Vdim, phi.shape[1]], initial_value=tf.norm, name="b")
        output = tf.nn.softmax(tf.matmul(V, phi) + b)
        return output
    

if __name__ == "__main__":
    img = Image.open("/home/zzx/work/DeepMLG/neuralNets/UNetLevelSet/data/1.jpg").convert("L")
    rlsLayer = RLSLayer(img);
    rlsLayer.generate_sequence_data();
        
        
        
    
    