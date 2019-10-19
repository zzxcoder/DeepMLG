# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Softmax, Dense
from tensorflow.keras.models import Sequential

from neuralNets.INeuralNet import INeuralNet

# VGG16主干网，只保留前面的卷积层，去掉输入层、后面的全链接层和Softmax层
class VGG16Backbone(INeuralNet):
    
    def build_network(self):
        print("Build architecture for " + self._cfg['modelName'])
        
        model = Sequential()
        input_image = Input(self._cfg['inputSize'])
        
        # layer1
        model.add( Conv2D(64, (3,3), strides=(1,1), input_shape=input_image.shape[1:], padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
        model.add( Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
        model.add( MaxPooling2D((2,2)) )
        
        # layer2
#        model.add( Conv2D(128, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( Conv2D(128, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( MaxPooling2D((2,2)) )
#        
#        # layer3
#        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( Conv2D(256, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( MaxPooling2D((2,2)) )
#        
#        # layer4
#        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( MaxPooling2D((2,2)) )
#        
#        # layer5
#        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( Conv2D(512, (3,3), strides=(1,1), padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform') )
#        model.add( MaxPooling2D((2,2)) )
        
        return model
        

class VGGNet16(INeuralNet):
    
    def build_network(self):
        print("Build architecture for " + self._cfg['modelName'])
        
        backbone = VGG16Backbone(self._cfg)
        model = backbone.build_network()
        model.add( Flatten() )
#        model.add( Dense(4096, activation='relu') )
#        model.add( Dense(4096, activation='relu') )
        model.add( Dense(1000, activation='relu') )
        model.add( Dense(10, activation='softmax') )
        
        model.summary()
        return model