# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Input, Conv2D, Activation, add
from tensorflow.keras.models import Model

from neuralNets.INeuralNet import INeuralNet

class VDSRNet(INeuralNet):
        
    def build_network(self):
        print("Build architecture for " + self._cfg['modelName'])
        
#        input_img = Input(shape=self._cfg['inputSize'])
#        
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        model = Activation('relu')(model)
#        model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
#        
#        res_img = model
#        output_img = add([res_img, input_img])
#        model = Model(input_img, output_img)
#        return model

        
    