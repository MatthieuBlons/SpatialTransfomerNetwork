# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:51:36 2021

@author: MBlons
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from stl.transformer import Apply2DTform, Apply3DTform
from pkg.utils import _3Dtform_generation, _2Dtform_generation
#%% Custom Block
def Conv_Block2D(c, 
               filters, 
               name,
               kernel_size=(5, 5), 
               kernel_init=tf.keras.initializers.RandomNormal(stddev=0.01),
               strides=(1, 1),
               padding="same", 
               use_relu=False,
               use_bn=False,
               use_ln=False,
               use_dropout=False, 
               use_bias=True,
               drop_value=0.5,
):
    c = layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, use_bias=use_bias, kernel_initializer=kernel_init, name='{}_Conv'.format(name))(c)
    if use_bn:    
        c = layers.BatchNormalization(name='{}_BatchNorm'.format(name))(c)   
    if use_ln:    
        c = layers.LayerNormalization(name='{}_LayerNorm'.format(name))(c) 
    if use_relu:    
        c = layers.ReLU(name='{}_ReLU'.format(name))(c)
    if use_dropout:
        c = layers.Dropout(drop_value, name='{}_Dropout'.format(name))(c)   
    return c

def Conv_Block3D(c, 
               filters, 
               name,
               kernel_size=(5, 5, 5), 
               kernel_init=tf.keras.initializers.RandomNormal(stddev=0.01),
               strides=(1, 1, 1),
               padding="same", 
               use_relu=False,
               use_bn=False,
               use_ln=False,
               use_dropout=False, 
               use_bias=True,
               drop_value=0.5,
):
    c = layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, use_bias=use_bias, kernel_initializer=kernel_init, name='{}_Conv'.format(name))(c)
    if use_bn:    
        c = layers.BatchNormalization(name='{}_BatchNorm'.format(name))(c)   
    if use_ln:    
        c = layers.LayerNormalization(name='{}_LayerNorm'.format(name))(c) 
    if use_relu:    
        c = layers.ReLU(name='{}_ReLU'.format(name))(c)
    if use_dropout:
        c = layers.Dropout(drop_value, name='{}_Dropout'.format(name))(c)   
    return c
#%% Create Model
def create_FCN_3D(img_shape):
    kernel_init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    
    inputs = layers.Input(img_shape)          
    
    # Spatial Transformer 
    f0 = layers.Flatten()(inputs)
    f1 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense_{}'.format(1))(f0)
    f2 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense_{}'.format(2))(f1)
    f3 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense_{}'.format(3))(f2)
    tform = layers.Dense(12, activation='linear', kernel_initializer='zeros', bias_initializer=tf.constant_initializer([0,0,0,0,0,0,0,0,0,0,0,0]), name='Output_Tform')(f3)
    
    t0 = Apply3DTform(inputs.shape[1:])(inputs, tform, padding=False, interp='Trilinear') 
    p0 = tf.math.reduce_mean(t0, axis=3)
    p0 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(p0)
    
    # Fully Connected Classifier
    f = layers.Flatten()(p0)    
    f = layers.Dense(128, activation=tf.nn.relu)(f) 
    f = layers.Dense(128, activation=tf.nn.relu)(f)
    label = layers.Dense(10, activation=tf.nn.softmax)(f)
             
    # define model
    model = tf.keras.models.Model(inputs, 
                                  [label, p0, tform], 
                                  name='Classifier')       
    return model

def create_CNN_3D(img_shape):
    kernel_init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    
    inputs = layers.Input(img_shape)          
    
    # Spatial Transformer 
    c0 = Conv_Block3D(inputs, filters=64, name='ST_1', kernel_size=(5, 5, 5), kernel_init=kernel_init, strides=(1, 1, 1), padding="valid", use_relu=True)
    c0 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), name='ST_1_MaxPool')(c0)
    c0 = Conv_Block3D(c0, filters=64, name='ST_2', kernel_size=(5, 5, 5), kernel_init=kernel_init, strides=(1, 1, 1), padding="valid", use_relu=True)
    c0 = layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), name='ST_2_MaxPool')(c0)
    c0 = layers.Flatten()(c0)
    c0 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense')(c0)
    tform = layers.Dense(12, activation='linear', kernel_initializer='zeros', bias_initializer=tf.constant_initializer([0,0,0,0,0,0,0,0,0,0,0,0]), name='Output_Tform')(c0)

    t0 = Apply3DTform(inputs.shape[1:])(inputs, tform, padding=False, interp='Trilinear') 
    p0 = tf.math.reduce_max(t0, axis=3)
    p0 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(p0)
    
    # Classifier
    c = Conv_Block2D(p0, filters=32, name='Classifier_1', kernel_size=(9, 9), kernel_init=kernel_init, strides=(1, 1), padding="valid", use_relu=True)
    c = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Classifier_1_MaxPool')(c)
    c = Conv_Block2D(c, filters=32, name='Classifier_2', kernel_size=(7, 7), kernel_init=kernel_init, strides=(1, 1), padding="valid", use_relu=True)
    c = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Classifier_2_MaxPool')(c)
    c = layers.Flatten()(c)
    label = layers.Dense(10, activation=tf.nn.softmax, name='Output_Label')(c) 
    
    # define model
    model = tf.keras.models.Model(inputs, 
                                  [label, t0, tform], 
                                  name='ST_CNN')  
    
    return model
#%% Create fully connected Model
def create_FCN_2D(img_shape):
    # kernel initialization
    kernel_init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    # input layer
    inputs = layers.Input(img_shape)          
    # Spatial Transformer 
    f0 = layers.Flatten()(inputs)
    f1 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense_{}'.format(1))(f0)
    f2 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense_{}'.format(2))(f1)
    f3 = layers.Dense(32, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense_{}'.format(3))(f2)
    tform = layers.Dense(6, activation='linear', kernel_initializer='zeros', bias_initializer=tf.constant_initializer([0,0,0,0,0,0]), name='Output_Tform')(f3)
    # Apply tform
    t0 = Apply2DTform(inputs.shape[1:])(inputs, tform, padding=False, interp='Bilinear') 
    t0 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(t0)
    # Classifier
    f = layers.Flatten()(t0)    
    f = layers.Dense(256, activation=tf.nn.relu, kernel_initializer=kernel_init)(f)
    f = layers.Dense(128, activation=tf.nn.relu, kernel_initializer=kernel_init)(f)
    label = layers.Dense(10, activation=tf.nn.softmax)(f) 
    # define model
    model = tf.keras.models.Model(inputs, 
                                  [label, t0, tform], 
                                  name='ST_FCN')  
    
    return model
#%% Create convolutional Model
def create_CNN_2D(img_shape):
    # kernel initialization
    kernel_init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)
    # input layer
    inputs = layers.Input(img_shape)          
    # Spatial Transformer 
    c0 = Conv_Block2D(inputs, filters=20, name='ST_1', kernel_size=(5, 5), kernel_init=kernel_init, strides=(1, 1), padding="valid", use_relu=True)
    c0 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='ST_1_MaxPool')(c0)
    c0 = Conv_Block2D(c0, filters=20, name='ST_2', kernel_size=(5, 5), kernel_init=kernel_init, strides=(1, 1), padding="valid", use_relu=True)
    c0 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='ST_2_MaxPool')(c0)
    c0 = layers.Flatten()(c0)
    c0 = layers.Dense(20, activation=tf.nn.relu, kernel_initializer=kernel_init, name='ST_Dense')(c0)
    tform = layers.Dense(6, activation='linear', kernel_initializer='zeros', bias_initializer=tf.constant_initializer([0,0,0,0,0,0]), name='Output_Tform')(c0)
    # Apply tform
    t0 = Apply2DTform(inputs.shape[1:])(inputs, tform, padding=False, interp='Bilinear') 
    t0 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), name='ST_AveragePool')(t0)
    # Classifier
    c = Conv_Block2D(t0, filters=64, name='Classifier_1', kernel_size=(9, 9), kernel_init=kernel_init, strides=(1, 1), padding="valid", use_relu=True)
    c = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Classifier_1_MaxPool')(c)
    c = Conv_Block2D(t0, filters=32, name='Classifier_2', kernel_size=(7, 7), kernel_init=kernel_init, strides=(1, 1), padding="valid", use_relu=True)
    c = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='Classifier_2_MaxPool')(c)
    c = layers.Flatten()(c)
    label = layers.Dense(10, activation=tf.nn.softmax, name='Output_Label')(c) 
    # define model
    model = tf.keras.models.Model(inputs, 
                                  [label, t0, tform], 
                                  name='ST_CNN')  
    
    return model
#%% 2D Data generator class
class Data2DGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, Digits, Classes, list_IDs, batch_size, input_dim=(32,32), output_dim=(32,32), n_channels=1, shuffle=True, use_tform=False, use_noise=False):
        'Initialization'        
        self.Digits = Digits
        self.Classes = Classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.use_noise = use_noise
        self.use_tform = use_tform
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)] 

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
       
    def Add_noise(self, Img, loc=0, scale=0.3/3):
        add_noise = np.random.normal(loc=loc, scale=scale, size=Img.shape)
        Img = Img + add_noise.astype('float32')
        return Img
               
    def __data_generation(self, list_IDs_temp):
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        Imgs = np.empty((self.batch_size, *self.input_dim, self.n_channels), dtype='float32')
        Labels = np.empty((self.batch_size, 10), dtype='float32')
        
        for i, ID in enumerate(list_IDs_temp):
                Imgs[i,:,:,:] = self.Digits[ID, :, :, :]
                Labels[i,:] = self.Classes[ID, :]
        
        if self.use_noise:
            Imgs = self.Add_noise(Imgs, loc=0.5, scale=0.5/3)
        
        if self.use_tform:    
            # Dataset Distortion 
            tform = _2Dtform_generation(batch=self.batch_size, dim = self.output_dim, 
                                      limit_tx =16, limit_ty=16, limit_r=45, limit_scale=0.3, limit_shear=0)        
            Imgs = Apply2DTform((*self.output_dim, 1))(Imgs, tform, padding=True, interp='Bilinear') 
            
        return [Imgs, Labels]  
    
#%% 3D Data generator class
class Data3DGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    
    def __init__(self, Digits, Classes, list_IDs, batch_size, input_dim=(28,28), output_dim=(60,60,60), n_channels=1, shuffle=True, use_noise=False, use_tform=False):
        'Initialization'        
        self.Digits = Digits
        self.Classes = Classes
        self.use_tform = use_tform
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.use_noise = use_noise
        
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[index*int(self.batch_size):(index+1)*int(self.batch_size)] 

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
       
    def Add_noise(self, Img, loc=0, scale=0.3/3):
        add_noise = np.random.normal(loc=loc, scale=scale, size=Img.shape)
        Img = Img + add_noise.astype('float32')
        return Img
               
    def __data_generation(self, list_IDs_temp):
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        Imgs = np.empty((self.batch_size, *self.input_dim, self.n_channels), dtype='float32')
        Labels = np.empty((self.batch_size, 10), dtype='float32')

        for i, ID in enumerate(list_IDs_temp):
                Imgs[i,:,:,:] = self.Digits[ID, :, :, :]
                Labels[i,:] = self.Classes[ID, :]

        if self.use_noise:
            Imgs = self.Add_noise(Imgs, loc=0.5, scale=0.5/3)
            
        # Add the channel dim at the end. (N, H, W, C) --> (N, H, W, C, 1)
        Imgs = Imgs[..., None]
        Imgs = tf.keras.backend.permute_dimensions(Imgs, (0,1,2,4,3))

        # Dataset Distortion 
        if self.use_tform:    
            # Dataset Distortion 
            tform = _3Dtform_generation(batch=self.batch_size, 
                                        dim = self.output_dim, 
                                        limit_t=[16,16,16], 
                                        limit_r=[90,90,90], 
                                        limit_scale=0.2)        
            
            Imgs = Apply3DTform((*self.output_dim, 1))(Imgs, tform, padding=True, interp='Trilinear') 
        else:
            Id = tf.cast(tf.tile(tf.expand_dims([1,0,0,0,1,0,0,0,1,0,0,0], axis=0), [self.batch_size, 1]), dtype='float32')
            Imgs = Apply3DTform((*self.output_dim, 1))(Imgs, Id, padding=True, interp='Trilinear')
        
        return [Imgs, Labels]   
    