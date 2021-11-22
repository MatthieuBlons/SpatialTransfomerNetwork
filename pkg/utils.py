# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:46:24 2021

@author: MBlons
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import random
from skimage import transform
from stl.transformer import Apply2DTform, Apply2DDispField
#%% Custom tools
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() 

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds" %tempTimeInterval )
    return tempTimeInterval  

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
def _2Dtform_generation(batch=1, dim=(28,28), limit_tx=16, limit_ty=16, limit_r=45, limit_scale=0.1, limit_shear=0.1):
    
    # Translation              
    tx = tf.random.uniform([batch, 1], -limit_tx, limit_tx)/dim[0]*2
    ty = tf.random.uniform([batch, 1], -limit_ty, limit_ty)/dim[1]*2     
    
    trans = tf.concat([tx, ty], axis=-1)
        
    # Rotation
    r = tf.random.uniform([batch, 1], -limit_r, limit_r)*(np.pi/180) #deg to rad
    
    cos = tf.tile(tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(r)
    sin = tf.tile(tf.cast(tf.expand_dims([0,-1,1,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(r)
    rot = cos+sin

    # Scaling
    sx = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
    sy = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
    
    scalex = tf.tile(tf.cast(tf.expand_dims([1,0,0,0], axis=0), dtype='float32'), [batch, 1])*sx
    scaley = tf.tile(tf.cast(tf.expand_dims([0,0,0,1], axis=0), dtype='float32'), [batch, 1])*sy
    scale = tf.tile(tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32'), [batch, 1])+scalex+scaley

    # Shear
    sx = tf.random.uniform([batch, 1], -limit_shear, limit_shear)
    sy = tf.random.uniform([batch, 1], -limit_shear, limit_shear)
    
    shearx = tf.tile(tf.cast(tf.expand_dims([0,1,0,0], axis=0), dtype='float32'), [batch, 1])*sx
    sheary = tf.tile(tf.cast(tf.expand_dims([0,0,1,0], axis=0), dtype='float32'), [batch, 1])*sy
    shear = tf.tile(tf.cast(tf.expand_dims([1,0,0,1], axis=0), dtype='float32'), [batch, 1])+shearx+sheary
    
    # Combine 
    matrix = tf.linalg.matmul(tf.reshape(shear, [batch, 2, 2]), tf.linalg.matmul(tf.reshape(scale, [batch, 2, 2]), tf.reshape(rot, [batch, 2, 2])))

    tform = tf.concat([tf.reshape(matrix, [batch, 4]), trans], axis=-1)
    
    return tform  

def _3Dtform_generation(batch=1, dim = (32,32,32), limit_t = [16,16,16],  limit_r=[20,20,20], limit_scale=0.8):
  
    # Translation              
    tx = tf.random.uniform([batch, 1], -limit_t[0], limit_t[0])/dim[0]*2
    ty = tf.random.uniform([batch, 1], -limit_t[1], limit_t[1])/dim[1]*2     
    tz = tf.random.uniform([batch, 1], -limit_t[2], limit_t[2])/dim[1]*2  
    
    trans = tf.concat([tx, ty, tz], axis=-1)
        
    # Rotation                   
    rx = tf.random.uniform([batch, 1], -limit_r[0], limit_r[0])*(np.pi/180) #deg to rad
    ry = tf.random.uniform([batch, 1], -limit_r[1], limit_r[1])*(np.pi/180) #deg to rad
    rz = tf.random.uniform([batch, 1], -limit_r[2], limit_r[2])*(np.pi/180) #deg to rad 

        
    cosx = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,1], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(rx)
    sinx = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,0,-1,0,1,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(rx)
    rotx = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,0], axis=0), dtype='float32'), [batch, 1])+cosx+sinx
                   
    cosy = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,1], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(ry)
    siny = tf.tile(tf.cast(tf.expand_dims([0,0,1,0,0,0,-1,0,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(ry)
    roty = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,0], axis=0), dtype='float32'), [batch, 1])+cosy+siny
    
    cosz = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*tf.math.cos(rz)
    sinz = tf.tile(tf.cast(tf.expand_dims([0,-1,0,1,0,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*tf.math.sin(rz)
    rotz = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,0,0,0,0,1], axis=0), dtype='float32'), [batch, 1])+cosz+sinz
    
    # Combine rotation
    rot = tf.linalg.matmul(tf.reshape(rotz, [batch, 3, 3]), tf.linalg.matmul(tf.reshape(roty, [batch, 3, 3]), tf.reshape(rotx, [batch, 3, 3])))
   
    # Scaling
    sx = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
    sy = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
    sz = tf.random.uniform([batch, 1], -limit_scale, limit_scale)
    
    scalex = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,0,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*sx
    scaley = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,1,0,0,0,0], axis=0), dtype='float32'), [batch, 1])*sy
    scalez = tf.tile(tf.cast(tf.expand_dims([0,0,0,0,0,0,0,0,1], axis=0), dtype='float32'), [batch, 1])*sz
    
    scale = tf.tile(tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,1], axis=0), dtype='float32'), [batch, 1])+scalex+scaley+scalez
           
    # Combine 
    matrix = tf.linalg.matmul(tf.reshape(scale, [batch, 3, 3]), rot)
    
    tform = tf.concat([tf.reshape(matrix, [batch, 9]), trans], axis=-1)
    
    return tform 
           
def save_models(model, save_dir, epoch_number):            
    tf.keras.models.save_model(model, os.path.join(save_dir, 'model_on_epoch_{}'.format(epoch_number)), overwrite=True, include_optimizer=False, save_format='tf')         
    model.save(os.path.join(save_dir, 'model_on_epoch_{}.h5'.format(epoch_number)), overwrite=True, include_optimizer=True)  
    
    
def summary(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary 

def schedule(epoch, decay, boundary, optimizer):
    if epoch % boundary == 0:                      
        g_lr = tf.keras.backend.get_value(optimizer.learning_rate)
        tf.keras.backend.set_value(optimizer.learning_rate, g_lr*decay)           
    return tf.keras.backend.get_value(optimizer.learning_rate)

def plot_2Dimages(model, Digits, Labels, method='tform'):
    nb, nh, nw, nc = Digits.shape
    
    Labels = np.argmax(Labels, axis=-1)
    if method=='tform':
        Prediction, T0, tform = model(Digits, training=False)
        Prediction = np.argmax(Prediction, axis=-1)
        Digit_tform = Apply2DTform((nh, nw, nc))(Digits, tform, padding=True, interp='Bilinear')
        
    elif method=='dispfield':
        Prediction, T0, disp = model(Digits, training=False)
        Prediction = np.argmax(Prediction, axis=-1)
        Digit_tform = Apply2DDispField((nh, nw, nc))(Digits, disp, padding=True, interp='Bilinear')
            
    # Convert the tensors to images.
    T0 = T0.numpy().squeeze(axis=-1)
    T0 = T0.astype(np.float32)
    
    Digits = Digits.numpy().squeeze(axis=-1)
    Digits = Digits.astype(np.float32)
    
    Digit_tform = Digit_tform.numpy().squeeze(axis=-1)
    Digit_tform = Digit_tform.astype(np.float32)
    # Plot images.
    fig = plt.figure(figsize=(3 * 1.7, nb * 1.7))
    images_list = [Digits, Digit_tform, T0]
    
    for i in range(nb):
        for j in range(3):
            ax = fig.add_subplot(nb, 3, i * 3 + j + 1)
            if j == 0:
                ax.set_title('Label = {}'.format(Labels[i]), fontsize=20)
            if j == 1:
                ax.set_title('Label = {}'.format(Labels[i]), fontsize=20)
            if j == 2:
                ax.set_title('Label = {}'.format(Prediction[i]), fontsize=20)
            ax.set_axis_off()
            ax.imshow(images_list[j][i], vmin=0, vmax=1, cmap='hot')

    plt.tight_layout()
    plt.show()    

def plot_3Dimages(model, Digits, Labels, method='tform'):
    nb, nh, nw, nd, nc = Digits.shape
    
    Labels = np.argmax(Labels, axis=-1)
    
    Prediction, P0, tform = model(Digits, training=False)
    Prediction = np.argmax(Prediction, axis=-1)

    Digits = tf.math.reduce_max(Digits, axis=3)
    
    # Convert the tensors to images.
    P0 = P0.numpy().squeeze(axis=-1)
    P0 = P0.astype(np.float32)
    Digits = Digits.numpy().squeeze(axis=-1)
    Digits = Digits.astype(np.float32)

    # Plot images.
    fig = plt.figure(figsize=(2 * 1.7, nb * 1.7))
    images_list = [Digits, P0]
    
    for i in range(nb):
        for j in range(2):
            ax = fig.add_subplot(nb, 2, i * 2 + j + 1)
            if j == 0:
                ax.set_title('Label = {}'.format(Labels[i]), fontsize=20)
            if j == 1:
                ax.set_title('Label = {}'.format(Prediction[i]), fontsize=20)
            ax.set_axis_off()
            ax.imshow(images_list[j][i], vmin=0, vmax=1, cmap='hot')

    plt.tight_layout()
    plt.show()         
    
def load_data(N_train=10000, N_test=1000, shape=(32,32)):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = (x_train[:N_train, :, :], y_train[:N_train]), (x_test[:N_test, :, :], y_test[:N_test])
    
    # Scale the image to [0, 1] range.
    x_train = (x_train.astype(np.float32) / 255.)
    x_test = (x_test.astype(np.float32) / 255.)
    
    # Resize images from (28, 28) to (32, 32).
    x_train = transform.resize(x_train, (x_train.shape[0], shape[1], shape[1]))
    x_test = transform.resize(x_test, (x_test.shape[0], shape[1], shape[1]))
    
    # Add channel
    x_train = x_train[..., None]
    x_test = x_test[..., None]    
    
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)