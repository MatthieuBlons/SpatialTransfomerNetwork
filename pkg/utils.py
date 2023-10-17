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
import json
import datetime

#%% Custom tools
#%% 
def my_range(start, step, end):
    """
    Count from start to end with a step lenght equal to step
    """
    while start<=end:
        yield start
        start += step

#%%        
class Timer(object):
    """
    Stopwatch  timer
    """
    def __init__(self, name=None, verbose=0):
        self.name = name
        self.verbose = verbose
        self.TicToc = self.TicTocGenerator()

    def TicTocGenerator(self): # add verbose arg
        # Generator that returns time differences
        ti = 0           # initial time
        tf = time.time() # final time
        while True:
            ti = tf
            tf = time.time()
            yield tf-ti # returns the time difference

    def toc(self, tempBool=True):
        tempTimeInterval = next(self.TicToc)
        if tempBool:
            if self.verbose == 1:
                print( "Elapsed time: %f seconds." %tempTimeInterval )
        return tempTimeInterval    
    
    def tic(self):
        # Records a time in TicToc, marks the beginning of a time interval
        self.toc(False) 

#%%   
def summary(model):
    """
    Use for writing summary data of a neural network, for use in analysis and visualization
    
    Input
    -----
    model<tf.model>
    
    Output
    -----
    short_model_summary<string>
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return short_model_summary 
#%%
class Save_(object):
    """
    Use for saving a neural network after traning + metadata
    
    """
    def __init__(self, savepath, overwrite, verbose=0):
        self.savepath = savepath
        self.overwrite = overwrite
        self.verbose = verbose
        self.date = datetime.datetime.now()
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        self.model_dir = os.path.join(self.savepath, '{}-{}-{}_{}h-{}m'.format(self.date.day, self.date.month, self.date.year, self.date.hour, self.date.minute))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
    def save_model(self, model, name, save_format, include_optimizer):
        if save_format=='pb':
            # save as .pb
            filepath = os.path.join(self.model_dir, name)
            filepath = bytes(filepath, 'utf-8').decode('utf-8', 'ignore')
            tf.keras.models.save_model(model, filepath, overwrite=self.overwrite, include_optimizer=include_optimizer, save_format=save_format)         
        elif save_format=='h5':       
            # save as .h5
            filepath = os.path.join(self.model_dir, name)
            filepath = bytes(filepath, 'utf-8').decode('utf-8', 'ignore')
            tf.keras.models.save_model(model, filepath, overwrite=self.overwrite, include_optimizer=include_optimizer, save_format=save_format) 
        
        if self.verbose==1:
            print('model saved at : ' + filepath)
        
    def save_model_summary(self, model):
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        
        Summary = self.model_dir + '\model_Summary.txt' 
        if self.overwrite:
            fichier = open(Summary, "w")
        else:
            fichier = open(Summary, "a")
        fichier.write(short_model_summary)
        fichier.close()
        if self.verbose==1:
            print('model summary writen at : ' + Summary)
            
    def save_dict2json(self, dictionary, name):
        JsonMeta = os.path.join(self.model_dir,  name + '.json')
        with open(JsonMeta, 'w') as outfile:
            json.dump(dictionary, outfile)
              
    def write_dict2txt(self, dictionary, name):
        Summary = os.path.join(self.model_dir, name + '.txt')
        if self.overwrite:
            fichier = open(Summary, "w")
        else:
            fichier = open(Summary, "a")
        
        fichier.write(name + ' : \n\n')
        for key, value in dictionary.items(): 
            fichier.write('%s:%s\n' % (key, value))
            
        fichier.close()
        if self.verbose==1:
            print(name + ' writen at : ' + Summary)
#%%        
class ReduceLr(object):
    """
    Use for reducing the learning rate during training
    
    """
    def __init__(self, optimizer, verbose=0):
        self.optimizer = optimizer
        self.wait = 0
        self.epoch= 0
        self.verbose=verbose
    
    def ReduceOnPlateau(self, monitor, factor=0.1, patience=5, min_delta=0.0001, min_lr=0):
        self.epoch +=1
        self.wait += 1
        if self.wait>patience:
            old_lr = tf.keras.backend.get_value(self.optimizer.learning_rate)
            if old_lr>min_lr:
                indx = self.epoch-patience
                diff = [i-j for i, j in zip(monitor[indx-1:-1], monitor[indx:])]
                monitor_test = np.array(diff) < min_delta
                if np.all(monitor_test):
                    new_lr = old_lr * factor
                    new_lr = max(new_lr, min_lr)
                    tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr) 
                    if self.verbose > 0:
                        print('Epoch {}: ReduceLROnPlateau reducing learning rate to {:.8f}'.format(self.epoch, new_lr))
                    self.wait = 0
            
    def ReduceOnEpoch(self, boundary, factor=0.1, min_lr=0):
        self.epoch +=1
        if self.epoch>1:
            if self.epoch==boundary:
                old_lr = tf.keras.backend.get_value(self.optimizer.learning_rate)
                if old_lr>min_lr:
                    new_lr = old_lr * factor
                    new_lr = max(new_lr, min_lr)
                    tf.keras.backend.set_value(self.optimizer.learning_rate, new_lr) 
                    if self.verbose > 0:
                        print('Epoch {}: ReduceLROnPlateau reducing learning rate to {:.8f}'.format(self.epoch, new_lr))
       
#%%
def mosaic(sampleID, vol1, cmap1 = "hot", lim1=[None,None], vol2 = None, cmap2 = None, lim2=[None,None], alpha = None, title='mosaic'):
    """
    Display slices /last dim of a volume or a montage (two volumes superimposed)
    
    """
    # Get n samlpes
    N = len(sampleID)
    nrows = np.int(np.ceil(np.sqrt(N)))
    ncols = nrows
    # Init figure
    fig, ax = plt.subplots(figsize=(12,12), nrows=nrows, ncols=ncols)
    fig.suptitle(title)
    ax = ax.reshape((nrows*ncols))
    if vol2 is None:
        for cnt in range(nrows*ncols):
            if cnt < N:
                ax[cnt].imshow(vol1[:,:,sampleID[cnt]], cmap=cmap1, vmin=lim1[0], vmax=lim1[1])
                ax[cnt].set_title('{}'.format(sampleID[cnt]), fontsize=8, pad=0)
                ax[cnt].get_xaxis().set_visible(False)
                ax[cnt].get_yaxis().set_visible(False)
            else:
                ax[cnt].set_axis_off()
    else:
        for cnt in range(nrows*ncols):
            if cnt < N:
                ax[cnt].imshow(vol1[:,:,sampleID[cnt]], cmap=cmap1, vmin=lim1[0], vmax=lim1[1])
                ax[cnt].imshow(vol2[:,:,sampleID[cnt]], cmap=cmap2, alpha=alpha, vmin=lim2[0], vmax=lim2[1])
                ax[cnt].set_title('{}'.format(sampleID[cnt]), fontsize=8,  pad=0)
                ax[cnt].get_xaxis().set_visible(False)
                ax[cnt].get_yaxis().set_visible(False)
            else:
                ax[cnt].set_axis_off()
    return fig      
    






