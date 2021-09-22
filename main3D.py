# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:57:33 2020

@author: MBlons
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import sys
sys.path.append(r'E:\MB\Projets\Spatial_Transformer\MNIST_DigitRecog_SpatialTransformer')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import datetime

from pkg.utils import tic, toc, _3Dtform_generation, save_models, load_data, plot_3Dimages, schedule, summary
from pkg.nn import create_FCN_3D, Data3DGenerator
from stl.transformer import Apply3DTform
#%% GPU config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) # allow memory growth
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=5000)]) # start with 1000 Mo
  except RuntimeError as e:
    print(e)
#%%
@tf.function          
def CategoricalCrossentropy(y_true, y_pred):
    loss = tf.keras.backend.mean(tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')(y_true, y_pred))
    return loss

@tf.function
def train_step(model, Imgs, Labels, criterion, optimizer, metric):    
    # train generator    
    with tf.GradientTape() as reg_tape:
        # Prediction
        [prediction, t0, tform] = model(Imgs, training=True)       
        # Calculate loss
        loss = criterion(Labels, prediction)  
        # Calculate accuracy 
        metric.update_state(Labels, prediction)
        acc = metric.result()
    # Compute gardient w.r.t the discriminator loss
    gradients_of_registrator = reg_tape.gradient(loss, model.trainable_variables)
    # Update the weights of the critic using the critic optimizer (Back propagation)
    optimizer.apply_gradients(zip(gradients_of_registrator, model.trainable_variables))    
    return [loss, acc]

@tf.function
def test_step(model, Imgs, Labels, criterion, metric):        
    # Prediction
    [prediction, t0, tform] = model(Imgs, training=False)       
    # Calculate loss
    loss = criterion(Labels, prediction) 
    # Calculate accuracy 
    metric.update_state(Labels, prediction)
    acc = metric.result()
    return [loss, acc]

#%%
def main(args):
    print('PREP...')
    loss = {}
    loss['Train'] = []
    loss['Valid'] = []
    accuracy = {}
    accuracy['Train'] = []
    accuracy['Valid'] = []
    
    # Load preprocessed training and testing data.
    (x_train, y_train), (x_test, y_test) = load_data(args.N_train, args.N_test, shape=args.init_dim)
    
    # Select some images from the test set to show sample results.
    ids = tf.constant(np.random.choice(x_test.shape[0], replace=False,
                                       size=args.num_samples))
    x_sample = tf.gather(x_test, ids)
    y_sample = tf.gather(y_test, ids)
    # Dataset Distortion 
    tform_sample = _3Dtform_generation(batch=args.num_samples, dim=args.input_dim, 
                                       limit_t=[16,16,16], limit_r=[90,90,90], limit_scale=0.2)
    # Add the channel dim at the end. (N, H, W, C) --> (N, H, W, C, 1)
    x_sample = x_sample[..., None]
    x_sample = tf.keras.backend.permute_dimensions(x_sample, (0,1,2,4,3))    
    x_sample = Apply3DTform((*args.input_dim, 1))(x_sample, tform_sample, padding=True, interp='Trilinear') 
    
    if not args.use_tform :
        # Dataset Distortion 
        tic()
        tform_train = _3Dtform_generation(batch=args.N_train, dim=args.input_dim, 
                                           limit_t=[16,16,16], limit_r=[90,90,90], limit_scale=0.2)
        toc()
        print('tform generated')
        # Add the channel dim at the end. (N, H, W, C) --> (N, H, W, C, 1)
        x_train = x_train[..., None]
        x_train = tf.keras.backend.permute_dimensions(x_train, (0,1,2,4,3))   

        tic()
        x_train = Apply3DTform((*args.input_dim, 1))(x_train, tform_train, padding=True, interp='Trilinear') 
        toc()
        print('Distortion Done')
        
    # Shuffle and batch the dataset.    
    N = x_train.shape[0]-1   
    partition = {}
    partition['train']=[]
    partition['valid']=[]
    for i in np.sort(random.sample(range(N), N)):
        a = random.uniform(0,1)
        if a < args.train_ratio :
            partition['train'].append(i+1)
        else:
            partition['valid'].append(i+1)
            
    # Create a model instance.
    model = create_FCN_3D(img_shape=(*args.input_dim, 1))
    if args.display_summary:
        model.summary()
        
    # Select optimizer and loss function.
    optimizer = args.optimizer(args.lr)
    criterion = CategoricalCrossentropy
    metric = tf.keras.metrics.CategoricalAccuracy()
    
    plot_3Dimages(model, x_sample, y_sample)
    print('Training START')
    now = datetime.datetime.now()
    # Train and evaluate the model.
    for epoch in range(args.epochs):  

        start = time.time()
        if args.plot_result:
            if int(epoch+1) % int(args.freq) == 0:
                plot_3Dimages(model, x_sample, y_sample)
        
        if args.use_schedule:
            boundary = np.int(np.ceil(args.epochs - args.epochs*args.boundary))
            actual_g_lr = schedule(epoch+1, boundary=boundary, decay=args.lr_decay, optimizer=optimizer) 
            print('epoch = {}/{} // learning rate = {}'.format(epoch+1, args.epochs, actual_g_lr))
            
        train_dataset = Data3DGenerator(Digits=x_train, 
                                        Classes=y_train, 
                                        list_IDs=partition['train'], 
                                        batch_size=args.batch_size, 
                                        input_dim=args.init_dim, 
                                        output_dim=args.input_dim, 
                                        n_channels=1, 
                                        shuffle=args.shuffle, 
                                        use_noise=args.use_noise, 
                                        use_tform=args.use_tform)   
        
        for [Imgs, Labels] in train_dataset:
                        
            loss_on_iteration, acc_on_iteration = train_step(model, Imgs, Labels, criterion, optimizer, metric)
            loss['Train'].append(np.array(loss_on_iteration))
            accuracy['Train'].append(np.array(acc_on_iteration))
            
            indx = np.sort(random.sample(range(len(partition['valid'])), args.batch_size))    

            valid_dataset = Data3DGenerator(Digits=x_train, 
                                            Classes=y_train, 
                                            list_IDs=[partition['valid'][indx[i]] for i in range(args.batch_size)], 
                                            batch_size=args.batch_size, 
                                            input_dim=args.init_dim, 
                                            output_dim=args.input_dim, 
                                            n_channels=1,
                                            shuffle=args.shuffle, 
                                            use_noise=args.use_noise,
                                            use_tform=args.use_tform)               
            
            for [Imgs, Labels] in valid_dataset:
                          
                losses_on_iteration, acc_on_iteration = test_step(model, Imgs, Labels, criterion, metric)
                loss['Valid'].append(np.array(losses_on_iteration))
                accuracy['Valid'].append(np.array(acc_on_iteration))
            
        print ('Time for epoch {} is {:.1f} sec'.format(epoch + 1, time.time()-start))
        print ('[Train] // loss = {:.4f} , acc = {:.4f}'.format(loss['Train'][-1], accuracy['Train'][-1]))
        print ('[Valid] // loss = {:.4f} , acc = {:.4f}'.format(loss['Valid'][-1], accuracy['Valid'][-1]))
    print('\n')
    
    # Show sample results.
    if args.plot_result:
        plot_3Dimages(model, x_sample, y_sample)
        
    # Save the trained model.
    if args.save_model:
        if not os.path.exists(args.ModelDir):
            os.makedirs(args.ModelDir)
        save_dir = os.path.join(args.ModelDir, 'Model_M{}_D{}_h{}_m{}'.format(now.month, now.day, now.hour, now.minute))
        if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
        save_models(model, save_dir, args.epochs)
        # save meta
        Param_file = save_dir + '\PARAM.txt' # training param
        fichier = open(Param_file, "a")
        Text =  ("DATE : {}/{}/{}".format(now.day, now.month, now.year),
                "Image_dim = {}".format(args.input_dim), 
                "N_train_images = {}".format(args.N_train),
                "N_test_images = {}".format(args.N_test),               
                "batch_size = {}".format(args.batch_size),
                "Suffle = {}".format(args.shuffle),
                "Epochs = {}".format(args.epochs),          
                "Loss = {}".format(criterion),
                "Accuracy = {}".format(metric),
                "Optimizer = {}".format(args.optimizer),  
                "Learning Rate = {}".format(args.lr), 
                "Use Schedule = {}".format(args.use_schedule),
                "Boundary = {}".format(args.boundary),
                "Learning Rate decay = {}".format(args.lr_decay),
                "Training Time = {}s".format(training_time))
        To_write = "\n".join(Text)
        
        fichier.write(To_write)
        fichier.close()
        
        fichier_losses = save_dir + '\Train_Losses.txt'
        with open(fichier_losses, "a") as file:
            for i in range(len(np.array(loss['Train']))):
                to_write = '{}\n'.format(np.array(loss['Train'])[i])   
                to_write = to_write.replace('[', '')
                to_write = to_write.replace(']', '')
                file.write(to_write) 
        
        fichier_losses = save_dir + '\Valid_Losses.txt'
        with open(fichier_losses, "a") as file:
            for i in range(len(np.array(loss['Valid']))):
                to_write = '{}\n'.format(np.array(loss['Valid'])[i])   
                to_write = to_write.replace('[', '')
                to_write = to_write.replace(']', '')
                file.write(to_write)  
                
        # Details : [tf.reduce_mean(mse), tf.reduce_mean(perceptual), tf.reduce_mean(min_true), tf.reduce_mean(min_pred), tf.reduce_mean(max_true), tf.reduce_mean(max_pred)]         
        fichier_losses = save_dir + '\Accuracy_Train.txt'
        with open(fichier_losses, "a") as file:
            for i in range(len(np.array(accuracy['Train']))):
                to_write = '{}\n'.format(np.array2string(np.array(accuracy['Train'])[i], precision=4, floatmode='fixed'))   
                to_write = to_write.replace('[', '')
                to_write = to_write.replace(']', '')
                file.write(to_write) 
        
        fichier_losses = save_dir + '\Accuracy_Valid.txt'
        with open(fichier_losses, "a") as file:
            for i in range(len(np.array(accuracy['Valid']))):
                to_write = '{}\n'.format(np.array2string(np.array(accuracy['Valid'])[i], precision=4, floatmode='fixed'))   
                to_write = to_write.replace('[', '')
                to_write = to_write.replace(']', '')
                file.write(to_write)     
                
        Summary = save_dir + '\Model_Summary.txt' 
        fichier = open(Summary, "a")
        fichier.write(summary(model))
        fichier.close()
        
    return model, loss, accuracy
#%%    
if __name__ == '__main__':

    class Args():
        init_dim = (28, 28)
        input_dim = (60, 60, 60)
        N_train = 5000
        N_test = 100
        train_ratio = 0.9
        batch_size = 16
        epochs = 1 # 150k iterations
        optimizer = tf.keras.optimizers.Adam
        lr = 0.001
        num_samples = 5
        save_model = False  
        use_schedule = False
        boundary = 0.33
        lr_decay = 0.1
        shuffle = True
        use_noise= False
        use_tform= True
        plot_result =True
        display_summary=True
        freq = 10
        ModelDir =r'E:\MB\Projets\Spatial_Transformer\MNIST_DigitRecog_SpatialTransformer\Trained_Models\MNIST\3D_Digits_Recog'
        
        
tic()  
args = Args()
model, loss, accuracy = main(args)    
training_time=toc()    

loss_train = np.array(loss['Train'])
loss_valid = np.array(loss['Valid'])
acc_train = np.array(accuracy['Train'])
acc_valid = np.array(accuracy['Valid'])

epoch = np.linspace(1, len(loss_train), len(loss_train), endpoint=False, dtype='int32')
fig, ax = plt.subplots(nrows=2, ncols=1)
fig.subplots_adjust(hspace=0.5)
ax[0].plot(epoch, loss_train, 'r', epoch, loss_valid, 'b')
ax[0].legend(['Train', 'Valid'])
ax[0].set_ylabel('loss')
ax[0].set_xlabel('epoch')
ax[0].set_title('Loss')
ax[1].plot(epoch, acc_train, 'r', epoch, acc_valid, 'b')
ax[1].legend(['Train', 'Valid'])
ax[1].set_ylabel('acc')
ax[1].set_xlabel('epoch')
ax[1].set_title('Acc')
#%%
# Load preprocessed training and testing data.
(x_train, y_train), (x_test, y_test) = load_data(100, 100)
 
# Select some images from the test set to show sample results.
ids = tf.constant(np.random.choice(x_test.shape[0], replace=False,
                                   size=1))

x_sample = tf.gather(x_test, ids)
y_sample = tf.gather(y_test, ids)

# Dataset Distortion 

# Id = tf.cast(tf.expand_dims([1,0,0,0,1,0,0,0,1,0,0,0], axis=0), dtype='float32')
tform_sample = _3Dtform_generation(batch=1, dim=args.input_dim, 
                          limit_t=[16,16,16], limit_r=[45,45,45], limit_scale=0.3)

# Add the channel dim at the end. (N, H, W, C) --> (N, H, W, C, 1)
x_sample = x_sample[..., None]
x_sample = tf.keras.backend.permute_dimensions(x_sample, (0,1,2,4,3))
x_sample = Apply3DTform((*args.input_dim, 1))(x_sample, tform_sample, padding=True, interp='Trilinear') 
    
Prediction, P0, tform = model(x_sample, training=False)
print('tform={}'.format(tform))
Prediction = np.argmax(Prediction, axis=-1)
print('Prediction={}'.format(int(Prediction)))

fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2)
ax[0].imshow(np.squeeze(tf.reduce_sum(x_sample, 3).numpy()))
ax[0].set_title('Label = {}'.format(np.argmax(y_sample)), fontsize=20)
ax[1].imshow(np.squeeze(P0.numpy()))
ax[1].set_title('Pred Label = {}'.format(int(Prediction)), fontsize=20)
