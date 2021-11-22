# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:51:00 2021

@author: MBlons
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import sys
sys.path.append(r'E:\MB\Projets\Spatial_Transformer\MNIST_DigitRecog_SpatialTransformer')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from pkg.utils import tic, toc, _2Dtform_generation, save_models, load_data, plot_2Dimages, schedule, summary
from stl.transformer import Apply2DTform
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
modelpath = r'E:\MB\Projets\Spatial_Transformer\MNIST_DigitRecog_SpatialTransformer\Trained_Models\MNIST\2D_Digits_Recog.h5'
model = tf.keras.models.load_model(modelpath, custom_objects={"ApplyTform2D": Apply2DTform}, compile=False)
model.summary()
#%%Load preprocessed training and testing data.
num_samples = 10
_, (x_test, y_test) = load_data(1, 1000, shape=(28,28)) # modif load_data to be able to pas 0 for ntrain/Ntest
print('Data Loaded')
# Select some images from the test set to show sample results.
ids = tf.constant(np.random.choice(x_test.shape[0], replace=False, size=num_samples))
x_sample = tf.gather(x_test, ids)
y_sample = tf.gather(y_test, ids)
# test sample Distortion 
tform_sample = _2Dtform_generation(batch=num_samples, dim = (64,64), 
                                   limit_tx =16, limit_ty=16, limit_r=45, limit_scale=0.3, limit_shear=0)
x_sample = Apply2DTform((64,64,1))(x_sample, tform_sample, padding=True, interp='Bilinear') 

# prediction.
plot_2Dimages(model, x_sample, y_sample)

Prediction, T0, tform = model(x_sample, training=False)

print('tform={}'.format(tform[0]))
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(np.squeeze(x_sample[0,:,:,:].numpy()), cmap='hot', vmin=0, vmax=1)
ax[1].imshow(np.squeeze(T0[0,:,:,:].numpy()), cmap='hot', vmin=0, vmax=1)