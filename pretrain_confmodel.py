# PRE-TRAIN THE CONF MODEL ON DS1 
#############################################################################

import numpy as np
from glob import glob
import librosa
from scipy.io import wavfile
import csv
import random
import os,sys
import math
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import mir_eval
import gc
import matplotlib.pyplot as plt
import time

from utils import *
from models import *

#############################################################################

train_audio_files = # path to training data in source domain DS1 (.npy files)
val_audio_files = # path to validation data in source domain DS1 (.npy files)
pitch_files = # path to pitch files in source domain DS1 (.npy files)

#############################################################################

batch_size = 8
Nfft = 1024
win_len = 1024
hop_len = 80
win_size = 500
bpo = 96

#############################################################################

pitch_range = get_cenfreq(51.91,1975.53,bpo)
pitch_range = np.concatenate([np.zeros(1),pitch_range])

#############################################################################

pitch_files_map = build_pitch_files_map(pitch_files)

train_pitch_files = np.array([])
for audio_file in train_audio_files:
    train_pitch_files = np.append(train_pitch_files,pitch_files_map[os.path.basename(os.path.splitext(audio_file)[0])])    
   
val_pitch_files = []
for audio_file in val_audio_files:
    val_pitch_files.append(pitch_files_map[os.path.basename(os.path.splitext(audio_file)[0])]) 

#############################################################################

pretrain_basemodel = melody_extraction()
pretrain_basemodel.load_weights('path to the weights of pretrained model')

conf_model = ConfidenceModel(model)
conf_model.build_graph([win_size,513,1]).summary()
conf_model.layers[0].trainable = False

#############################################################################

optimizer = keras.optimizers.Adam(learning_rate=1.e-5)
loss_fn = tf.keras.losses.MeanSquaredError()

#############################################################################

def l_conf(yt, yp, yc):
    yc_star = tf.zeros((yt.shape[0], yt.shape[1]))
    for i in range(yt.shape[0]):
        true_indx = tf.cast(tf.argmax(yt[i], axis=-1),dtype=tf.int32)
        pred_indx = tf.argmax(yp[i], axis=-1)
        max_conf = tf.reduce_max(yp[i],axis=-1)  

        c_star = tf.gather_nd(yp[i], tf.stack([tf.range(len(true_indx)), true_indx], axis=-1))
        c_star = c_star/max_conf  #--> TCP/MCP
        c_star = tf.expand_dims(c_star, axis=0)
        yc_star = tf.tensor_scatter_nd_update(yc_star, tf.constant([[i]]), c_star)
    
    yc_star = tf.expand_dims(yc_star, axis=-1) 
    loss = loss_fn(yc_star, yc)   
    return loss

def train_step(x,y):
    with tf.GradientTape() as tape:
        y_pre = pretrain_basemodel.call(x)  
        y_conf = conf_model.call(x)  
        loss = l_conf(y,y_pre,y_conf)
    grads = tape.gradient(loss,conf_model.trainable_variables)
    optimizer.apply_gradients(zip(grads,conf_model.trainable_variables))
    return loss

def test_step(x,y):
    y_pre = pretrain_basemodel.call(x)
    y_conf = conf_model.call(x)
    loss = l_conf(y,y_pre,y_conf)
    return loss 

#############################################################################

epochs = 200

val_dataset = tf.data.Dataset.from_tensor_slices((val_audio_files,val_pitch_files)).shuffle(len(val_audio_files)).cache()
val_dataset = val_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32),
                                                tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(1).prefetch(tf.data.AUTOTUNE) 

mean = np.load('tot_mean.npy')
std = np.load('tot_std.npy')  

train_loss = []
val_loss = []

for epoch in range(epochs):
    print(f'Epoch...{epoch+1}')  
    tot_time = np.array([])   
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_audio_files,train_pitch_files)).shuffle(len(train_audio_files))
    train_dataset = train_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32),
                                                    tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size) 
        
    for step,batch in enumerate(train_dataset):
        x,y = batch
        x = (x-mean)/std        
        x = x[:,:,:,tf.newaxis]
        loss_value = train_step(x,y)
    
    for step,batch in enumerate(val_dataset):
        x,y = batch
        x = (x-mean)/std 
        x = x[:,:,:,tf.newaxis]
        val_loss_value = test_step(x,y)
     
    train_loss.append(loss_value)
    val_loss.append(val_loss_value)
    
    if (epoch+1)%10==0:
        conf_model.save_weights(filepath.format(epoch=(epoch+1)),save_format='tf')
        print('Model Saved')      


    