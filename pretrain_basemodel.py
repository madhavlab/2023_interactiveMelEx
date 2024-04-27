# PRE-TRAIN THE BASE MODEL ON DS1 
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

train_audio_files = # path to training data DS1 (.npy files)
val_audio_files = # path to validation data DS1 (.npy files)
pitch_files = # path to pitch files in DS1 (.npy files)

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

model = melody_extraction()
model.build_graph([win_size,513,1]).summary()

#############################################################################

weights = get_weights(pitch_files,len(pitch_range))

#############################################################################

optimizer = keras.optimizers.Adam(learning_rate=1.e-5)
loss_fn = keras.losses.CategoricalCrossentropy()
test_loss_fn = keras.losses.CategoricalCrossentropy()
# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
test_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()

#############################################################################

def custom_loss(yt, yp):
    w = tf.gather(weights, tf.argmax(yt, axis=-1))
    loss = loss_fn(yt, yp, sample_weight=w)
    return loss

def train_step(x,y):
    with tf.GradientTape() as tape:
        logits = model.call(x)
        loss = custom_loss(y,logits)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    train_acc_metric.update_state(y,logits)
    return loss 

def test_step(x,y):
    with tf.GradientTape() as tape:
        logits = model.call(x)
        loss = test_loss_fn(y,logits)
        test_acc_metric.update_state(y,logits)
    return loss    

#############################################################################

epochs = 450

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
    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    
    for step,batch in enumerate(val_dataset):
        x,y = batch
        x = (x-mean)/std 
        x = x[:,:,:,tf.newaxis]
        val_loss_value = test_step(x,y)
     
    train_loss.append(loss_value)
    val_loss.append(val_loss_value)
    
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    
    if (epoch+1)%10==0:
        model.save_weights(filepath.format(epoch=(epoch+1)),save_format='tf')
        print('Model Saved')      


    