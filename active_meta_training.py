# ACTIVE-META-TRAINING ON THE SOURCE DATASET DS2 
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

train_audio_files = # path to test data in source domain DS2 (.npy files)
pitch_files = # path to pitch files in source domain DS2 (.npy files)

#############################################################################

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

#############################################################################

meta_model_pre = melody_extraction()
meta_model_pre.build_graph([win_size,513,1]).summary()
meta_model_pre.load_weights('path to pre-trained base model')  #load the pre-train weights
for i in range(len(meta_model_pre.layers)-1):
    meta_model_pre.layers[i].trainable = False


meta_model_conf = ConfidenceModel()
meta_model_conf.build_graph([win_size,513,1]).summary()
meta_model_conf.layers[0].trainable=False
meta_model_conf.load_weights('path to pre-trained conf model')

#############################################################################

#meta-training
meta_acc_metric = tf.keras.metrics.CategoricalAccuracy()
alpha = 1.e-5
beta = 1.e-5
loss_fn = tf.keras.losses.CategoricalCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
inner_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)
conf_inner_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)
outer_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
conf_outer_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

inner_step = 10 # number of inner-loop updates
EPOCHS = 400


def get_support_query(x):
    x = tf.convert_to_tensor(x)
    y = meta_model_conf.call(x)
    y = tf.reshape(y,-1)

    # Get indices of the least confident values
    neg_tensor = tf.negative(y)
    _, indices = tf.math.top_k(neg_tensor, k=10)
    
    # Get indices of the remaining values
    all_indices = tf.range(tf.size(y))
    remaining_indices = tf.where(tf.logical_not(tf.reduce_any(tf.equal(all_indices[:, None], indices), axis=1)))    
    return tf.cast(indices,tf.int64), tf.cast(remaining_indices,tf.int64)
    

def custom_loss(yt,yp,ts):
    l = 0.2 #lambda
    w_mask = tf.zeros(win_size, dtype=tf.float32)
    w_mask = tf.tensor_scatter_nd_update(w_mask, tf.expand_dims(ts, 1), tf.ones_like(ts, dtype=tf.float32))
    w_mask = tf.expand_dims(w_mask,0)

    gt_classes = tf.reduce_sum(tf.cast(yt, tf.float32), axis=(0, 1))
    gt_cratio = 1.0 / (gt_classes / tf.reduce_max(gt_classes))
    gt_weights = tf.where(tf.math.is_finite(gt_cratio), gt_cratio, tf.constant(0.0))

    yp_indices = tf.argmax(yp,axis=-1)
    yp_onehot = tf.one_hot(yp_indices, depth=len(pitch_range))

    p_classes = tf.reduce_sum(tf.cast(yp_onehot, tf.float32), axis=(0, 1))
    p_cratio = 1.0 / (p_classes / tf.reduce_max(p_classes))
    p_weights = tf.where(tf.math.is_finite(p_cratio), p_cratio, tf.constant(0.0))

    delta_weights = tf.abs(tf.divide(tf.subtract(gt_weights,p_weights),gt_weights))
    up_weights = tf.multiply(gt_weights,tf.exp(l*delta_weights))   
  
    w = tf.gather(up_weights, tf.argmax(yt, axis=-1))
    w = tf.multiply(w,w_mask)
    loss = loss_fn(yt, yp, sample_weight=w)
    return loss


def conf_loss(yt,yp,yc,ts):
    yc_star = tf.zeros((yt.shape[0],win_size),dtype=tf.float32)
    
    w_mask = tf.zeros(win_size, dtype=tf.float32)
    w_mask = tf.tensor_scatter_nd_update(w_mask,tf.expand_dims(ts,1),tf.ones_like(ts,dtype=tf.float32))
    w_mask = tf.expand_dims(w_mask,0)

    true_indx = tf.argmax(yt,axis=-1)
    true_indx = tf.reduce_max(true_indx,axis=0)
    row_no = tf.range(0,win_size,1)
    row_no = tf.cast(row_no,tf.int32)
    true_indx = tf.cast(true_indx,tf.int32)
    indexing = tf.concat([row_no[:,tf.newaxis],true_indx[:,tf.newaxis]], axis=1) 
    
    c_star = tf.gather_nd(yp[0],indexing)
    c_star = c_star[tf.newaxis,:,tf.newaxis]
    
    loss = mse(c_star,yc,sample_weight=tf.constant(w_mask))
    return loss   

def support_pretrain_step(x,y,ts):
    start = time.time()
    # reset_weights()
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = x[tf.newaxis,:,:,:]
    y = y[tf.newaxis,:,:]
    with tf.device('/gpu:0'):
        for _ in range(inner_step):
            with tf.GradientTape(persistent=True) as tape:       
                ys_hat,_ = meta_model_pre.call(x)
                loss = custom_loss(y,ys_hat,ts)
                # print(f'support pre loss...{loss}')
            grads = tape.gradient(loss,meta_model_pre.trainable_variables) 
            inner_optimizer.apply_gradients(zip(grads,meta_model_pre.trainable_variables))            
            end = time.time()    
    # print(f'training loop time..{end-start}') 
    return loss

def support_conftrain_step(x,y,ts):
    start = time.time()
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = x[tf.newaxis,:,:,:]
    y = y[tf.newaxis,:,:]
    with tf.device('/gpu:0'):
        for _ in range(inner_step):
            with tf.GradientTape(persistent=True) as tape:       
                ys_hat,_ = meta_model_pre.call(x)
                yc_hat = meta_model_conf.call(x)
                loss = conf_loss(y,ys_hat,yc_hat,ts)
                # print('support conf loss',loss)
            grads = tape.gradient(loss,meta_model_conf.trainable_variables) 
            conf_inner_optimizer.apply_gradients(zip(grads,meta_model_conf.trainable_variables))            
    end = time.time()    
    # print(f'training loop time..{end-start}') 
    return loss

def calc_rpa(y,yp,tq):
    tq = tf.reshape(tq, shape=(tq.shape[0]))
    for i in range(tf.shape(y)[0]):
        true_indx = tf.argmax(y[i],axis=-1)
        pred_indx = tf.argmax(yp[i],axis=-1)       
        
        gfv = pitch_range[true_indx]
        efv = pitch_range[pred_indx]
        
        
        gfv = tf.gather(gfv,tq)
        efv = tf.gather(efv,tq)        
        
        t = np.array([i*0.01 for i in range(len(gfv))])
        
        (ref_v, ref_c,est_v, est_c) = mir_eval.melody.to_cent_voicing(t,gfv.numpy(),t,efv.numpy())

        RPA = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,est_v, est_c)
        RCA = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,est_v, est_c)
        OA = mir_eval.melody.overall_accuracy(ref_v, ref_c,est_v, est_c)
    return RPA,RCA,OA

def query_pretrain_step(x,y,tq):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = x[tf.newaxis,:,:,:]
    y = y[tf.newaxis,:,:]
    with tf.device('/gpu:0'):
        with tf.GradientTape(persistent=True) as tape:  
            yq_hat,_ = meta_model_pre.call(x)
            yc_hat = meta_model_conf.call(x)
            loss1 = custom_loss(y,yq_hat,tq)
            loss2 = conf_loss(y,yq_hat,yc_hat,tq)

        meta_model_pre.set_weights(meta_pre_weights)
        meta_model_conf.set_weights(meta_conf_weights)
        grads = tape.gradient(loss1,meta_model_pre.trainable_variables) 
        outer_optimizer.apply_gradients(zip(grads,meta_model_pre.trainable_variables)) 
                
        grads = tape.gradient(loss2,meta_model_conf.trainable_variables)   
        conf_outer_optimizer.apply_gradients(zip(grads,meta_model_conf.trainable_variables))            
          
    # calculate rpa   
    yq_hat,_ = meta_model_pre.call(x)
    rpa_query,rca_query,oa_query = calc_rpa(y,yq_hat,tq)
    return loss1.numpy(),loss2.numpy(),rpa_query,rca_query,oa_query

    
mean = np.load('tot_mean.npy')
std = np.load('tot_std.npy')   

tot_pre_loss = []
tot_conf_loss = [] 
rpa_epoch = []
rca_epoch = []
oa_epoch = [] 

for e in range(EPOCHS):
    # print(f'Epoch..{e+1}')
    task_acc = np.array([])
    task_oa = np.array([])
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_audio_files,train_pitch_files)).shuffle(len(train_audio_files))
    train_dataset = train_dataset.map(lambda wav,pitch: (tf.py_function(preprocess_wav,[wav],tf.float32),
                                                    tf.py_function(preprocess_pitch,[pitch],tf.float32)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    
    olo_pre_loss = np.array([])
    olo_conf_loss = np.array([])
    rpa_batch =[]
    rca_batch =[]
    oa_batch =[]

    
    for step,batch in enumerate(train_dataset):
        # print(f'For episode..{step+1}')
        X,Y = batch
        X = (X-mean)/std
        X = X[:,:,:,tf.newaxis]
        
        print(f'Epoch..{e+1} episode..{step+1} Loop..{l+1}')
        meta_pre_weights = meta_model_pre.get_weights()
        meta_conf_weights = meta_model_conf.get_weights()
        
        # ILO            
        meta_model_conf.set_weights(meta_conf_weights)
        meta_model_pre.set_weights(meta_pre_weights)
        
        support_indices, query_indices = tf.map_fn(get_support_query,[X],(tf.int64, tf.int64))           
        pre_inner_loss = tf.map_fn(lambda x: tf.py_function(support_pretrain_step,x,tf.float32),(X,Y,support_indices),tf.float32)
        conf_inner_loss = tf.map_fn(lambda x: tf.py_function(support_conftrain_step,x,tf.float32),(X,Y,support_indices),tf.float32)

        # OLO 
        output = tf.map_fn(lambda x: tf.py_function(query_pretrain_step,x,[tf.float32,tf.float32,tf.float32,tf.float32]),(X,Y,query_indices),fn_output_signature=[tf.float32,tf.float32,tf.float32,tf.float32])
        pre_outer_loss, conf_outer_loss,rpa,rca,oa = output
        print(f'After adaptation..{rpa} {rca} {oa}')
        olo_pre_loss=np.append(olo_pre_loss,pre_outer_loss.numpy())
        olo_conf_loss=np.append(olo_conf_loss,conf_outer_loss.numpy())
        rpa_batch.append(rpa)
        rca_batch.append(rca)
        oa_batch.append(oa)
        
    tot_pre_loss = np.append(tot_pre_loss,np.mean(olo_pre_loss)) 
    tot_conf_loss = np.append(tot_conf_loss,np.mean(olo_conf_loss))
    rpa_epoch.append(np.mean(np.array(rpa_batch)))
    rca_epoch.append(np.mean(np.array(rca_batch)))
    oa_epoch.append(np.mean(np.array(oa_batch)))

    if (e+1)%10==0:
        meta_model_pre.save_weights(filepath_pre.format(epoch=(e+1)),save_format='tf')
        meta_model_conf.save_weights(filepath_conf.format(epoch=(e+1)),save_format='tf')
        print('Models Saved')  
 