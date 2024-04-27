import numpy as np
import os

def get_cenfreq(StartFreq,StopFreq,NumPerOct):
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    return central_freq

def get_weights(pitch_files, pitch_range):

    class_counts = np.arange(pitch_range)

    count_dict = {}
    for num in class_counts:
        if not num in count_dict:
            count_dict[num]=0

    for i in range(len(pitch_files)):
        y_one_hot = np.load(pitch_files[i])
        classes = np.argmax(y_one_hot,axis=-1)
        for num in classes:
            count_dict[num]+=1       
        
    tot_class_count = np.array([])
    for i in count_dict.keys():
        tot_class_count = np.append(tot_class_count,count_dict[i])
        
    ratio = tot_class_count/np.max(tot_class_count)
    weights = [1/c if c!=0 else 0 for c in ratio]
    return weights


def build_pitch_files_map(pitch_files):
    pitch_files_map = {}
    for pitch_file in pitch_files:
        pitch_filename = os.path.basename(os.path.splitext(pitch_file)[0])
        pitch_files_map[pitch_filename] = pitch_file
    return pitch_files_map  

def preprocess_wav(wav_path):
    X = np.load(wav_path.numpy().decode())
    return X

def preprocess_pitch(pitch_path):
    X = np.load(pitch_path.numpy().decode())
    return X