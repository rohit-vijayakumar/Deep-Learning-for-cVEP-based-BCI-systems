import os
import glob
import h5py
import copy
import numpy as np
import mne
import random
import warnings
import scipy.io
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt
from sklearn.preprocessing import StandardScaler

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def standardize_data(X):
    scaler = StandardScaler()
    standardized_data = np.zeros((X.shape))
    for ch in range(0,X.shape[2]):
        data = scaler.fit_transform(X[:,:,ch])
        standardized_data[:,:,ch] = data

    return standardized_data

def bandpass_filter_data(X, lfreq, hfreq, sfreq):
    trial_time = 2.1

    if X.shape[3] == 256:
        preprocessed_data = np.array([]).reshape(0,X.shape[1],X.shape[2],X.shape[3])

        for subject in range(0,5):
            filtered_data = X[subject].copy()
            for chan in range(0,filtered_data.shape[2]):
                chan_data = filtered_data[:,:,chan]
                hp_signal = butter_highpass_filter(chan_data, cutoff=lfreq, fs=sfreq, order=2)
                lp_signal = butter_lowpass_filter(hp_signal, cutoff=hfreq, fs=sfreq, order=6) 
                b, a = signal.iirnotch(50.0, 30.0, sfreq)
                notched_signal = signal.filtfilt(b, a, lp_signal)
                filtered_data[:,:,chan] = notched_signal

            preprocessed_data = np.vstack((preprocessed_data, filtered_data[np.newaxis,...]))


        return preprocessed_data
                    

    elif X.shape[3] == 8:

        preprocessed_data = np.array([]).reshape(0,X.shape[1],X.shape[2],X.shape[3])

        for subject in range(0,30):
            filtered_data = X[subject].copy()
            for chan in range(0,filtered_data.shape[2]):
                chan_data = filtered_data[:,:,chan]
                hp_signal = butter_highpass_filter(chan_data, cutoff=lfreq, fs=sfreq, order=2)
                lp_signal = butter_lowpass_filter(hp_signal, cutoff=hfreq, fs=sfreq, order=6) 
                b, a = signal.iirnotch(50.0, 30.0, sfreq)
                notched_signal = signal.filtfilt(b, a, lp_signal)
                filtered_data[:,:,chan] = notched_signal


            preprocessed_data = np.vstack((preprocessed_data, filtered_data[np.newaxis,...]))

        return preprocessed_data
                          
    else:
        warnings.warn("Unsupported Dataset")
        
def remove_bad_channels(X):
    rejected_chans = []
    for subj in range(0,X.shape[0]):
        avg_trials = np.average(X[subj],axis=0)
        std = np.std(avg_trials,axis=0)
        for i in range(0,X.shape[3]):
            if(std[i]>3):
                X[:,:,:,i] = 0
                rejected_chans.append(i)

    accepted_chans = np.array([])
    for i in range(0,X.shape[3]):
        if i not in rejected_chans:
            accepted_chans = np.append(accepted_chans,i)

    accepted_chans = accepted_chans.astype('int')
    
    return X, accepted_chans

def augment_data_chan(X, Ys, Yt):
    mu, sigma = 0, 1
    noise = np.random.normal(mu, sigma, [X.shape[0],X.shape[1], X.shape[2], X.shape[3]]) 

    X = np.concatenate((X,X+noise),axis=-1)

    Ys = np.concatenate((Ys,Ys),axis=-1)
    Yt = Yt[...,np.newaxis]
    Yt = np.concatenate((Yt,Yt),axis=-1)

    return X, Ys, Yt

def augment_data_trial(X, Ys, Yt):
    mu, sigma = 0, 1
    noise = np.random.normal(mu, sigma, [X.shape[0],X.shape[1], X.shape[2], X.shape[3]]) 

    X = np.concatenate((X,X+noise),axis=1)

    Ys = np.concatenate((Ys,Ys),axis=1)
    Yt = Yt[...,np.newaxis]
    Yt = np.concatenate((Yt,Yt),axis=1)

    return X, Ys, Yt