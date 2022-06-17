import numpy as np
from scipy.interpolate import griddata
import scipy.io
from scipy import signal
from data_preprocessing import *
from data_loader import load_data
import warnings
from metrics import *
import os
import glob
import h5py
import numpy as np
import mne
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2,l1
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from multi_objective_cnn import *

def run_transfer_learning(dataset,mode,model):
    with open('./datasets/{}.pickle'.format(dataset), 'rb') as handle:
        data = pickle.load(handle)

        X = data['X']
        Ys = data['Ys']
        Yt = data['Yt']

    # Preprocessing data
    low_cutoff = 2
    high_cutoff = 30
    sfreq = 240
    X = bandpass_filter_data(X, low_cutoff, high_cutoff, sfreq)

    if(dataset=='8_channel_cVEP'):
        dataset_txt = '8-channel dataset'
        n_subjects = 30
        n_classes = 21
        n_channels = 8
        bbox_l = 1.01
        mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
        codes = mat['codes'].astype('float32')
        codebook = np.moveaxis(codes,1,0).astype('float32')

    if(dataset=='256_channel_cVEP'):
        dataset_txt = '256-channel dataset'
        n_subjects = 5
        n_classes = 36
        n_channels = 256
        bbox_l = 1.07
        codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
        codes = np.moveaxis(codebook,1,0)

        X, rejected_chans = remove_bad_channels(X)
        X, Ys, Yt = augment_data_chan(X, Ys, Yt)

    results = {}
    for i in range(0,n_subjects):
        #print("Train on subject {}".format(i+1))
        results[i+1] = []
        X_new = X[i]
        ys_new = Ys[i]
        yt_new = Yt[i]
        
        if dataset =='8_channel_cVEP':
            yt_new = yt_new[..., np.newaxis]
            
        y_new= np.concatenate((yt_new,ys_new), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2,stratify=y_new[:,0], shuffle= True)
        
        if dataset =='8_channel_cVEP':
            ys_train = y_train[:,1:]
            ys_test = y_test[:,1:]

            yt_train = y_train[:,0]
            yt_test = y_test[:,0]
            
        if(dataset=='256_channel_cVEP'):
            
            X_train1 = X_train[:,:,:256]
            X_train2 = X_train[:,:,256:512]

            X_test1 = X_test[:,:,:256]
            X_test2 = X_test[:,:,256:512]

            X_train = np.concatenate((X_train1,X_train2), axis=0)
            X_test = np.concatenate((X_test1,X_test2), axis=0)[:216]

            yt_train1 = y_train[:,0]
            yt_train2 = y_train[:,1]
            ys_train1 = y_train[:,2:128]
            ys_train2 = y_train[:,128:254]

            yt_test1 = y_test[:,0]
            yt_test2 = y_test[:,1]
            ys_test1 = y_test[:,2:128]
            ys_test2 = y_test[:,128:254]

            
            ys_train = np.concatenate((ys_train1,ys_train2), axis=0)
            yt_train = np.concatenate((yt_train1,yt_train2), axis=0)
            ys_test = np.concatenate((ys_test1,ys_test2), axis=0)[:216]
            yt_test = np.concatenate((yt_test1,yt_test2), axis=0)[:216]

        X_train = standardize_data(X_train)
        X_test = standardize_data(X_test)

        yt_train = to_categorical(yt_train)
        yt_test = to_categorical(yt_test)

        multi_objective_cnn_model = build_multi_objective_cnn_model(n_channels,n_classes)
        checkpoint_filepath = './saved_models/{}/{}/{}/S{}/'.format(model,dataset,mode,i+1)
        multi_objective_cnn_model.load_weights(checkpoint_filepath).expect_partial()
        
        loss, _,_, seq_accuracy, category_accuracy = multi_objective_cnn_model.evaluate(x = X_test, y = {"sequence": ys_test, 
                                                                                        "category": yt_test}, verbose=0)
        results[i+1].append(category_accuracy)
        n_trials = 100
        for trial in range(1,n_trials):
            print("Train on subject {} trial {}".format(i+1,trial))
            X_train_sample = X_train[:trial]
            ys_train_sample = ys_train[:trial]
            yt_train_sample = yt_train[:trial]

            multi_objective_cnn_model = build_multi_objective_cnn_model(n_channels,n_classes)

            for j, layer in enumerate(multi_objective_cnn_model.layers):
                if(j==17 or j==18):
                    multi_objective_cnn_model.layers[j].trainable = True
                else:
                    multi_objective_cnn_model.layers[j].trainable = False 

            checkpoint_filepath = './saved_models/{}/{}/{}/S{}/'.format(model,dataset,mode,i+1)
            multi_objective_cnn_model.load_weights(checkpoint_filepath).expect_partial()
            callback = EarlyStopping(monitor='category_loss', patience=10)

            multi_objective_cnn_model.fit(x = X_train_sample, y = {"sequence": ys_train_sample, "category": yt_train_sample}, batch_size = 5, 
                      epochs = 50, verbose=0, callbacks=[callback])

            loss, _,_, seq_accuracy, category_accuracy = multi_objective_cnn_model.evaluate(x = X_test, y = {"sequence": ys_test, 
                                                                                        "category": yt_test}, verbose=0)

            results[i+1].append(category_accuracy)

        plt.rcParams["figure.figsize"] = (10,5)
        acc_samples = results[i+1]
        samples = np.arange(0,len(acc_samples))
        plt.plot(samples,acc_samples)
        plt.xticks(np.arange(0,n_trials+1,10))
        plt.yticks(np.arange(0,1.09,0.1))
        plt.ylim((0,1.09))
        plt.xlabel('Number of trials')
        plt.ylabel('Accuracy')
        plt.title("Transfer learning for subject {} in {}".format(i+1,dataset_txt))
        plt.grid(False)
        plt.grid(True)
        
        filename = "./visualizations/Transfer learning/{}_S{}.png".format(dataset,i+1)
        os.makedirs(os.path.dirname(filename), exist_ok=True)   
        plt.savefig(filename) 
        plt.close()
        
    filename = './results/Transfer learning/{}.pickle'.format(dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    NUM_COLORS = n_subjects+1
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    sns.reset_orig() 
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 
    fig, ax = plt.subplots(figsize=(20,10))
    for l in results.keys():
        acc_samples = results[l]
        samples = np.arange(0,len(acc_samples))
        lines = ax.plot(samples,acc_samples,label=l)
        lines[0].set_color(clrs[l])
        lines[0].set_linestyle(LINE_STYLES[l%NUM_STYLES])

    ax.set_xticks(np.arange(0,n_trials+1,10))
    ax.set_yticks(np.arange(0,1.09,0.1))
    ax.set_ylim((0,1.09))
    ax.set_xlabel('Number of trials')
    ax.set_ylabel('Accuracy')
    ax.set_title("Transfer learning for {}".format(dataset_txt))

    ax.legend(fontsize=13,bbox_to_anchor=(bbox_l, 1.01))
    plt.grid(False)
    plt.grid(True)

    filename = "./visualizations/Transfer learning/{}_all.png".format(dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename) 
    plt.close()
    
datasets = ['8_channel_cVEP','256_channel_cVEP']
model = 'multi_objective_cnn'
mode = 'loso_subject'

for dataset in datasets:
    run_transfer_learning(dataset,mode,model)