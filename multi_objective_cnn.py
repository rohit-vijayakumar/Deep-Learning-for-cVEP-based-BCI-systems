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


def build_multi_objective_cnn_model(n_channels):
    inputs = Input(shape=(504,n_channels), dtype="float32")
    x = Reshape((504,n_channels,1))(inputs)
    x = Masking(mask_value=0.)(x)
    x = Conv2D(filters=8, kernel_size=(1,n_channels), strides=(1,1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    
    x = Conv2D(filters=4, kernel_size=(48,1), strides=(2,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)               
    x = Dropout(0.25)(x)
    
    x = Conv2D(filters=2, kernel_size=(12,1), strides=(2,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)              
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x6 = Dense(126, activation=activation_fn)(x)
    x1 = Dropout(0.25)(x6)
    
    output1 = Dense(126, activation="sigmoid", name = 'sequence')(x1)
    
    output2 = Dense(21, activation="softmax", name = 'category')(x1)

    model = Model(inputs, outputs=[output1,output2])

    losses = {"sequence": 'binary_crossentropy',
              "category": "categorical_crossentropy"}
    metric = {"sequence": "accuracy",
              "category": "accuracy"}

    model.compile(loss=losses,
                optimizer='adam',
                metrics=metric)

    return model

def train_multi_objective_cnn(dataset,mode,model, X_train, ys_train,yt_train,X_val, ys_val,yt_val,n_subjects,n_classes, current_subj, current_fold):
    warnings.filterwarnings("ignore")
    results = {}
    if(dataset=='8_channel_cVEP'):
        n_subjects = 30
        n_classes = 20
        n_channels = 8
        mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
        codes = mat['codes'].astype('float32')
        codebook = np.moveaxis(codes,1,0).astype('float32')

    elif(dataset=='256_channel_cVEP'):
        n_subjects = 5
        n_classes = 36
        n_channels = 256
        codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
        codes = np.moveaxis(codebook,1,0)
    else:
        warnings.warn("Unsupported dataset")

    model_multi_objective_cnn = build_multi_objective_cnn_model(n_channels)
    callback = EarlyStopping(monitor='val_category_loss', patience=10)
        
    if(current_fold!=None):
        current_f = '_f'+ str(current_fold+1)
        checkpoint_filepath = './saved_models/{}/{}/{}/S{}{}/'.format(dataset,mode,model,current_subj+1,current_f)
    else:
        checkpoint_filepath = './saved_models/{}/{}/{}/S{}/'.format(dataset,mode,model,current_subj+1)
        
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_category_accuracy',
    mode='max',
    save_best_only=True)

    history = model_multi_objective_cnn.fit(x = X_train, y = {"sequence": ys_train, "category": yt_train}, batch_size = 256, 
                  epochs = 100, verbose=2, validation_data=(X_val, {"sequence": ys_val, "category": yt_val}), 
                            callbacks=[callback, model_checkpoint_callback])

    return model_multi_objective_cnn, history.history

def evaluate_multi_objective_cnn(model_multi_objective_cnn, dataset,mode,model,X_test,ys_test,yt_test,n_subjects,n_classes,codebook):
    results = {}
    loss, _,_, seq_accuracy, category_accuracy = model_multi_objective_cnn.evaluate(x = X_test, y = {"sequence": ys_test, 
                                                                                    "category": yt_test}, verbose=0)
    results['sequence_accuracy'] = np.array(seq_accuracy)
    results['category_accuracy'] = np.array(category_accuracy)


    if(mode!='cross_subject'):

        pred_s, pred_c = model_multi_objective_cnn.predict(X_test)
        y_acc_all = []
        cm_all = np.zeros((len(pred_s),2,2))
        for j in range(len(pred_s)):
            pred_seq = pred_s[j]
            y_true = ys_test[j]
            y_pred = pred_seq>=0.5
            y_pred = y_pred.astype('int')
            y_acc = accuracy_score(y_true, y_pred)
            
            cm_s = confusion_matrix(y_true, y_pred,labels=[0,1])
            cm_all[j] = cm_s
            y_acc_all.append(y_acc)
        
        pred_c = np.argmax(pred_c, axis=1)
        cm_c = confusion_matrix(yt_test, pred_c)

        results['sequence_cm'] = np.array(cm_all)
        results['category_cm'] = np.array(cm_c)

        acc_time_step =[]
        acc_time_step_r =[]
        itr_time_step = []
        pred_time_step = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
        pred_time_step_r = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
        for k in range(0, X_test.shape[1]):
            X_test_new = X_test.copy()
            X_test_new[:,k:,:] = 0

            _, pred = model_multi_objective_cnn.predict(X_test_new)
            prediction = np.argmax(pred,axis=1)

            acc = 100*(np.sum(prediction == np.argmax(yt_test,axis=1))/len(prediction))
            acc_time_step.append(acc)
            
            accuracy = acc/100
            num_trials = X_test_new.shape[0]
            time_min = (X_test_new.shape[0]* k*(2.1/504)*(1/60))
            itr = calculate_ITR(n_classes, accuracy, time_min, num_trials)
            itr_time_step.append(itr)
            
            results['pred_time_step'] = {}
            
            pred_time_step[:,k] = pred
            
        for k in range(0, X_test.shape[1]):
            X_test_new = X_test.copy()
            r = 504-k-1
            X_test_new[:,:r,:] = 0
            
            _, pred = model_multi_objective_cnn.predict(X_test_new)
            prediction = np.argmax(pred,axis=1)
            
            acc = 100*(np.sum(prediction == np.argmax(yt_test,axis=1))/len(prediction))
            acc_time_step_r.append(acc)
            
            results['pred_time_step_r'] = {}
            
            pred_time_step_r[:,k] = pred
                  

        results['variable_time_steps'] = np.array(acc_time_step)
        results['variable_time_steps_r'] = np.array(acc_time_step_r)
        results['ITR_time_steps'] = np.array(itr_time_step)

        results['pred_time_step'] = pred_time_step
        results['pred_time_step_r'] = pred_time_step_r

    return results

def run_multi_objective_cnn(dataset,mode,model):            

    if(mode=='cross_subject'):
        results = {}
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
            n_subjects = 30
            n_classes = 20
            mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
            codes = mat['codes'].astype('float32')
            codebook = np.moveaxis(codes,1,0).astype('float32')

        if(dataset=='256_channel_cVEP'):
            n_subjects = 5
            n_classes = 36
            codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
            codes = np.moveaxis(codebook,1,0)

            X, accepted_chans = remove_bad_channels(X)
            X, Ys, Yt = augment_data_trial(X, Ys, Yt)


        for i in range(0,n_subjects):
            results[i+1] = {}

            X_new = X[i]
            ys_new = Ys[i]
            yt_new = Yt[i]
            yt_new = np.argmax(yt_new,axis=1)
            y_new= np.concatenate((yt_new[..., np.newaxis],ys_new), axis=1)

            X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.2,stratify=y_new[:,0], shuffle= True)

            X_train = standardize_data(X_train)
            X_val = standardize_data(X_val)

            ys_train = y_train[:,1:]
            ys_val = y_val[:,1:]

            yt_train = y_train[:,0]
            yt_val = y_val[:,0]

            yt_train = to_categorical(yt_train)
            yt_val = to_categorical(yt_val)
            yt_test = to_categorical(Yt_test)


            model_multi_objective_cnn, model_history = train_multi_objective_cnn(dataset,mode,model, X_train, ys_train, yt_train, X_val, ys_val, yt_val, n_subjects, n_classes, i, None)
            results[i+1]['history'] = model_history

            for j in range(0,n_subjects):
                results[i+1][j+1] = {}
                X_test = X[j]
                X_test = standardize_data(X_test)
                ys_test = Ys[j]
                yt_test= Yt[j]

                results_eval = evaluate_multi_objective_cnn(model_multi_objective_cnn, dataset,mode,model,X_test,ys_test, yt_test, n_subjects,n_classes,codebook)

                print("Train on subject {} test on subject {} category_accuracy: {}".format(i+1,j+1,results_eval['category_accuracy']))
                results[i+1][j+1]['category_accuracy'] = results_eval['category_accuracy']
                results[i+1][j+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                if(mode!='cross_subject'):
                    results[i+1][j+1]['variable_time_steps'] = results_eval['variable_time_steps']
                    results[i+1][j+1]['variable_time_steps_r'] = results_eval['variable_time_steps_r']
                    results[i+1][j+1]['ITR_time_steps'] = results_eval['ITR_time_steps']

                    results[i+1][j+1]['pred_time_step'] = results_eval['pred_time_step']
                    results[i+1][j+1]['pred_time_step_r'] = results_eval['pred_time_step_r']

        with open('./results/{}/{}/{}/eeg2code.pickle'.format(dataset,mode,model), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif(mode=='within_subject' or mode=='loso_subject'):
        results = {}
        if(dataset=='8_channel_cVEP'):
            n_subjects = 30
            n_classes = 20
            mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
            codes = mat['codes'].astype('float32')
            codebook = np.moveaxis(codes,1,0).astype('float32')

        if(dataset=='256_channel_cVEP'):
            n_subjects = 5
            n_classes = 36
            codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
            codes = np.moveaxis(codebook,1,0)

        for i in range(0,n_subjects):
            results[i+1] = {}
            for fold in range(0,15):
                results[i+1][fold+1] = {}
                
                data = load_data(mode,dataset,model,i,fold)
                X_train = data['X_train']
                X_val = data['X_val']
                X_test = data['X_test']

                ys_train = data['ys_train']
                ys_val = data['ys_val']
                ys_test = data['ys_test']

                yt_train = data['yt_train']
                yt_val = data['yt_val']
                yt_test = data['yt_test']

                model_multi_objective_cnn, model_history = train_multi_objective_cnn(dataset,mode,model, X_train, ys_train, yt_train, X_val, ys_val, yt_val, n_subjects, n_classes, i, fold)
                results[i+1][fold+1]['history'] = model_history

                results_eval = evaluate_multi_objective_cnn(model_multi_objective_cnn, dataset,mode,model,X_test,ys_test, yt_test, n_subjects,n_classes,codebook)

                print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']))
                results[i+1][fold+1]['category_accuracy'] = results_eval['category_accuracy']
                results[i+1][fold+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                if(mode!='cross_subject'):
                    results[i+1][fold+1]['variable_time_steps'] = results_eval['variable_time_steps']
                    results[i+1][fold+1]['variable_time_steps_r'] = results_eval['variable_time_steps_r']
                    results[i+1][fold+1]['ITR_time_steps'] = results_eval['ITR_time_steps']

                    results[i+1][fold+1]['pred_time_step'] = results_eval['pred_time_step']
                    results[i+1][fold+1]['pred_time_step_r'] = results_eval['pred_time_step_r']

        with open('./results/{}/{}/{}/eeg2code.pickle'.format(dataset,mode,model), 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        warnings.warn("Unsupported mode")

datasets = ['8_channel_cVEP','256_channel_cVEP']
modes = ['within_subject','loso_subject','cross_subject']
model = "multi_objective_cnn"

for dataset in datasets:
    for mode in modes: 
        print('\n------Running {} for dataset {} in mode {}-----\n'.format(model, dataset, mode))
        run_eeg2code(dataset, mode, model)