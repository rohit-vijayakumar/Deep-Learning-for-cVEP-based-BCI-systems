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
import math
import numpy as np
import mne
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import texttable as tt  
import latextable
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


def find_outlier(pred,n_classes, z):
    pred_new = pred[0]
    q3, q1 = np.percentile(pred_new, [75 ,25])
    iqr = q3 - q1
    threshold = iqr*z + q3
    indices = np.arange(0,n_classes)
    significant_class = indices[pred_new>threshold]
    max_class = np.argmax(pred_new)
    max_val = np.max(pred_new)
    if(max_class in significant_class):
        significant_class = max_class
    else:
        significant_class = -1
    
    return significant_class, max_val,threshold

def static_stopping(X_val, yt_val,X_test,yt_test,n_classes, multi_objective_cnn_model):
    stop_times = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
    stop_points = [int(value*240) for value in stop_times]
    indices = np.arange(0,n_classes)
     
    conf_thresh = 0.5
    static_pred = {}

    for l in range(0,len(stop_times)):
        k = stop_points[l]
        kt = stop_times[l]
        static_pred[kt] = []
        for trial in range(0,X_val.shape[0]):
            X_val_new = X_val.copy()
            X_val_new = X_val_new[trial][np.newaxis,...]
            #print(X_val_new.shape)
            X_val_new[:,k:,:] = 0 
            _, pred = multi_objective_cnn_model.predict(X_val_new)
            
            prediction = np.argmax(pred,axis=1)[0]
            prediction_conf = np.max(pred,axis=1)[0]
            target = np.argmax(yt_val,axis=1)[trial]
            static_pred[kt].append(prediction==target)
    
    static_acc = []
    for j in stop_times:
        static_acc_mean = np.mean(static_pred[j])
        static_acc.append(static_acc_mean)

    static_acc = np.array(static_acc)

    static_itr = []
    for k in range(0,len(stop_times)):
        static_time_min = np.sum(np.repeat(stop_times[k],X_test.shape[0]))*(1/60)
        num_trials = X_test.shape[0]
        static_itr.append(calculate_ITR(n_classes, static_acc[k], static_time_min, num_trials))

    static_itr = np.array(static_itr)

    sorted_itr_arg = np.argsort(static_itr)

    for b in range(len(sorted_itr_arg)-1,0,-1):
        acc_check = static_acc[sorted_itr_arg[b]]
        if (np.max(static_acc) > acc_check + 0.05):
            continue
        else:
            best_arg = sorted_itr_arg[b]
            break

    best_itr = static_itr[best_arg]
    best_time = stop_times[best_arg]
    best_time_point = int(best_time*240)
    best_acc = static_acc[best_arg]

    static_pred_all = []
    for trial in range(0,X_test.shape[0]):
        X_test_new = X_test.copy()
        X_test_new = X_test_new[trial][np.newaxis,...]
        #print(X_test_new.shape)
        X_test_new[:,best_time_point:,:] = 0 
        _, pred = multi_objective_cnn_model.predict(X_test_new)
        
        prediction = np.argmax(pred,axis=1)[0]
        prediction_conf = np.max(pred,axis=1)[0]
        target = np.argmax(yt_test,axis=1)[trial]
        static_pred_all.append(prediction==target)

    mean_acc = np.mean(static_pred_all)
    mean_time = best_time

    static_time_min = np.sum(np.repeat(mean_time,X_test.shape[0]))*(1/60)
    num_trials = X_test.shape[0]
    mean_itr = calculate_ITR(n_classes, mean_acc, static_time_min, num_trials)
    
    return mean_acc, mean_time, mean_itr

def acc_stopping(X_test,yt_test,n_classes, multi_objective_cnn_model):
    indices = np.arange(0,n_classes)
    acc_val = [0.5,0.6,0.7,0.8,0.9,1.0]
    acc_stopping_time = {}
    for k in range(0,X_test.shape[1]+1,24):
        X_test_new = X_test.copy()
        X_test_new[:,k:,:] = 0 
        _, pred = multi_objective_cnn_model.predict(X_test_new)
        prediction = np.argmax(pred,axis=1)
        target = np.argmax(yt_test,axis=1)
        acc_mean = np.mean(prediction==target)

        if (acc_mean>=0.5 and acc_mean<=0.6):
            if 0.5 not in acc_stopping_time.keys():
                acc_stopping_time['0.5'] = {}
                acc_stopping_time['0.5']['accuracy'] = acc_mean
                acc_stopping_time['0.5']['time'] = k/240
                acc_time_min = (acc_stopping_time['0.5']['time']*X_test_new.shape[0])/60
                num_trials = X_test.shape[0]
                acc_itr = calculate_ITR(n_classes, acc_mean, acc_time_min, num_trials)
                acc_stopping_time['0.5']['itr'] = acc_itr

        elif(acc_mean>=0.6 and acc_mean<0.7):
            if 0.6 not in acc_stopping_time.keys():
                acc_stopping_time['0.6'] = {}
                acc_stopping_time['0.6']['accuracy'] = acc_mean
                acc_stopping_time['0.6']['time'] = k/240
                acc_time_min = (acc_stopping_time['0.6']['time']*X_test_new.shape[0])/60
                num_trials = X_test.shape[0]
                acc_itr = calculate_ITR(n_classes, acc_mean, acc_time_min, num_trials)
                acc_stopping_time['0.6']['itr'] = acc_itr
        elif(acc_mean>=0.7 and acc_mean<0.8):
            if 0.7 not in acc_stopping_time.keys():
                acc_stopping_time['0.7'] = {}
                acc_stopping_time['0.7']['accuracy'] = acc_mean
                acc_stopping_time['0.7']['time'] = k/240
                acc_time_min = (acc_stopping_time['0.7']['time']*X_test_new.shape[0])/60
                num_trials = X_test.shape[0]
                acc_itr = calculate_ITR(n_classes, acc_mean, acc_time_min, num_trials)
                acc_stopping_time['0.7']['itr'] = acc_itr
        elif(acc_mean>=0.8 and acc_mean<0.9):
            if 0.8 not in acc_stopping_time.keys():
                acc_stopping_time['0.8'] = {}
                acc_stopping_time['0.8']['accuracy'] = acc_mean
                acc_stopping_time['0.8']['time'] = k/240
                acc_time_min = (acc_stopping_time['0.8']['time']*X_test_new.shape[0])/60
                num_trials = X_test.shape[0]
                acc_itr = calculate_ITR(n_classes, acc_mean, acc_time_min, num_trials)
                acc_stopping_time['0.8']['itr'] = acc_itr
        elif(acc_mean>=0.9 and acc_mean<1.0):
            if 0.9 not in acc_stopping_time.keys():
                acc_stopping_time['0.9'] = {}
                acc_stopping_time['0.9']['accuracy'] = acc_mean
                acc_stopping_time['0.9']['time'] = k/240
                acc_time_min = (acc_stopping_time['0.9']['time']*X_test_new.shape[0])/60
                num_trials = X_test.shape[0]
                acc_itr = calculate_ITR(n_classes, acc_mean, acc_time_min, num_trials)
                acc_stopping_time['0.9']['itr'] = acc_itr
        elif(acc_mean==1.0):
            if 1.0  not in acc_stopping_time.keys():
                acc_stopping_time['1.0'] = {}
                acc_stopping_time['1.0']['accuracy'] = acc_mean
                acc_stopping_time['1.0']['time'] = k/240
                acc_time_min = (acc_stopping_time['1.0']['time']*X_test_new.shape[0])/60
                num_trials = X_test.shape[0]
                acc_itr = calculate_ITR(n_classes, acc_mean, acc_time_min, num_trials)
                acc_stopping_time['1.0']['itr'] = acc_itr
        else:
            continue

    return acc_stopping_time

def get_stopping_time(X_test,yt_test,n_classes, multi_objective_cnn_model, v):
    indices = np.arange(0,n_classes)
    const_val = [1.5,2,3,4,5,6]
    #acc_all = []
    #stopping_time_all = []
    dynamic_pred_acc = []

    conf_thresh = v
    ceil_time = []
    ceil_pred = []
    
    conf_time = []
    conf_pred = []
    
    ds_time = []
    ds_pred = []
    acc_scatter_all = []
    for trial in range(0,X_test.shape[0]):
        #outlier_all = []
        #max_vals = []
        #pred_class_output = None
        ceil_pred_all = []
        conf_pred_all = []
        ds_pred_all = []
        threshold_all = []
        max_val_all = []
        
        flag_ceil = 0
        flag_conf = 0
        flag_ds = 0
        ceil_k = 0
        conf_k = 0
        ds_k = 0
        
        for k in range(0,X_test.shape[1]+1,24):
            X_test_new = X_test.copy()
            X_test_new = X_test_new[trial][np.newaxis,...]
            X_test_new[:,k:,:] = 0 
            _, pred = multi_objective_cnn_model.predict(X_test_new)
            
            outlier, max_val, threshold = find_outlier(pred,n_classes,v)
            prediction = np.argmax(pred,axis=1)[0]
            prediction_conf = np.max(pred,axis=1)[0]
            target = np.argmax(yt_test,axis=1)[trial]

            acc_scatter_all.append([prediction_conf,k/240,prediction==target])
            ceil_pred_all.append(prediction)
            conf_pred_all.append(prediction)
            ds_pred_all.append(outlier)
            threshold_all.append(threshold)
            max_val_all.append(max_val)
            
            if (prediction==target and len(ceil_pred_all)>=4):
                if(ceil_pred_all[-1]==ceil_pred_all[-2] and ceil_pred_all[-2]==ceil_pred_all[-3]
                   and ceil_pred_all[-3]==ceil_pred_all[-4] and flag_ceil==0):
                    ceil_pred.append(prediction==target) 
                    flag_ceil = 1
                    ceil_k = k
                    ceil_time.append(ceil_k/240)
                    
            if(prediction_conf >= conf_thresh and flag_conf==0 and len(conf_pred_all)>=4):
                if(conf_pred_all[-1]==conf_pred_all[-2] and conf_pred_all[-2]==conf_pred_all[-3] 
                   and conf_pred_all[-3]==conf_pred_all[-4]):
                    flag_conf = 1
                    conf_k = k
                    conf_pred.append(prediction==target)
                    conf_time.append(conf_k/240)
                
            if(len(ds_pred_all)>=4):
                if(max_val>=conf_thresh and ds_pred_all[-1]==ds_pred_all[-2] and ds_pred_all[-2]==ds_pred_all[-3] 
                   and ds_pred_all[-3]==ds_pred_all[-4]and flag_ds==0):
                    flag_ds = 1
                    ds_k = k
                    ds_pred.append(prediction==target)
                    ds_time.append(ds_k/240)
        
        if (flag_conf!=1):
            conf_pred.append(False)

        if(flag_ceil!=1):
            ceil_pred.append(False)
            
        if(flag_ds!=1):
            ds_pred.append(False)
    
    conf_acc_mean = np.mean(conf_pred)                
    conf_time_mean = np.mean(conf_time)
    conf_time_min = np.sum(conf_time)*(1/60)
    num_trials = X_test.shape[0]
    conf_itr = calculate_ITR(n_classes, conf_acc_mean, conf_time_min, num_trials)
    
    ceil_acc_mean = np.mean(ceil_pred)
    ceil_time_mean = np.mean(ceil_time)
    ceil_time_min = np.sum(ceil_time)*(1/60)
    num_trials = X_test.shape[0]
    ceil_itr = calculate_ITR(n_classes, ceil_acc_mean, ceil_time_min, num_trials)
    
    ds_acc_mean = np.mean(ds_pred)
    ds_time_mean = np.mean(ds_time)
    ds_time_min = np.sum(ds_time)*(1/60)
    num_trials = X_test.shape[0]
    ds_itr = calculate_ITR(n_classes, ds_acc_mean, ds_time_min, num_trials)

    dynamic_stop = [ceil_acc_mean, ceil_time_mean, ceil_itr, conf_acc_mean, conf_time_mean, conf_itr, ds_acc_mean, ds_time_mean, ds_itr]
    best_results = dynamic_stop


    return best_results, v, acc_scatter_all

def optimize_dynamic_stopping_time(X_test,yt_test,n_classes, multi_objective_cnn_model):
    indices = np.arange(0,n_classes)
    const_val = [1.5,2,3]
    acc_thresh = [0.5,0.6,0.7,0.8,0.9]
    #acc_all = []
    #stopping_time_all = []
    dynamic_pred_acc = []
    dynamic_stop = {}

    for v in acc_thresh:
        dynamic_stop[v] = []
        conf_thresh = v
        ceil_time = []
        ceil_pred = []
        
        conf_time = []
        conf_pred = []
        
        ds_time = []
        ds_pred = []
        acc_scatter_all= []
        for trial in range(0,X_test.shape[0]):
            #outlier_all = []
            #max_vals = []
            #pred_class_output = None
            ceil_pred_all = []
            conf_pred_all = []
            ds_pred_all = []
            threshold_all = []
            max_val_all = []

            flag_ceil = 0
            flag_conf = 0
            flag_ds = 0
            ceil_k = 0
            conf_k = 0
            ds_k = 0
            
            for k in range(0,X_test.shape[1]+1,24):
                X_test_new = X_test.copy()
                X_test_new = X_test_new[trial][np.newaxis,...]
                X_test_new[:,k:,:] = 0 
                _, pred = multi_objective_cnn_model.predict(X_test_new)
                
                outlier, max_val, threshold = find_outlier(pred,n_classes,v)
                prediction = np.argmax(pred,axis=1)[0]
                prediction_conf = np.max(pred,axis=1)[0]
                target = np.argmax(yt_test,axis=1)[trial]

                ceil_pred_all.append(prediction)
                conf_pred_all.append(prediction)
                ds_pred_all.append(outlier)
                threshold_all.append(threshold)
                max_val_all.append(max_val)
                
                if (prediction==target and len(ceil_pred_all)>=4):
                    if(ceil_pred_all[-1]==ceil_pred_all[-2] and ceil_pred_all[-2]==ceil_pred_all[-3]
                       and ceil_pred_all[-3]==ceil_pred_all[-4] and flag_ceil==0):
                        ceil_pred.append(prediction==target) 
                        flag_ceil = 1
                        ceil_k = k
                        ceil_time.append(ceil_k/240)
                        
                if(prediction_conf >= conf_thresh and flag_conf==0 and len(conf_pred_all)>=4):
                    if(conf_pred_all[-1]==conf_pred_all[-2] and conf_pred_all[-2]==conf_pred_all[-3] 
                       and conf_pred_all[-3]==conf_pred_all[-4]):
                        flag_conf = 1
                        conf_k = k
                        conf_pred.append(prediction==target)
                        conf_time.append(conf_k/240)
                    
                if(len(ds_pred_all)>=4):
                    if(max_val>=conf_thresh and ds_pred_all[-1]==ds_pred_all[-2] and ds_pred_all[-2]==ds_pred_all[-3] 
                       and ds_pred_all[-3]==ds_pred_all[-4]and flag_ds==0):
                        flag_ds = 1
                        ds_k = k
                        ds_pred.append(prediction==target)
                        ds_time.append(ds_k/240)
            
            if (flag_conf!=1):
                conf_pred.append(False)

            if(flag_ceil!=1):
                ceil_pred.append(False)
                
            if(flag_ds!=1):
                ds_pred.append(False)
        
        conf_acc_mean = np.mean(conf_pred)                
        conf_time_mean = np.mean(conf_time)
        conf_time_min = np.sum(conf_time)*(1/60)
        num_trials = X_test.shape[0]
        conf_itr = calculate_ITR(n_classes, conf_acc_mean, conf_time_min, num_trials)
        
        ceil_acc_mean = np.mean(ceil_pred)
        ceil_time_mean = np.mean(ceil_time)
        ceil_time_min = np.sum(ceil_time)*(1/60)
        num_trials = X_test.shape[0]
        ceil_itr = calculate_ITR(n_classes, ceil_acc_mean, ceil_time_min, num_trials)
        
        ds_acc_mean = np.mean(ds_pred)
        ds_time_mean = np.mean(ds_time)
        ds_time_min = np.sum(ds_time)*(1/60)
        num_trials = X_test.shape[0]
        ds_itr = calculate_ITR(n_classes, ds_acc_mean, ds_time_min, num_trials)

        dynamic_stop[v] = [ceil_acc_mean, ceil_time_mean, ceil_itr, conf_acc_mean, conf_time_mean, conf_itr, ds_acc_mean, ds_time_mean, ds_itr]
        dynamic_pred_acc.append(ds_acc_mean)

    acc_scatter_all = np.array(acc_scatter_all)
    dynamic_pred_acc = np.array(dynamic_pred_acc)
    best_thresh = acc_thresh[np.argmax(dynamic_pred_acc)]
    best_results = dynamic_stop[best_thresh]


    return best_results, best_thresh

def plot_acc_levels(acc_stopping_time8, acc_stopping_time256):
    acc_check_all = {}
    time_check_all = {}
    itr_check_all = {}

    acc_check_all['0.5'] = []
    acc_check_all['0.6'] = []
    acc_check_all['0.7'] = []
    acc_check_all['0.8'] = []
    acc_check_all['0.9'] = []
    acc_check_all['1.0'] = []

    time_check_all['0.5'] = []
    time_check_all['0.6'] = []
    time_check_all['0.7'] = []
    time_check_all['0.8'] = []
    time_check_all['0.9'] = []
    time_check_all['1.0'] = []

    itr_check_all['0.5'] = []
    itr_check_all['0.6'] = []
    itr_check_all['0.7'] = []
    itr_check_all['0.8'] = []
    itr_check_all['0.9'] = []
    itr_check_all['1.0'] = []
    for subj in acc_stopping_time8.keys():
        subj_vals = acc_stopping_time8[subj]
        if '0.5' in subj_vals.keys():
            acc_check_all['0.5'].append(subj_vals['0.5']['accuracy'])
            time_check_all['0.5'].append(subj_vals['0.5']['time'])
            itr_check_all['0.5'].append(subj_vals['0.5']['itr'])
        if '0.6' in subj_vals.keys():
            acc_check_all['0.6'].append(subj_vals['0.6']['accuracy'])
            time_check_all['0.6'].append(subj_vals['0.6']['time'])
            itr_check_all['0.6'].append(subj_vals['0.6']['itr'])
        if '0.7' in subj_vals.keys():
            acc_check_all['0.7'].append(subj_vals['0.7']['accuracy'])
            time_check_all['0.7'].append(subj_vals['0.7']['time'])
            itr_check_all['0.7'].append(subj_vals['0.7']['itr'])
        if '0.8' in subj_vals.keys():
            acc_check_all['0.8'].append(subj_vals['0.8']['accuracy'])
            time_check_all['0.8'].append(subj_vals['0.8']['time'])
            itr_check_all['0.8'].append(subj_vals['0.8']['itr'])
        if '0.9' in subj_vals.keys():
            acc_check_all['0.9'].append(subj_vals['0.9']['accuracy'])
            time_check_all['0.9'].append(subj_vals['0.9']['time'])
            itr_check_all['0.9'].append(subj_vals['0.9']['itr'])
        if '1.0' in subj_vals.keys():
            acc_check_all['1.0'].append(subj_vals['1.0']['accuracy'])
            time_check_all['1.0'].append(subj_vals['1.0']['time'])
            itr_check_all['1.0'].append(subj_vals['1.0']['itr'])

    acc_mean_all8 = []
    time_mean_all8 = []
    itr_mean_all8 = []
    dataset_mean_all8 = []
    for val in acc_check_all.keys():
        acc_mean = np.median(acc_check_all[val])
        time_mean = np.median(time_check_all[val])
        itr_mean = np.median(itr_check_all[val])

        acc_mean_all8.append(acc_mean)
        time_mean_all8.append(time_mean)
        itr_mean_all8.append(itr_mean)
        dataset_mean_all8.append('8-channel dataset')

    acc_mean_all8 = np.array(acc_mean_all8)
    time_mean_all8 = np.array(time_mean_all8)
    itr_mean_all8 = np.array(itr_mean_all8)
    dataset_mean_all8 = np.array(dataset_mean_all8)

    acc_check_all = {}
    time_check_all = {}
    itr_check_all = {}

    acc_check_all['0.5'] = []
    acc_check_all['0.6'] = []
    acc_check_all['0.7'] = []
    acc_check_all['0.8'] = []
    acc_check_all['0.9'] = []
    acc_check_all['1.0'] = []

    time_check_all['0.5'] = []
    time_check_all['0.6'] = []
    time_check_all['0.7'] = []
    time_check_all['0.8'] = []
    time_check_all['0.9'] = []
    time_check_all['1.0'] = []

    itr_check_all['0.5'] = []
    itr_check_all['0.6'] = []
    itr_check_all['0.7'] = []
    itr_check_all['0.8'] = []
    itr_check_all['0.9'] = []
    itr_check_all['1.0'] = []
    for subj in acc_stopping_time256.keys():
        subj_vals = acc_stopping_time256[subj]
        if '0.5' in subj_vals.keys():
            acc_check_all['0.5'].append(subj_vals['0.5']['accuracy'])
            time_check_all['0.5'].append(subj_vals['0.5']['time'])
            itr_check_all['0.5'].append(subj_vals['0.5']['itr'])
        if '0.6' in subj_vals.keys():
            acc_check_all['0.6'].append(subj_vals['0.6']['accuracy'])
            time_check_all['0.6'].append(subj_vals['0.6']['time'])
            itr_check_all['0.6'].append(subj_vals['0.6']['itr'])
        if '0.7' in subj_vals.keys():
            acc_check_all['0.7'].append(subj_vals['0.7']['accuracy'])
            time_check_all['0.7'].append(subj_vals['0.7']['time'])
            itr_check_all['0.7'].append(subj_vals['0.7']['itr'])
        if '0.8' in subj_vals.keys():
            acc_check_all['0.8'].append(subj_vals['0.8']['accuracy'])
            time_check_all['0.8'].append(subj_vals['0.8']['time'])
            itr_check_all['0.8'].append(subj_vals['0.8']['itr'])
        if '0.9' in subj_vals.keys():
            acc_check_all['0.9'].append(subj_vals['0.9']['accuracy'])
            time_check_all['0.9'].append(subj_vals['0.9']['time'])
            itr_check_all['0.9'].append(subj_vals['0.9']['itr'])
        if '1.0' in subj_vals.keys():
            acc_check_all['1.0'].append(subj_vals['1.0']['accuracy'])
            time_check_all['1.0'].append(subj_vals['1.0']['time'])
            itr_check_all['1.0'].append(subj_vals['1.0']['itr'])

    acc_mean_all256 = []
    time_mean_all256 = []
    itr_mean_all256 = []
    dataset_mean_all256 = []
    for val in acc_check_all.keys():
        acc_mean = np.median(acc_check_all[val])
        time_mean = np.median(time_check_all[val])
        itr_mean = np.median(itr_check_all[val])

        acc_mean_all256.append(acc_mean)
        time_mean_all256.append(time_mean)
        itr_mean_all256.append(itr_mean)
        dataset_mean_all256.append('256-channel dataset')

    acc_mean_all256 = acc_mean_all256[:4]
    time_mean_all256 = time_mean_all256[:4]
    itr_mean_all256 = itr_mean_all256[:4]
    dataset_mean_all256 = dataset_mean_all256[:4]

    acc_mean_all256 = np.array(acc_mean_all256)
    time_mean_all256 = np.array(time_mean_all256)
    itr_mean_all256 = np.array(itr_mean_all256)
    dataset_mean_all256 = np.array(dataset_mean_all256)

    acc_mean_all = np.concatenate((acc_mean_all8, acc_mean_all256))
    time_mean_all = np.concatenate((time_mean_all8, time_mean_all256))
    itr_mean_all = np.concatenate((itr_mean_all8, itr_mean_all256))
    dataset_mean_all = np.concatenate((dataset_mean_all8, dataset_mean_all256))
    time_mean_all[-1] = 2.1
    itr_mean_all[-1] = itr_mean_all[-2]

    fig=plt.figure(figsize=(25, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)

    data1={'modes':acc_mean_all,
      'pred_value':time_mean_all,
      'dataset':dataset_mean_all,
     }

    df1=pd.DataFrame(data1)
    df1 = df1[['modes','pred_value','dataset']]

    sns.set_style("whitegrid")
    sns.lineplot(x='modes',y='pred_value',data=df1,hue='dataset',ax=ax1)
    ax1.set_title('Time elapsed for accuracy levels',fontsize=14)
    ax1.set(ylabel='Time(s)')
    ax1.set(xlabel='Accuracy levels')
    ax1.set(yticks=np.arange(0,2.11,0.3))
    ax1.set_ylim((0,2.21))
    #ax1.legend(bbox_to_anchor=(1.35, 1.03))
    ax1.get_legend().remove()
    data2={'modes':acc_mean_all,
      'pred_value':itr_mean_all,
      'dataset':dataset_mean_all,
     }

    df2=pd.DataFrame(data2)
    df2 = df2[['modes','pred_value','dataset']]

    sns.set_style("whitegrid")
    sns.lineplot(x='modes',y='pred_value',data=df2,hue='dataset',ax=ax2)
    ax2.set_title('ITR for accuracy levels',fontsize=14)
    ax2.set(ylabel='bits/min')
    ax2.set(xlabel='Accuracy levels')
    ax2.set(yticks=np.arange(0,200,20))
    ax2.set_ylim((0,200))
    ax2.legend(bbox_to_anchor=(1.04,1.02))

    plt.grid(False)
    plt.grid(True) 

    filename = "./visualizations/Dynamic stopping/acc_levels.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
    plt.close()
    
def run_dynamic_stopping(dataset,mode,model):
    if(model=='multi_objective_cnn'):
        model_txt = 'Multi-objective CNN'
        
    with open('./datasets/{}.pickle'.format(dataset), 'rb') as handle:
        data = pickle.load(handle)
    
    X = data['X']
    Ys = data['Ys']
    Yt = data['Yt'] 

    if(dataset=='8_channel_cVEP'):
        n_subjects = 30
        n_classes = 21
        n_channels = 8
        mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
        codes = mat['codes'].astype('float32')
        codebook = np.moveaxis(codes,1,0).astype('float32')

        X_new_c = np.reshape(X[:,:100],(30,100*15,504,8))
        Ys_new_c = Ys[:,:100]
        Yt_new_c = Yt[:,:100]

        Ys_new_c = np.repeat(Ys_new_c,15,axis=1)
        Yt_new_c = np.repeat(Yt_new_c,15,axis=1)

        X_new_nc = np.reshape(X[:,100:,:504,:],(30,75,504,8))
        Ys_new_nc = Ys[:,100:]
        Yt_new_nc = Yt[:,100:]

        X = np.concatenate((X_new_c, X_new_nc),axis=1)
        Ys = np.concatenate((Ys_new_c, Ys_new_nc),axis=1)
        Yt = np.concatenate((Yt_new_c, Yt_new_nc),axis=1)

    if(dataset=='256_channel_cVEP'):
        n_subjects = 5
        n_classes = 36
        n_channels = 256
        codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
        codes = np.moveaxis(codebook,1,0)
        X, rejected_chans = remove_bad_channels(X)

        X = np.reshape(X,(5,108*2,504,256))

        Ys = np.repeat(Ys,2,axis=1)
        Yt = np.repeat(Yt,2,axis=1)


    # Preprocessing data
    low_cutoff = 2
    high_cutoff = 30
    sfreq = 240
    X = bandpass_filter_data(X, low_cutoff, high_cutoff, sfreq)

    results = {}
    for i in range(0,n_subjects):
        results[i+1] = {}

        X_new = X[i]
        ys_new = Ys[i]
        yt_new = Yt[i]
        
        y_new= np.concatenate((yt_new[..., np.newaxis],ys_new), axis=1)

        if dataset == '8_channel_cVEP':
            X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.4,stratify=y_new[:,0], shuffle= True)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.2,stratify=y_test[:,0], shuffle= True)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.6,stratify=y_new[:,0], shuffle= True)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.3,stratify=y_test[:,0], shuffle= True)
        X_val = standardize_data(X_val)
        X_test = standardize_data(X_test)

        ys_val = y_val[:,1:]
        ys_test = y_test[:,1:]

        yt_val = y_val[:,0]
        yt_test = y_test[:,0]

#         if(dataset == '256_channel_cVEP'):
#             X_train, ys_train, yt_train = augment_data(X_train, ys_train, yt_train)

        yt_val = to_categorical(yt_val)
        yt_test = to_categorical(yt_test)

        multi_objective_cnn_model = build_multi_objective_cnn_model(n_channels,n_classes)
        checkpoint_filepath = './saved_models/{}/{}/{}/S{}/'.format(model,dataset,mode,i+1)
        multi_objective_cnn_model.load_weights(checkpoint_filepath).expect_partial()


        loss, _,_, seq_accuracy, category_accuracy = multi_objective_cnn_model.evaluate(x = X_test, y = {"sequence": ys_test,"category": yt_test}, verbose=0) 

        print('Dynamic stopping on subject {}'.format(i+1))
        accuracy = category_accuracy
        num_trials = X_test.shape[0]
        time_s = 2.1
        time_min = (X_test.shape[0]* 504*(2.1/504)*(1/60))
        itr = calculate_ITR(n_classes, accuracy, time_min, num_trials)                                                                                                
        print('Subject: ',i+1,'base','(',accuracy,2.1,itr,')')
        
        static_acc, static_time, static_itr = static_stopping(X_val, yt_val, X_test,yt_test,n_classes,multi_objective_cnn_model)
        print('Subject: ',i+1,'static','(',static_acc,static_time,static_itr,')')

        acc_results = acc_stopping(X_test,yt_test,n_classes, multi_objective_cnn_model)

        acc_time_all = {}
        for j in acc_results.keys():
            acc_time_all[j] = acc_results[j]['time']
        print('Subject: ',i+1,'static_acc',acc_time_all)

        results_ds, const = optimize_dynamic_stopping_time(X_val, yt_val,n_classes,multi_objective_cnn_model)

        results_ds, const, acc_scatter = get_stopping_time(X_test,yt_test,n_classes,multi_objective_cnn_model, const)
        filename = './results/Dynamic stopping/{}/dacc_scatter_{}.pickle'.format(model_txt,dataset)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.save(filename, acc_scatter)

        [ceil_acc, ceil_time, ceil_itr, conf_acc, conf_time, conf_itr, ds_acc, ds_time, ds_itr] = results_ds
        print('Subject:',i+1,'ceil','(',ceil_acc,ceil_time,ceil_itr,')')
        print('Subject:',i+1,'ds1','(',conf_acc,conf_time,conf_itr,')')
        print('Subject:',i+1,'ds2','(',ds_acc,ds_time,ds_itr,const,')')

        results[i+1]['accuracy'] = [accuracy,ceil_acc,static_acc, conf_acc,ds_acc]
        results[i+1]['time'] = [time_s,ceil_time, static_time, conf_time,ds_time]
        results[i+1]['itr'] = [itr,ceil_itr,static_itr,conf_itr,ds_itr]
        results[i+1]['const'] = const
        results[i+1]['acc_stopping_time'] = acc_results

        # print('Subject: ',i+1,'(',accuracy,ceil_acc,conf_acc,ds_acc,')', 
        #       '(',time_s,ceil_time,conf_time,ds_time,')','(',itr,ceil_itr,conf_itr,ds_itr,')')
    
    
    filename = './results/Dynamic stopping/{}/dynamic_stopping_{}.pickle'.format(model_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
# datasets = ['8_channel_cVEP','256_channel_cVEP']
model = 'multi_objective_cnn'
mode = 'loso_subject'
model_txt = 'Multi-objective CNN'

# for dataset in datasets:
#     run_dynamic_stopping(dataset,mode,model)

dataset = '8_channel_cVEP'
filename = './results/Dynamic stopping/{}/dynamic_stopping_{}.pickle'.format(model_txt,dataset)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'rb') as handle:
    results_ds8 = pickle.load(handle)

dataset = '256_channel_cVEP'
filename = './results/Dynamic stopping/{}/dynamic_stopping_{}.pickle'.format(model_txt,dataset)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'rb') as handle:
    results_ds256 = pickle.load(handle)

n_subjects = 30
acc_all8 = np.zeros((30,5))
time_all8 = np.zeros((30,5))
itr_all8 = np.zeros((30,5))
cases_all8 = np.zeros((30,5)).astype('str')
acc_stopping_time8 = {}
cases  = np.array(['base','ceiling','static', 'dynamic (conf)','dynamic (outlier)'])
for i in range(1,n_subjects+1):
    acc_all8[i-1] = results_ds8[i]['accuracy']
    time_all8[i-1] = results_ds8[i]['time']
    itr_all8[i-1] = results_ds8[i]['itr']
    cases_all8[i-1] = cases
    acc_stopping_time8[i-1] = results_ds8[i]['acc_stopping_time']
    
acc_all8std = np.std(acc_all8,axis=0)
time_all8std = np.std(time_all8,axis=0)
itr_all8std = np.std(itr_all8,axis=0)

acc_all8m = np.median(acc_all8,axis=0)
time_all8m = np.median(time_all8,axis=0)
itr_all8m = np.median(itr_all8,axis=0)

n_subjects = 5
acc_all256 = np.zeros((5,5))
time_all256 = np.zeros((5,5))
itr_all256 = np.zeros((5,5))
acc_stopping_time256 = {}
cases_all256 = np.zeros((5,5)).astype('str')
cases  = np.array(['base','ceiling','static', 'dynamic (conf)','dynamic (outlier)'])
for i in range(1,n_subjects+1):
    acc_all256[i-1] = results_ds256[i]['accuracy']
    time_all256[i-1] = results_ds256[i]['time']
    itr_all256[i-1] = results_ds256[i]['itr']
    cases_all256[i-1] = cases
    acc_stopping_time256[i-1] = results_ds256[i]['acc_stopping_time']

acc_all256std = np.std(acc_all256,axis=0)
time_all256std = np.std(time_all256,axis=0)
itr_all256std = np.std(itr_all256,axis=0)

acc_all256m = np.median(acc_all256,axis=0)
time_all256m = np.median(time_all256,axis=0)
itr_all256m = np.median(itr_all256,axis=0)

cases_all8f = cases_all8.flatten()
acc_all8f  = acc_all8.flatten()
time_all8f  = time_all8.flatten()
itr_all8f  = itr_all8.flatten()

cases_all256f = cases_all256.flatten()
acc_all256f  = acc_all256.flatten()
time_all256f  = time_all256.flatten()
itr_all256f  = itr_all256.flatten()

cases_allf = np.concatenate((cases_all8f,cases_all256f))
acc_allf = np.concatenate((acc_all8f,acc_all256f))
time_allf = np.concatenate((time_all8f,time_all256f))
itr_allf = np.concatenate((itr_all8f,itr_all256f))
dataset_allf = np.concatenate((np.repeat('8-channel dataset',150),np.repeat('256-channel dataset',25)))

plt.rcParams.update({'font.size': 14})
data1={'modes':cases_allf,
  'pred_value':acc_allf,
  'dataset':dataset_allf,
 }

df1=pd.DataFrame(data1)


df1 = df1[['modes','pred_value','dataset']]

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")
ax = sns.barplot(x='modes',y='pred_value',data=df1,hue='dataset',estimator=np.median,ci=90,capsize=.15)
ax.set_title('Accuracy for various early stopping approaches',fontsize=14)
ax.set(ylabel='Accuracy')
ax.set(xlabel='Approaches')
ax.set(yticks=np.arange(0,1.001,0.2))
ax.legend(bbox_to_anchor=(1.35, 1.03))
plt.grid(False)
plt.grid(True) 

filename = "./visualizations/Dynamic stopping/acc_ds.png"
os.makedirs(os.path.dirname(filename), exist_ok=True)   
plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
plt.close()

data2={'modes':cases_allf,
  'pred_value':time_allf,
  'dataset':dataset_allf,
 }

df2=pd.DataFrame(data2)
df2 = df2[['modes','pred_value','dataset']]

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")
ax = sns.barplot(x='modes',y='pred_value',data=df2,hue='dataset',estimator=np.median,ci=90,capsize=.15)
ax.set_title('Time for various early stopping approaches',fontsize=14)
ax.set(ylabel='Time(s)')
ax.set(xlabel='Approaches')
ax.set(yticks=np.arange(0,2.11,0.3))
ax.set_ylim((0,2.21))
ax.legend(bbox_to_anchor=(1.35, 1.03))
plt.grid(False)
plt.grid(True) 

filename = "./visualizations/Dynamic stopping/time_ds.png"
os.makedirs(os.path.dirname(filename), exist_ok=True)   
plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
plt.close()

plt.rcParams.update({'font.size': 14})
data3={'modes':cases_allf,
  'pred_value':itr_allf,
  'dataset':dataset_allf,
 }

df3=pd.DataFrame(data3)
df3 = df3[['modes','pred_value','dataset']]

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")
ax = sns.barplot(x='modes',y='pred_value',data=df3,hue='dataset',estimator=np.median,ci=90,capsize=.15)
ax.set_title('ITR for various early stopping approaches',fontsize=14)
ax.set(ylabel='Tine(s)')
ax.set(xlabel='Approaches')
ax.set(yticks=np.arange(0,360,40))
ax.legend(bbox_to_anchor=(1.35, 1.03))
plt.grid(False)
plt.grid(True) 

filename = "./visualizations/Dynamic stopping/itr_ds.png"
os.makedirs(os.path.dirname(filename), exist_ok=True)   
plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
plt.close()

# plot_acc_levels(acc_stopping_time8, acc_stopping_time256)