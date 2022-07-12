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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def build_inception_model(n_channels,n_classes):
    activation_fn = 'elu'
    n_filters = 8
    inputs = Input(shape=(504,n_channels), dtype="float32")
    inpr = Reshape((504,n_channels,1))(inputs)
    mask = Masking(mask_value=0.)(inpr)
    conv1 = Conv2D(filters=n_filters, kernel_size=(64,1), strides=(1,1), padding='same')(mask)
    bn1 = BatchNormalization()(conv1)
    actv1 = Activation(activation_fn)(bn1)
    drop1 = Dropout(0.25)(actv1)
    
    conv2 = Conv2D(filters=n_filters, kernel_size=(32,1), strides=(1,1), padding='same')(mask)
    bn2 = BatchNormalization()(conv2)
    actv2 = Activation(activation_fn)(bn2)
    drop2 = Dropout(0.25)(actv2)
    
    conv3 = Conv2D(filters=n_filters, kernel_size=(16,1), strides=(1,1), padding='same')(mask)
    bn3 = BatchNormalization()(conv3)
    actv3 = Activation(activation_fn)(bn3)
    drop3 = Dropout(0.25)(actv3)
    
    dconv1 = DepthwiseConv2D(kernel_size=(1,n_channels),strides=(1, 1), padding="valid", depth_multiplier=1)(drop1)
    dconv2 = DepthwiseConv2D(kernel_size=(1,n_channels),strides=(1, 1), padding="valid", depth_multiplier=1)(drop2)
    dconv3 = DepthwiseConv2D(kernel_size=(1,n_channels),strides=(1, 1), padding="valid", depth_multiplier=1)(drop3)
    
    conc1 = Concatenate(axis=3)([dconv1, dconv2, dconv3])
    pool1 = AveragePooling2D(pool_size=(4, 1), strides=None, padding="valid")(conc1)
    
    conv4 = Conv2D(filters=n_filters, kernel_size=(16,1), strides=(1,1), padding='same')(pool1)
    bn4 = BatchNormalization()(conv4)
    actv4 = Activation(activation_fn)(bn4)
    drop4 = Dropout(0.25)(actv4)
    
    conv5 = Conv2D(filters=n_filters, kernel_size=(8,1), strides=(1,1), padding='same')(pool1)
    bn5 = BatchNormalization()(conv5)
    actv5 = Activation(activation_fn)(bn5)
    drop5 = Dropout(0.25)(actv5)
    
    conv6 = Conv2D(filters=n_filters, kernel_size=(4,1), strides=(1,1), padding='same')(pool1)
    bn6 = BatchNormalization()(conv6)
    actv6 = Activation(activation_fn)(bn6)
    drop6 = Dropout(0.25)(actv6)
    
    conc2 = Concatenate(axis=3)([drop4, drop5, drop6])
    pool2 = AveragePooling2D(pool_size=(2, 1), strides=None, padding="valid")(conc2)
    
    #conc3 = Add()([conc2, pool1])
    conv7 = Conv2D(filters=12, kernel_size=(8,1), strides=(1,1), padding='same')(pool2)
    pool3 = AveragePooling2D(pool_size=(2, 1), strides=None, padding="valid")(conv7)
    
    conv8 = Conv2D(filters=4, kernel_size=(4,1), strides=(1,1), padding='same')(pool3)
    #conv9 = Conv2D(filters=1, kernel_size=(4,1), strides=(1,1), padding='same')(conv8)
    pool4 = AveragePooling2D(pool_size=(2, 1), strides=None, padding="valid")(conv8)
    
    flatten1 = Flatten()(conv8)
    drop7 = Dropout(0.25)(flatten1)

    #flatten2 = Flatten()(conv9)
    # dense1 = Dense(126, activation=activation_fn)(flatten1)
    # drop8 = Dropout(0.25)(dense1)
    
    #output1 = Dense(126, activation="sigmoid", name = 'sequence')(drop8)
    outputs = Dense(n_classes, activation="softmax", name = 'category')(drop7)

    model = Model(inputs,outputs)

    losses = {"category": "categorical_crossentropy"}
    metric = {"category": "accuracy"}

    model.compile(loss=losses,
                optimizer='adam',
                metrics=metric)

    return model

def train_inception(dataset,mode,model, X_train, ys_train,yt_train,X_val, ys_val,yt_val,n_subjects,n_classes, current_subj, current_fold):
    warnings.filterwarnings("ignore")
    results = {}
    if(dataset=='8_channel_cVEP'):
        n_subjects = 30
        n_classes = 21
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

    model_inception = build_inception_model(n_channels,n_classes)
    callback = EarlyStopping(monitor='val_loss', patience=100)
        
    if(current_fold!=None):
        current_f = '_f'+ str(current_fold+1)        
        checkpoint_filepath = './saved_models/{}/{}/{}/S{}{}/'.format(model,dataset,mode,current_subj+1,current_f)
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)   
    else:
        checkpoint_filepath = './saved_models/{}/{}/{}/S{}/'.format(model,dataset,mode,current_subj+1)
        os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)   
        
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    history = model_inception.fit(x = X_train, y = {"category": yt_train}, batch_size = 32, 
                  epochs = 100, verbose=0, validation_data=(X_val, {"category": yt_val}), 
                            callbacks=[callback, model_checkpoint_callback])

    return model_inception, history.history

def evaluate_inception(model_inception, dataset,mode,model,X_test,ys_test,yt_test,n_subjects,n_classes,codebook):
    results = {}
    loss, category_accuracy = model_inception.evaluate(x = X_test, y = {"category": yt_test}, verbose=0)
    
    category_accuracy = category_accuracy
    #seq_accuracy = seq_accuracy*100
    #results['sequence_accuracy'] = np.array(seq_accuracy)
    results['category_accuracy'] = np.array(category_accuracy)

    accuracy = category_accuracy
    num_trials = X_test.shape[0]
    time_min = (X_test.shape[0]* 504*(2.1/504)*(1/60))
    itr = calculate_ITR(n_classes, accuracy, time_min, num_trials)
    results['ITR'] = np.array(itr)

    pred_c = model_inception.predict(X_test)
    
    pred_c_all = np.argmax(pred_c, axis=1)
    cm_c = confusion_matrix(np.argmax(yt_test,axis=1), pred_c_all)

    precision = precision_score(np.argmax(yt_test,axis=1), pred_c_all, average='weighted')
    recall = recall_score(np.argmax(yt_test,axis=1), pred_c_all, average='weighted')
    f1 = f1_score(np.argmax(yt_test,axis=1), pred_c_all, average='weighted')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for n in range(n_classes):
        fpr[n], tpr[n], _ = roc_curve(yt_test[:, n], pred_c[:, n])
        roc_auc[n] = auc(fpr[n], tpr[n])

    results['category_cm'] = np.array(cm_c)
    results['recall'] = np.array(recall)
    results['precision'] = np.array(precision)
    results['f1_score'] = np.array(f1)
    results['category_cm'] = np.array(cm_c)
    results['fpr'] = fpr
    results['tpr'] = tpr
    results['auc'] = roc_auc

    if(mode!='cross_subject'):

        pred_c = model_inception.predict(X_test)
        pred_c = np.argmax(pred_c, axis=1)
        cm_c = confusion_matrix(np.argmax(yt_test,axis=1), pred_c)

        results['category_cm'] = np.array(cm_c)

        acc_time_step =[]
        acc_time_step_r =[]
        itr_time_step = []
        pred_time_step = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
        pred_time_step_r = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
        for k in range(0, X_test.shape[1]):
            X_test_new = X_test.copy()
            X_test_new[:,k:,:] = 0
            pred = model_inception.predict(X_test_new)
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
            
    #     # for k in range(0, X_test.shape[1]):
    #     #     X_test_new = X_test.copy()
    #     #     r = 504-k-1
    #     #     X_test_new[:,:r,:] = 0
            
    #     #     pred = model_inception.predict(X_test_new)
    #     #     prediction = np.argmax(pred,axis=1)
            
    #     #     acc = 100*(np.sum(prediction == np.argmax(yt_test,axis=1))/len(prediction))
    #     #     acc_time_step_r.append(acc)
            
    #     #     results['pred_time_step_r'] = {}
            
    #     #     pred_time_step_r[:,k] = pred
                  

        results['variable_time_steps'] = np.array(acc_time_step)
        #results['variable_time_steps_r'] = np.array(acc_time_step_r)
        results['ITR_time_steps'] = np.array(itr_time_step)

        results['pred_time_step'] = pred_time_step
        #results['pred_time_step_r'] = pred_time_step_r

    return results

def run_inception(dataset,mode,model): 
    filename = "./results/{}/{}/{}/{}_{}_run.txt".format(model,dataset,mode,model,mode)
    os.makedirs(os.path.dirname(filename), exist_ok=True)           
    run_f = open(filename, "w")
    if(mode=='cross_subject'):
        results = {}
        with open('./datasets/{}.pickle'.format(dataset), 'rb') as handle:
            data = pickle.load(handle)

        X = data['X']
        Ys = data['Ys']
        Yt = data['Yt']

        if(dataset=='8_channel_cVEP'):
            n_subjects = 30
            n_classes = 21
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

        for i in range(0,n_subjects):
            results[i+1] = {}

            X_new = X[i]
            ys_new = Ys[i]
            yt_new = Yt[i]
       
            y_new= np.concatenate((yt_new[..., np.newaxis],ys_new), axis=1)

            X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.2,stratify=y_new[:,0], shuffle= True)

            X_train = standardize_data(X_train)
            X_val = standardize_data(X_val)

            ys_train = y_train[:,1:]
            ys_val = y_val[:,1:]

            yt_train = y_train[:,0]
            yt_val = y_val[:,0]

            if(dataset == '256_channel_cVEP'):
                X_train, ys_train, yt_train = augment_data(X_train, ys_train, yt_train)

            yt_train = to_categorical(yt_train)
            yt_val = to_categorical(yt_val)

            model_inception, model_history = train_inception(dataset,mode,model, X_train, ys_train, yt_train, X_val, ys_val, yt_val, n_subjects, n_classes, i, None)
            results[i+1]['history'] = model_history

            for j in range(0,n_subjects):
                results[i+1][j+1] = {}
                if(j!=i):
                    X_test = X[j]
                    X_test = standardize_data(X_test)
                    ys_test = Ys[j]
                    yt_test= Yt[j]
                    yt_test = to_categorical(yt_test)

                else:
                    X_test = X_val
                    ys_test = ys_val
                    yt_test = yt_val

                if (dataset == '256_channel_cVEP'):
                    X_test = X_test
                    ys_test = ys_test
                    yt_test = yt_test

                results_eval = evaluate_inception(model_inception, dataset,mode,model,X_test,ys_test, yt_test, n_subjects,n_classes,codebook)

                print("Train on subject {} test on subject {} category_accuracy: {}".format(i+1,j+1,results_eval['category_accuracy']))
                print("Train on subject {} test on subject {} category_accuracy: {}".format(i+1,j+1,results_eval['category_accuracy']),file=run_f)
                results[i+1][j+1]['category_accuracy'] = results_eval['category_accuracy']
                results[i+1][j+1]['ITR'] = results_eval['ITR']
                results[i+1][j+1]['category_cm'] =  results_eval['category_cm']
                results[i+1][j+1]['recall'] = results_eval['recall']
                results[i+1][j+1]['precision'] = results_eval['precision']
                results[i+1][j+1]['f1_score'] = results_eval['f1_score']
                results[i+1][j+1]['fpr'] = results_eval['fpr']
                results[i+1][j+1]['tpr'] = results_eval['tpr']
                results[i+1][j+1]['auc'] = results_eval['auc']

                # if(mode!='cross_subject'):
                #     results[i+1][j+1]['variable_time_steps'] = results_eval['variable_time_steps']
                #     #results[i+1][j+1]['variable_time_steps_r'] = results_eval['variable_time_steps_r']
                #     results[i+1][j+1]['ITR_time_steps'] = results_eval['ITR_time_steps']
                #     results[i+1][j+1]['ITR'] = results_eval['ITR']
                #     results[i+1][j+1]['pred_time_step'] = results_eval['pred_time_step']
                #     #results[i+1][j+1]['pred_time_step_r'] = results_eval['pred_time_step_r']


        filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif(mode=='within_subject' or mode=='loso_subject'):
        results = {}
        if(dataset=='8_channel_cVEP'):
            n_subjects = 30
            n_classes = 21
            n_folds = 5
            n_channels = 8
            mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
            codes = mat['codes'].astype('float32')
            codebook = np.moveaxis(codes,1,0).astype('float32')

        if(dataset=='256_channel_cVEP'):
            n_subjects = 5
            n_classes = 36
            n_folds = 3
            n_channels = 256
            codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
            codes = np.moveaxis(codebook,1,0)

        for i in range(0,n_subjects):
            results[i+1] = {}
            for fold in range(0,n_folds):
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

                # if (dataset == '256_channel_cVEP'):
                #     X_test = X_test
                #     ys_test = ys_test[:216]
                #     yt_test = yt_test[:216]

                # if (len(yt_train.shape)!=2):
                #     yt_train = to_categorical(yt_train)
                #     yt_val = to_categorical(yt_val)
                #     yt_test = to_categorical(yt_test)

                if(mode == 'within_subject'):
                    model_inception, model_history = train_inception(dataset,mode,model, X_train, ys_train, yt_train, X_val, ys_val, yt_val, n_subjects, n_classes, i, fold)
                    results[i+1][fold+1]['history'] = model_history

                else:
                    if(fold==0):
                        model_inception, model_history = train_inception(dataset,mode,model, X_train, ys_train, yt_train, X_val, ys_val, yt_val, n_subjects, n_classes, i, None)
                        results[i+1]['history'] = model_history

                model_inception = build_inception_model(n_channels,n_classes)

                if(mode=='loso_subject'):
                    filename = './saved_models/{}/{}/{}/S{}/'.format(model,dataset,mode,i+1)
                else:
                    current_f = '_f'+ str(fold+1)    
                    filename = './saved_models/{}/{}/{}/S{}{}/'.format(model,dataset,mode,i+1,current_f)

                model_inception.load_weights(filename).expect_partial()

                results_eval = evaluate_inception(model_inception, dataset,mode,model,X_test,ys_test, yt_test, n_subjects,n_classes,codebook)

                print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']))
                print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']),file=run_f)
                results[i+1][fold+1]['category_accuracy'] = results_eval['category_accuracy']
                results[i+1][fold+1]['ITR'] = results_eval['ITR']
                results[i+1][fold+1]['category_cm'] =  results_eval['category_cm']
                results[i+1][fold+1]['recall'] = results_eval['recall']
                results[i+1][fold+1]['precision'] = results_eval['precision']
                results[i+1][fold+1]['f1_score'] = results_eval['f1_score']
                results[i+1][fold+1]['fpr'] = results_eval['fpr']
                results[i+1][fold+1]['tpr'] = results_eval['tpr']
                results[i+1][fold+1]['auc'] = results_eval['auc']
                # #results[i+1][fold+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                if(mode!='cross_subject'):
                    results[i+1][fold+1]['variable_time_steps'] = results_eval['variable_time_steps']
                    #results[i+1][fold+1]['variable_time_steps_r'] = results_eval['variable_time_steps_r']
                    results[i+1][fold+1]['ITR_time_steps'] = results_eval['ITR_time_steps']
                    results[i+1][fold+1]['pred_time_step'] = results_eval['pred_time_step']
                    #results[i+1][fold+1]['pred_time_step_r'] = results_eval['pred_time_step_r']

        filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        warnings.warn("Unsupported mode")

# # datasets = ['8_channel_cVEP','256_channel_cVEP']
# # modes = ['within_subject','loso_subject','cross_subject']
# model = "inception"

# for dataset in datasets:
#     for mode in modes: 
#         print('\n------Running {} for dataset {} in mode {}-----\n'.format(model, dataset, mode))
#         run_inception(dataset, mode, model)