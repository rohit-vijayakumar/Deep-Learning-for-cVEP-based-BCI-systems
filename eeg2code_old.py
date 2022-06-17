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
from sklearn.preprocessing import label_binarize

def epoch_data(X, Ys, n_subjects, n_classes):
    
    n_samples = 60 #150ms
    n_steps = 4
    n_trials = int(X.shape[1]/n_steps)

    data_X_trial = np.array([]).reshape(0,n_trials,n_samples,X.shape[2])
    for trial in range(X.shape[0]):
        data_X = np.array([]).reshape(0,n_samples,X.shape[2])
        for bit in range(Ys.shape[1]):
            y_bit = Ys[trial][bit]
            x_window = np.roll(X[trial], -bit*4, axis=0)[:n_samples]
            data_X = np.vstack((data_X, x_window[np.newaxis,...]))
            
        data_X_trial = np.vstack((data_X_trial, data_X[np.newaxis,...]))
    
    #data_X_trial = np.reshape(data_X_trial,(X.shape[0]*n_trials,n_samples,X.shape[2]))[..., np.newaxis]
    data_Ys_trial = Ys
    
    return data_X_trial, data_Ys_trial

def build_eeg2code_model(n_channels):
    model = Sequential()
    # permute input so that it is as in EEG2Code paper

    model.add(Permute((3,2,1), input_shape=(n_samples,n_channels,1)))
    # layer1
    model.add(Conv2D(16, kernel_size=(n_channels, 1), padding='valid', strides=(1, 1), data_format='channels_first', activation='relu'))
    model.add(BatchNormalization(axis=1, scale=False, center=False))
    #model.add(Permute((2,1,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
    # layer2
    model.add(Conv2D(8,kernel_size=(1, 64),data_format='channels_first',padding='same'))
    model.add(BatchNormalization(axis=1,scale=False, center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
    model.add(Dropout(0.5))
    # layer3
    model.add(Conv2D(4,kernel_size=(5, 5),data_format='channels_first',padding='same'))
    model.add(BatchNormalization(axis=1,scale=False,center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first',padding='same'))
    model.add(Dropout(0.5))
    # layer4
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # layer5
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    return model

def train_eeg2code(dataset,mode,model, X_train, ys_train,X_val, ys_val,n_subjects,n_classes, current_subj, current_fold):
    warnings.filterwarnings("ignore")
    results = {}
    if(dataset=='8_channel_cVEP'):
        n_subjects = 30
        n_classes = 20
        n_channels = 8
        n_samples = 60
        mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
        codes = mat['codes'].astype('float32')
        codebook = np.moveaxis(codes,1,0).astype('float32')

    elif(dataset=='256_channel_cVEP'):
        n_subjects = 5
        n_classes = 36
        n_channels = 256
        n_samples = 60
        codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
        codes = np.moveaxis(codebook,1,0)
    else:
        warnings.warn("Unsupported dataset")

    model_eeg2code = build_eeg2code_model(n_channels,n_samples)
    callback = EarlyStopping(monitor='val_loss', patience=10)
        
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

    history = model_eeg2code.fit(X_train, ys_train, batch_size = 128, epochs = 100, verbose=0, validation_data = (X_val, ys_val), callbacks=[callback, model_checkpoint_callback])

    return model_eeg2code, history.history

def evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test,ys_test,yt_test,n_subjects,n_classes,codebook):
    results = {}
    pred_class_arr = []
    pred_seq_arr = []
    for trial in range (X_test.shape[0]):
        _, seq_acc = model_eeg2code.evaluate(X_test[trial][..., np.newaxis], ys_test[trial], verbose=0)
        pred_seq_arr.append(seq_acc)
        pred_seq = model_eeg2code.predict(X_test[trial][..., np.newaxis])[:,0]
        target_seq = ys_test[trial,:]
        corr_arr = []
        for k in range(len(codebook)):
            corr = np.corrcoef(pred_seq,codebook[k])[0, 1:]
            corr_arr.append(corr)
            
        pred_class = np.argmax(corr_arr)
        pred_class_arr.append(pred_class)
    
    category_acc = (np.sum(pred_class_arr == yt_test)/len(pred_class_arr))
    sequence_acc = np.mean(pred_seq_arr)

    results['sequence_accuracy'] = np.array(sequence_acc)
    results['category_accuracy'] = np.array(category_acc)

    accuracy = category_acc
    num_trials = X_test.shape[0]
    time_min = (X_test.shape[0]* 504*(2.1/504)*(1/60))
    itr = calculate_ITR(n_classes, accuracy, time_min, num_trials)
    results['ITR'] = np.array(itr)

    precision = precision_score(yt_test, pred_class_arr, average='weighted')
    recall = recall_score(yt_test, pred_class_arr, average='weighted')
    f1 = f1_score(yt_test, pred_class_arr, average='weighted')

    cm_c = confusion_matrix(yt_test, pred_class_arr)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    yt_test_categorical = label_binarize(yt_test, classes = np.arange(0,n_classes))
    prediction_categorical = label_binarize(pred_class_arr,classes = np.arange(0,n_classes))
    for n in range(n_classes):
        fpr[n], tpr[n], _ = roc_curve(yt_test_categorical[:, n], prediction_categorical[:, n])
        roc_auc[n] = auc(fpr[n], tpr[n])

    results['category_cm'] = np.array(cm_c)
    results['recall'] = np.array(recall)
    results['precision'] = np.array(precision)
    results['f1_score'] = np.array(f1)
    results['sequence_cm'] = np.array(cm_all)
    results['category_cm'] = np.array(cm_c)
    results['fpr'] = fpr
    results['tpr'] = tpr
    results['auc'] = roc_auc

    return results

    # if(mode!='cross_subject'):
    #     acc_time_step =[]
    #     acc_time_step_r =[]
    #     itr_time_step = []
    #     pred_time_step = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
    #     pred_time_step_r = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
    #     for k in range(1, X_test.shape[1]):
    #         print(k)
    #         X_test_new = X_test[:,:k,:] 
    #         ys_test_new = ys_test[:,:k] 

    #         pred_class_arr = []
    #         pred_seq_arr = []
    #         corr_all = np.zeros((X_test_new.shape[0],n_classes))
    #         for trial in range (X_test_new.shape[0]):
    #             _, seq_acc = model_eeg2code.evaluate(X_test_new[trial][..., np.newaxis], ys_test_new[trial], verbose=0)
    #             pred_seq_arr.append(seq_acc*100)
    #             pred_seq = model_eeg2code.predict(X_test_new[trial][..., np.newaxis])[:,0]
    #             target_seq = ys_test_new[trial,:]
    #             corr_arr = []
    #             for k1 in range(len(codebook)):
    #                 corr = np.corrcoef(pred_seq,codebook[k1,:k])[0, 1:]
    #                 corr_arr.append(corr)
                
    #             corr_all[trial] = corr_arr
    #             pred_class = np.argmax(corr_arr)
    #             pred_class_arr.append(pred_class)
                
    #         category_acc = 100*(np.sum(pred_class_arr == yt_test)/len(pred_class_arr))
    #         sequence_acc = np.mean(pred_seq_arr)
            
    #         acc_time_step.append(category_acc)

            
    #         accuracy = category_acc/100
    #         num_trials = X_test_new.shape[0]
    #         time_min = (X_test_new.shape[0]* k*(2.1/126)*(1/60))
    #         itr = calculate_ITR(n_classes, accuracy, time_min, num_trials)
    #         itr_time_step.append(itr)


    #         pred_time_step[:,k] = corr_all
            
    #     for k in range(1, X_test.shape[1]):
    #         r = 126-k-1
    #         X_test_new = X_test[:,r:,:] 
    #         ys_test_new = ys_test[:,r:] 

    #         pred_class_arr = []
    #         pred_seq_arr = []
    #         corr_all = np.zeros((X_test_new.shape[0],n_classes))
    #         for trial in range (X_test_new.shape[0]):
    #             _, seq_acc = model_eeg2code.evaluate(X_test_new[trial][..., np.newaxis], ys_test_new[trial], verbose=0)
    #             pred_seq_arr.append(seq_acc*100)
    #             pred_seq = model_eeg2code.predict(X_test_new[trial][..., np.newaxis])[:,0]
    #             target_seq = ys_test_new[trial,:]
    #             corr_arr = []
    #             for k1 in range(len(codebook)):
    #                 corr = np.corrcoef(pred_seq,codebook[k1,r:])[0, 1:]
    #                 corr_arr.append(corr)
                
    #             corr_all[trial] = corr_arr
    #             pred_class = np.argmax(corr_arr)
    #             pred_class_arr.append(pred_class)
                
    #         category_acc = 100*(np.sum(pred_class_arr == yt_test)/len(pred_class_arr))
    #         sequence_acc = np.mean(pred_seq_arr)
            
    #         acc_time_step_r.append(category_acc)
    #         pred_time_step_r[:,k] = corr_all
        
    #     results['variable_time_steps'] = np.array(acc_time_step)
    #     results['variable_time_steps_r'] = np.array(acc_time_step_r)
    #     results['ITR_time_steps'] = np.array(itr_time_step)

    #     results['pred_time_step'] = pred_time_step
    #     results['pred_time_step_r'] = pred_time_step_r

def run_eeg2code(dataset,mode,model):            
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

        # Preprocessing data
        low_cutoff = 2
        high_cutoff = 30
        sfreq = 240
        X = bandpass_filter_data(X, low_cutoff, high_cutoff, sfreq)

        if(dataset=='8_channel_cVEP'):
            n_subjects = 30
            n_classes = 20
            if (model=='eeg2code'):
                X = X[:,:1500,:,:]
                Ys = Ys[:,:1500]
                Yt = Yt[:,:1500]
            mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
            codes = mat['codes'].astype('float32')
            codebook = np.moveaxis(codes,1,0).astype('float32')

        if(dataset=='256_channel_cVEP'):
            n_subjects = 5
            n_classes = 36
            codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
            codes = np.moveaxis(codebook,1,0)

            #X, accepted_chans = remove_bad_channels(X)
            #X, Ys, Yt = augment_data_trial(X, Ys, Yt)


        for i in range(0,n_subjects):
            results[i+1] = {}

            X_new = X[i]
            ys_new = Ys[i]
            yt_new = Yt[i].flatten()
            y_new= np.concatenate((yt_new[..., np.newaxis],ys_new), axis=1)

            X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.2,stratify=y_new[:,0], shuffle= True)

            X_train = standardize_data(X_train)
            X_val = standardize_data(X_val)

            ys_train = y_train[:,1:]
            ys_val = y_val[:,1:]

            yt_train = y_train[:,0]
            yt_val = y_val[:,0]


            X_train_epoched, Ys_train_epoched = epoch_data(X_train, ys_train, n_subjects, n_classes)
            X_train_epoched = np.reshape(X_train_epoched,(int(X_train_epoched.shape[0]*X_train_epoched.shape[1]),X_train_epoched.shape[2],X_train_epoched.shape[3]))[..., np.newaxis]
            Ys_train_epoched = np.reshape(Ys_train_epoched, (int(Ys_train_epoched.shape[0]*Ys_train_epoched.shape[1]),1))

            X_val_epoched, Ys_val_epoched = epoch_data(X_val, ys_val, n_subjects, n_classes)
            X_val_epoched = np.reshape(X_val_epoched,(X_val_epoched.shape[0]*X_val_epoched.shape[1],X_val_epoched.shape[2],X_val_epoched.shape[3]))[..., np.newaxis]
            Ys_val_epoched = np.reshape(Ys_val_epoched, (Ys_val_epoched.shape[0]*Ys_val_epoched.shape[1],1))

            model_eeg2code, model_history = train_eeg2code(dataset,mode,model, X_train_epoched, Ys_train_epoched,X_val_epoched, Ys_val_epoched, n_subjects, n_classes, i, None)
            results[i+1]['history'] = model_history

            for j in range(0,n_subjects):
                results[i+1][j+1] = {}
                if(j!=i):
                    X_test = X[j]
                    X_test = standardize_data(X_test)
                    ys_test = Ys[j]
                    yt_test= Yt[j]

                else:
                    X_test = X_val
                    ys_test = ys_val
                    yt_test = yt_val

                if (dataset == '256_channel_cVEP'):
                    X_test = X_test[:216]
                    ys_test = ys_test[:216]
                    yt_test = yt_test[:216]

                X_test_epoched, Ys_test_epoched = epoch_data(X_test, ys_test, n_subjects, n_classes)

                results_eval = evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test_epoched,Ys_test_epoched, yt_test, n_subjects,n_classes,codebook)

                print("Train on subject {} test on subject {} category_accuracy: {}".format(i+1,j+1,results_eval['category_accuracy']))
                print("Train on subject {} test on subject {} category_accuracy: {}".format(i+1,j+1,results_eval['category_accuracy']),file=run_f)
                results[i+1][j+1]['category_accuracy'] = results_eval['category_accuracy']
                results[i+1][j+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
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
                #     results[i+1][j+1]['variable_time_steps_r'] = results_eval['variable_time_steps_r']
                #     results[i+1][j+1]['ITR_time_steps'] = results_eval['ITR_time_steps']
                    
                #     results[i+1][j+1]['pred_time_step'] = results_eval['pred_time_step']
                #     results[i+1][j+1]['pred_time_step_r'] = results_eval['pred_time_step_r']

        filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
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

                yt_test = np.argmax(yt_test,axis=1)
                if (dataset == '256_channel_cVEP'):
                    X_test = X_test[:216]
                    ys_test = ys_test[:216]
                    yt_test = yt_test[:216]

                X_test_epoched, Ys_test_epoched = epoch_data(X_test, ys_test, n_subjects, n_classes)

                if(mode == 'within_subject'):
                    X_train_epoched, Ys_train_epoched = epoch_data(X_train, ys_train, n_subjects, n_classes)

                    X_train_epoched = np.reshape(X_train_epoched,(int(X_train_epoched.shape[0]*X_train_epoched.shape[1]),X_train_epoched.shape[2],X_train_epoched.shape[3]))[..., np.newaxis]
                    Ys_train_epoched = np.reshape(Ys_train_epoched, (int(Ys_train_epoched.shape[0]*Ys_train_epoched.shape[1]),1))

                    X_val_epoched, Ys_val_epoched = epoch_data(X_val, ys_val, n_subjects, n_classes)
                    X_val_epoched = np.reshape(X_val_epoched,(X_val_epoched.shape[0]*X_val_epoched.shape[1],X_val_epoched.shape[2],X_val_epoched.shape[3]))[..., np.newaxis]
                    Ys_val_epoched = np.reshape(Ys_val_epoched, (Ys_val_epoched.shape[0]*Ys_val_epoched.shape[1],1))

                    model_eeg2code, model_history = train_eeg2code(dataset,mode,model, X_train_epoched, Ys_train_epoched, X_val_epoched, Ys_val_epoched,n_subjects, n_classes, i, fold)
                    results[i+1][fold+1]['history'] = model_history
            

                else:
                    X_test_epoched, Ys_test_epoched = epoch_data(X_test, ys_test, n_subjects, n_classes)
                    if(fold==0):
                        X_train_epoched, Ys_train_epoched = epoch_data(X_train, ys_train, n_subjects, n_classes)

                        X_train_epoched = np.reshape(X_train_epoched,(int(X_train_epoched.shape[0]*X_train_epoched.shape[1]),X_train_epoched.shape[2],X_train_epoched.shape[3]))[..., np.newaxis]
                        Ys_train_epoched = np.reshape(Ys_train_epoched, (int(Ys_train_epoched.shape[0]*Ys_train_epoched.shape[1]),1))

                        X_val_epoched, Ys_val_epoched = epoch_data(X_val, ys_val, n_subjects, n_classes)
                        X_val_epoched = np.reshape(X_val_epoched,(X_val_epoched.shape[0]*X_val_epoched.shape[1],X_val_epoched.shape[2],X_val_epoched.shape[3]))[..., np.newaxis]
                        Ys_val_epoched = np.reshape(Ys_val_epoched, (Ys_val_epoched.shape[0]*Ys_val_epoched.shape[1],1))

                        model_eeg2code, model_history = train_eeg2code(dataset,mode,model, X_train_epoched, Ys_train_epoched, X_val_epoched, Ys_val_epoched,n_subjects, n_classes, i, fold)
                        results[i+1]['history'] = model_history

                results_eval = evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test_epoched,Ys_test_epoched, yt_test, n_subjects,n_classes,codebook)

                print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']))
                print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']),file=run_f)
                results[i+1][fold+1]['category_accuracy'] = results_eval['category_accuracy']
                results[i+1][fold+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                results[i+1][fold+1]['ITR'] = results_eval['ITR']
                results[i+1][fold+1]['category_cm'] =  results_eval['category_cm']
                results[i+1][fold+1]['recall'] = results_eval['recall']
                results[i+1][fold+1]['precision'] = results_eval['precision']
                results[i+1][fold+1]['f1_score'] = results_eval['f1_score']
                results[i+1][fold+1]['fpr'] = results_eval['fpr']
                results[i+1][fold+1]['tpr'] = results_eval['tpr']
                results[i+1][fold+1]['auc'] = results_eval['auc']

                # if(mode!='cross_subject'):
                #     results[i+1][fold+1]['variable_time_steps'] = results_eval['variable_time_steps']
                #     results[i+1][fold+1]['variable_time_steps_r'] = results_eval['variable_time_steps_r']
                #     results[i+1][fold+1]['ITR_time_steps'] = results_eval['ITR_time_steps']
                    
                #     results[i+1][fold+1]['pred_time_step'] = results_eval['pred_time_step']
                #     results[i+1][fold+1]['pred_time_step_r'] = results_eval['pred_time_step_r']

        filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        warnings.warn("Unsupported mode")

datasets = ['256_channel_cVEP','8_channel_cVEP']
modes = ['loso_subject','cross_subject','within_subject']
model = "eeg2code"

for dataset in datasets:
    for mode in modes: 
        print('\n------Running {} for dataset {} in mode {}-----\n'.format(model, dataset, mode))
        run_eeg2code(dataset, mode, model)