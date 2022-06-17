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
    n_samples = 60
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

def train_eeg2code(model_eeg2code,dataset,mode,model, X_train, ys_train,X_val, ys_val,n_subjects,n_classes, current_subj, current_fold):
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

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()

    # yt_test_categorical = label_binarize(yt_test, classes = np.arange(0,n_classes))
    # prediction_categorical = label_binarize(pred_class_arr,classes = np.arange(0,n_classes))
    # for n in range(n_classes):
    #     fpr[n], tpr[n], _ = roc_curve(yt_test_categorical[:, n], prediction_categorical[:, n])
    #     roc_auc[n] = auc(fpr[n], tpr[n])

    # results['category_cm'] = np.array(cm_c)
    results['recall'] = np.array(recall)
    results['precision'] = np.array(precision)
    results['f1_score'] = np.array(f1)
    # #results['sequence_cm'] = np.array(cm_all)
    # #results['category_cm'] = np.array(cm_c)
    # results['fpr'] = fpr
    # results['tpr'] = tpr
    # results['auc'] = roc_auc


    # if(mode!='cross_subject'):
    #     acc_time_step =[]
    #     acc_time_step_r =[]
    #     itr_time_step = []
    #     pred_time_step = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
    #     pred_time_step_r = np.zeros((X_test.shape[0],X_test.shape[1],n_classes))
    #     for k in range(1, X_test.shape[1]):
    #         #print(k)
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
                
    #         category_acc = (np.sum(pred_class_arr == yt_test)/len(pred_class_arr))
    #         sequence_acc = np.mean(pred_seq_arr)
            
    #         acc_time_step.append(category_acc)

            
    #         accuracy = category_acc
    #         num_trials = X_test_new.shape[0]
    #         time_min = (X_test_new.shape[0]* k*(2.1/126)*(1/60))
    #         itr = calculate_ITR(n_classes, accuracy, time_min, num_trials)
    #         itr_time_step.append(itr)


    #     results['variable_time_steps'] = np.array(acc_time_step)
    #     results['ITR_time_steps'] = np.array(itr_time_step)
    #     results['pred_time_step'] = pred_time_step

    return results

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

        if(dataset=='8_channel_cVEP'):
            n_subjects = 30
            n_classes = 20
            n_channels = 8

            mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
            codes = mat['codes'].astype('float32')
            codebook = np.moveaxis(codes,1,0).astype('float32')

        if(dataset=='256_channel_cVEP'):
            n_subjects = 5
            n_classes = 36
            n_channels = 256
            codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
            codes = np.moveaxis(codebook,1,0)

            #X, accepted_chans = remove_bad_channels(X)
            #X, Ys, Yt = augment_data_trial(X, Ys, Yt)


        for i in range(0,n_subjects):
            results[i+1] = {}

            with open('./datasets/epoched_data/{}_epoched_S{}.pickle'.format(dataset,i+1), 'rb') as handle:
                data = pickle.load(handle)

                X = data['X']
                Ys = data['Ys']
                Yt = data['Yt'] 

            X_new = X
            ys_new = Ys
            yt_new = Yt.flatten()
            y_new= np.concatenate((yt_new[..., np.newaxis],ys_new), axis=1)

            X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.2,stratify=y_new[:,0], shuffle= True)

            del X_new

            X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],X_train.shape[2],X_train.shape[3]))
            X_val = np.reshape(X_val, (X_val.shape[0]*X_val.shape[1],X_val.shape[2],X_val.shape[3]))

            X_train = standardize_data(X_train)
            X_val = standardize_data(X_val)

            ys_train = y_train[:,1:]
            ys_val = y_val[:,1:]

            yt_train = y_train[:,0]
            yt_val = y_val[:,0]

            ys_train = np.reshape(ys_train,(ys_train.shape[0]*ys_train.shape[1]))
            ys_val = np.reshape(ys_val,(ys_val.shape[0]*ys_val.shape[1]))

            model_eeg2code = build_eeg2code_model(n_channels)
            model_eeg2code, model_history = train_eeg2code(model_eeg2code,dataset,mode,model, X_train[...,np.newaxis], ys_train,X_val[...,np.newaxis], ys_val, n_subjects, n_classes, i, None)
            results[i+1]['history'] = model_history

            del X_train
            for j in range(0,n_subjects):
                results[i+1][j+1] = {}
                if(j!=i):

                    with open('./datasets/epoched_data/{}_epoched_S{}.pickle'.format(dataset,j+1), 'rb') as handle:
                        data = pickle.load(handle)

                    X = data['X']
                    Ys = data['Ys']
                    Yt = data['Yt'] 

                    X_test = X
                    X_test = standardize_epoched_data(X_test)
                    ys_test = Ys
                    yt_test= Yt

                else:
                    X_test = X_val
                    ys_test = ys_val
                    yt_test = yt_val

                if (dataset == '256_channel_cVEP'):
                    X_test = X_test[:216]
                    ys_test = ys_test[:216]
                    yt_test = yt_test[:216]

                results_eval = evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test,ys_test, yt_test, n_subjects,n_classes,codebook)

                del X_val, X_test

                print("Train on subject {} test on subject {} category_accuracy: {}".format(i+1,j+1,results_eval['category_accuracy']))
                print("Train on subject {} test on subject {} category_accuracy: {}".format(i+1,j+1,results_eval['category_accuracy']),file=run_f)
                results[i+1][j+1]['category_accuracy'] = results_eval['category_accuracy']
                results[i+1][j+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                results[i+1][j+1]['ITR'] = results_eval['ITR']
                #results[i+1][j+1]['category_cm'] =  results_eval['category_cm']
                results[i+1][j+1]['recall'] = results_eval['recall']
                results[i+1][j+1]['precision'] = results_eval['precision']
                results[i+1][j+1]['f1_score'] = results_eval['f1_score']
                #results[i+1][j+1]['fpr'] = results_eval['fpr']
                #results[i+1][j+1]['tpr'] = results_eval['tpr']
                #results[i+1][j+1]['auc'] = results_eval['auc']
                # if(mode!='cross_subject'):
                #results[i+1][j+1]['variable_time_steps'] = results_eval['variable_time_steps']
                #results[i+1][j+1]['ITR_time_steps'] = results_eval['ITR_time_steps']   
                #results[i+1][j+1]['pred_time_step'] = results_eval['pred_time_step']
                #     results[i+1][j+1]['variable_time_steps_r'] = results_eval['variable_time_steps_r']

                #     results[i+1][j+1]['pred_time_step_r'] = results_eval['pred_time_step_r']

        filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    elif(dataset=='8_channel_cVEP'):
        results = {}
        n_subjects = 30
        n_classes = 20
        n_channels = 8
        mat = scipy.io.loadmat('./datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat')
        codes = mat['codes'].astype('float32')
        codebook = np.moveaxis(codes,1,0).astype('float32')

        if(mode=='within_subject'):
            subjects = np.arange(1,31)
            for i in range(0,n_subjects):
                results[i+1] = {}
                with open('./datasets/epoched_data/{}_epoched_S{}.pickle'.format(dataset,i+1), 'rb') as handle:
                    data = pickle.load(handle)

                X = data['X']
                Ys = data['Ys']
                Yt = data['Yt'] 

                X_cv = np.array([]).reshape(0,100,126,60,8)
                Ys_cv = np.array([]).reshape(0,100,126)
                Yt_cv = np.array([]).reshape(0,100)
                for f in range(0,15):
                    X_fold = X[f::15]
                    Ys_fold = Ys[f::15]
                    Yt_fold = Yt[f::15]


                    X_cv = np.vstack((X_cv, X_fold[np.newaxis,...]))
                    Ys_cv = np.vstack((Ys_cv, Ys_fold[np.newaxis,...]))
                    Yt_cv = np.vstack((Yt_cv, Yt_fold[np.newaxis,...]))

            
                for fold in range(0,15):
                    results[i+1][fold+1] = {}
                    #print(i+1,fold+1)
                    data = {}
                    #print("Training on subject {} fold {} ".format(i+1,fold+1))
                    X_train_f = np.concatenate((X_cv[0:fold], X_cv[fold+1:15]))
                    Ys_train_f = np.concatenate((Ys_cv[0:fold], Ys_cv[fold+1:15]))
                    Yt_train_f = np.concatenate((Yt_cv[0:fold], Yt_cv[fold+1:15]))

                    X_train_folds = np.reshape(X_train_f, (X_train_f.shape[0]*X_train_f.shape[1],126,60,8))
                    Ys_train_folds = np.reshape(Ys_train_f, (X_train_f.shape[0]*X_train_f.shape[1],126))
                    Yt_train_folds = np.reshape(Yt_train_f, (X_train_f.shape[0]*X_train_f.shape[1]))

                    X_test = X_cv[fold:fold+1][0]
                    Ys_test = Ys_cv[fold:fold+1][0]
                    Yt_test = Yt_cv[fold:fold+1][0]


                    Y_train_folds = np.concatenate((Yt_train_folds[..., np.newaxis],Ys_train_folds), axis=1)

                    X_train, X_val, y_train, y_val = train_test_split(X_train_folds, Y_train_folds, test_size=0.2, 
                                                                      stratify=Yt_train_folds, shuffle= True)

                    del X_train_f, X_train_folds
                    
                    X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],X_train.shape[2],X_train.shape[3]))
                    X_val = np.reshape(X_val, (X_val.shape[0]*X_val.shape[1],X_val.shape[2],X_val.shape[3]))

                    X_train = standardize_data(X_train)
                    X_val = standardize_data(X_val)
                    X_test = standardize_epoched_data(X_test)

                    ys_train = y_train[:,1:]
                    ys_val = y_val[:,1:]

                    yt_train = y_train[:,0]
                    yt_val = y_val[:,0]

                    ys_train = np.reshape(ys_train,(ys_train.shape[0]*ys_train.shape[1]))
                    ys_val = np.reshape(ys_val,(ys_val.shape[0]*ys_val.shape[1]))
                    
                    model_eeg2code = build_eeg2code_model(n_channels)
                    model_eeg2code, model_history = train_eeg2code(model_eeg2code,dataset,mode,model, X_train[...,np.newaxis], ys_train,X_val[...,np.newaxis], ys_val, n_subjects, n_classes, i, fold)
                    results[i+1][fold+1]['history'] = model_history

                    results_eval = evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test,Ys_test, Yt_test, n_subjects,n_classes,codebook)
                
                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']))
                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']),file=run_f)

                    results[i+1][fold+1]['category_accuracy'] = results_eval['category_accuracy']
                    results[i+1][fold+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                    results[i+1][fold+1]['ITR'] = results_eval['ITR']
                    #results[i+1][fold+1]['category_cm'] =  results_eval['category_cm']
                    results[i+1][fold+1]['recall'] = results_eval['recall']
                    results[i+1][fold+1]['precision'] = results_eval['precision']
                    results[i+1][fold+1]['f1_score'] = results_eval['f1_score']
                   # results[i+1][fold+1]['fpr'] = results_eval['fpr']
                    #results[i+1][fold+1]['tpr'] = results_eval['tpr']
                    #results[i+1][fold+1]['auc'] = results_eval['auc']
                    #results[i+1][fold+1]['variable_time_steps'] = results_eval['variable_time_steps']
                    #results[i+1][fold+1]['ITR_time_steps'] = results_eval['ITR_time_steps']   
                    #results[i+1][fold+1]['pred_time_step'] = results_eval['pred_time_step']

                    del X_train, X_val, X_test

                del X_cv

            filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif(mode=='loso_subject'):

            subjects = np.arange(1,31)
            for i in range(0,n_subjects):
                results[i+1] = {}

                for f in range(0,15):
                    X = np.array([]).reshape(0,100,126,60,8)
                    Ys = np.array([]).reshape(0,100,126)
                    Yt = np.array([]).reshape(0,100,1)
                    for j in range(0,n_subjects):
                        if(j!=i):
                            with open('./datasets/epoched_data/{}_epoched_S{}.pickle'.format(dataset,j+1), 'rb') as handle:
                                data = pickle.load(handle)

                            X_subj = data['X'][f::15]
                            Ys_subj = data['Ys'][f::15]
                            Yt_subj = data['Yt'][f::15][...,np.newaxis]

                            X = np.vstack((X, X_subj[np.newaxis,...]))
                            Ys = np.vstack((Ys, Ys_subj[np.newaxis,...]))
                            Yt = np.vstack((Yt, Yt_subj[np.newaxis,...]))

                    #print("Leaving out subject ",i+1)
                    X_new = np.reshape(X, (X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4]))
                    Ys_new = np.reshape(Ys, (Ys.shape[0]*Ys.shape[1],Ys.shape[2]))
                    Yt_new = np.reshape(Yt, (Yt.shape[0]*Yt.shape[1],Yt.shape[2]))         

                    Y_new = np.concatenate((Yt_new,Ys_new), axis=1)

                    X_train, X_val, y_train, y_val = train_test_split(X_new, Y_new, test_size=0.2, stratify=Y_new[:,0], shuffle= True)

                    del X_new

                    X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],X_train.shape[2],X_train.shape[3]))
                    X_val = np.reshape(X_val, (X_val.shape[0]*X_val.shape[1],X_val.shape[2],X_val.shape[3]))

                    X_train = standardize_data(X_train)
                    X_val = standardize_data(X_val)

                    ys_train = y_train[:,1:]
                    ys_val = y_val[:,1:]

                    yt_train = y_train[:,0]
                    yt_val = y_val[:,0]

                    ys_train = np.reshape(ys_train,(ys_train.shape[0]*ys_train.shape[1]))
                    ys_val = np.reshape(ys_val,(ys_val.shape[0]*ys_val.shape[1]))

                    if(f==0):
                        model_eeg2code = build_eeg2code_model(n_channels)
                    
                        model_eeg2code, model_history = train_eeg2code(model_eeg2code, dataset,mode,model, X_train[...,np.newaxis], ys_train, X_val[...,np.newaxis], ys_val,n_subjects, n_classes, i, None)
                        print("Trained eeg2code left out subject {}, batch {}".format(i+1,f+1))
                        results[i+1]['history'] = model_history

                    del X_train, X_val


                with open('./datasets/epoched_data/{}_epoched_S{}.pickle'.format(dataset,i+1), 'rb') as handle:
                    data = pickle.load(handle)

                X = data['X']
                Ys = data['Ys']
                Yt = data['Yt'] 

                X_cv = np.array([]).reshape(0,100,126,60,8)
                Ys_cv = np.array([]).reshape(0,100,126)
                Yt_cv = np.array([]).reshape(0,100)
                for f in range(0,15):
                    X_fold = X[f::15]
                    Ys_fold = Ys[f::15]
                    Yt_fold = Yt[f::15]

                    X_cv = np.vstack((X_cv, X_fold[np.newaxis,...]))
                    Ys_cv = np.vstack((Ys_cv, Ys_fold[np.newaxis,...]))
                    Yt_cv = np.vstack((Yt_cv, Yt_fold[np.newaxis,...]))

                for fold in range(0,15):
                    results[i+1][fold+1] = {}
                    #print(i+1,fold+1)

                    X_test = X_cv[fold:fold+1][0]
                    Ys_test = Ys_cv[fold:fold+1][0]
                    Yt_test = Yt_cv[fold:fold+1][0]

                    X_test = standardize_epoched_data(X_test)

                    results_eval = evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test,Ys_test, Yt_test, n_subjects,n_classes,codebook)


                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']))
                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']),file=run_f)
                    results[i+1][fold+1]['category_accuracy'] = results_eval['category_accuracy']
                    results[i+1][fold+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                    results[i+1][fold+1]['ITR'] = results_eval['ITR']
                    #results[i+1][fold+1]['category_cm'] =  results_eval['category_cm']
                    results[i+1][fold+1]['recall'] = results_eval['recall']
                    results[i+1][fold+1]['precision'] = results_eval['precision']
                    results[i+1][fold+1]['f1_score'] = results_eval['f1_score']
                   # results[i+1][fold+1]['fpr'] = results_eval['fpr']
                    #results[i+1][fold+1]['tpr'] = results_eval['tpr']
                    #results[i+1][fold+1]['auc'] = results_eval['auc']
                    #results[i+1][fold+1]['variable_time_steps'] = results_eval['variable_time_steps']
                    #results[i+1][fold+1]['ITR_time_steps'] = results_eval['ITR_time_steps']   
                    #results[i+1][fold+1]['pred_time_step'] = results_eval['pred_time_step']

                del X_test, X_cv

            filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            warnings.warn("Unsupported mode")

    elif(dataset=='256_channel_cVEP'):
        results = {}
        n_subjects = 5
        n_classes = 36
        n_channels = 256
        codebook = np.load('./datasets/256_channel_cVEP/Scripts/codebook_36t.npy')[:n_classes]
        codes = np.moveaxis(codebook,1,0)

        if(mode=='within_subject'):
            for i in range(0,n_subjects):
                results[i+1] = {}
                with open('./datasets/epoched_data/{}_epoched_S{}.pickle'.format(dataset,i+1), 'rb') as handle:
                    data = pickle.load(handle)

                X = data['X']
                Ys = data['Ys']
                Yt = data['Yt'] 

                foldn = 15
                for fold in range(0,foldn):
                    results[i+1][fold+1] = {}
                    data = {}
                    #print("Training on subject {} fold {} ".format(i+1,fold+1))

                    X_train_f = X
                    Ys_train_f = Ys
                    Yt_train_f = Yt

                    X_train_f1 = X_train_f[:216]
                    X_train_f2 = X_train_f[216:]
                    X_trainf3 = np.concatenate((X_train_f1,X_train_f2), axis=-1)
            
                    Ys_train_f1 = Ys_train_f[:216]
                    Ys_train_f2 = Ys_train_f[216:]
                    Ys_train_f3 = np.concatenate((Ys_train_f1,Ys_train_f2), axis=-1)

                    Yt_train_f1 = Yt_train_f[:216]
                    Yt_train_f2 = Yt_train_f[216:]
                    Yt_train_f3 = np.concatenate((Yt_train_f1,Yt_train_f2), axis=-1)

                    Y_train_f = np.concatenate((Yt_train_f3,Ys_train_f3), axis=1)

                    del X_train_f1, X_train_f2, X_train_f

                    X_train_fold, X_val_fold, y_train_fold, y_val_fold  = train_test_split(X_trainf3, Y_train_f, 
                                                                                            test_size=0.35, 
                                                                                            stratify=Y_train_f[:,0], 
                                                                                            shuffle= True)

                    X_val_fold, X_test_fold, y_val_fold, y_test_fold = train_test_split(X_val_fold, y_val_fold, 
                                                                                            test_size=0.5, 
                                                                                            stratify=y_val_fold[:,0], 
                                                                                            shuffle= True)

                    X_train1 = X_train_fold[:,:,:,:256]
                    X_train2 = X_train_fold[:,:,:,256:512]

                    X_val1 = X_val_fold[:,:,:,:256]
                    X_val2 = X_val_fold[:,:,:,256:512]

                    X_test1 = X_test_fold[:,:,:,:256]
                    X_test2 = X_test_fold[:,:,:,256:512]

                    X_train = np.concatenate((X_train1,X_train2), axis=0)
                    X_val = np.concatenate((X_val1,X_val2), axis=0)
                    X_test = np.concatenate((X_test1,X_test2), axis=0)

                    del X_test1, X_test2, X_train1, X_train2, X_val1, X_val2, X_train_fold, X_val_fold, X_test_fold, X_trainf3

                    X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],X_train.shape[2],X_train.shape[3]))
                    X_val = np.reshape(X_val, (X_val.shape[0]*X_val.shape[1],X_val.shape[2],X_val.shape[3]))

                    X_train = standardize_data(X_train)
                    X_val = standardize_data(X_val)
                    X_test = standardize_epoched_data(X_test)

                    yt_train1 = y_train_fold[:,0]
                    yt_train2 = y_train_fold[:,1]
                    ys_train1 = y_train_fold[:,2:128]
                    ys_train2 = y_train_fold[:,128:254]

                    yt_test1 = y_test_fold[:,0]
                    yt_test2 = y_test_fold[:,1]
                    ys_test1 = y_test_fold[:,2:128]
                    ys_test2 = y_test_fold[:,128:254]

                    yt_val1 = y_val_fold[:,0]
                    yt_val2 = y_val_fold[:,1]
                    ys_val1 = y_val_fold[:,2:128]
                    ys_val2 = y_val_fold[:,128:254]

                    ys_train = np.concatenate((ys_train1,ys_train2), axis=0)
                    yt_train = np.concatenate((yt_train1,yt_train2), axis=0)

                    ys_test = np.concatenate((ys_test1,ys_test2), axis=0)
                    yt_test = np.concatenate((yt_test1,yt_test2), axis=0)

                    ys_val = np.concatenate((ys_val1,ys_val2), axis=0)
                    yt_val = np.concatenate((yt_val1,yt_val2), axis=0)

                    ys_train = np.reshape(ys_train,(ys_train.shape[0]*ys_train.shape[1]))
                    ys_val = np.reshape(ys_val,(ys_val.shape[0]*ys_val.shape[1]))

                    model_eeg2code = build_eeg2code_model(n_channels)
                    model_eeg2code, model_history = train_eeg2code(model_eeg2code,dataset,mode,model, X_train[...,np.newaxis], ys_train, X_val[...,np.newaxis], ys_val,n_subjects, n_classes, i, None)
                    results[i+1]['history'] = model_history

                    results_eval = evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test,ys_test, yt_test, n_subjects,n_classes,codebook)

                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']))
                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']),file=run_f)
                    results[i+1][fold+1]['category_accuracy'] = results_eval['category_accuracy']
                    results[i+1][fold+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                    results[i+1][fold+1]['ITR'] = results_eval['ITR']
                   # results[i+1][fold+1]['category_cm'] =  results_eval['category_cm']
                    results[i+1][fold+1]['recall'] = results_eval['recall']
                    results[i+1][fold+1]['precision'] = results_eval['precision']
                    results[i+1][fold+1]['f1_score'] = results_eval['f1_score']
                    #results[i+1][fold+1]['fpr'] = results_eval['fpr']
                    #results[i+1][fold+1]['tpr'] = results_eval['tpr']
                    #results[i+1][fold+1]['auc'] = results_eval['auc']
                    #results[i+1][fold+1]['variable_time_steps'] = results_eval['variable_time_steps']
                    #results[i+1][fold+1]['ITR_time_steps'] = results_eval['ITR_time_steps']   
                    #results[i+1][fold+1]['pred_time_step'] = results_eval['pred_time_step']

                    del X_train, X_val, X_test


            filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


        elif(mode=='loso_subject'):
            for i in range(0,n_subjects):
                results[i+1] = {}
                X = np.array([]).reshape(0,216,126,60,256)
                Ys = np.array([]).reshape(0,216,126)
                Yt = np.array([]).reshape(0,216,1)

                for j in range(0,n_subjects):
                    if(j!=i):
                        with open('./datasets/epoched_data/{}_epoched_S{}.pickle'.format(dataset,j+1), 'rb') as handle:
                            data = pickle.load(handle)
                        
                        X_subj = data['X'][:216]
                        Ys_subj = data['Ys'][:216]
                        Yt_subj = data['Yt'][:216]

                        X = np.vstack((X, X_subj[np.newaxis,...]))
                        Ys = np.vstack((Ys, Ys_subj[np.newaxis,...]))
                        Yt = np.vstack((Yt, Yt_subj[np.newaxis,...]))

                #print("Leaving out subject ",i+1)
                X_new = np.reshape(X, (X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4]))
                Ys_new = np.reshape(Ys, (Ys.shape[0]*Ys.shape[1],Ys.shape[2]))
                Yt_new = np.reshape(Yt, (Yt.shape[0]*Yt.shape[1],Yt.shape[2]))   

                Y_new = np.concatenate((Yt_new,Ys_new), axis=1)

                X_train, X_val, y_train, y_val = train_test_split(X_new, Y_new, test_size=0.2, 
                                                                  stratify=Y_new[:,0], shuffle= True)

                del X_new
                X_train = np.reshape(X_train, (X_train.shape[0]*X_train.shape[1],X_train.shape[2],X_train.shape[3]))
                X_val = np.reshape(X_val, (X_val.shape[0]*X_val.shape[1],X_val.shape[2],X_val.shape[3]))

                X_train = standardize_data(X_train)
                X_val = standardize_data(X_val)

                ys_train = y_train[:,1:]
                ys_val = y_val[:,1:]

                yt_train = y_train[:,0]
                yt_val = y_val[:,0]

                ys_train = np.reshape(ys_train,(ys_train.shape[0]*ys_train.shape[1]))
                ys_val = np.reshape(ys_val,(ys_val.shape[0]*ys_val.shape[1]))

                if(j==0):
                    model_eeg2code = build_eeg2code_model(n_channels)

                model_eeg2code, model_history = train_eeg2code(model_eeg2code,dataset,mode,model, X_train[...,np.newaxis], ys_train, X_val[...,np.newaxis], ys_val,n_subjects, n_classes, i, None)
                results[i+1]['history'] = model_history

                del X_train, X_val
                
                with open('./datasets/{}_epoched_S{}.pickle'.format(dataset,i+1), 'rb') as handle:
                    data = pickle.load(handle)

                X = data['X'][:216]
                Ys = data['Ys'][:216]
                Yt = data['Yt'][:216] 

                for f in range(0,15):
                    results[i+1][fold+1] = {}
                    X_fold = X
                    Ys_fold = Ys
                    Yt_fold = Yt

                    Y_fold = np.concatenate((Yt_fold,Ys_fold), axis=1)

                    X_test_fold, _, y_test_fold, _ = train_test_split(X_fold, Y_fold, test_size=0.8, 
                                                                      stratify=Y_fold[:,0], shuffle= True)
                    X_test = standardize_data(X_test_fold)

                    yt_test = y_test_fold[:,0]
                    ys_test = y_test_fold[:,1:]

                    X_test = standardize_epoched_data(X_test)

                    results_eval = evaluate_eeg2code(model_eeg2code, dataset,mode,model,X_test,ys_test, yt_test, n_subjects,n_classes,codebook)

                    del X_fold, X_test_fold, X_test

                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']))
                    print("Subject {} fold {} category_accuracy: {}".format(i+1,fold+1,results_eval['category_accuracy']),file=run_f)
                    results[i+1][fold+1]['category_accuracy'] = results_eval['category_accuracy']
                    results[i+1][fold+1]['sequence_accuracy'] = results_eval['sequence_accuracy']
                    results[i+1][fold+1]['ITR'] = results_eval['ITR']
                    #results[i+1][fold+1]['category_cm'] =  results_eval['category_cm']
                    results[i+1][fold+1]['recall'] = results_eval['recall']
                    results[i+1][fold+1]['precision'] = results_eval['precision']
                    results[i+1][fold+1]['f1_score'] = results_eval['f1_score']
                    #results[i+1][fold+1]['fpr'] = results_eval['fpr']
                    #results[i+1][fold+1]['tpr'] = results_eval['tpr']
                    #results[i+1][fold+1]['auc'] = results_eval['auc']
                    #results[i+1][fold+1]['variable_time_steps'] = results_eval['variable_time_steps']
                    #results[i+1][fold+1]['ITR_time_steps'] = results_eval['ITR_time_steps']   
                    #results[i+1][fold+1]['pred_time_step'] = results_eval['pred_time_step']


            filename = './results/{}/{}/{}/{}_{}.pickle'.format(model,dataset,mode,model,mode)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


        else:
            warnings.warn("Unsupported mode")

    else:
        warnings.warn("Unsupported dataset")


datasets = ['8_channel_cVEP','256_channel_cVEP']
modes = ['loso_subject','cross_subject'] #'within_subject'
model = "eeg2code"

for mode in modes:
    for dataset in datasets: 
        print('\n------Running {} for dataset {} in mode {}-----\n'.format(model, dataset, mode))
        run_eeg2code(dataset, mode, model)