import os
import glob
import copy
import time
import math
import pickle
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import warnings
from data_preprocessing import*
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.np_utils import to_categorical

def epoch_data(X, Ys, Yt, n_subjects, n_classes):
    
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
    
    return data_X_trial

def load_raw_dataset(data_path, label_path, sfreq, dfreq, num_chans):

    if num_chans == 256:
        trials_256ch = 108
        samps_256ch = 1008

        codebook36 = np.load(label_path)
        all_subject_data = np.array([]).reshape(0,trials_256ch,samps_256ch,num_chans)
        all_label_data = np.array([]).reshape(0,trials_256ch)
        all_stim_seq_data = np.array([]).reshape(0,trials_256ch,126)
        for i, subject in enumerate(data_path):
            
            data = scipy.io.loadmat(subject)
            subject_data = data['data'][0][0][0]
            subject_data = np.moveaxis(subject_data,2,0)
            subject_data = np.moveaxis(subject_data,2,1)
            label_data = data['data'][0][0][1][0]-1
            #label_data = np.repeat(label_data,2,axis=0)
                
            samps = 1008
            resampled_trials = np.array([]).reshape(0,samps,subject_data.shape[2])
            for trial in range(len(subject_data)):
                resampled_data = scipy.signal.resample(subject_data[trial], samps)
                resampled_trials = np.vstack((resampled_trials, resampled_data[np.newaxis, ...]))
            
            #subject_data1 = np.reshape(resampled_trials,(trials_256ch,samps_256ch,num_chans))
            all_subject_data = np.vstack((all_subject_data, resampled_trials[np.newaxis,...]))
            all_label_data = np.vstack((all_label_data, label_data[np.newaxis,...]))
            stim_seq_data = codebook36[all_label_data[0].astype('int'),:]
            all_stim_seq_data = np.vstack((all_stim_seq_data, stim_seq_data[np.newaxis,...]))

            print("Loaded data from Subject{}".format(i+1))

        all_data={}
        all_data['X'] = all_subject_data
        all_data['Yt'] = all_label_data
        all_data['Ys'] = all_stim_seq_data
        return all_data


    elif num_chans == 8:
        trials_8ch = 1500
        samps_8ch = 504
        subject_data = []

        subj_labels = np.array([]).reshape(0,20)

        mat = scipy.io.loadmat(label_path)
        codes = mat['codes'].astype('float32')
        codebook20 = np.moveaxis(codes,1,0)
        subject_no_stim_data = np.array([]).reshape(0,num_chans,512)
        subject_no_stim_data1 = np.array([]).reshape(0,90,240,num_chans)
        for i, subject in enumerate(data_path):
            files = os.listdir(subject)
            subject_block_data = []
            for file in files:
                if file.startswith('block'):   
                    data_files_path = os.path.join(subject, file)
                    data_files = os.listdir(data_files_path)

                    # Reading gdf data
                    for data_file in data_files:

                        if data_file == 'trainlabels.mat':
                            data_file_path = os.path.join(data_files_path, data_file)
                            data_dict = {}
                            f = h5py.File(data_file_path, 'r')
                            for k, v in f.items():
                                data_dict[k] = np.array(v)
                            f.close()
                            data_array = data_dict['v'].astype('int')
                            data_array = np.reshape(data_array, (len(data_array)))
                            subj_labels = np.vstack((subj_labels,data_array[np.newaxis,...]))



                        elif data_file.endswith('.gdf'):
                            data_file_path = os.path.join(data_files_path, data_file)
                            raw_mne_data = mne.io.read_raw_gdf(data_file_path, verbose=0)
                            raw_data = raw_mne_data.get_data()

                            # Slicing data
                            onset_instances = []
                            for instance in range(0,raw_data.shape[1]):
                                status = raw_data[0,instance]
                                if status == 257:
                                    onset_instances.append(instance)

                            onset_instances.append(len(raw_data[1]))


                            trial_time = 31.5
                            code_time = 2.1

                            code_size = int(sfreq*code_time)+1
                            target_size = int(sfreq*trial_time)

                            all_trials = np.array([]).reshape(0,target_size,num_chans)
                            single_trial = np.array([]).reshape(0,num_chans)
                            counter = 0
                            excess_counter = 0
                            for q in range(len(onset_instances)-1):
                                num_datapoints = onset_instances[q+1] - onset_instances[q]
                                counter += 1

                                data = raw_data[1:,onset_instances[q]:onset_instances[q+1]]
                                if (data.shape[1]<700 and data.shape[1]>=512):
                                        f = data.shape[1]-512
                                        no_stim_data = data[:,f:]
                                        subject_no_stim_data = np.vstack((subject_no_stim_data, no_stim_data[np.newaxis, ...]))

                                if (counter<=15): # 2.1s of data
                                    data = raw_data[1:,onset_instances[q]:onset_instances[q+1]]

                                    #print(data.shape)
                                    if (len(data[0])>1600):
                                        if excess_counter<3:
                                            data = data[:,:1076]
                                        else:
                                            data = data[:,:1075]
                                        single_trial = np.vstack((single_trial,data.T))
                                        all_trials = np.vstack((all_trials,single_trial[np.newaxis, ...]))
                                        single_trial = np.array([]).reshape(0,num_chans)
                                        counter = 0
                                        excess_counter = 0
                                        continue

                                    if (len(data[0])>=code_size and excess_counter<3):
                                        data = data[:,:1076]
                                        excess_counter+=1
                                    else:  
                                        data = data[:,:1075]

                                    single_trial = np.vstack((single_trial,data.T))

                                elif (counter>15): # 1 second cue
                                    current_size = len(single_trial)

                                    if current_size < target_size:
                                        required_instances = target_size - current_size
                                        cue_data = raw_data[1:,onset_instances[i]:onset_instances[i]+required_instances]
                                        single_trial = np.vstack((single_trial,cue_data.T))
                                    elif current_size > target_size:
                                        single_trial = single_trial[:target_size,:]

                                    all_trials = np.vstack((all_trials,single_trial[np.newaxis, ...]))
                                    single_trial = np.array([]).reshape(0,num_chans)
                                    counter = 0
                                    excess_counter = 0

                            samps = 7560
                            resampled_trials = np.array([]).reshape(0,samps,num_chans)
                            for trial in range(len(all_trials)):
                                resampled_data = scipy.signal.resample(all_trials[trial], samps)
                                resampled_trials = np.vstack((resampled_trials, resampled_data[np.newaxis, ...]))

                            subject_block_data.append(resampled_trials)

            block_data1 = np.array(subject_no_stim_data)
            block_data1 = np.moveaxis(block_data1,2,1)

            samps1 = 240
            num_chans = 8
            n_trials1 = 90
            resampled_trials1 = np.array([]).reshape(0,samps1,num_chans)
            for trial1 in range(block_data1.shape[0]):
                resampled_data1 = scipy.signal.resample(block_data1[trial1], samps1)
                resampled_trials1 = np.vstack((resampled_trials1, resampled_data1[np.newaxis, ...]))
            resampled_trials1 = resampled_trials1[:n_trials1,:,:]

            subject_no_stim_data1 = np.vstack((subject_no_stim_data1,resampled_trials1[np.newaxis, ...]))

            subject_data.append(np.array(subject_block_data))
            print("Loaded data from Subject{}".format(i+1))


        subject_no_stim_data2 = np.zeros((30,90,7560,8))
        subject_no_stim_data2[:,:,:240,:] = subject_no_stim_data1

        all_subject_data = np.array(subject_data)

        all_subject_data = np.reshape(all_subject_data,(all_subject_data.shape[0],5,20,7560,8))
        all_subject_data = np.reshape(all_subject_data,(all_subject_data.shape[0],100,7560,8))
        all_subject_data = np.concatenate((all_subject_data,subject_no_stim_data2[:,:75,:,:]),axis=1)


        subj_labels = np.reshape(subj_labels,(all_subject_data.shape[0],100))-1
        subject_no_stim_labels = np.ones((all_subject_data.shape[0],75))*20
        all_subj_labels = np.concatenate((subj_labels,subject_no_stim_labels),axis=1)


        subj_code_labels = codebook20[subj_labels[:,:].astype('int'),:]
        subject_no_stim_code_labels = np.zeros((30,75,126))
        all_subj_stim_seq = np.concatenate((subj_code_labels,subject_no_stim_code_labels),axis=1)

        all_data={}
        all_data['X'] = all_subject_data
        all_data['Yt'] = all_subj_labels
        all_data['Ys'] = all_subj_stim_seq

        return all_data

    else:
        warnings.warn("Unsupported Dataset")
        

def load_preprocessed_data_cv(dataset, mode, model):
    if dataset == '8_channel_cVEP':
        n_subjects = 30
        
        # Loading data
        with open('./datasets/8_channel_cVEP.pickle', 'rb') as handle:
            data_8ch = pickle.load(handle)
            X = data_8ch['X']
            Yt = data_8ch['Yt']
            Ys = data_8ch['Ys']

        #print(X.shape)
        # Preprocessing data
        low_cutoff = 2
        high_cutoff = 30
        sfreq = 240
        X = bandpass_filter_data(X, low_cutoff, high_cutoff, sfreq)
        if (model == 'eeg2code'):
            n_classes = 20

            # X = X[:,:1500,:,:]
            # Yt = Yt[:,:1500]
            # Ys = Ys[:,:1500]

            # data = {}
            # for subj in range(0,n_subjects):
            #     X_epoched = epoch_data(X[subj], Ys[subj], Yt[subj], n_subjects, n_classes)
            #     data ['X'] = X_epoched
            #     data['Ys'] = Ys[subj]
            #     data['Yt'] = Yt[subj]

            #     filename = './datasets/epoched_data/8_channel_cVEP_epoched_S{}.pickle'.format(subj+1)
            #     os.makedirs(os.path.dirname(filename), exist_ok=True)
            #     with open(filename, 'wb') as handle:
            #         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #     print("Epoched subject {}  for 8_channel_cVEP".format(subj+1))

        else:
            if (mode=='within_subject'):
                n_folds = 5
                fold_steps = 20
                fold_c_steps = 15
                subjects = np.arange(1,n_subjects+1)
                for i in range(0,n_subjects):
                    X_cv = np.array([]).reshape(0,315,504,8)
                    Ys_cv = np.array([]).reshape(0,315,126)
                    Yt_cv = np.array([]).reshape(0,315)

                    for f in range(0,n_folds):
                        fold_s = int(f*fold_steps)
                        fold_c = int(f*fold_c_steps)

                        X_fold = X[i][:100][fold_s:fold_s+fold_steps]
                        Ys_fold = Ys[i][:100][fold_s:fold_s+fold_steps]
                        Yt_fold = Yt[i][:100][fold_s:fold_s+fold_steps]

                        Ys_fold = np.repeat(Ys_fold,15,axis=0)
                        Yt_fold = np.repeat(Yt_fold,15,axis=0)

                        X_nostim_fold = X[i][100:][fold_c:fold_c+fold_c_steps]
                        Ys_nostim_fold = Ys[i][100:][fold_c:fold_c+fold_c_steps]
                        Yt_nostim_fold = Yt[i][100:][fold_c:fold_c+fold_c_steps]

                        X_fold = np.reshape(X_fold,(20*15,504,8))
                        X_nostim_fold = X_nostim_fold[:,:504,:]

                        X_f = np.concatenate((X_fold,X_nostim_fold),axis=0)
                        Ys_f = np.concatenate((Ys_fold,Ys_nostim_fold),axis=0)
                        Yt_f = np.concatenate((Yt_fold,Yt_nostim_fold),axis=0)

                        X_cv = np.vstack((X_cv, X_f[np.newaxis,...]))
                        Ys_cv = np.vstack((Ys_cv, Ys_f[np.newaxis,...]))
                        Yt_cv = np.vstack((Yt_cv, Yt_f[np.newaxis,...]))

                    for fold in range(0,n_folds):
                        #print(i+1,fold+1)
                        data = {}
                        #print("Training on subject {} fold {} ".format(i+1,fold+1))
                        X_train_f = np.concatenate((X_cv[0:fold], X_cv[fold+1:n_folds]))
                        Ys_train_f = np.concatenate((Ys_cv[0:fold], Ys_cv[fold+1:n_folds]))
                        Yt_train_f = np.concatenate((Yt_cv[0:fold], Yt_cv[fold+1:n_folds]))

                        X_train_folds = np.reshape(X_train_f, (X_train_f.shape[0]*X_train_f.shape[1],504,8))
                        Ys_train_folds = np.reshape(Ys_train_f, (X_train_f.shape[0]*X_train_f.shape[1],126))
                        Yt_train_folds = np.reshape(Yt_train_f, (X_train_f.shape[0]*X_train_f.shape[1]))

                        X_test = X_cv[fold:fold+1][0]
                        Ys_test = Ys_cv[fold:fold+1][0]
                        Yt_test = Yt_cv[fold:fold+1][0]


                        Y_train_folds = np.concatenate((Yt_train_folds[..., np.newaxis],Ys_train_folds), axis=1)

                        X_train, X_val, y_train, y_val = train_test_split(X_train_folds, Y_train_folds, test_size=0.25, 
                                                                          stratify=Yt_train_folds, shuffle= True)

                        X_train = standardize_data(X_train)
                        X_val = standardize_data(X_val)
                        X_test = standardize_data(X_test)

                        ys_train = y_train[:,1:]
                        ys_val = y_val[:,1:]

                        yt_train = y_train[:,0]
                        yt_val = y_val[:,0]

                        yt_train = to_categorical(yt_train)
                        yt_val = to_categorical(yt_val)
                        yt_test = to_categorical(Yt_test)
                            
                            
                        data['X_train'] = X_train
                        data['X_val'] = X_val
                        data['X_test'] = X_test
                        data['ys_train'] = ys_train
                        data['ys_val'] = ys_val
                        data['ys_test'] = Ys_test
                        data['yt_train'] = yt_train
                        data['yt_val'] = yt_val
                        data['yt_test'] = yt_test

                        filename = './datasets/8_channel_cVEP_within_subject/S{}_f{}.pickle'.format(i+1,fold+1)
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        with open(filename, 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                        del data
                    del X_train, X_val, X_test, X_fold, X_f, X_cv

            elif(mode=='loso_subject'):
                n_folds = 5
                fold_steps = 20
                fold_c_steps = 15
                subjects = np.arange(1,31)
                for i in range(0,n_subjects):
                    data = {}
                    #print("Leaving out subject ",i+1)
                    X_new = np.concatenate((X[0:i], X[i+1:30]))
                    Ys_new = np.concatenate((Ys[0:i], Ys[i+1:30]))
                    Yt_new = np.concatenate((Yt[0:i], Yt[i+1:30]))
                    
                    X_new_c = np.reshape(X_new[:,:100],(29,100*15,504,8))
                    Ys_new_c = Ys_new[:,:100]
                    Yt_new_c = Yt_new[:,:100]

                    Ys_new_c = np.repeat(Ys_new_c,15,axis=1)
                    Yt_new_c = np.repeat(Yt_new_c,15,axis=1)

                    X_new_nc = np.reshape(X_new[:,100:,:504,:],(29,75,504,8))
                    Ys_new_nc = Ys_new[:,100:]
                    Yt_new_nc = Yt_new[:,100:]

                    X_new = np.concatenate((X_new_c, X_new_nc),axis=1)
                    Ys_new = np.concatenate((Ys_new_c, Ys_new_nc),axis=1)
                    Yt_new = np.concatenate((Yt_new_c, Yt_new_nc),axis=1)

                    Y_new = np.concatenate((Yt_new[..., np.newaxis],Ys_new), axis=2)

                    X_new1 = np.moveaxis(X_new,1,0)
                    Y_new1 = np.moveaxis(Y_new,1,0)
                    X_train, X_val, y_train, y_val = train_test_split(X_new1, Y_new1, test_size=0.25, stratify=Y_new1[:,:,0], shuffle= True)

                    train_n = X_train.shape[0]
                    train_s = X_train.shape[1]
                    val_n = X_val.shape[0]
                    val_s = X_val.shape[1]

                    X_train = np.moveaxis(X_train,1,0)
                    X_train = np.reshape(X_train, (train_n*train_s,504,8))
                    X_val = np.moveaxis(X_val,1,0)
                    X_val = np.reshape(X_val, (val_n*val_s,504,8))
                    y_train = np.moveaxis(y_train,1,0)
                    y_train = np.reshape(y_train, (train_n*train_s,127))
                    y_val = np.moveaxis(y_val,1,0)
                    y_val = np.reshape(y_val, (val_n*val_s,127))

                    X_train = standardize_data(X_train)
                    X_val = standardize_data(X_val)

                    ys_train = y_train[:,1:]
                    ys_val = y_val[:,1:]

                    yt_train = y_train[:,0]
                    yt_val = y_val[:,0]

                    yt_train = to_categorical(yt_train)
                    yt_val = to_categorical(yt_val)
                    
                 
                    data['X_train'] = X_train
                    data['ys_train'] = ys_train
                    data['yt_train'] = yt_train
                    data['X_val'] = X_val
                    data['ys_val'] = ys_val
                    data['yt_val'] = yt_val

                    filename = './datasets/8_channel_cVEP_loso_subject/S{}.pickle'.format(i+1)
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                    del data, X_train, X_val, X_new, X_new1
                    
                    X_cv = np.array([]).reshape(0,315,504,8)
                    Ys_cv = np.array([]).reshape(0,315,126)
                    Yt_cv = np.array([]).reshape(0,315)

                    for f in range(0,n_folds):
                        fold_s = int(f*fold_steps)
                        fold_c = int(f*fold_c_steps)

                        X_fold = X[i][:100][fold_s:fold_s+fold_steps]
                        Ys_fold = Ys[i][:100][fold_s:fold_s+fold_steps]
                        Yt_fold = Yt[i][:100][fold_s:fold_s+fold_steps]

                        Ys_fold = np.repeat(Ys_fold,15,axis=0)
                        Yt_fold = np.repeat(Yt_fold,15,axis=0)

                        X_nostim_fold = X[i][100:][fold_c:fold_c+fold_c_steps]
                        Ys_nostim_fold = Ys[i][100:][fold_c:fold_c+fold_c_steps]
                        Yt_nostim_fold = Yt[i][100:][fold_c:fold_c+fold_c_steps]

                        X_fold = np.reshape(X_fold,(20*15,504,8))
                        X_nostim_fold = X_nostim_fold[:,:504,:]

                        X_f = np.concatenate((X_fold,X_nostim_fold),axis=0)
                        Ys_f = np.concatenate((Ys_fold,Ys_nostim_fold),axis=0)
                        Yt_f = np.concatenate((Yt_fold,Yt_nostim_fold),axis=0)

                        X_cv = np.vstack((X_cv, X_f[np.newaxis,...]))
                        Ys_cv = np.vstack((Ys_cv, Ys_f[np.newaxis,...]))
                        Yt_cv = np.vstack((Yt_cv, Yt_f[np.newaxis,...]))


                    for fold in range(0,n_folds):
                        #print(i+1,fold+1)
                        data = {}
                        X_test = X_cv[fold:fold+1][0]
                        Ys_test = Ys_cv[fold:fold+1][0]
                        Yt_test = Yt_cv[fold:fold+1][0]

                        X_test = standardize_data(X_test)
                        yt_test = to_categorical(Yt_test)
                 
                        data['X_test'] = X_test
                        data['ys_test'] = Ys_test
                        data['yt_test'] = yt_test

                        filename = './datasets/8_channel_cVEP_loso_subject/S{}_f{}.pickle'.format(i+1,fold+1)
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        with open(filename, 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    del data, X_fold, X_f, X_cv, X_test
            
            else:
                warnings.warn("Unsupported mode")   
    
    elif (dataset == '256_channel_cVEP'):
        n_subjects = 5
        n_classes = 36
        # Loading data
        with open('./datasets/256_channel_cVEP.pickle', 'rb') as handle:
            data_256ch = pickle.load(handle)
            X = data_256ch['X']
            Yt = data_256ch['Yt']
            Ys = data_256ch['Ys']
        
        # Preprocessing data
        low_cutoff = 2
        high_cutoff = 30
        sfreq = 240
        X = bandpass_filter_data(X, low_cutoff, high_cutoff, sfreq)

        if (model == 'eeg2code'):
            # X, rejected_chans = remove_bad_channels(X)
            # X, Ys, Yt = augment_data_trial(X, Ys, Yt)
            # n_subjects = 5
            n_classes = 36

            # data = {}
            # for subj in range(0,n_subjects):
            #     X_epoched= epoch_data(X[subj], Ys[subj], Yt[subj], n_subjects, n_classes)

            #     data ['X'] = X_epoched
            #     data['Ys'] = Ys[subj]
            #     data['Yt'] = Yt[subj]

            #     filename = './datasets/epoched_data/256_channel_cVEP_epoched_S{}.pickle'.format(subj+1)
            #     os.makedirs(os.path.dirname(filename), exist_ok=True)
            #     with open(filename, 'wb') as handle:
            #         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #     print("Epoched subject {}  for 256_channel_cVEP".format(subj+1))
        
        else:

            if (mode=='within_subject'):
                #X, Ys, Yt = augment_data_chan(X, Ys, Yt)
                n_folds = 3
                fold_steps = 36

                for i in range(0,n_subjects):
                    X_cv = np.array([]).reshape(0,72,504,256)
                    Ys_cv = np.array([]).reshape(0,72,126)
                    Yt_cv = np.array([]).reshape(0,72)

                    for f in range(0,n_folds):
                        fold_s = int(f*fold_steps)

                        X_fold = X[i][fold_s:fold_s+fold_steps]
                        Ys_fold = Ys[i][fold_s:fold_s+fold_steps]
                        Yt_fold = Yt[i][fold_s:fold_s+fold_steps]

                        Ys_fold = np.repeat(Ys_fold,2,axis=0)
                        Yt_fold = np.repeat(Yt_fold,2,axis=0)

                        X_fold = np.reshape(X_fold,(36*2,504,256))

                        X_cv = np.vstack((X_cv, X_fold[np.newaxis,...]))
                        Ys_cv = np.vstack((Ys_cv, Ys_fold[np.newaxis,...]))
                        Yt_cv = np.vstack((Yt_cv, Yt_fold[np.newaxis,...]))

                    for fold in range(0,n_folds):
                        #print(i+1,fold+1)
                        data = {}
                        #print("Training on subject {} fold {} ".format(i+1,fold+1))
                        X_train_f = np.concatenate((X_cv[0:fold], X_cv[fold+1:n_folds]))
                        Ys_train_f = np.concatenate((Ys_cv[0:fold], Ys_cv[fold+1:n_folds]))
                        Yt_train_f = np.concatenate((Yt_cv[0:fold], Yt_cv[fold+1:n_folds]))

                        X_train_folds = np.reshape(X_train_f, (X_train_f.shape[0]*X_train_f.shape[1],504,256))
                        Ys_train_folds = np.reshape(Ys_train_f, (X_train_f.shape[0]*X_train_f.shape[1],126))
                        Yt_train_folds = np.reshape(Yt_train_f, (X_train_f.shape[0]*X_train_f.shape[1]))

                        X_test = X_cv[fold:fold+1][0]
                        Ys_test = Ys_cv[fold:fold+1][0]
                        Yt_test = Yt_cv[fold:fold+1][0]


                        Y_train_folds = np.concatenate((Yt_train_folds[..., np.newaxis],Ys_train_folds), axis=1)

                        X_train, X_val, y_train, y_val = train_test_split(X_train_folds, Y_train_folds, test_size=0.25, 
                                                                          stratify=Yt_train_folds, shuffle= True)

                        X_train = standardize_data(X_train)
                        X_val = standardize_data(X_val)
                        X_test = standardize_data(X_test)

                        ys_train = y_train[:,1:]
                        ys_val = y_val[:,1:]

                        yt_train = y_train[:,0]
                        yt_val = y_val[:,0]
                        
                        #X_train = remove_bad_channels(X_train)
                        X_train, ys_train, yt_train = augment_data(X_train, ys_train, yt_train)

                        yt_train = to_categorical(yt_train)
                        yt_val = to_categorical(yt_val)
                        yt_test = to_categorical(Yt_test)

                        data['X_train'] = X_train
                        data['ys_train'] = ys_train
                        data['yt_train'] = yt_train
                        data['X_val'] = X_val
                        data['ys_val'] = ys_val
                        data['yt_val'] = yt_val
                        data['X_test'] = X_test
                        data['ys_test'] = Ys_test
                        data['yt_test'] = yt_test
                        
                        filename = './datasets/256_channel_cVEP_within_subject/S{}_f{}.pickle'.format(i+1,fold+1)
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        with open(filename, 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                        del data, X_train, X_test
                
            elif(mode=='loso_subject'):
                n_folds = 3
                fold_steps = 36
                for i in range(0,n_subjects):
                    data = {}
                    #print("Leaving out subject ",i+1)
                    X_new = np.concatenate((X[0:i], X[i+1:30]))
                    Ys_new = np.concatenate((Ys[0:i], Ys[i+1:30]))
                    Yt_new = np.concatenate((Yt[0:i], Yt[i+1:30]))
                    
                    X_new = np.reshape(X_new,(4,108*2,504,256))
                    Ys_new = np.repeat(Ys_new,2,axis=1)
                    Yt_new = np.repeat(Yt_new,2,axis=1)

                    Y_new = np.concatenate((Yt_new[..., np.newaxis],Ys_new), axis=2)

                    X_new1 = np.moveaxis(X_new,1,0)
                    Y_new1 = np.moveaxis(Y_new,1,0)
 
                    X_train, X_val, y_train, y_val = train_test_split(X_new1, Y_new1, test_size=0.25, stratify=Y_new1[:,:,0], shuffle= True)

                    train_n = X_train.shape[0]
                    train_s = X_train.shape[1]
                    val_n = X_val.shape[0]
                    val_s = X_val.shape[1]

                    X_train = np.moveaxis(X_train,1,0)
                    X_train = np.reshape(X_train, (train_n*train_s,504,256))
                    X_val = np.moveaxis(X_val,1,0)
                    X_val = np.reshape(X_val, (val_n*val_s,504,256))
                    y_train = np.moveaxis(y_train,1,0)
                    y_train = np.reshape(y_train, (train_n*train_s,127))
                    y_val = np.moveaxis(y_val,1,0)
                    y_val = np.reshape(y_val, (val_n*val_s,127))

                    X_train = standardize_data(X_train)
                    X_val = standardize_data(X_val)


                    ys_train = y_train[:,1:]
                    ys_val = y_val[:,1:]

                    yt_train = y_train[:,0]
                    yt_val = y_val[:,0]

                    X_train, ys_train, yt_train = augment_data(X_train, ys_train, yt_train)

                    yt_train = to_categorical(yt_train)
                    yt_val = to_categorical(yt_val)


                    data['X_train'] = X_train
                    data['ys_train'] = ys_train
                    data['yt_train'] = yt_train
                    data['X_val'] = X_val
                    data['ys_val'] = ys_val
                    data['yt_val'] = yt_val

                    filename = './datasets/256_channel_cVEP_loso_subject/S{}.pickle'.format(i+1)
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'wb') as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    del data, X_train, X_val, X_new, X_new1

                    X_cv = np.array([]).reshape(0,72,504,256)
                    Ys_cv = np.array([]).reshape(0,72,126)
                    Yt_cv = np.array([]).reshape(0,72)

                    for f in range(0,n_folds):
                        fold_s = int(f*fold_steps)

                        X_fold = X[i][fold_s:fold_s+fold_steps]
                        Ys_fold = Ys[i][fold_s:fold_s+fold_steps]
                        Yt_fold = Yt[i][fold_s:fold_s+fold_steps]

                        Ys_fold = np.repeat(Ys_fold,2,axis=0)
                        Yt_fold = np.repeat(Yt_fold,2,axis=0)

                        X_fold = np.reshape(X_fold,(36*2,504,256))

                        X_cv = np.vstack((X_cv, X_fold[np.newaxis,...]))
                        Ys_cv = np.vstack((Ys_cv, Ys_fold[np.newaxis,...]))
                        Yt_cv = np.vstack((Yt_cv, Yt_fold[np.newaxis,...]))


                    for fold in range(0,n_folds):
                        #print(i+1,fold+1)
                        data = {}
                        X_test = X_cv[fold:fold+1][0]
                        Ys_test = Ys_cv[fold:fold+1][0]
                        Yt_test = Yt_cv[fold:fold+1][0]

                        X_test = standardize_data(X_test)
                        yt_test = to_categorical(Yt_test)
                 
                        data['X_test'] = X_test
                        data['ys_test'] = Ys_test
                        data['yt_test'] = yt_test

                        filename = './datasets/256_channel_cVEP_loso_subject/S{}_f{}.pickle'.format(i+1,fold+1)
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        with open(filename, 'wb') as handle:
                            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                warnings.warn("Unsupported mode")
                
    else:
        warnings.warn("Unsupported datset")

def load_data(mode,dataset,model,i,fold):
    if (mode=='within_subject'):
        with open('./datasets/{}_{}/S{}_f{}.pickle'.format(dataset,mode,i+1,fold+1), 'rb') as handle:
            data = pickle.load(handle)

            if (dataset=='8_channel_cVEP'):
                
                X_train = data['X_train']
                X_val = data['X_val']
                X_test = data['X_test']

                ys_train = data['ys_train']
                ys_val = data['ys_val']
                ys_test = data['ys_test']

                yt_train = data['yt_train']
                yt_val = data['yt_val']
                yt_test = data['yt_test']


                if (model=='cca'):
                    X_val = signal.resample(data['X_val'], 126, axis=1) 


                    Yt_train = np.argmax(yt_train, axis=1)
                    X_train = X_train[Yt_train!=20]
                    ys_train = ys_train[Yt_train!=20]
                    yt_train = yt_train[Yt_train!=20][:,:20]
                    yt_train = np.argmax(yt_train, axis=1)

                    Yt_val = np.argmax(yt_val, axis=1)
                    X_val = X_val[Yt_val!=20]
                    ys_val = ys_val[Yt_val!=20]
                    yt_val = yt_val[Yt_val!=20][:,:20]
                    yt_val = np.argmax(yt_val, axis=1)

                    Yt_test = np.argmax(yt_test, axis=1)
                    X_test = X_test[Yt_test!=20]
                    ys_test = ys_test[Yt_test!=20]
                    yt_test = yt_test[Yt_test!=20][:,:20]
                    yt_test = np.argmax(yt_test, axis=1)

            elif(dataset=='256_channel_cVEP'):

                X_train = data['X_train']
                X_val= data['X_val']
                X_test = data['X_test']

                ys_train = data['ys_train']
                ys_val = data['ys_val']
                ys_test = data['ys_test']

                yt_train = data['yt_train']
                yt_val = data['yt_val']
                yt_test = data['yt_test']

                if (model=='cca'):
                    X_train = signal.resample(data['X_train'], 126, axis=1)
                    X_val = signal.resample(data['X_val'], 126, axis=1)
                    X_test = signal.resample(data['X_test'], 126, axis=1) 

                    yt_train = np.argmax(yt_train, axis=1)
                    yt_val= np.argmax(yt_val, axis=1)
                    yt_test = np.argmax(yt_test, axis=1)

            else:
                warnings.warn("Unsupported mode")

    elif(mode=='loso_subject'):
        with open('./datasets/{}_{}/S{}_f{}.pickle'.format(dataset,mode,i+1,fold+1), 'rb') as handle:
            data = pickle.load(handle)


            X_test = data['X_test']
            ys_test = data['ys_test']
            yt_test = data['yt_test']

            if (dataset=='8_channel_cVEP' and model=='cca'):

                Yt_test = np.argmax(yt_test, axis=1)
                X_test = X_test[Yt_test!=20]
                ys_test = ys_test[Yt_test!=20]
                yt_test = yt_test[Yt_test!=20][:,:20]

                yt_test = np.argmax(yt_test, axis=1)

            with open('./datasets/{}_{}/S{}.pickle'.format(dataset,mode,i+1), 'rb') as handle:
                data = pickle.load(handle)   

                X_train = data['X_train']
                X_val = data['X_val']

                ys_train = data['ys_train']
                ys_val = data['ys_val']

                yt_train = data['yt_train']
                yt_val = data['yt_val'] 

                if (dataset=='8_channel_cVEP' and model=='cca'):

                    Yt_train = np.argmax(yt_train, axis=1)
                    X_train = X_train[Yt_train!=20]
                    ys_train = ys_train[Yt_train!=20]
                    yt_train = yt_train[Yt_train!=20][:,:20]
                    

                    Yt_val = np.argmax(yt_val, axis=1)
                    X_val = X_val[Yt_val!=20]
                    ys_val = ys_val[Yt_val!=20]
                    yt_val = yt_val[Yt_val!=20][:,:20]
                    
                    yt_train = np.argmax(yt_train, axis=1) 
                    yt_val = np.argmax(yt_val, axis=1)
    else:
        warnings.warn("Unsupported mode")

    loaded_data = {}
    loaded_data['X_train'] = X_train
    loaded_data['X_val'] = X_val
    loaded_data['X_test'] = X_test

    loaded_data['ys_train'] = ys_train
    loaded_data['ys_val'] = ys_val
    loaded_data['ys_test'] = ys_test

    loaded_data['yt_train'] = yt_train
    loaded_data['yt_val'] = yt_val
    loaded_data['yt_test'] = yt_test

    return loaded_data

# # Loading 8 channel data
# data_8ch_path = glob.iglob('./datasets/8_channel_cVEP/sourcedata/offline/*')
# label_8ch_path = './datasets/8_channel_cVEP/resources/mgold_61_6521_flip_balanced_20.mat'

# sfreq = 512
# dfreq = 240
# num_chans = 8
# data_8ch = load_raw_dataset(data_8ch_path, label_8ch_path, sfreq, dfreq, num_chans)

# with open('./datasets/8_channel_cVEP.pickle', 'wb') as handle:
#     pickle.dump(data_8ch, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

# # Loading 256 channel data
# data_256ch_path = glob.iglob('./datasets/256_channel_cVEP/data/*')
# label_256ch_path = './datasets/256_channel_cVEP/Scripts/codebook_36t.npy'

# sfreq = 360
# dfreq = 240
# num_chans = 256
# data_256ch = load_raw_dataset(data_256ch_path, label_256ch_path, sfreq, dfreq, num_chans)


# with open('./datasets/256_channel_cVEP.pickle', 'wb') as handle:
#     pickle.dump(data_256ch, handle, protocol=pickle.HIGHEST_PROTOCOL)


datasets = ['256_channel_cVEP'] #'8_channel_cVEP', '256_channel_cVEP'
modes = ['within_subject','loso_subject'] #'within_subject'
model = 'cca'

for dataset in datasets:
    for mode in modes:
        preprocessed_data = load_preprocessed_data_cv(dataset, mode, model)