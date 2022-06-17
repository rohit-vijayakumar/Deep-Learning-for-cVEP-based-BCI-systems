import os
import glob
import copy
import time
import pickle
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
#from data_loader import*
from data_preprocessing import*

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
      
def mean_no_none(l):
    l_no_none = [el for el in l if el is not None]
    return sum(l_no_none) / len(l_no_none)

def aggregate_dicts(dicts, operation=lambda x: sum(x) / len(x)):
    """
    Aggregate a sequence of dictionaries to a single dictionary using `operation`. `Operation` should
    reduce a list of all values with the same key. Keyrs that are not found in one dictionary will
    be mapped to `None`, `operation` can then chose how to deal with those.
    """
    all_keys = set().union(*[el.keys() for el in dicts])
    return {k: operation([dic.get(k, None) for dic in dicts]) for k in all_keys}

def fix_roc(results,dataset):
	if(dataset =='8_channel_cVEP'):
		n_subjects = 30
	else:
		n_subjects = 5

	for i in range(1,n_subjects+1):
		for j in range(1,16):
			fpr_f = results[i][j]['fpr']
			for k in fpr_f.keys():
				if(len(fpr_f[k])<3):
					fpr_f[k] = np.array([0,0,1])
					results[i][j]['fpr'] = fpr_f
				if(len(tpr_f[k])<3):
					tpr_f[k] = np.array([0,0,1])
					results[i][j]['tpr'] = tpr_f

	return results

def get_results(results,dataset,eval_mode):
    if(dataset =='8_channel_cVEP'):
        n_subjects = 30
    else:
        n_subjects = 5

    result_val = {}
    reuslt_avg ={}
    for s in range(1,n_subjects+1):
        if s not in result_val.keys():
            result_val[s] = []

        avg_results = []
        for f in range(1,16):
            result_val[s].append(results[s][f][eval_mode])
            avg_results.append(results[s][f][eval_mode])

        avg_results = np.mean(avg_results)
        reuslt_avg[s] = avg_results

    result_val_arr = np.array(list(result_val.values()))
    result_val_mean = list(np.mean(result_val_arr, axis=0))
    result_val['mean'] = []
    for f in range(0,15):
        result_val['mean'].append(result_val_mean[f])
    
    group = []
    result_val_grouped = []
    for i in result_val.keys():
        result_val_list = result_val[i]
        
        for j in range(1,len(result_val_list)):
            group.append(i)
            result_val_grouped.append(float(result_val_list[j]))
    
    return [group,result_val_grouped,reuslt_avg]

# def get_sequence_acc(results,dataset):
#     if(dataset =='8_channel_cVEP'):
#         n_subjects = 30
#     else:
#         n_subjects = 5
#     sequence_accuracy = {}
#     for s in range(1,n_subjects+1):
#         if s not in sequence_accuracy.keys():
#             sequence_accuracy[s] = []
#         for f in range(1,16):
#             sequence_accuracy[s].append(results[s][f]['sequence_accuracy'])

#     sequence_accuracy_arr = np.array(list(sequence_accuracy.values()))
#     sequence_accuracy_mean = list(np.mean(sequence_accuracy_arr, axis=0))
#     sequence_accuracy['mean'] = []
#     for f in range(0,15):
#         sequence_accuracy['mean'].append(sequence_accuracy_mean[f])
    
#     return sequence_accuracy

def boxplot_multi_objective_cnn(results,dataset,eval_mode):
    df = pd.DataFrame({'Subjects':results[dataset]['within_subject'][0],'within-subject':results[dataset]['within_subject'][1],
                           'loso-subject': results[dataset]['loso_subject'][1]})
        
    df = df[['Subjects','within-subject','loso-subject']]

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.set_style("whitegrid")
    dd=pd.melt(df,id_vars=['Subjects'],value_vars=['within-subject','loso-subject'],var_name='Models')
    if(dataset=='8_channel_cVEP'):
    	ax = sns.boxplot(x='Subjects',y='value',data=dd,hue='Models')
    else:
    	ax = sns.boxplot(x='Subjects',y='value',data=dd,hue='Models',width=0.25)
        
    if dataset == '8_channel_cVEP':
        dataset_txt = '8 channel dataset'
    else:
        dataset_txt = '256 channel dataset'

    if eval_mode == 'category_accuracy':
    	eval_txt = 'Category accuracy'	
    elif eval_mode == 'sequence_accuracy':
    	eval_txt = 'Sequence accuracy'
    elif eval_mode == 'precision':
    	eval_txt = 'Precision'
    elif eval_mode == 'recall':
    	eval_txt = 'Recall'
    elif eval_mode == 'f1_score':
    	eval_txt = 'F1 score'
    else:
    	eval_txt = 'ITR'

    ax.set_title('{} of multi-objective cnn on {}'.format(eval_txt,dataset_txt))
    ax.set(ylabel=eval_txt)
    if(eval_mode!='ITR'):
    	ax.set(yticks=np.arange(0,1.001,0.1))
    else:
    	ax.set(yticks=np.arange(0,150,10))

    plt.grid(False)
    ax.xaxis.grid()
    ax.yaxis.grid() 

    if(dataset=='8_channel_cVEP'):
    	ax.legend(bbox_to_anchor=(1.01, 1.0))
    else:
    	ax.legend(bbox_to_anchor=(1.01, 1.0))

    filename = "./visualizations/Box plots/Multi-objective cnn/{}_{}.png".format(eval_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename) 

def boxplot_results(results,eval_mode):
    df = pd.DataFrame({'Subjects':results[dataset][mode]['cca'][0],'cca':results[dataset][mode]['cca'][1],
                           'multi-objective cnn': results[dataset][mode]['multi_objetcive_cnn'][1]})
        
    df = df[['Subjects','cca','multi-objective cnn']]

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.set_style("whitegrid")
    dd=pd.melt(df,id_vars=['Subjects'],value_vars=['cca','multi-objective cnn'],var_name='Models')
    ax = sns.boxplot(x='Subjects',y='value',data=dd,hue='Models')
    
    if mode == 'within_subject':
        mode_txt = 'Within subject'
    else:
        mode_txt = 'LOSO subject'
        
    if dataset == '8_channel_cVEP':
        dataset_txt = '8 channel dataset'
    else:
        dataset_txt = '256 channel dataset'

    if eval_mode == 'category_acc':
    	eval_txt = 'Category accuracy'	
    elif eval_mode == 'precision':
    	eval_txt = 'Precision'
    elif eval_mode == 'recall':
    	eval_txt = 'Recall'
    elif eval_mode == 'f1_score':
    	eval_txt = 'F1 score'
    else:
    	eval_txt = 'ITR'

    ax.set_title('{} {} for {}'.format(mode_txt,eval_txt,dataset_txt))
    ax.set(ylabel=eval_txt)
    if(eval_mode!='ITR'):
    	ax.set(yticks=np.arange(0,1.001,0.1))
    else:
    	ax.set(yticks=np.arange(0,150,10))

    ax.legend(bbox_to_anchor=(1.01, 1.0))
    ax.yaxis.grid()
    ax.xaxis.grid() 
    
    filename = "./visualizations/Box plots/Comparison/{}_{}_{}.png".format(eval_txt,mode,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename) 


def plot_roc_curve(results, model, dataset, mode):
	if mode == 'within_subject':
		mode_txt = 'Within subject'
	else:
		mode_txt = 'LOSO subject'

	if(dataset =='8_channel_cVEP'):
		dataset_txt = '8 channel dataset'
		if(model=='multi_objective_cnn'):
			model_txt = 'multi-objective cnn'
			n_classes = 21
		else:
			model_txt = 'cca'
			n_classes = 20
		n_subjects = 30
	else:
		dataset_txt = '256 channel dataset'
		n_subjects = 5
		n_classes = 36

	fpr_all = []
	tpr_all = []
	roc_auc_all = []
	for subj in range(1,n_subjects+1):
		fpr = results['fpr'][subj]
		tpr = results['tpr'][subj]
		roc_auc = results['auc'][subj]

		fpr_all.append(fpr)
		tpr_all.append(tpr)
		roc_auc_all.append(roc_auc)

		for i in range(n_classes):
		    plt.plot(fpr[i], tpr[i], lw=1.5,
		             label='ROC curve of class {0} (area = {1:0.2f})'
		             ''.format(i+1, roc_auc[i]))
		plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
		plt.xlim([-0.05, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		ax.set_title('{} ROC curve for {}'.format(mode_txt,dataset_txt))
		plt.title('ROC curve for subject {}'.format(subj))
		plt.legend(bbox_to_anchor=(1.05, 1.0))

		filename = "./visualizations/ROC curves/{}/{}/S{}.png".format(model,dataset,subj)
		os.makedirs(os.path.dirname(filename), exist_ok=True)   
		plt.savefig(filename)


