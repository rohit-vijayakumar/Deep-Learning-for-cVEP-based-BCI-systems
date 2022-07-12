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
import texttable as tt  
import latextable
from matplotlib.patches import PathPatch
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
   
def get_mean_results(results,dataset,eval_mode):
    if(dataset =='8_channel_cVEP'):
        n_subjects = 30
        n_folds = 5
    else:
        n_subjects = 5
        n_folds = 3

    result_subj = np.zeros((n_subjects,n_folds))
    for s in range(1,n_subjects+1):
        fold_results = []
        for f in range(1,n_folds+1):
            fold_results.append(results[s][f][eval_mode])

        result_subj[s-1] = fold_results
    
    result_mean = np.mean(result_subj,axis=0)

    return result_mean

def get_results(results,dataset,eval_mode):
    if(dataset =='8_channel_cVEP'):
        n_subjects = 30
        n_folds = 5
    else:
        n_subjects = 5
        n_folds = 3

    result_val = {}
    reuslt_avg ={}
    for s in range(1,n_subjects+1):
        if s not in result_val.keys():
            result_val[s] = []

        avg_results = []
        for f in range(1,n_folds+1):
            result_val[s].append(results[s][f][eval_mode])
            avg_results.append(results[s][f][eval_mode])

        avg_results = np.mean(avg_results)
        reuslt_avg[s] = avg_results


    result_val_arr = np.array(list(result_val.values()))
    result_val_mean = list(np.mean(result_val_arr, axis=0))
    result_val['mean'] = []
    for f in range(0,n_folds):
        result_val['mean'].append(result_val_mean[f])
    
    group = []
    result_val_grouped = []
    for i in result_val.keys():
        result_val_list = result_val[i]
        for j in range(0,len(result_val_list)):
            group.append(i)
            result_val_grouped.append(float(result_val_list[j]))
    
    return [group,result_val_grouped,reuslt_avg]

# def boxplot_mean_results(results,eval_mode):

    
#     group1 = np.repeat('8-channel dataset',5)
#     group2 = np.repeat('256-channel dataset',3)
    
#     chance_lvl1 = np.repeat(0.05,5)
#     chance_lvl2 = np.repeat(0.0277,3)
    
#     group = np.concatenate((group1,group2),axis=0)
#     chance_lvl = np.concatenate((chance_lvl1,chance_lvl2),axis=0)
    
#     df = pd.DataFrame({'Datasets':group,
#                        'within-subject': results['within_subject'],
#                         'loso-subject': results['loso_subject']})

#     df = df[['Datasets','within-subject','loso-subject']]
#     df2 = pd.DataFrame({'Datasets':group,'chance-level':chance_lvl})
    
#     plt.rcParams.update({'font.size': 14})
#     fig, ax = plt.subplots(figsize=(7, 7))
#     sns.set_style("whitegrid")
#     dd=pd.melt(df,id_vars=['Datasets'],value_vars=['within-subject','loso-subject'],var_name='modes')
    
#     dd2=pd.melt(df2,id_vars=['Datasets'],value_vars=['chance-level'],var_name='modes2')
    
#     sns.boxplot(x='Datasets',y='value',data=dd,hue='modes', width=0.2)
#     adjust_box_widths(fig, 0.8)
#     if eval_mode == 'category_accuracy':
#         eval_txt = 'Category accuracy'
#         sns.scatterplot(x='Datasets',y='value',data=dd2, hue='modes2')
        
#     elif eval_mode == 'sequence_accuracy':
#         eval_txt = 'Sequence accuracy'
#     elif eval_mode == 'precision':
#         eval_txt = 'Precision'
#     elif eval_mode == 'recall':
#         eval_txt = 'Recall'
#     elif eval_mode == 'f1_score':
#         eval_txt = 'F1 score'
#     else:
#         eval_txt = 'ITR'

#     ax.set_title('{} of dual-objective CNN'.format(eval_txt),fontsize=14)
#     ax.set(ylabel=eval_txt)
#     if(eval_mode!='ITR'):
#         ax.set(yticks=np.arange(0,1.001,0.2))
#     else:
#         ax.set(yticks=np.arange(0,150,20))

#     plt.grid(False)
#     plt.grid(True) 
    
#     ax.legend(bbox_to_anchor=(1.4, 1.02))

#     filename = "./visualizations/Mean results2/Multi-objective cnn/{}.png".format(eval_txt)
#     os.makedirs(os.path.dirname(filename), exist_ok=True)   
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
#     plt.close()

def boxplot_mean_results(results1, results2, eval_mode):

    fontsize = 18
    group1 = np.repeat('8-channel dataset',5)
    group2 = np.repeat('256-channel dataset',3)
    
    chance_lvl1 = np.repeat(0.05,5)
    chance_lvl2 = np.repeat(0.0277,3)
    
    group = np.concatenate((group1,group2),axis=0)
    chance_lvl = np.concatenate((chance_lvl1,chance_lvl2),axis=0)
    
    df1 = pd.DataFrame({'Datasets':group,
                       'within-subject': results1['within_subject'],
                        'LOSO': results1['loso_subject']})

    df1 = df1[['Datasets','within-subject','LOSO']]
    
    df2 = pd.DataFrame({'Datasets':group,
                       'within-subject': results2['within_subject'],
                        'LOSO': results2['loso_subject']})

    df2 = df2[['Datasets','within-subject','LOSO']]
    
    dfclvl = pd.DataFrame({'Datasets':group,'chance-level':chance_lvl})
    
    plt.rcParams.update({'font.size': fontsize})
    
    
    fig=plt.figure(figsize=(25, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    
    sns.set_style("whitegrid")
    dd1=pd.melt(df1,id_vars=['Datasets'],value_vars=['within-subject','LOSO'],var_name='modes')
    dd2=pd.melt(df2,id_vars=['Datasets'],value_vars=['within-subject','LOSO'],var_name='modes')
        
    ddclvl=pd.melt(dfclvl,id_vars=['Datasets'],value_vars=['chance-level'],var_name='modes2')
    
    adjust_box_widths(fig, 0.8)
    if eval_mode == 'accuracy':
        eval_txt1 = 'Category accuracy'
        eval_txt2 = 'Sequence accuracy'
        eval_txt = 'accuracy'
        sns.scatterplot(x='Datasets',y='value',data=ddclvl, hue='modes2',ax=ax1)
        
    elif eval_mode == 'precision':
        eval_txt1 = 'Precision'
        eval_txt2 = 'Recall'
        eval_txt = 'precison_recall'
    elif eval_mode == 'precision':
        eval_txt1 = 'Precision'
        eval_txt2 = 'Recall'
    elif eval_mode == 'itr':
        eval_txt1 = 'F1-score'
        eval_txt2 = 'ITR'
        eval_txt = 'f1score_itr'
    else:
        eval_txt = 'ITR'
        eval_txt = 'f1score_itr'

    sns.boxplot(x='Datasets',y='value',data=dd1,hue='modes', width=0.2,ax=ax1)
    sns.boxplot(x='Datasets',y='value',data=dd2,hue='modes', width=0.2,ax=ax2)

    ax1.set_title('{} of dual-objective CNN'.format(eval_txt1),fontsize=fontsize)
    ax2.set_title('{} of dual-objective CNN'.format(eval_txt2),fontsize=fontsize)
    ax1.set(ylabel=eval_txt1)
    if eval_mode == 'itr':
        ax2.set(ylabel='bits/min')
    else:
        ax2.set(ylabel=eval_txt2)
    
    if(eval_mode!='itr'):
        ax1.set(yticks=np.arange(0,1.001,0.2))
        ax2.set(yticks=np.arange(0,1.001,0.2))
    else:
        ax1.set(yticks=np.arange(0,1.001,0.2))
        ax2.set(yticks=np.arange(0,150,20))
    
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax1.grid(False)
    ax1.grid(True) 
    ax2.grid(False)
    ax2.grid(True) 
    
    handles, labels = ax1.get_legend_handles_labels()
    handles_new = handles.copy()
    if(len(handles)==3):
        handles_new[0] = handles[1]
        handles_new[1] = handles[2]
        handles_new[2] = handles[0]
    ax2.legend(bbox_to_anchor=(1.55, 1.02),fontsize=fontsize, handles=handles_new)
    plt.subplots_adjust(wspace=0.4)
    #fig.tight_layout() 
    filename = "./visualizations/Mean results/Multi-objective cnn/{}.png".format(eval_txt)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
    plt.close()

def boxplot_multi_objective_cnn(results,dataset,eval_mode):
    fsize=18
    if dataset == '8_channel_cVEP':
        dataset_txt = '8-channel dataset'
        n_subjects = 30
        chance_lvl = 0.05
    else:
        dataset_txt = '256-channel dataset'
        n_subjects = 5
        chance_lvl = 0.0277

    df = pd.DataFrame({'Subjects':results[dataset]['within_subject'][0],'within-subject':results[dataset]['within_subject'][1],
                           'LOSO': results[dataset]['loso_subject'][1]})
        
    df = df[['Subjects','within-subject','LOSO']]

    df2 = pd.DataFrame({'Subjects':np.arange(0,n_subjects+1),'chance-level':np.repeat(chance_lvl, n_subjects+1)})
        
    df2 = df2[['Subjects','chance-level']]

    plt.rcParams.update({'font.size': fsize})
    fig, ax = plt.subplots(figsize=(30, 20))
    sns.set_style("whitegrid")
    dd=pd.melt(df,id_vars=['Subjects'],value_vars=['within-subject','LOSO'],var_name='Models')
    
    dd2=pd.melt(df2,id_vars=['Subjects'],value_vars=['chance-level'],var_name='Models2')
    if(dataset=='8_channel_cVEP'):
    	sns.boxplot(x='Subjects',y='value',data=dd,hue='Models')
    else:
        sns.boxplot(x='Subjects',y='value',data=dd,hue='Models',width=0.25)

    adjust_box_widths(fig, 0.8)
    if eval_mode == 'category_accuracy':
        eval_txt = 'Category accuracy'
        sns.lineplot(x='Subjects',y='value',data=dd2,linestyle=':', hue='Models2')
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

    ax.set_title('{} of dual-objective CNN on {}'.format(eval_txt,dataset_txt),fontsize=fsize)
    ax.set(ylabel=eval_txt)
    if(eval_mode!='ITR'):
    	ax.set(yticks=np.arange(0,1.001,0.1))
    else:
    	ax.set(yticks=np.arange(0,150,10))

    plt.grid(False)
    plt.grid(True) 

    if(dataset=='8_channel_cVEP' and eval_mode!='ITR'):
        c1 = 1.13
        c2 = 1.01
    elif(eval_mode=='ITR'):
        c1 = 1.13
        c2 = 1.01
    else:
        c1 = 1.13
        c2 = 1.01

    leg = ax.legend(bbox_to_anchor=(c1, c2))
    if(eval_mode =='category_accuracy'):
        leg_lines = leg.get_lines()
        leg_lines[0].set_linestyle(":")

    filename = "./visualizations/Box plots/Multi-objective cnn/{}_{}.png".format(eval_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
    plt.close()

def get_latex_table(category_accuracy,sequence_accuracy,precision,recall,f1_score,itr):
    modes = ['within_subject','loso_subject']
    datasets = ['8_channel_cVEP','256_channel_cVEP']
    for dataset in datasets:
        if(dataset=='8_channel_cVEP'):
            n_subjects = 30
            dataset_txt = '8-channel dataset'
            c= 5
        else:
            dataset_txt = '256-channel dataset'
            n_subjects = 5
            c = 3

        for mode in modes:
            results_table = np.zeros((n_subjects+2,7)).astype('object')
            results_table_std = np.zeros((n_subjects+2,7)).astype('object')
            columns = np.array(['Subject','category accuracy','sequence accuracy','precision','recall','f1-score','ITR'])
            num_decimals = 2
            results_table[0,:] = columns
            results_table[1:,0] = np.append(np.arange(1,n_subjects+1),'mean')

            results_table_std[0,:] = columns
            results_table_std[1:,0] = np.append(np.arange(1,n_subjects+1),'mean')

            data = category_accuracy[dataset][mode][1]
            mean = []
            std = []
            for f in range(0,len(data),c): 
                mean.append(np.mean(data[f:f+c]))
                std.append(np.std(data[f:f+c]))
            results_table[1:,1] = np.around(np.array(mean),num_decimals)
            results_table_std[1:,1] = np.around(np.array(std),num_decimals)

            data = sequence_accuracy[dataset][mode][1]
            mean = []
            std = []
            for f in range(0,len(data),c): 
                mean.append(np.mean(data[f:f+c]))
                std.append(np.std(data[f:f+c]))

            results_table[1:,2] = np.around(np.array(mean),num_decimals)
            results_table_std[1:,2] = np.around(np.array(std),num_decimals)

            data = precision[dataset][mode][1]
            mean = []
            std = []
            for f in range(0,len(data),c): 
                mean.append(np.mean(data[f:f+c]))
                std.append(np.std(data[f:f+c]))

            results_table[1:,3] = np.around(np.array(mean),num_decimals)
            results_table_std[1:,3] = np.around(np.array(std),num_decimals)

            data = recall[dataset][mode][1]
            mean = []
            std = []
            for f in range(0,len(data),c): 
                mean.append(np.mean(data[f:f+c]))
                std.append(np.std(data[f:f+c]))

            results_table[1:,4] = np.around(np.array(mean),num_decimals)
            results_table_std[1:,5] = np.around(np.array(std),num_decimals)

            data = f1_score[dataset][mode][1]
            mean = []
            std = []
            for f in range(0,len(data),c): 
                mean.append(np.mean(data[f:f+c]))
                std.append(np.std(data[f:f+c]))

            results_table[1:,5] = np.around(np.array(mean),num_decimals)
            results_table_std[1:,5] = np.around(np.array(std),num_decimals)

            data = itr[dataset][mode][1]
            mean = []
            std = []
            for f in range(0,len(data),c): 
                mean.append(np.mean(data[f:f+c]))
                std.append(np.std(data[f:f+c]))

            results_table[1:,6] = np.around(np.array(mean),num_decimals)
            results_table_std[1:,6] = np.around(np.array(std),num_decimals)

            df_mean = pd.DataFrame(results_table[1:],columns = list(results_table[0,:]))
            df_std = pd.DataFrame(results_table_std[1:],columns = list(results_table_std[0,:]))
            df_mean_std = df_mean.astype(str) + u"\u00B1" + df_std.astype(str)

            subj_row = np.append(np.arange(1,n_subjects+1),'mean')
            for k in range(1,len(subj_row)+1):
                if k<=9:
                    subj_row[k-1] = 'S0'+str(k)
                elif(k==len(subj_row)):
                    subj_row[k-1] = 'mean'
                else:
                    subj_row[k-1] = 'S'+str(k)
            df_mean_std['Subject'] = subj_row
            df_mean_std = df_mean_std.to_numpy()

            df_mean_std_new = np.zeros((df_mean_std.shape[0]+1,df_mean_std.shape[1])).astype('object')
            df_mean_std_new[0][1:] = list(results_table[0,:])[1:]
            df_mean_std_new[1:] = df_mean_std
            df_mean_std_new[0][0] = ""
            if(mode=='within_subject'):
                mode_txt = 'Within-subject'
            else:
                mode_txt = 'LOSO'

            table_1 = tt.Texttable()
            table_1.add_rows(df_mean_std_new)
            table_1.draw()
            table_latex = latextable.draw_latex(table_1, caption="{} results of dual-objective CNN on {}".format(mode_txt,dataset_txt), 
                                        label="dual_objective_cnn_{}_{}".format(mode,dataset))

            filename = './visualizations/Latex tables/{}_{}.txt'.format(mode,dataset)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.write(table_latex)

def plot_varaible_time_steps(dataset,mode,results):
    fsize=18
    if(dataset=='8_channel_cVEP'):
        n_folds = 5
        n_subjects = 30
        dataset_txt = '8-channel dataset'
    else:
        n_folds = 3
        n_subjects = 5
        dataset_txt = '256-channel dataset'

    if(mode == 'within_subject'):
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO'

    results_variable_time_steps = {}
    for subj in range(1,n_subjects+1):
        variable_time_steps_all = []
        for fold in range(1,n_folds+1):
            variable_time_steps_all.append(results[subj][fold]['variable_time_steps'])

        variable_time_steps_all = np.array(variable_time_steps_all)
        variable_time_steps = np.mean(variable_time_steps_all,axis=0)
        
        results_variable_time_steps[subj] = variable_time_steps

    NUM_COLORS = n_subjects+1
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    time = list(np.round(np.arange(0,2.2,0.1),2))

    sns.reset_orig() 
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 

    plt.rcParams.update({'font.size': fsize})
    fig, ax = plt.subplots(figsize=(20,10))
    for l in results_variable_time_steps.keys():
        
        acc_samples = results_variable_time_steps[l]
        samples = np.arange(0,len(acc_samples))
        lines = ax.plot(samples,acc_samples,label=l)
        lines[0].set_color(clrs[l])
        lines[0].set_linestyle(LINE_STYLES[l%NUM_STYLES])

    ax.set_xticks(np.linspace(0,504,len(time)), time)
    ax.set_yticks(np.arange(0,1.09,0.1))
    ax.set_ylim((0,1.09))
    ax.set_xlim((5,504))
    ax.set_xlabel('Time (s)',fontsize=fsize)
    ax.set_ylabel('Accuracy',fontsize=fsize)
    ax.set_title("{} category accuracy over time steps of dual-objective CNN on {}".format(mode_txt,dataset_txt),fontsize=fsize)

    ax.legend(bbox_to_anchor=(1.12, 1.02),fontsize=fsize)
    plt.grid(False)
    plt.grid(True) 

    filename = "./visualizations/Performance over time-steps/Multi-objective cnn/{}_{}_time_steps.png".format(mode_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()   

def plot_itr_time_steps(dataset,mode,results):
    fsize=18
    if(dataset=='8_channel_cVEP'):
        n_folds = 5
        n_subjects = 30
        dataset_txt = '8-channel dataset'
    else:
        n_folds = 3
        n_subjects = 5
        dataset_txt = '256-channel dataset'

    if(mode == 'within_subject'):
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO'

    results_variable_time_steps = {}
    for subj in range(1,n_subjects+1):
        variable_time_steps_all = []
        for fold in range(1,n_folds+1):
            variable_time_steps_all.append(results[subj][fold]['ITR_time_steps'])

        variable_time_steps_all = np.array(variable_time_steps_all)
        variable_time_steps = np.mean(variable_time_steps_all,axis=0)
        results_variable_time_steps[subj] = variable_time_steps

    NUM_COLORS = n_subjects+1
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    time = list(np.round(np.arange(0,2.2,0.1),2))

    sns.reset_orig() 
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 

    plt.rcParams.update({'font.size': fsize})
    fig, ax = plt.subplots(figsize=(20,10))
    for l in results_variable_time_steps.keys():
        
        acc_samples = results_variable_time_steps[l]
        samples = np.arange(0,len(acc_samples))
        lines = ax.plot(samples,acc_samples,label=l)
        lines[0].set_color(clrs[l])
        lines[0].set_linestyle(LINE_STYLES[l%NUM_STYLES])

    ax.set_xticks(np.linspace(0,504,len(time)), time)
    ax.set_yticks(np.arange(0,380,20))
    ax.set_ylim((0,380))
    ax.set_xlim((5,504))
    ax.set_xlabel('Time (s)',fontsize=fsize)
    ax.set_ylabel('ITR (bits/min)',fontsize=fsize)
    ax.set_title("{} ITR over time steps of dual-objective CNN on {}".format(mode_txt,dataset_txt),fontsize=fsize)

    ax.legend(bbox_to_anchor=(1.12, 1.02),fontsize=fsize)
    plt.grid(False)
    plt.grid(True) 

    filename = "./visualizations/Performance over time-steps/Multi-objective cnn/{}_{}_ITR_time_steps.png".format(mode_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()

def plot_confusion_matrix(dataset,mode,results):
    fsize = 18
    if(dataset=='8_channel_cVEP'):
        n_folds = 5
        n_subjects = 30
        n_rep = 15
        n_classes = 21
        dataset_txt = '8-channel dataset'
    else:
        n_folds = 3
        n_subjects = 5
        n_rep = 2
        n_classes = 36
        dataset_txt = '256-channel dataset'

    if(mode == 'within_subject'):
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO'

    cm_all = np.zeros((n_subjects*n_folds,n_classes,n_classes))
    c = 0
    for subj in range(1,n_subjects+1):
        for fold in range(1,n_folds+1):
            cm = results[subj][fold]['category_cm']
            cm_all[c]= cm.astype('int')/n_rep
            c+=1

    cm_all = np.sum(cm_all,axis=0)/c
    cm_all = np.round(cm_all,2)

    sns.set(font_scale=2.0)
    plt.rcParams.update({'font.size': fsize})
    plt.figure(figsize=(30,30))
    labels = np.arange(1,n_classes+1)
    cm_plot = sns.heatmap(cm_all, annot=True, xticklabels=labels, 
                          yticklabels=labels, cmap='Reds',annot_kws={"size": fsize})
    cm_plot.set(xlabel='Predicted labels', ylabel='True labels')
    plt.title('{} normalized confusion matrix for category prediction across subjects on {}'.format(mode_txt, dataset_txt),fontsize=fsize)

    filename = "./visualizations/Confusion matrix/Multi-objective cnn/{}_{}_cm_category.png".format(mode_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()

    cm_all = np.zeros((n_subjects*n_folds,2,2))
    c = 0
    for subj in range(1,n_subjects+1):
        for fold in range(1,n_folds+1):
            cm = results[subj][fold]['sequence_cm']
            cm = np.mean(cm,axis=0)
            cm_all[c]= cm.astype('int')/63
            c+=1

    cm_all = np.sum(cm_all,axis=0)/c
    cm_all = np.round(cm_all,2)
    plt.rcParams.update({'font.size': fsize})
    plt.figure(figsize=(20,20))
    labels = np.arange(0,2)
    cm_plot = sns.heatmap(cm_all, annot=True, xticklabels=labels, 
                          yticklabels=labels, cmap='Reds')
    cm_plot.set(xlabel='Predicted labels', ylabel='True labels')
    plt.title('{} normalized confusion matrix for sequence prediction across subjects on {}'.format(mode_txt, dataset_txt),fontsize=fsize)

    filename = "./visualizations/Confusion matrix/Multi-objective cnn/{}_{}_cm_sequence.png".format(mode_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()

def plot_roc_curve(results, model, dataset, mode):
    fsize=18
    if mode == 'within_subject':
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO'

    if(model=='multi_objective_cnn'):
        model_txt = 'Multi-objective cnn'
    else:
        model_txt = 'cca'
        n_classes = 20

    if(dataset =='8_channel_cVEP'):
        dataset_txt = '8-channel dataset'
        n_classes = 21
        n_subjects = 30
    else:
        dataset_txt = '256-channel dataset'
        n_subjects = 5
        n_classes = 36

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': fsize})
    for subj in range(1,n_subjects+1):
        fpr = results[subj][1]['fpr']
        tpr = results[subj][1]['tpr']
        roc_auc = results[subj][1]['auc']
        
        # m = max(map(len, fpr.values()))
        # fpr = np.array([np.pad(v, (0, m - len(v)), 'constant') for v in fpr.values()])
        
        # m = max(map(len, tpr.values()))
        # tpr = np.array([np.pad(v, (0, m - len(v)), 'constant') for v in tpr.values()])
        
        fig, ax = plt.subplots(figsize=(10,10))
        NUM_COLORS = n_classes+1
        LINE_STYLES = ['solid', 'dashed', 'dotted']
        
        clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 
        
        NUM_STYLES = len(LINE_STYLES)
        for i in range(1,n_classes):
            lines = ax.plot(fpr[i], tpr[i], lw=1.5,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i+1, roc_auc[i]))
            
            lines[0].set_color(clrs[i])
            lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
            
        ax.plot([0, 1], [0, 1], linestyle = '--', lw=1.5, color = 'k')
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate',fontsize=fsize)
        ax.set_ylabel('True Positive Rate',fontsize=fsize)
        ax.set_title('{} ROC curve for subject {} in {} '.format(mode_txt,subj,dataset_txt),fontsize=fsize)
        ax.legend(bbox_to_anchor=(1.05, 1.0),fontsize=fsize)
        ax.grid(False)
        ax.grid(True)

        filename = "./visualizations/ROC curves/{}/{}/{}_{}_S{}.png".format(model_txt,mode_txt,mode_txt,dataset,subj)
        os.makedirs(os.path.dirname(filename), exist_ok=True)   
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent=True) 
        plt.close()

def get_mean_time_steps(results, dataset, mode,model, eval_mode):
    if(dataset=='8_channel_cVEP'):
        n_folds = 5
        n_subjects = 30
        dataset_txt = '8-channel'
    else:
        n_folds = 3
        n_subjects = 5
        dataset_txt = '256-channel'

    if(mode == 'within_subject'):
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO-subject'

    results_variable_time_steps = {}
    for subj in range(1,n_subjects+1):
        variable_time_steps_all = []
        for fold in range(1,n_folds+1):
            variable_time_steps_all.append(results[subj][fold][eval_mode])
            if(model=='eeg2code'):
                break

        variable_time_steps_all = np.array(variable_time_steps_all)
        variable_time_steps = np.mean(variable_time_steps_all,axis=0)
        
        results_variable_time_steps[subj] = variable_time_steps
    
    results_variable_time_steps = np.array(list(results_variable_time_steps.values()))
    results_variable_time_steps = np.mean(results_variable_time_steps,axis=0)
    
    return results_variable_time_steps

# def plot_time_steps(results,eval_mode):
#     sns.reset_orig() 
#     fsize=12
#     plt.rcParams.update({'font.size': fsize})
#     fig, ax = plt.subplots(figsize=(7,5))
#     order = ['8_channel_cVEP_within_subject','8_channel_cVEP_loso_subject','256_channel_cVEP_within_subject','256_channel_cVEP_loso_subject']
#     for i,l in enumerate(order):
#         if l == '8_channel_cVEP_within_subject':
#             txt = '8-channel within-subject'
#         if l == '8_channel_cVEP_loso_subject':
#             txt = '8-channel loso-subject'
#         if l == '256_channel_cVEP_within_subject':
#             txt = '256-channel within-subject'
#         if l == '256_channel_cVEP_loso_subject':
#             txt = '256-channel loso-subject'
            
#         NUM_COLORS = 4+1
#         LINE_STYLES = ['solid', 'solid', 'dotted', 'dotted']
#         NUM_STYLES = len(LINE_STYLES)

#         time = list(np.round(np.arange(0,2.1,0.2),2)[1:])
#         time.insert(0,0.0)

        
#         clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 

#         acc_samples = results[order[i]]
#         if eval_mode != 'Category accuracy':
#             acc_samples[:5] = 0
#         samples = np.arange(0,len(acc_samples))
#         lines = ax.plot(samples,acc_samples,label=txt)
#         lines[0].set_color(clrs[i])
#         lines[0].set_linestyle(LINE_STYLES[i])

#     ax.set_xticks(np.arange(0,504,50), time)
#     ax.set_xlabel('Time (s)',fontsize=fsize)
    
#     if eval_mode == 'Category accuracy':
#         ax.set_yticks(np.arange(0,1.09,0.2))
#         ax.set_ylim((0,1.09))
#         ax.set_ylabel('Accuracy',fontsize=fsize)
#         ax.set_title("Category accuracy over time of dual-objective CNN",fontsize=fsize)
#         ax.legend(bbox_to_anchor=(1.01, 1.02),fontsize=fsize)
#     else:
#         ax.set_yticks(np.arange(0,300,60))
#         ax.set_ylim((0,300))
#         ax.set_ylabel('ITR (bits/min)',fontsize=fsize)
#         ax.set_title("ITR over time of dual-objective CNN",fontsize=fsize)
#         ax.legend(bbox_to_anchor=(1.55, 1.02),fontsize=fsize)

#     plt.grid(False)
#     plt.grid(True) 
#     #plt.show()

#     filename = "./visualizations/Mean results/Multi-objective cnn/{}_time_steps.png".format(eval_mode)
#     os.makedirs(os.path.dirname(filename), exist_ok=True)   
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
#     plt.close()   

def plot_mean_time_steps(results1,results2):
    sns.reset_orig() 
    fsize=20
    plt.rcParams.update({'font.size': fsize})
    fig=plt.figure(figsize=(30, 6))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    
    order = ['8_channel_cVEP_within_subject','8_channel_cVEP_loso_subject','256_channel_cVEP_within_subject','256_channel_cVEP_loso_subject']
    for i,l in enumerate(order):
        if l == '8_channel_cVEP_within_subject':
            txt = '8-channel within-subject'
        if l == '8_channel_cVEP_loso_subject':
            txt = '8-channel LOSO'
        if l == '256_channel_cVEP_within_subject':
            txt = '256-channel within-subject'
        if l == '256_channel_cVEP_loso_subject':
            txt = '256-channel LOSO'
            
        NUM_COLORS = 4+1
        LINE_STYLES = ['solid', 'solid', 'dashdot', 'dashdot']
        NUM_STYLES = len(LINE_STYLES)

        time = list(np.round(np.arange(0,2.1,0.3),2))

        #clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 
        clrs = ["tab:blue","tab:orange","tab:blue","tab:orange"]

        acc_samples1 = results1[order[i]]
        acc_samples2 = results2[order[i]]
        #acc_samples2[:5] = 0
        
        samples1 = np.arange(0,len(acc_samples1))
        samples2 = np.arange(0,len(acc_samples2))
        
        lines1 = ax1.plot(samples1,acc_samples1,label=txt)
        lines1[0].set_color(clrs[i])
        lines1[0].set_linestyle(LINE_STYLES[i])
        
        lines2 = ax2.plot(samples2,acc_samples2,label=txt)
        lines2[0].set_color(clrs[i])
        lines2[0].set_linestyle(LINE_STYLES[i])


    ax1.set_xticks(np.linspace(0,504,len(time)), time)
    ax1.set_xlabel('Time (s)',fontsize=fsize)
    ax2.set_xticks(np.linspace(0,504,len(time)), time)
    ax2.set_xlabel('Time (s)',fontsize=fsize)
    
    ax1.set_yticks(np.arange(0,1.09,0.2))
    ax1.set_ylim((0,1.09))
    ax1.set_xlim((5,504))
    ax1.set_ylabel('Accuracy',fontsize=fsize)
    ax1.set_title("Accuracy over time of dual-objective CNN",fontsize=fsize)
    ax1.legend(bbox_to_anchor=(1.01, 1.02),fontsize=fsize)

    
    ax2.set_yticks(np.arange(0,300,60))
    ax2.set_ylim((0,300))
    ax2.set_xlim((5,504))
    ax2.set_ylabel('bits/min',fontsize=fsize)
    ax2.set_title("ITR over time of dual-objective CNN",fontsize=fsize)
    ax2.legend(bbox_to_anchor=(1.65, 1.02),fontsize=fsize)

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax1.grid(False)
    ax1.grid(True) 
    ax2.grid(False)
    ax2.grid(True) 
    ax2.legend(bbox_to_anchor=(1.8, 1.02),fontsize=fsize)
    plt.subplots_adjust(wspace=0.3)
    #fig.tight_layout()
    #plt.show()

    filename = "./visualizations/Mean results/Multi-objective cnn/performance_time_steps.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close() 

def plot_comparison_time_steps(results1,results2,eval_mode):
    sns.reset_orig() 

    fsize=18
    plt.rcParams.update({'font.size': fsize})
    fig=plt.figure(figsize=(30, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    
    order = ['cca_within_subject','cca_loso_subject',
    'eeg2code_within_subject','eeg2code_loso_subject',
    'inception_within_subject','inception_loso_subject',
    'multi_objective_cnn_within_subject', 'multi_objective_cnn_loso_subject']
    
    palette = {
    'cca_within_subject': 'tab:blue',
    'cca_loso_subject': 'tab:blue',   
    'eeg2code_within_subject': 'tab:orange',
    'eeg2code_loso_subject': 'tab:orange',    
    'inception_within_subject': 'tab:red',
    'inception_loso_subject': 'tab:red',
    'multi_objective_cnn_within_subject': 'tab:green',
    'multi_objective_cnn_loso_subject': 'tab:green'}
        
    dataset_txt = '8-channel dataset'
    
    for i,l in enumerate(order):
        if l == 'cca_within_subject':
            txt = 'CCA (within-subject)'
        if l == 'cca_loso_subject':
            txt = 'CCA (LOSO)'
        if l == 'eeg2code_within_subject':
            txt = 'EEG2Code (within-subject)'
        if l == 'eeg2code_loso_subject':
            txt = 'EEG2Code (LOSO)'
        if l == 'inception_within_subject':
            txt = 'EEG-Inception (within-subject)'
        if l == 'inception_loso_subject':
            txt = 'EEG-Inception (LOSO)'
        if l == 'multi_objective_cnn_within_subject':
            txt = 'dual-objective CNN (within-subject)'
        if l == 'multi_objective_cnn_loso_subject':
            txt = 'dual-objective CNN (LOSO)'
        
        clrs = palette[l]
        LINE_STYLES = ['solid', 'dashdot', 'solid', 'dashdot','solid', 'dashdot','solid', 'dashdot']
        NUM_STYLES = len(LINE_STYLES)

        time = list(np.round(np.arange(0,2.1,0.3),2))
        #time.append(2.1)

        acc_samples = results1[order[i]]
        if(i==4 or i==5):
            if(eval_mode=='Category accuracy'):
                acc_samples=acc_samples/100
        #acc_samples[:3] = 0
        if (len(acc_samples)==126):
            samples = np.linspace(0,504,126)
        else:
            samples = np.arange(0,len(acc_samples))
            
        lines = ax1.plot(samples,acc_samples,label=txt)
        lines[0].set_color(clrs)
        lines[0].set_linestyle(LINE_STYLES[i])
    
    ax1.set_xticks(np.linspace(0,504,len(time)), time)
    ax1.set_xlabel('Time (s)',fontsize=fsize)
    ax1.set_xlim((5,504))
    if eval_mode == 'Category accuracy':
        ax1.set_yticks(np.arange(0,1.09,0.2))
        ax1.set_ylim((0,1.09))
        ax1.set_ylabel('Accuracy',fontsize=fsize)
        ax1.set_title("Category accuracy over time on {}".format(dataset_txt),fontsize=fsize)
        ax1.legend(bbox_to_anchor=(1.05, 1.02),fontsize=fsize)
    else:
        ax1.set_yticks(np.arange(0,300,60))
        ax1.set_ylim((0,300))
        ax1.set_ylabel('bits/min',fontsize=fsize)
        ax1.set_title("ITR over time on {}".format(dataset_txt),fontsize=fsize)
        ax1.legend(bbox_to_anchor=(1.05, 1.02),fontsize=fsize)

    ax1.grid(False)
    ax1.grid(True)
    ax1.get_legend().remove()
    
    dataset_txt = '256-channel dataset'
    
    for i,l in enumerate(order):
        if l == 'cca_within_subject':
            txt = 'CCA (within-subject)'
        if l == 'cca_loso_subject':
            txt = 'CCA (LOSO)'
        if l == 'eeg2code_within_subject':
            txt = 'EEG2Code (within-subject)'
        if l == 'eeg2code_loso_subject':
            txt = 'EEG2Code (LOSO)'
        if l == 'inception_within_subject':
            txt = 'EEG-Inception (within-subject)'
        if l == 'inception_loso_subject':
            txt = 'EEG-Inception (LOSO)'
        if l == 'multi_objective_cnn_within_subject':
            txt = 'dual-objective CNN (within-subject)'
        if l == 'multi_objective_cnn_loso_subject':
            txt = 'dual-objective CNN (LOSO)'
        
        clrs = palette[l]
        LINE_STYLES = ['solid', 'dashdot', 'solid', 'dashdot','solid', 'dashdot','solid', 'dashdot']
        NUM_STYLES = len(LINE_STYLES)

        time = list(np.round(np.arange(0,2.1,0.3),2))
        #time.append(2.1)

        acc_samples = results2[order[i]]
        if(i==4 or i==5):
            if(eval_mode=='Category accuracy'):
                acc_samples=acc_samples/100
        #acc_samples[:3] = 0
        if (len(acc_samples)==126):
            samples = np.linspace(0,504,126)
        else:
            samples = np.arange(0,len(acc_samples))
            
        lines = ax2.plot(samples,acc_samples,label=txt)
        lines[0].set_color(clrs)
        lines[0].set_linestyle(LINE_STYLES[i])
    
    ax2.set_xticks(np.linspace(0,504,len(time)), time)
    ax2.set_xlabel('Time (s)',fontsize=fsize)
    ax2.set_xlim((5,504))
    if eval_mode == 'Category accuracy':
        ax2.set_yticks(np.arange(0,1.09,0.2))
        ax2.set_ylim((0,1.09))
        ax2.set_ylabel("")
        ax2.set_title("Category accuracy over time on {}".format(dataset_txt),fontsize=fsize)
        ax2.legend(bbox_to_anchor=(1.05, 1.02),fontsize=fsize)
    else:
        ax2.set_yticks(np.arange(0,300,60))
        ax2.set_ylim((0,300))
        ax2.set_ylabel("")
        ax2.set_title("ITR over time on {}".format(dataset_txt),fontsize=fsize)
        ax2.legend(bbox_to_anchor=(1.05, 1.02),fontsize=fsize)

    ax2.grid(False)
    ax2.grid(True)
    plt.subplots_adjust(wspace=0.3)

    filename = "./visualizations/Mean results/Comparison/{}_time_steps.png".format(eval_mode)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()   

def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

def boxplot_comparison_results(results,dataset,mode,eval_mode):
    fsize=20
    df = pd.DataFrame({'Subjects':results[dataset][mode]['cca'][0],'CCA':results[dataset][mode]['cca'][1],
                            'EEG2Code': results[dataset][mode]['eeg2code'][1],
                            'EEG-Inception': results[dataset][mode]['inception'][1],
                           'dual-objective CNN': results[dataset][mode]['multi_objective_cnn'][1]})

    df = df[['Subjects','CCA','EEG2Code','EEG-Inception','dual-objective CNN']]
    plt.rcParams.update({'font.size': fsize})
    palette = {
    'CCA': 'tab:blue',
    'EEG2Code': 'tab:orange',
    'EEG-Inception': 'tab:red',
    'dual-objective CNN': 'tab:green',
}
    fig, ax = plt.subplots(figsize=(40, 20))
    sns.set_style("whitegrid")
    dd=pd.melt(df,id_vars=['Subjects'],value_vars=['CCA','EEG2Code','EEG-Inception','dual-objective CNN'],var_name='Models')
    if(dataset=='8_channel_cVEP'):
        sns.boxplot(x='Subjects',y='value',data=dd,hue='Models',palette=palette)
        adjust_box_widths(fig, 0.8)
    else:
        sns.boxplot(x='Subjects',y='value',data=dd,hue='Models',width=0.25, palette=palette)
        adjust_box_widths(fig, 0.8)

    if mode == 'within_subject':
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO'

    if dataset == '8_channel_cVEP':
        dataset_txt = '8-channel dataset'
    else:
        dataset_txt = '256-channel dataset'

    if eval_mode == 'category_acc':
        eval_txt = 'Category accuracy'  
    elif eval_mode == 'precision':
        eval_txt = 'Precision'
    elif eval_mode == 'recall':
        eval_txt = 'Recall'
    elif eval_mode == 'f1_score':
        eval_txt = 'F1-score'
    else:
        eval_txt = 'ITR'
    
    ax.set_title('{} {} on {}'.format(mode_txt,eval_txt,dataset_txt),fontsize=fsize)
    ax.set(ylabel=eval_txt)
    if(eval_mode!='ITR'):
        if(dataset!='256_channel_cVEP'):
            ax.set(yticks=np.arange(0,1.001,0.1))
            ax.legend(bbox_to_anchor=(1.01, 1.01))
        else:
            ax.set(yticks=np.arange(0,1.001,0.1))
            ax.legend(bbox_to_anchor=(1.13, 1.01))
    else:
        if(dataset!='256_channel_cVEP'):
            ax.set(yticks=np.arange(0,170,10))
            ax.legend(bbox_to_anchor=(1.13, 1.01))
        else:
            ax.set(yticks=np.arange(0,170,10))
            ax.legend(bbox_to_anchor=(1.13, 1.01))

    plt.grid(False)
    plt.grid(True) 

    filename = "./visualizations/Box plots/Comparison/{}_{}_{}.png".format(eval_txt,mode,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
    plt.close()

def boxplot_comparison_mean_results(results1,results2,eval_mode):

    fontsize = 20
    dataset_txt = '8-channel dataset'
    nreps = 5
    
    dataset_txt2 = '256-channel dataset'
    nreps2 = 3
        
    group1 = np.repeat('within-subject',nreps)
    group2 = np.repeat('loso-subject',nreps)
    
    
    group = np.concatenate((group1,group2),axis=0)
    df = pd.DataFrame({'Modes':group,
                       'CCA': results1['cca'],
                       'EEG2Code': results1['eeg2code'],
                       'EEG-Inception': results1['inception'],
                        'dual-objective CNN': results1['multi_objective_cnn']})

    df = df[['Modes','CCA','EEG2Code','EEG-Inception','dual-objective CNN']]
    
    plt.rcParams.update({'font.size': fontsize})

    fig=plt.figure(figsize=(25, 7))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)

    sns.set_style("whitegrid")
    dd=pd.melt(df,id_vars=['Modes'],value_vars=['CCA','EEG2Code','EEG-Inception','dual-objective CNN'],var_name='models')
    
    palette = {
    'CCA': 'tab:blue',
    'EEG2Code': 'tab:orange',
    'EEG-Inception': 'tab:red',
    'dual-objective CNN': 'tab:green',
}
    sns.boxplot(x='Modes',y='value',data=dd,hue='models',palette=palette, width=0.3,ax=ax1)
    adjust_box_widths(fig, 0.8)
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
        

    ax1.set_title('{} on {}'.format(eval_txt,dataset_txt),fontsize=fontsize)
    if(eval_mode!='ITR'):
        ax1.set(ylabel=eval_txt)
        ax1.set(yticks=np.arange(0,1.001,0.2))
    else:
        ax1.set(ylabel='bits/min')
        ax1.set(yticks=np.arange(0,150,20))

    plt.grid(False)
    plt.grid(True) 
    
    #ax.legend(bbox_to_anchor=(1.45, 1.02))
    ax1.get_legend().remove()
    
    dataset_txt = '256-channel dataset'
    nreps = 3
        
    group1 = np.repeat('within-subject',nreps)
    group2 = np.repeat('loso-subject',nreps)
    
    
    group = np.concatenate((group1,group2),axis=0)
    df = pd.DataFrame({'Modes':group,
                       'CCA': results2['cca'],
                       'EEG2Code': results2['eeg2code'],
                       'EEG-Inception': results2['inception'],
                        'dual-objective CNN': results2['multi_objective_cnn']})

    df = df[['Modes','CCA','EEG2Code','EEG-Inception','dual-objective CNN']]

    sns.set_style("whitegrid")
    dd=pd.melt(df,id_vars=['Modes'],value_vars=['CCA','EEG2Code','EEG-Inception','dual-objective CNN'],var_name='models')
    
    sns.boxplot(x='Modes',y='value',data=dd,hue='models',palette=palette, width=0.3,ax=ax2)
    adjust_box_widths(fig, 0.8)
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

    ax2.set_title('{} on {}'.format(eval_txt,dataset_txt),fontsize=fontsize)
    ax2.set(ylabel="")
    if(eval_mode!='ITR'):
        ax2.set(yticks=np.arange(0,1.001,0.2))
    else:
        ax2.set(yticks=np.arange(0,150,20))

    ax1.grid(False)
    ax1.grid(True) 
    
    ax2.grid(False)
    ax2.grid(True) 
    
    ax2.legend(bbox_to_anchor=(1.65, 1.02))
    plt.subplots_adjust(wspace=0.3)
    
    filename = "./visualizations/Mean results/Comparison/{}.png".format(eval_txt)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5,transparent=True) 
    plt.close()

def plot_comparison_varaible_time_steps(dataset,mode,results,model):
    fsize=20
    if(model=='cca'):
        model_txt ='CCA'
    elif(model=='eeg2code'):
        model_txt = 'EEG2Code'
    elif(model=='inception'):
        model_txt = 'EEG-Inception'
    else:
        model_txt = 'dual-objective CNN'  
    if(dataset=='8_channel_cVEP'):
        n_folds = 5
        n_subjects = 30
        dataset_txt = '8-channel dataset'
    else:
        n_folds = 3
        n_subjects = 5
        dataset_txt = '256-channel dataset'

    if(mode == 'within_subject'):
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO'

    results_variable_time_steps = {}
    for subj in range(1,n_subjects+1):
        variable_time_steps_all = []
        for fold in range(1,n_folds+1):
            variable_time_steps_all.append(results[subj][fold]['variable_time_steps'])
            if(model=='eeg2code'):
                break

        variable_time_steps_all = np.array(variable_time_steps_all)
        variable_time_steps = np.mean(variable_time_steps_all,axis=0)
        
        results_variable_time_steps[subj] = variable_time_steps
    NUM_COLORS = n_subjects+1
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    time = list(np.round(np.arange(0,2.2,0.1),2))

    sns.reset_orig() 
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 

    plt.rcParams.update({'font.size': fsize})
    fig, ax = plt.subplots(figsize=(20,10))
    for l in results_variable_time_steps.keys():
        
        acc_samples = results_variable_time_steps[l]
        if(model=='inception'):
            acc_samples = acc_samples/100
        if(len(acc_samples)>126):
            t_len = 504
            t_step = 25
        else:
            t_len = 126
            t_step = 6
        #acc_samples[:3] = 0
        samples = np.arange(0,len(acc_samples))
        lines = ax.plot(samples,acc_samples,label=l)
        lines[0].set_color(clrs[l])
        lines[0].set_linestyle(LINE_STYLES[l%NUM_STYLES])

    ax.set_xticks(np.linspace(0,t_len,len(time)), time)
    ax.set_yticks(np.arange(0,1.09,0.1))
    ax.set_ylim((0,1.09))
    ax.set_xlim((5,t_len))
    ax.set_xlabel('Time (s)',fontsize=fsize)
    ax.set_ylabel('Accuracy',fontsize=fsize)
    ax.set_title("{} category accuracy over time steps of {} on {}".format(mode_txt,model_txt,dataset_txt),fontsize=fsize)
    
    if(dataset=='8_channel_cVEP'):
        ax.legend(bbox_to_anchor=(1.02, 1.01),fontsize=fsize)
    else:
        ax.legend(bbox_to_anchor=(1.1, 1.01),fontsize=fsize)
       
    plt.grid(False)
    plt.grid(True)
    
    filename = "./visualizations/Performance over time-steps/Comparison/{}_{}_{}_time_steps.png".format(model_txt,mode_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()

def plot_comparison_itr_time_steps(dataset,mode,results,model):
    fsize=20
    if(model=='cca'):
        model_txt ='CCA'
    elif(model=='eeg2code'):
        model_txt = 'EEG2Code'
    elif(model=='inception'):
        model_txt = 'EEG-Inception'
    else:
        model_txt = 'dual-objective CNN'  
    if(dataset=='8_channel_cVEP'):
        n_folds = 5
        n_subjects = 30
        dataset_txt = '8-channel dataset'
    else:
        n_folds = 3
        n_subjects = 5
        dataset_txt = '256-channel dataset'

    if(mode == 'within_subject'):
        mode_txt = 'Within-subject'
    else:
        mode_txt = 'LOSO'

    results_variable_time_steps = {}
    for subj in range(1,n_subjects+1):
        variable_time_steps_all = []
        for fold in range(1,n_folds+1):
            variable_time_steps_all.append(results[subj][fold]['ITR_time_steps'])
            if(model=='eeg2code'):
                break

        variable_time_steps_all = np.array(variable_time_steps_all)
        variable_time_steps = np.mean(variable_time_steps_all,axis=0)
        results_variable_time_steps[subj] = variable_time_steps

    NUM_COLORS = n_subjects+1
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    time = list(np.round(np.arange(0,2.2,0.1),2))

    sns.reset_orig() 
    clrs = sns.color_palette('husl', n_colors=NUM_COLORS) 

    plt.rcParams.update({'font.size': fsize})
    fig, ax = plt.subplots(figsize=(20,10))
    for l in results_variable_time_steps.keys():
        acc_samples = results_variable_time_steps[l]
        if(len(acc_samples)>126):
            t_len = 504
            t_step = 25
        else:
            t_len = 126
            t_step = 6

        #acc_samples[:3] = 0
        samples = np.arange(0,len(acc_samples))
        lines = ax.plot(samples,acc_samples,label=l)
        lines[0].set_color(clrs[l])
        lines[0].set_linestyle(LINE_STYLES[l%NUM_STYLES])

    ax.set_xticks(np.linspace(0,t_len,len(time)), time)
    if(model=='eeg2code' and mode=='within_subject'):
        ax.set_yticks(np.arange(0,700,50))
        ax.set_ylim((0,700))
    else:
        ax.set_yticks(np.arange(0,420,20))
        ax.set_ylim((0,420))
    ax.set_xlim((5,t_len))
    ax.set_xlabel('Time (s)',fontsize=fsize)
    ax.set_ylabel('bits/min',fontsize=fsize)
    ax.set_title("{} ITR over time steps of {} on {}".format(mode_txt,model_txt,dataset_txt),fontsize=fsize)

    if(dataset=='8_channel_cVEP'):
        ax.legend(bbox_to_anchor=(1.12, 1.01),fontsize=fsize)
    else:
        ax.legend(bbox_to_anchor=(1.1, 1.01),fontsize=fsize)
    plt.grid(False)
    plt.grid(True) 

    filename = "./visualizations/Performance over time-steps/Comparison/{}_{}_{}_ITR_time_steps.png".format(model_txt,mode_txt,dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()

def plot_model_history(results_history, dataset):
    if(dataset=='8_channel_cVEP'):
        n_subjects=30
        n_folds = 5
        dataset_txt = '8-channel dataset'
    else:
        n_subjects = 5
        n_folds = 3
        dataset_txt = '256-channel dataset'

    results = results_history['within_subject']

    loss_folds_subj = []
    sequence_loss_folds_subj = []
    category_loss_folds_subj = []
    sequence_accuracy_folds_subj = []
    category_accuracy_folds_subj = []

    val_loss_folds_subj = []
    val_sequence_loss_folds_subj = []
    val_category_loss_folds_subj = []
    val_sequence_accuracy_folds_subj = []
    val_category_accuracy_folds_subj = []
    for i in range(1,n_subjects+1):
        loss_folds = []
        sequence_loss_folds = []
        category_loss_folds = []
        sequence_accuracy_folds = []
        category_accuracy_folds = []

        val_loss_folds = []
        val_sequence_loss_folds = []
        val_category_loss_folds = []
        val_sequence_accuracy_folds = []
        val_category_accuracy_folds = []

        for j in range(1,n_folds+1):
            loss_folds.append(results[i][j]['history']['loss'])
            sequence_loss_folds.append(results[i][j]['history']['sequence_loss'])
            category_loss_folds.append(results[i][j]['history']['category_loss'])
            sequence_accuracy_folds.append(results[i][j]['history']['sequence_accuracy'])
            category_accuracy_folds.append(results[i][j]['history']['category_accuracy'])

            val_loss_folds.append(results[i][j]['history']['val_loss'])
            val_sequence_loss_folds.append(results[i][j]['history']['val_sequence_loss'])
            val_category_loss_folds.append(results[i][j]['history']['val_category_loss'])
            val_sequence_accuracy_folds.append(results[i][j]['history']['val_category_accuracy'])
            val_category_accuracy_folds.append(results[i][j]['history']['val_category_accuracy'])

        loss_folds_mean = np.mean(np.array(loss_folds),axis=0)
        sequence_loss_folds_mean = np.mean(np.array(sequence_loss_folds),axis=0)
        category_loss_folds_mean = np.mean(np.array(category_loss_folds),axis=0)
        sequence_accuracy_folds_mean = np.mean(np.array(sequence_accuracy_folds),axis=0)
        category_accuracy_folds_mean = np.mean(np.array(category_accuracy_folds),axis=0)

        val_loss_folds_mean = np.mean(np.array(val_loss_folds),axis=0)
        val_sequence_loss_folds_mean = np.mean(np.array(val_sequence_loss_folds),axis=0)
        val_category_loss_folds_mean = np.mean(np.array(val_category_loss_folds),axis=0)
        val_sequence_accuracy_folds_mean = np.mean(np.array(val_sequence_accuracy_folds),axis=0)
        val_category_accuracy_folds_mean = np.mean(np.array(val_category_accuracy_folds),axis=0)

        loss_folds_subj.append(loss_folds_mean)
        sequence_loss_folds_subj.append(sequence_loss_folds_mean)
        category_loss_folds_subj.append(category_loss_folds_mean)
        sequence_accuracy_folds_subj.append(sequence_accuracy_folds_mean)
        category_accuracy_folds_subj.append(category_accuracy_folds_mean)

        val_loss_folds_subj.append(val_loss_folds_mean)
        val_sequence_loss_folds_subj.append(val_sequence_loss_folds_mean)
        val_category_loss_folds_subj.append(val_category_loss_folds_mean)
        val_sequence_accuracy_folds_subj.append(val_sequence_accuracy_folds_mean)
        val_category_accuracy_folds_subj.append(val_category_accuracy_folds_mean)


    loss_folds_subj_mean_ws = np.mean(np.array(loss_folds_subj),axis=0)
    sequence_loss_folds_subj_mean_ws = np.mean(np.array(sequence_loss_folds_subj),axis=0)
    category_loss_folds_subj_mean_ws = np.mean(np.array(category_loss_folds_subj),axis=0)
    sequence_accuracy_folds_subj_mean_ws = np.mean(np.array(sequence_accuracy_folds_subj),axis=0)
    category_accuracy_folds_subj_mean_ws = np.mean(np.array(category_accuracy_folds_subj),axis=0)

    val_loss_folds_subj_mean_ws = np.mean(np.array(val_loss_folds_subj),axis=0)
    val_sequence_loss_folds_subj_mean_ws = np.mean(np.array(val_sequence_loss_folds_subj),axis=0)
    val_category_loss_folds_subj_mean_ws = np.mean(np.array(val_category_loss_folds_subj),axis=0)
    val_sequence_accuracy_folds_subj_mean_ws = np.mean(np.array(val_sequence_accuracy_folds_subj),axis=0)
    val_category_accuracy_folds_subj_mean_ws = np.mean(np.array(val_category_accuracy_folds_subj),axis=0)

    results = results_history['loso_subject']

    loss_folds = []
    sequence_loss_folds = []
    category_loss_folds = []
    sequence_accuracy_folds = []
    category_accuracy_folds = []

    val_loss_folds = []
    val_sequence_loss_folds = []
    val_category_loss_folds = []
    val_sequence_accuracy_folds = []
    val_category_accuracy_folds = []
    for i in range(1,n_subjects+1):
        loss_folds = []
        sequence_loss_folds = []
        category_loss_folds = []
        sequence_accuracy_folds = []
        category_accuracy_folds = []

        val_loss_folds = []
        val_sequence_loss_folds = []
        val_category_loss_folds = []
        val_sequence_accuracy_folds = []
        val_category_accuracy_folds = []

        loss_folds.append(results[i]['history']['loss'])
        sequence_loss_folds.append(results[i]['history']['sequence_loss'])
        category_loss_folds.append(results[i]['history']['category_loss'])
        sequence_accuracy_folds.append(results[i]['history']['sequence_accuracy'])
        category_accuracy_folds.append(results[i]['history']['category_accuracy'])

        val_loss_folds.append(results[i]['history']['val_loss'])
        val_sequence_loss_folds.append(results[i]['history']['val_sequence_loss'])
        val_category_loss_folds.append(results[i]['history']['val_category_loss'])
        val_sequence_accuracy_folds.append(results[i]['history']['val_category_accuracy'])
        val_category_accuracy_folds.append(results[i]['history']['val_category_accuracy'])

    loss_folds_subj_mean_loso = np.mean(np.array(loss_folds),axis=0)
    sequence_loss_folds_subj_mean_loso = np.mean(np.array(sequence_loss_folds),axis=0)
    category_loss_folds_subj_mean_loso = np.mean(np.array(category_loss_folds),axis=0)
    sequence_accuracy_folds_subj_mean_loso = np.mean(np.array(sequence_accuracy_folds),axis=0)
    category_accuracy_folds_subj_mean_loso = np.mean(np.array(category_accuracy_folds),axis=0)

    val_loss_folds_subj_mean_loso = np.mean(np.array(val_loss_folds),axis=0)
    val_sequence_loss_folds_subj_mean_loso = np.mean(np.array(val_sequence_loss_folds),axis=0)
    val_category_loss_folds_subj_mean_loso = np.mean(np.array(val_category_loss_folds),axis=0)
    val_sequence_accuracy_folds_subj_mean_loso = np.mean(np.array(val_sequence_accuracy_folds),axis=0)
    val_category_accuracy_folds_subj_mean_loso = np.mean(np.array(val_category_accuracy_folds),axis=0)

    fsize=20
    plt.rcParams.update({'font.size': fsize})
    fig=plt.figure(figsize=(30, 6))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)

    LINE_STYLES = ['solid', 'solid', 'dashdot', 'dashdot']

    loss_samples_ws1 = sequence_loss_folds_subj_mean_ws
    loss_samples_ws2 = category_loss_folds_subj_mean_ws
    loss_samples_val_ws1 = val_sequence_loss_folds_subj_mean_ws
    loss_samples_val_ws2 = val_category_loss_folds_subj_mean_ws
    
    loss_samples1_loso = sequence_loss_folds_subj_mean_loso
    loss_samples2_loso = category_loss_folds_subj_mean_loso
    loss_samples_val1_loso = val_sequence_loss_folds_subj_mean_loso
    loss_samples_val2_loso = val_category_loss_folds_subj_mean_loso

    acc_samples_ws1 = sequence_accuracy_folds_subj_mean_ws
    acc_samples_ws2 = category_accuracy_folds_subj_mean_ws
    acc_samples_val_ws1 = val_sequence_accuracy_folds_subj_mean_ws
    acc_samples_val_ws2 = val_category_accuracy_folds_subj_mean_ws
    
    acc_samples_loso1 = sequence_accuracy_folds_subj_mean_loso
    acc_samples_loso2 = category_accuracy_folds_subj_mean_loso
    acc_samples_val_loso1 = val_sequence_accuracy_folds_subj_mean_loso
    acc_samples_val_loso2 = val_category_accuracy_folds_subj_mean_loso

    samples = np.arange(0,100)

    lines_loss_ws1 = ax1.plot(samples,loss_samples_ws2,label='within-subject train category')
    lines_loss_ws1[0].set_color("tab:blue")
    lines_loss_ws1[0].set_linestyle('solid')

    lines_val_loss_ws2 = ax1.plot(samples,loss_samples_val_ws2,label='within-subject val category')
    lines_val_loss_ws2[0].set_color("tab:blue")
    lines_val_loss_ws2[0].set_linestyle('dashdot')
    
    lines_loss_loso1 = ax1.plot(samples,loss_samples2_loso,label='LOSO train category')
    lines_loss_loso1[0].set_color("tab:orange")
    lines_loss_loso1[0].set_linestyle('solid')

    lines_val_loss_loso2 = ax1.plot(samples,loss_samples_val2_loso,label='LOSO val category')
    lines_val_loss_loso2[0].set_color("tab:orange")
    lines_val_loss_loso2[0].set_linestyle('dashdot')


    lines_acc_ws1 = ax2.plot(samples,acc_samples_ws2,label='within-subject train category')
    lines_acc_ws1[0].set_color("tab:blue")
    lines_acc_ws1[0].set_linestyle('solid')

    lines_val_acc_ws2 = ax2.plot(samples,acc_samples_val_ws2,label='within-subject val category')
    lines_val_acc_ws2[0].set_color("tab:blue")
    lines_val_acc_ws2[0].set_linestyle('dashdot')
    
    lines_acc_loso1 = ax2.plot(samples,acc_samples_loso2,label='LOSO train category')
    lines_acc_loso1[0].set_color("tab:orange")
    lines_acc_loso1[0].set_linestyle('solid')

    lines_val_acc_loso2 = ax2.plot(samples,acc_samples_val_loso2,label='LOSO val category')
    lines_val_acc_loso2[0].set_color("tab:orange")
    lines_val_acc_loso2[0].set_linestyle('dashdot')


    ax1.set_xticks(np.arange(0,101,20))
    ax1.set_xlabel('Iterations',fontsize=fsize)
    ax2.set_xticks(np.arange(0,101,20))
    ax2.set_xlabel('Iterations',fontsize=fsize)

    ax1.set_yticks(np.arange(0,4.01,0.8))
    ax1.set_ylim((0,4.01))
    ax1.set_ylabel('Loss',fontsize=fsize)
    ax1.set_title("Loss on {}".format(dataset_txt),fontsize=fsize)
    ax1.legend(bbox_to_anchor=(1.01, 1.02),fontsize=fsize)


    ax2.set_yticks(np.arange(0,1.01,0.2))
    ax2.set_ylim((0,1.01))
    ax2.set_ylabel('Accuracy',fontsize=fsize)
    ax2.set_title("Accuracy on {}".format(dataset_txt),fontsize=fsize)
    ax2.legend(bbox_to_anchor=(1.65, 1.02),fontsize=fsize)

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax1.grid(False)
    ax1.grid(True) 
    ax2.grid(False)
    ax2.grid(True) 
    ax2.legend(bbox_to_anchor=(1.05, 1.02),fontsize=fsize)
    plt.subplots_adjust(wspace=0.3)

    filename = "./visualizations/Mean results/Multi-objective cnn/{}_model_history.png".format(dataset)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5, transparent = True) 
    plt.close()

def comparison_significance_test(category_accuracy, datasets):
    from scipy.stats import ttest_ind
    significance_testing = {}
    for i in datasets:
        stat_all = []
        p_all = []
        flag_all = []
        history_all = []
        perfromance_all = []
        mean1_all = []
        mean2_all = []
        significance_testing[i] = {}

        if (i=='8_channel_cVEP'):
            c = 5
        else:
            c = 3
        data_cnn_ws = category_accuracy[i]['multi_objective_cnn'][:c]
        data_cnn_ls = category_accuracy[i]['multi_objective_cnn'][c:]

        data_cca_ws = category_accuracy[i]['cca'][:c]
        data_cca_ls = category_accuracy[i]['cca'][c:]

        data_eeg2code_ws = category_accuracy[i]['eeg2code'][:c]
        data_eeg2code_ls = category_accuracy[i]['eeg2code'][c:]

        data_inception_ws = category_accuracy[i]['inception'][:c]
        data_inception_ls = category_accuracy[i]['inception'][c:]

        stat, p = ttest_ind(data_cnn_ws, data_cca_ws)
        stat_all.append(stat)
        p_all.append(p)
        alpha = 0.05
        if p > alpha:
            flag = 0
        else:
            flag = 1
        flag_all.append(flag)
        history_all.append('cnn_cca_ws')
        mean1_all.append(np.mean(data_cnn_ws))
        mean2_all.append(np.mean(data_cca_ws))
        if(np.mean(data_cnn_ws)>np.mean(data_cca_ws)):
            perfromance_all.append(1)
        else:
            perfromance_all.append(0) 

        stat, p = ttest_ind(data_cnn_ws, data_eeg2code_ws)
        stat_all.append(stat)
        p_all.append(p)
        alpha = 0.05
        if p > alpha:
            flag = 0
        else:
            flag = 1
        flag_all.append(flag)
        history_all.append('cnn_eeg2code_ws')
        mean1_all.append(np.mean(data_cnn_ws))
        mean2_all.append(np.mean(data_eeg2code_ws))
        if(np.mean(data_cnn_ws)>np.mean(data_eeg2code_ws)):
            perfromance_all.append(1)
        else:
            perfromance_all.append(0)

        stat, p = ttest_ind(data_cnn_ws, data_inception_ws)
        stat_all.append(stat)
        p_all.append(p)
        alpha = 0.05
        if p > alpha:
            flag = 0
        else:
            flag = 1
        flag_all.append(flag)
        history_all.append('cnn_inception_ws')
        mean1_all.append(np.mean(data_cnn_ws))
        mean2_all.append(np.mean(data_inception_ws))
        if(np.mean(data_cnn_ws)>np.mean(data_inception_ws)):
            perfromance_all.append(1)
        else:
            perfromance_all.append(0)

        stat, p = ttest_ind(data_cnn_ls, data_cca_ls)
        stat_all.append(stat)
        p_all.append(p)
        alpha = 0.05
        if p > alpha:
            flag = 0
        else:
            flag = 1
        flag_all.append(flag)
        history_all.append('cnn_cca_ls')
        mean1_all.append(np.mean(data_cnn_ls))
        mean2_all.append(np.mean(data_cca_ls))
        if(np.mean(data_cnn_ls)>np.mean(data_cca_ls)):
            perfromance_all.append(1)
        else:
            perfromance_all.append(0)

        stat, p = ttest_ind(data_cnn_ls, data_eeg2code_ls)
        stat_all.append(stat)
        p_all.append(p)
        alpha = 0.05
        if p > alpha:
            flag = 0
        else:
            flag = 1
        flag_all.append(flag)
        history_all.append('cnn_eeg2code_ls')
        mean1_all.append(np.mean(data_cnn_ls))
        mean2_all.append(np.mean(data_eeg2code_ls))
        if(np.mean(data_cnn_ls)>np.mean(data_eeg2code_ls)):
            perfromance_all.append(1)
        else:
            perfromance_all.append(0)


        stat, p = ttest_ind(data_cnn_ls, data_inception_ls)
        stat_all.append(stat)
        p_all.append(p)
        alpha = 0.05
        if p > alpha:
            flag = 0
        else:
            flag = 1
        flag_all.append(flag)
        history_all.append('cnn_inception_ls')
        mean1_all.append(np.mean(data_cnn_ls))
        mean2_all.append(np.mean(data_inception_ls))
        if(np.mean(data_cnn_ls)>np.mean(data_inception_ls)):
            perfromance_all.append(1)
        else:
            perfromance_all.append(0)


        for k in range(0,len(history_all)):
            significance_testing[i][history_all[k]] = {}
            significance_testing[i][history_all[k]]['t-statistic'] = np.round(stat_all[k],3)
            significance_testing[i][history_all[k]]['p'] = np.round(p_all[k],3)
            significance_testing[i][history_all[k]]['significance'] = flag_all[k]
            significance_testing[i][history_all[k]]['performance'] = perfromance_all[k]
            significance_testing[i][history_all[k]]['mean1'] = mean1_all[k]
            significance_testing[i][history_all[k]]['mean2'] = mean2_all[k]

        filename = './visualizations/Significance testing/Comparison/significance_{}.txt'.format(i)
        os.makedirs(os.path.dirname(filename), exist_ok=True)       
        with open(filename,'w') as f: 
            for key, value in significance_testing[i].items(): 
                f.write('%s:%s\n' % (key, value))

def cnn_significance_test(category_accuracy, modes):
    from scipy.stats import ttest_ind
    significance_testing = {}
    
    stat_all = []
    p_all = []
    flag_all = []
    history_all = []
    perfromance_all = []
    mean1_all = []
    mean2_all = []
    significance_testing = {}

    data_cnn_ws8 = category_accuracy['within_subject'][:5]
    data_cnn_ws256 = category_accuracy['within_subject'][5:]

    data_cnn_ls8 = category_accuracy['loso_subject'][:5]
    data_cnn_ls256 = category_accuracy['loso_subject'][5:]

    stat, p = ttest_ind(data_cnn_ws8, data_cnn_ls8)
    stat_all.append(stat)
    p_all.append(p)
    alpha = 0.05
    if p > alpha:
        flag = 0
    else:
        flag = 1
    flag_all.append(flag)
    history_all.append('cnn_ws_ls8')
    mean1_all.append(np.mean(data_cnn_ws8))
    mean2_all.append(np.mean(data_cnn_ls8))
    if(np.mean(data_cnn_ws8)>np.mean(data_cnn_ls8)):
        perfromance_all.append(1)
    else:
        perfromance_all.append(0) 

    stat, p = ttest_ind(data_cnn_ws256, data_cnn_ls256)
    stat_all.append(stat)
    p_all.append(p)
    alpha = 0.05
    if p > alpha:
        flag = 0
    else:
        flag = 1
    flag_all.append(flag)
    history_all.append('cnn_ws_ls256')
    mean1_all.append(np.mean(data_cnn_ws256))
    mean2_all.append(np.mean(data_cnn_ls256))
    if(np.mean(data_cnn_ws256)>np.mean(data_cnn_ls256)):
        perfromance_all.append(1)
    else:
        perfromance_all.append(0)

    for k in range(0,len(history_all)):
        significance_testing[history_all[k]] = {}
        significance_testing[history_all[k]]['t-statistic'] = np.round(stat_all[k],3)
        significance_testing[history_all[k]]['p'] = np.round(p_all[k],3)
        significance_testing[history_all[k]]['significance'] = flag_all[k]
        significance_testing[history_all[k]]['performance'] = perfromance_all[k]
        significance_testing[history_all[k]]['mean1'] = mean1_all[k]
        significance_testing[history_all[k]]['mean2'] = mean2_all[k]

    filename = './visualizations/Significance testing/Multi-objective CNN/significance.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)       
    with open(filename,'w') as f: 
        for key, value in significance_testing.items(): 
            f.write('%s:%s\n' % (key, value))