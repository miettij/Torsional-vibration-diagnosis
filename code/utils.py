from collections import OrderedDict
import datetime
import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_filepaths(root_dir):


    subdirs = os.listdir(root_dir)
    filepaths = []
    if '.DS_Store' in subdirs:
        subdirs.remove('.DS_Store')
    for subdir in subdirs:
        files = os.listdir(root_dir+subdir)
        if '.DS_Store' in files:
            files.remove('.DS_Store')

        for file in files:
            path = os.path.join(root_dir,subdir,file)
            filepaths.append(path)
    return filepaths

def get_meta(filepath):
    
    labeldict = {'no_sim_csv':0,
                '001_1sim_csv':1,
                '001_2sim_csv':2,
                '001_3sim_csv':3,
                '003_1sim_csv':4,
                '003_2sim_csv':5,
                '003_3sim_csv':6,
                '005_1sim_csv':7,
                '005_2sim_csv':8,
                '005_3sim_csv':9,
                }

    simtype = filepath.split('/')[-2]

    return labeldict[simtype]


def save_history(history,SAVEPATH,model_name):
    fig,axes = plt.subplots()
    axes.set_yscale("log")
    l1, = axes.plot(history['train'],label = "Training error")
    l2, = axes.plot(history['val'], label = "Validation error")
    plt.legend(handles=[l1,l2])
    axes.set_title('Convergence of '+model_name)
    axes.set_ylabel('CrossEntropyLoss')
    axes.set_xlabel('Epoch')
    fig.savefig(SAVEPATH+'/'+model_name+'.png')
    plt.close(fig)


if __name__ == '__main__':
    filepaths = get_filepaths('../original.tmp/')
    #print(filepaths)
    print(len(filepaths))
    load_data(filepaths,0.33,0.55,debug_flag=True)
