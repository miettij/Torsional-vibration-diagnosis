from torch.utils.data import Dataset

import pandas as pd
import numpy as  np
from numpy import diff
import torch

def get_velocity(Y,t):
    """returns DY/dt, used to compute angular velocity (Hz) from encoder angle
    signal and time signal."""
    dydx = diff(Y)/diff(t)
    return dydx/360

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

def load_data(filepaths, start, stop, debug_flag = False, input_channels = ['acc1', 'acc2', 'acc3', 'acc4']):
    datalist = []
    enc_keys = []
    for i, key in enumerate(input_channels):
        #print(i, key)
        if 'enc' in key:

            enc_keys.append(input_channels[i])
            #print('popped',i , input_channels[i],'\n')
    input_channels = [x for x in input_channels if x not in enc_keys]
    # print("Input channels:\n",input_channels)
    # print("enc keys: ",enc_keys)
    for filepath in filepaths:
        #print(filepath)
        data = pd.read_csv(filepath, sep = ';')
        #print(data)

        if len(input_channels)>0:
            acc_and_t_data = data[input_channels][:-1001].to_numpy()
            #print(acc_and_t_data.shape)
        #print(data)
        if len(enc_keys) >0:
            enc_arrs = []
            for enc_key in enc_keys:
                idx = enc_key[-1]
                #print(idx, enc_key)
                anglekey = 'en'+idx+'angle'
                timekey = 'en'+idx+'time'
                velocity = get_velocity(np.abs(data[anglekey][:-1000]),data[timekey][:-1000])
                enc_arrs.append(velocity)
                #print(velocity.shape)

            enc_vels = np.array(enc_arrs).T
            #print(enc_vels.shape)

        if len(input_channels)>0 and len(enc_keys) > 0:
            data = np.hstack((acc_and_t_data, enc_vels))
        elif len(input_channels)>0:
            data = acc_and_t_data
        else:
            data = enc_vels
        #print(data.shape)


        len_data = data.shape[0]

        start_idx = int(len_data*start)
        if stop < 0:
            stop = 1
        stop_idx = int(len_data*stop)
        data = data[start_idx:stop_idx,:]
        #print(data)
        label = get_meta(filepath)
        #print(label)
        datalist.append((data,label))
        if debug_flag:
            print(data.dtype)
            print("debug_mode is on")
            break
    return datalist

def time_window_division(items, stride, window_len):
    iterable_items = []
    for item in items:
        start = 0
        stop = start+window_len
        #print(item[0].shape)
        arr_len = item[0].shape[0]
        #print(arr_len)
        arr = item[0].T

        label = item[1]
        while stop < arr_len-window_len:
            #print(start, stop)
            iterable_items.append((torch.Tensor(arr[:,start:stop]),label))
            start += stride
            stop = start+window_len
    return iterable_items


class Gear_Dataset(Dataset):
    """
    Simple Dataset for gear fault diagnosis.
    Note datasplit indexes:
    - With default datasplit arguments this class loads the complete data in the
    given filepaths.
    - Otherwise:
        datasplit start and stop parameters define the proportions of the loaded
        data. I.e. datasplit_start = 0, and datasplit_stop = 0.3 loads the first
        30 % of each file given by the filepaths.


    """
    def __init__(self, filepaths, args, datasplit_start = 0, datasplit_stop = -1, train_flag = True, input_channels = ['acc1', 'acc2', 'acc3', 'acc4']): #

        print('loading data')
        self.items = load_data(filepaths,datasplit_start, datasplit_stop, debug_flag = False, input_channels = input_channels)
        print("num items: ",len(self.items))
        if train_flag:
            print('train time window division:')
            self.items = time_window_division(self.items, args.tw_stride, args.tw_len)
        else:
            print('test time window division:')
            self.items = time_window_division(self.items, args.tw_len, args.tw_len)
        self.length = len(self.items)
        print('num items in dataset',self.length)

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        return self.items[idx]
