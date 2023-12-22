import torch
import numpy as np
import os
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
import datetime


class Dataset_Traj(Dataset):
    def __init__(self, root_path, flag='train', city='bj', size=None,embedding_model='HHGCLV3'):
        if city == 'bj':
            self.seq_len = 128
        else:
            self.seq_len = 64
        self.root_path = root_path
        self.city = city
        self.flag = flag
        self.embedding_model = embedding_model

        self.__read_data__()

    def __read_data__(self):
        traj_raw = pd.read_csv(self.root_path, delimiter=';')
        traj_road_ids = []
        traj_times = []
        #按照8：1：1切分
        split_list = [0,math.floor(0.8*len(traj_raw)),math.floor(0.9*len(traj_raw)),len(traj_raw)]
        flag_map = {'train': 0, 'val': 1, 'test': 2}
        i = split_list[flag_map[self.flag]]
        j = split_list[flag_map[self.flag]+1]
        self.traj_raw = traj_raw.iloc[i:j].reset_index(drop=True)
        
    
    def process_traj_raw(self, i):
        path = self.traj_raw.loc[i, 'path']
        path = path[1:len(path) - 1].split(',')
        path = [int(s) for s in path]
        pad_len = max(self.seq_len - len(path), 0)
        if pad_len > 0:
            traj_len = len(path)
            path = path + ([-100] * pad_len)
        else:
            path = path[:self.seq_len]
            traj_len = self.seq_len

        tlist = self.traj_raw.loc[i, 'tlist']
        tlist = tlist[1:len(tlist) - 1].split(',')
        tlist = [int(t) for t in tlist]
        if pad_len > 0:
            tlist = tlist + ([-100] * pad_len)
        else:
            tlist = tlist[:self.seq_len]
        
        traj_minute_indexs = []
        traj_week_indexs = []

        for traj_times_item in tlist:
            traj_minute_index, traj_week_index = self.process_timestamp(traj_times_item)
            traj_minute_indexs.append(traj_minute_index)
            traj_week_indexs.append(traj_week_index)

        start_timestamp = datetime.datetime.utcfromtimestamp(int(tlist[0]))
        end_timestamp = datetime.datetime.utcfromtimestamp(int(tlist[traj_len-1]))
        traj_eta = (end_timestamp - start_timestamp).seconds

        hours = traj_eta // 3600
        minutes = (traj_eta % 3600) // 60
        seconds = (traj_eta % 60)

        return path, traj_minute_indexs, traj_week_indexs, traj_len, traj_eta, [hours, minutes, seconds]
    
    def process_timestamp(self, traj_times_item_road):
        if traj_times_item_road == -100:
            return -100, -100
        road_timestamp = datetime.datetime.utcfromtimestamp(int(traj_times_item_road))
        return road_timestamp.hour * 60 + road_timestamp.minute + 1 , road_timestamp.weekday() + 1 
    
    def __getitem__(self, index):
        path, traj_minute_indexs, traj_week_indexs, traj_len, traj_eta, traj_eta_list = self.process_traj_raw(index)
        return torch.tensor(path), torch.tensor(traj_minute_indexs), torch.tensor(traj_week_indexs), torch.tensor(traj_len), torch.tensor(traj_eta), torch.tensor(traj_eta_list)
    
    def __len__(self):
        return len(self.traj_raw)
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) 
    
    




class Dataset_Traj_no_padding(Dataset):
    def __init__(self, root_path, flag='train', city='bj', size=None,embedding_model='HHGCLV3'):
        if city == 'bj':
            self.seq_len = 128
        else:
            self.seq_len = 64
        self.root_path = root_path
        self.city = city
        self.flag = flag
        self.embedding_model = embedding_model

        self.__read_data__()

    def __read_data__(self):
        traj_raw = pd.read_csv(self.root_path, delimiter=';')
        traj_road_ids = []
        traj_times = []
        #按照8：1：1切分
        split_list = [0,math.floor(0.8*len(traj_raw)),math.floor(0.9*len(traj_raw)),len(traj_raw)]
        flag_map = {'train': 0, 'val': 1, 'test': 2}
        i = split_list[flag_map[self.flag]]
        j = split_list[flag_map[self.flag]+1]
        self.traj_raw = traj_raw.iloc[i:j].reset_index(drop=True)
        
    
    def process_traj_raw(self, i):
        path = self.traj_raw.loc[i, 'path']
        path = path[1:len(path) - 1].split(',')
        path = [int(s) for s in path]
        pad_len = max(self.seq_len - len(path), 0)
        if pad_len > 0:
            traj_len = len(path)
            path = path 
        else:
            path = path[:self.seq_len]
            traj_len = self.seq_len

        tlist = self.traj_raw.loc[i, 'tlist']
        tlist = tlist[1:len(tlist) - 1].split(',')
        tlist = [int(t) for t in tlist]
        if pad_len > 0:
            tlist = tlist
        else:
            tlist = tlist[:self.seq_len]
        
        traj_minute_indexs = []
        traj_week_indexs = []

        for traj_times_item in tlist:
            traj_minute_index, traj_week_index = self.process_timestamp(traj_times_item)
            traj_minute_indexs.append(traj_minute_index)
            traj_week_indexs.append(traj_week_index)

        start_timestamp = datetime.datetime.utcfromtimestamp(int(tlist[0]))
        end_timestamp = datetime.datetime.utcfromtimestamp(int(tlist[traj_len-1]))
        traj_eta = (end_timestamp - start_timestamp).seconds

        hours = traj_eta // 3600
        minutes = (traj_eta % 3600) // 60
        seconds = (traj_eta % 60)

        return path, traj_minute_indexs, traj_week_indexs, traj_len, traj_eta, [hours, minutes, seconds]
    
    def process_timestamp(self, traj_times_item_road):

        road_timestamp = datetime.datetime.utcfromtimestamp(int(traj_times_item_road))
        return road_timestamp.hour * 60 + road_timestamp.minute + 1 , road_timestamp.weekday() + 1 
    
    def __getitem__(self, index):
        path, traj_minute_indexs, traj_week_indexs, traj_len, traj_eta, traj_eta_list = self.process_traj_raw(index)
        return torch.tensor(path), torch.tensor(traj_minute_indexs), torch.tensor(traj_week_indexs), torch.tensor(traj_len), torch.tensor(traj_eta), torch.tensor(traj_eta_list)
    
    def __len__(self):
        return len(self.traj_raw)
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) 
