#这个文件主要用于勘察各个数据集的格式以及具体内容

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
from xgboost import XGBClassifier


class show_data():
    def __init__(self, filepath, filename):
        self.filepath = filepath
        self.filename = filename
        self.dataset = np.load(self.filepath + '/' + self.filename)
        

    def shape(self):
        print(self.dataset.shape)   

    def content(self):
        print(self.dataset)    


class debug_dataloader():
    def __init__(self,filepath):
        self.filepath = filepath
        self.dataset = np.load(self.filepath)
        self.features = self.dataset[:,:-1]
        self.simulate_dataloader(15)
        

    def simulate_dataloader(self,item):
        anchor_features = self.features[item,:]
        anchor_label = self.dataset[item,-1]

        positive_list = np.array(np.where(self.dataset[:,-1] == anchor_label))
        positive_list = np.reshape(positive_list, positive_list.shape[1])
        positive_item = np.random.choice(positive_list,1)
        positive_features = self.features[positive_item[0],:]
        
        negative_list = np.array(np.where(self.dataset[:,-1] != anchor_label))
        
        negative_list = negative_list.reshape(negative_list.shape[1])
        negative_item = np.random.choice(negative_list,1)
        negative_features = self.features[negative_item[0],:]
        
        print(anchor_features.shape)
        print(positive_features.shape)
        print(negative_features.shape)



if __name__ =='__main__':
    # filepath = '/home/zhangtongze/homework/datasets_class/npy'
    # filename1 = 'ADNI_AD.npy'
    # filename2 = 'ADNI_NC.npy'
    # show_data(filepath, filename1).shape()
    # show_data(filepath, filename1).content()
    # show_data(filepath, filename2).shape()
    debug_dataloader('/home/zhangtongze/homework/datasets_class/processed_npy/second_proj/ppmi_raw_normalized_data.npy')
