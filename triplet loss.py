#triplet loss的具体使用：加载数据，划分三元组，网络结构设计，找到进一步的分类方法

import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
from xgboost import XGBClassifier

PATH = "/kaggle/input/digit-recognizer/"

torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()

embedding_dims = 2
batch_size = 32
epochs = 50


dataset = np.load('/home/zhangtongze/homework/datasets_class/processed_npy/second_proj/adni_raw_normalized_data.npy')
train_df = dataset[:int(dataset.shape[0] * 0.8),:]
test_df = dataset[int(dataset.shape[0] * 0.8):,:]

class MNIST(Dataset):
    def __init__(self, df, train=True, transform=None):
        self.is_train = train
        self.transform = transform
        self.dataset = df
        
        if self.is_train:            
            self.features = self.dataset[:,:-1]
            self.labels = self.dataset[:,-1]
        else:
            self.features = self.dataset[:,:-1]
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, item):
        anchor_features = self.features[item,:]
        
        if self.is_train:
            anchor_label = self.dataset[item,-1]

            positive_list = np.where(self.dataset[:,-1] == anchor_label)

            positive_item = random.choice(positive_list)
            positive_features = self.features[positive_item,:]
            
            negative_list = np.where(self.dataset[:,-1] != anchor_label)
            negative_item = random.choice(negative_list)
            negative_features = self.features[negative_item]
            
            if self.transform:
                anchor_features = self.transform(anchor_features)
                positive_features = self.transform(positive_features)
                negative_features = self.transform(negative_features)

            return anchor_features, positive_features, negative_features, anchor_label
        
        else:
            if self.transform:
                anchor_features = self.transform(anchor_features)
            return anchor_features

train_ds = MNIST(train_df, 
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

test_ds = MNIST(test_df, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


#这里的网络结构需要大改，暂时不能用
class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.PReLU(),
            nn.Linear(512, emb_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x
