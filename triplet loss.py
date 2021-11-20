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
batch_size = 1
epochs = 25


dataset = np.load('/home/zhangtongze/homework/datasets_class/processed_npy/second_proj/adni_raw_normalized_data.npy')
np.random.shuffle(dataset)
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
        
    def __getitem__(self, item):
        anchor_features = self.features[item,:]
        
        if self.is_train:
            anchor_label = self.dataset[item,-1]

            positive_list = np.where(self.dataset[:,-1] == anchor_label)

            positive_item = random.choice(positive_list)
            positive_features = self.features[positive_item,:]
            
            negative_list = np.where(self.dataset[:,-1] != anchor_label)
            negative_item = random.choice(negative_list)
            negative_features = self.features[negative_item,:]
            
            if self.transform:
                anchor_features = torch.from_numpy(anchor_features.astype(np.float))
                positive_features = torch.from_numpy(positive_features.astype(np.float))
                negative_features = torch.from_numpy(negative_features.astype(np.float))

            return anchor_features, positive_features, negative_features, anchor_label
        
        else:
            if self.transform:
                anchor_features = torch.from_numpy(anchor_features)
            return anchor_features

    def __len__(self):
        return self.dataset.shape[0]

train_ds = MNIST(train_df, 
                 train=True,
                 transform = True)
print(train_ds.dataset.shape)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)


test_ds = MNIST(test_df, train=False, transform=True)
print(test_ds.dataset.shape)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


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
            nn.Conv1d(1, 32, 4, stride = 2),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 4, stride = 2),
            nn.PReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(186, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
            nn.Linear(16, emb_dim)
        )
        
    def forward(self, x):
        #x = self.conv(x)
        # x = x.view(-1, 64*21*21)
        x = x.float()
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

model = Network(embedding_dims)
model.apply(init_weights)
model = torch.jit.script(model).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.jit.script(TripletLoss())

model.train()
for epoch in range(epochs):
    running_loss = []
    for step, (anchor_features, positive_features, negative_features, anchor_label) in enumerate(train_loader):
        anchor_features = anchor_features.to(device)
        positive_features = positive_features.to(device)
        negative_features = negative_features.to(device)
        
        optimizer.zero_grad()
        anchor_out = model(anchor_features)
        positive_out = model(positive_features)
        negative_out = model(negative_features)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.cpu().detach().numpy())
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))

    
train_results = []
labels = []

model.eval()
with torch.no_grad():
    for features, _, _, label in train_loader:
        train_results.append(model(features.to(device)).cpu().numpy())
        labels.append(label)
        
train_results = np.concatenate(train_results)
labels = np.concatenate(labels)
print(train_results.shape)

plt.figure(figsize=(15, 10), facecolor="azure")
for label in np.unique(labels):
    tmp = train_results[labels==label]
    plt.scatter(tmp[:, 0], tmp[:, 1], label=label)

plt.legend()
plt.show()
plt.savefig('/home/zhangtongze/homework/result_visulize/train_result.png')


tree = XGBClassifier(seed=2020)
tree.fit(train_results, labels)

test_results = []
test_labels = []

model.eval()
with torch.no_grad():
    for features in test_loader:
        test_results.append(model(features.to(device)).cpu().numpy())
        
test_results = np.concatenate(test_results)

plt.figure(figsize=(15, 10), facecolor="azure")
plt.scatter(test_results[:, 0], test_results[:, 1], label=label)
plt.savefig('/home/zhangtongze/homework/result_visulize/test_result.png')



test_results.shape
label = tree.predict(test_results)
print(label.shape)
print(label)
print(test_df[:,-1])

acc = 0
for i in range(label.shape[0]):
    if label[i] == test_df[i,-1]:
        acc += 1

print(acc)
