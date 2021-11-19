#这个文件主要用于数据预处理，写成一个一个的类，方便调用（自己处理自己的数据）


import os
from os import terminal_size

import h5py
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# 这里要进行的步骤：
# 1.导入数据，将二者整合成一个数据集
# 2.对数据进行归一化
# 3.给数据集加标签
class PPMI_normalization():
    def __init__(self):
        self.ppmi_nc_path = '/home/zhangtongze/homework/datasets_class/npy/PPMI_NC.npy'
        self.ppmi_pd_path = '/home/zhangtongze/homework/datasets_class/npy/PPMI_PD.npy'
        self.dataset_nc = np.load(self.ppmi_nc_path,allow_pickle=True)
        self.dataset_pd = np.load(self.ppmi_pd_path,allow_pickle = True)
        print(self.dataset_nc.shape)
        print(self.dataset_pd.shape)
        self.dataset_all = np.vstack((self.dataset_nc,self.dataset_pd))
        self.normalization()
        self.add_label()

    def normalization(self):
        dataset_mean = np.mean(self.dataset_all)
        dataset_var = np.var(self.dataset_all)
        self.dataset_normalize_all = (self.dataset_all - dataset_mean) / dataset_var

    def add_label(self):
        label = np.zeros((self.dataset_normalize_all.shape[0],1))
        for i in range(self.dataset_nc.shape[0]):
            label[i] = 1
        self.dataset_normalized_labeled = np.hstack((self.dataset_normalize_all,label))
        print(self.dataset_normalized_labeled.shape)
        np.save('/home/zhangtongze/homework/datasets_class/processed_npy/second_proj/' + 'ppmi_raw_normalized_data.npy',self.dataset_normalized_labeled)

class ADNI_normalization():
    def __init__(self):
        self.adni_ad_path = '/home/zhangtongze/homework/datasets_class/npy/ADNI_AD.npy'
        self.adni_nc_path = '/home/zhangtongze/homework/datasets_class/npy/ADNI_NC.npy'
        self.dataset_ad = np.load(self.adni_ad_path)
        self.dataset_nc = np.load(self.adni_nc_path)
        print(self.dataset_ad.shape)
        print(self.dataset_nc.shape)
        self.dataset_all = np.vstack((self.dataset_ad,self.dataset_nc))
        print(self.dataset_all.shape)
        self.normalization()
        self.add_label()

    def normalization(self):
        dataset_mean = np.mean(self.dataset_all)
        dataset_var = np.var(self.dataset_all)
        self.dataset_normalize_all = (self.dataset_all - dataset_mean) / dataset_var

    def add_label(self):
        label = np.zeros((self.dataset_normalize_all.shape[0],1))
        for i in range(self.dataset_ad.shape[0]):
            label[i] = 1
        self.dataset_normalized_labeled = np.hstack((self.dataset_normalize_all,label))
        print(self.dataset_normalized_labeled.shape)
        print(self.dataset_normalized_labeled)
        np.save('/home/zhangtongze/homework/datasets_class/processed_npy/second_proj/' + 'adni_raw_normalized_data.npy',self.dataset_normalized_labeled)



    
if __name__ =='__main__':
    # model = PPMI_normalization()
    ADNI_normalization()
