#这个文件主要用于勘察各个数据集的格式以及具体内容

import numpy as np


class show_data():
    def __init__(self, filepath, filename):
        self.filepath = filepath
        self.filename = filename
        self.dataset = np.load(self.filepath + '/' + self.filename)
        

    def shape(self):
        print(self.dataset.shape)   

    def content(self):
        print(self.dataset)    



if __name__ =='__main__':
    filepath = '/home/zhangtongze/homework/datasets_class/npy'
    filename1 = 'ADNI_AD.npy'
    filename2 = 'ADNI_NC.npy'
    show_data(filepath, filename1).shape()
    show_data(filepath, filename1).content()
    # show_data(filepath, filename2).shape()
