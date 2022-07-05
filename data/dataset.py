from torch.utils.data import Dataset
import pickle
import os
import numpy as np
import torch.nn.functional as F

import torch
import csv
import dask.dataframe as dd
osp = os.path

datadir = '/home/liyuan/LSTM/data/dataset'
class_list = [1,2,3,4,5]


class Emodata(Dataset):
    def __init__(self):
        super(Emodata,self).__init__
        self.classList = class_list
        self.lenCls = {}
        numData = []
        for cls in self.classList:
            tmpDir = osp.join(datadir,str(cls))
            len_ = len(os.listdir(tmpDir))
            self.lenCls[cls] = len_
            numData.append(len_)
        self.len = np.sum(numData)
    
    def reindex(self,index):
        [*num_c],[*len_c] = zip(*self.lenCls.items())
        i = 0
        while index - sum(len_c[:i+1]) >= 0:
            i = i+1
        index = index - sum(len_c[:i])
        cls = num_c[i]
        #print(cls,index)
        return cls,index 


    def __len__(self):
        return self.len

    def __getitem__(self,index):
        label,reid = self.reindex(index)
        filename = str(reid)+'.csv'
        datapath = osp.join(datadir,str(label),filename)

        data = dd.read_csv(datapath,encoding = 'UTF-8')
        data_tensor = torch.Tensor(data.values.compute()).float()
        # print(data.values.compute().shape)
        # print(data_tensor.size())
        #label = F.one_hot(torch.Tensor(label))
        return data_tensor,label
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = Emodata()
    dloader = DataLoader(data, batch_size = 1, shuffle=True)
    inputs, labels = next(iter(dloader))
    print(inputs.size(),labels)