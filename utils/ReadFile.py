import torch
import numpy as np
import pandas as pd


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


# if use cuda
use_cuda = torch.cuda.is_available()


class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]




def Load_Dataset(batch_size=128,seq_len=3,pre_len=1,dataset='SH_Park',norm=True): 
    # Specially designed for ATGCN
    # Normalized=True

    if dataset=='SH_Park':
        path='../data/SH_Park/SH_Park_10_d.csv'
    elif dataset=='CN_AQI':
        path='../data/CN_AQI/AQI_data.csv'
    elif dataset=='Metr-LA':
        path='../data/Metr-LA/Metr_LA.csv'
    elif dataset=='PeMS08':
        path='../data/PeMS08/PeMS08_Flow.csv' 



    raw_data=pd.read_csv(path).values #(time_slots,node_num)
    min_val=np.min(raw_data)
    max_val=np.max(raw_data)
    if norm==True:
        raw_data=(raw_data-min_val)/(max_val-min_val)

    time_slot=raw_data.shape[0]
    node_num=raw_data.shape[1]

    # create dataset by sliding windows
    x,y=[],[]
    for iter in range(time_slot-seq_len-pre_len+1):
        x.append(list(raw_data[iter:iter+seq_len,:]))     
        y.append(list(raw_data[iter+seq_len:iter+seq_len+pre_len,:]))

    print('Scene:'+dataset)
    print("Node Number: "+str(node_num))
    print("batch_size:"+str(batch_size))
    
   
    x=torch.from_numpy(np.array(x,dtype=float))
    y=torch.from_numpy(np.array(y,dtype=float))

    # convert dtype
    x=x.type(torch.float32) #(,seq_len,node_num)
    y=y.type(torch.float32)#(,pre_len,node_num)


    # split dataset
    data_set=MyDataset(x,y)
    train_ratio=0.6
    val_ratio=0.2
    num_sample = len(data_set)
    num_train = int(train_ratio * num_sample)
    num_val = int(val_ratio * num_sample)
    num_test = num_sample - num_train - num_val


    train_data, val_data, test_data = random_split(data_set, [num_train, num_val, num_test])


    # 创建DataLoader加载数据
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
  

    return train_loader,val_loader,test_loader,node_num,max_val,min_val




