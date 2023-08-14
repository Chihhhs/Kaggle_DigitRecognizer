import torch
import torch.nn as nn
from torch.utils.data import DataLoader ,Dataset
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
import csv
import pandas as pd

MODEL_FILE ="./model/Kaggle_Mnist.pt"
TRAIN_FILE = "./digit-recognizer/train.csv"
TEST_FILE = "./digit-recognizer/test.csv"

num_classes =10
num_epochs = 4
batch_size = 4 
classes = ('1' , '2' , '3' , '4', '5','6','7','8','9','0')

class TrainDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(TRAIN_FILE,delimiter=",",dtype=np.float32 ,skiprows=1)
        self.img = torch.from_numpy(xy[:, 1:].reshape(-1,1,28,28))
        self.label = torch.from_numpy(xy[:, 0])
        self.n_samples =xy.shape[0]
        
    def __getitem__(self,index):
        return self.img[index],self.label[index]
    
    def __len__(self):
        return self.n_samples


class TestDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(TEST_FILE,delimiter=",",dtype=np.float32 ,skiprows=1)
        self.img = torch.from_numpy(xy[:,0:].reshape(-1,28,28))
        self.n_samples =xy.shape[0]
        
    def __getitem__(self,index):
        return self.img[index]
    
    def __len__(self):
        return self.n_samples


data = TrainDataset()
data_loader =DataLoader(dataset=data , batch_size =batch_size,shuffle =True)


test = TestDataset()
test_loader = DataLoader(dataset=test ,batch_size = batch_size,shuffle =True)

# if os.path.exists(MODEL_FILE):
#     model.load_state_dict(torch.load(MODEL_FILE))

example = iter(test_loader)
sample,labels =next(example)
# sample.unsqueeze(1)

print(sample.shape)