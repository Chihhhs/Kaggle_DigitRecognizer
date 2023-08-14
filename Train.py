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

num_classes =10
num_epochs = 4
batch_size = 4 

class TrainDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(TRAIN_FILE,delimiter=",",dtype=np.float32 ,skiprows=1)
        self.img = torch.from_numpy(xy[:, 1:].reshape(-1,28,28))
        self.label = torch.from_numpy(xy[:, 0])
        self.n_samples =xy.shape[0]
        
    def __getitem__(self,index):
        return self.img[index],self.label[index]
    
    def __len__(self):
        return self.n_samples

data = TrainDataset()


dataloader =DataLoader(dataset=data , batch_size =batch_size,shuffle =True,num_workers=0)

# if os.path.exists(MODEL_FILE):
#     model.load_state_dict(torch.load(MODEL_FILE))


classes = ('1' , '2' , '3' , '4', '5','6','7','8','9','0')

class ConvNeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32,3)
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self,x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0),-1) 
        # print(x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
       
        return x

if __name__ =="__main__":      
    model = ConvNeuralNet(num_classes)
    criterion =nn.CrossEntropyLoss()
    optimizer =torch.optim.Adam(model.parameters(),lr =0.01)

    n_total_steps = len(dataloader)
    for epoch in range(num_epochs):
        for i ,(images,labels) in enumerate(dataloader):
            images = images.unsqueeze(1)  # Adds an extra channel dimension
            labels = labels.to(torch.long)
            # print(images.shape)
            
            outputs=model(images)
            loss =criterion(outputs,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if(i+1) %5000 ==0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    print("Finish Training.")

    torch.save(model.state_dict(),MODEL_FILE)

