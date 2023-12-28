import torch
import torch.nn as nn
from torch.utils.data import DataLoader ,Dataset
from sklearn.model_selection import train_test_split
import numpy as np

MODEL_FILE ="./model/Kaggle_Mnist.pt"
FILE = "./data/train.csv"

num_classes =10
num_epochs = 10
batch_size = 220

Total_data = np.loadtxt(FILE,delimiter=",",dtype=np.float32 ,skiprows=1)

# Spilt train and test
train_data, test_data = train_test_split(Total_data, test_size=0.2, random_state=42)

class CusDataset(Dataset):
    def __init__(self,data):
        # train_data loading
        self.img = torch.from_numpy(data[:, 1:].reshape(-1,1,28,28))
        self.label = torch.from_numpy(data[:, 0]).to(torch.int64)
        self.n_samples =data.shape[0] # batch_size
        
    def __getitem__(self,index):
        return self.img[index],self.label[index]
    
    def __len__(self):
        return self.n_samples

train_data = CusDataset(train_data)
test_data = CusDataset(test_data)

train_loader =DataLoader(dataset=train_data , batch_size =batch_size,shuffle =True,num_workers=0)
test_loader =DataLoader(dataset=test_data , batch_size =batch_size,shuffle =False,num_workers=0)

class ConvNeuralNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvNeuralNet, self).__init__()
        # output_size = (input_size + 2 * padding - kernel_size) / stride + 1
        self.conv1 = nn.Conv2d(1, 16, 5)  #28-5+1 =24
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32, 5)
        self.fc1 = nn.Linear(32*4*4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self,x):
        x = self.conv1(x) # 24
        x = self.relu(x) 
        x = self.pool(x) # 12
        x = self.conv2(x) # 8
        x = self.relu(x) 
        x = self.pool(x) # 4
        # print(x.shape)
        x = x.view(x.size(0),-1) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = ConvNeuralNet(num_classes)

def train():
    criterion =nn.CrossEntropyLoss()
    optimizer =torch.optim.Adam(model.parameters(),lr =0.001)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i ,(images,labels) in enumerate(train_loader):
            
            outputs=model(images)
            loss =criterion(outputs,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if(i+1) % 153 ==0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.8f}')
    print("Finish Training.")
    torch.save(model.state_dict(),MODEL_FILE)
    
def test():
    with torch.no_grad():
        n_correct =0
        n_samples =0
        for images , labels in test_loader:
            outputs =model(images)
            # max return (value , index)
            _,predicted = torch.max(outputs,1)
            n_samples += labels.size(0)
            n_correct +=(predicted ==labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network:{acc:.4f} %')

if __name__ =="__main__":      
    train()
    test()
