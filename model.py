import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

MODEL_FILE ="./model/Kaggle_Mnist.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784
hidden_size = 100
num_classes =10
num_epochs = 4 
batch_size =100
# Learning Rate Decay 
learning_rate =0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root ='./data' , train =True,transform=transforms.ToTensor(),download =True)
test_dataset = torchvision.datasets.MNIST(root ='./data' , train =False,transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset 
                                          ,batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset =test_dataset
                                         ,batch_size=batch_size,shuffle=False)

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

def train():
    model = ConvNeuralNet(num_classes)

    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))

    model.to(device)

    criterion =nn.CrossEntropyLoss()
    optimizer =torch.optim.Adam(model.parameters(),lr =learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i ,(images,labels) in enumerate(train_loader):
            outputs=model(images)
            loss =criterion(outputs,labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if(i+1) %600 ==0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    print("Finish Training.")

    # Save model
    torch.save(model.state_dict(),MODEL_FILE)

    print(len(test_loader))

    # write csv
    with torch.no_grad():
        n_correct =0
        n_samples =0
        for image , labels in test_loader:
            outputs =model(image)
            # max return (value , index)
            _,predicted = torch.max(outputs,1)
            n_samples += labels.size(0)
            n_correct +=(predicted ==labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network:{acc} %')

if __name__ =="__main__":
    train()