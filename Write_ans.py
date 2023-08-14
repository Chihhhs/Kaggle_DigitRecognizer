import torch
from Train import ConvNeuralNet
from torch.utils.data import DataLoader ,Dataset
import numpy as np
import os 


MODEL_FILE ="./model/Kaggle_Mnist.pt"
TEST_FILE = "./digit-recognizer/test.csv"

batch_size = 4
classes = ('1' , '2' , '3' , '4', '5','6','7','8','9','0')

class TestDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(TEST_FILE,delimiter=",",dtype=np.float32 ,skiprows=1)
        self.img = torch.from_numpy(xy.reshape(28,28))
        self.n_samples =xy.shape[0]
        
    def __getitem__(self,index):
        return self.img[index]
    
    def __len__(self):
        return self.n_samples
test = TestDataset()
test_loader = DataLoader(dataset=test ,batch_size = batch_size,shuffle =True,num_workers=0)

model = ConvNeuralNet()

if os.path.exists(MODEL_FILE):
    model =model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()


## 
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
    print(f'Accuracy of the network:{acc} %')
