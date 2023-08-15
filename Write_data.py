import torch
from Train import ConvNeuralNet
from torch.utils.data import DataLoader ,Dataset
import numpy as np
import csv
import os 


MODEL_FILE ="./model/Kaggle_Mnist.pt"
TEST_FILE = "./digit-recognizer/test.csv"

batch_size = 4

class TestDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt(TEST_FILE,delimiter=",",dtype=np.float32 ,skiprows=1)
        self.img = torch.from_numpy(xy.reshape(-1,1,28,28))
        self.n_samples =xy.shape[0]
        
    def __getitem__(self,index):
        return self.img[index]
    
    def __len__(self):
        return self.n_samples

def write_to_file(predictions):
    with open("./digit-recognizer/sample_submission.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['ImageId', 'Label'])
        for i, label in enumerate(predictions, start=1):
            csvwriter.writerow([i, label])

    print("Finish write.")

test = TestDataset()
test_loader = DataLoader(dataset=test ,batch_size = batch_size,shuffle =True,num_workers=0)

model = ConvNeuralNet()

if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

All_predect =[]
with torch.no_grad():
    for images in test_loader:
        outputs =model(images)
        _,predicted = torch.max(outputs,1)
        # print(type(predicted), predicted.shape)
        integer_list = [int(value) for value in predicted]
        # print(integer_list)
        All_predect.extend(integer_list)
        print(All_predect)

write_to_file(All_predect)
