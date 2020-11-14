import numpy as np
import pandas as pd
import torchsnooper
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import torch
import os
import time
import copy
from torch.optim import Adam
import argparse
import torch.nn.functional as F

def get_parser():
    parser = argparse.ArgumentParser(description='Machine learning for Muon Track')
    parser.add_argument("--load", type=int)
    parser.add_argument("--inputData", default='/junofs/users/wudr/SPMT_waveform/MLForGain/randList.txt')
    parser.add_argument("--output", default='/junofs/users/wudr/SPMT_waveform/MLForGain/myConvNet.pkl')
    parser.add_argument("--modelinput", default='/junofs/users/wudr/SPMT_waveform/MLForGain/myConvNet.pkl')
    parser.add_argument("--epoch", type=int)
    return parser

class MyDataSet(Dataset):
    def __init__(self, txt, transform=torch.as_tensor, loader=np.loadtxt):
        super(MyDataSet, self).__init__()
        indexf = open(txt, 'r')
        lines = indexf.readlines()
        datas = []
        for line in lines:
            words = line.split()
            if len(words) is not 2:
                continue
            datas.append((words[0], float(words[1])))
        self.datas = datas
        self.transform = transform
        self.loader = loader
    
    def __getitem__(self, index):
        path, label = self.datas[index]
        dataX = self.loader(path, dtype='d')
        dataX = np.expand_dims(dataX, 0)
        dataX = self.transform(dataX)
        return dataX, label
    
    def __len__(self):
        return len(self.datas)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.hidden1 = nn.Linear(199, 400, bias=True)
        self.hidden2 = nn.Linear(400, 200)
        self.hidden3 = nn.Linear(200, 100)
        self.hidden4 = nn.Linear(100, 50)
        self.regress = nn.Linear(50, 1)
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        output = self.regress(x)
        return output[:, 0, 0]

def trainModel(model, dataLoader, trainRate, criterion, optimizer, numEpochs, savePath):
    # model: Net model
    # dataLoader: Data loader of train data
    # trainRate: Ratio of train data to all data
    # criterion: Loss function
    # optimizer: Optimization method
    # numEpochs: Training times
    batchNum = len(dataLoader)
    trainBatchNum = round(batchNum * trainRate)

    bestModelWts = copy.deepcopy(model.state_dict())
    bestLoss = 100
    trainLossAll = []
    valLossAll = []
    since = time.time()
    for epoch in range(numEpochs):
        print('Epoch {}/{}'.format(epoch, numEpochs - 1))
        print('-' * 20)

        trainLoss = 0.0
        trainNum = 0
        valLoss = 0.0
        valNum = 0

        for step, (X, Y) in enumerate(dataLoader):
            if step < trainBatchNum:
                model.train()
                output = model(X)
                loss = criterion(output, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trainLoss += loss.item() * X.size(0)
                trainNum += X.size(0)
            else:
                model.eval()
                output = model(X)
                loss = criterion(output, Y)
                valLoss += loss.item() * X.size(0)
                valNum += X.size(0)
        trainLossAll.append(trainLoss / trainNum)
        valLossAll.append(valLoss / valNum)
        print('{} Train Loss: {:.4f} '.format(epoch, trainLossAll[-1]))
        print('{} Val Loss: {:.4f} '.format(epoch, valLossAll[-1]))

        if valLossAll[-1] < bestLoss:
            bestLoss = valLossAll[-1]
            bestModelWts = copy.deepcopy(model.state_dict())
        timeUse = time.time() - since
        print('Train and val complete in {:.0f}m {:.0f}'.format(timeUse // 60, timeUse % 60))
        torch.save(model, savePath)
    model.load_state_dict(bestModelWts)
    trainProcess = pd.DataFrame(
        data={
            "epoch": range(numEpochs),
            "trainLossAll": trainLossAll,
            "valLossAll": valLossAll,
        }
    )
    return model, trainProcess

if __name__ == "__main__":
    torch.set_default_tensor_type(torch.DoubleTensor)
    parser = get_parser()
    args = parser.parse_args()
    if args.load is 0:
        NetForGain = MyNet()
    else:
        NetForGain = torch.load(args.modelinput)

    trainlist = args.inputData
    # testlist = ''
    trainData = MyDataSet(trainlist)
    trainLoader = Data.DataLoader(
        dataset=trainData,
        batch_size=64,
        shuffle=True,
        num_workers=1,
    )
    # testData = MyDataSet(testlist)
    # testLoader = Data.DataLoader(
    #     dataset=testData,
    #     batch_size=64,
    # )

    optimizer = torch.optim.Adam(NetForGain.parameters(), lr=0.0003)
    lossFunc = nn.MSELoss()
    model, process = trainModel(NetForGain, trainLoader, 0.8, lossFunc, optimizer, args.epoch, args.output)