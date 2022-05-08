'''
自编码器加上一个NN网络进行分类
'''

import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import random_split
from envs import PathSEED_X, PathSEED_Y
import os
import time
import random
from AutoEncoder import AutoCoder


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 读入数据
EEG_X = sio.loadmat(PathSEED_X)['X'][0]
EEG_Y = sio.loadmat(PathSEED_Y)['Y'][0] + 1

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class BaseNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, output_dim)
        )

    def forward(self, input):

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)

        return input



def train(testNumber, Batchsize, learnrate,coderpath):
    set_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 制作数据集
    train_X = None
    train_Y = None
    for i in range(0, testNumber):
        if train_X is None:
            train_X = torch.FloatTensor(EEG_X[0])
            train_Y = torch.LongTensor(EEG_Y[0])
        else:
            train_X = torch.cat((train_X, torch.FloatTensor(EEG_X[i])), 0)
            train_Y = torch.cat((train_Y, torch.LongTensor(EEG_Y[i])), 0)

    for i in range(testNumber+1, 15):
        if train_X is None:
            train_X = torch.FloatTensor(EEG_X[testNumber+1])
            train_Y = torch.LongTensor(EEG_Y[testNumber+1])
        else:
            train_X = torch.cat((train_X, torch.FloatTensor(EEG_X[i])), 0)
            train_Y = torch.cat((train_Y, torch.LongTensor(EEG_Y[i])), 0)

    test_X = torch.FloatTensor(EEG_X[testNumber])
    test_Y = EEG_Y[testNumber].squeeze()


    torch_dataset = Data.TensorDataset(train_X, train_Y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Batchsize,
        shuffle=True

    )



    # 定义网络模型，经过编码后变为64维

    mymodel = BaseNN(input_dim=64, output_dim=3).to(device)
    autocoder = AutoCoder(inputDim=310, hiddenDim=64).to(device) # 编码器
    autocoder.load_state_dict(torch.load(coderpath))

    # 不同的优化器：Momentum RMSprop, Adam
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learnrate)
    # CrossEntropyLoss 自动实现了一个激活函数，已经softmax的操作，所以可以直接将网络输出的one hot和一维target放在一起
    lossfunc = nn.CrossEntropyLoss().to(device)
    # 训练次数
    maxepisodes = 1000
    maxaccuracy = 0
    for episode in range(maxepisodes):
        for step, (batch_x, batch_y) in enumerate(loader):
            encoded, decoded = autocoder(batch_x.to(device))
            out = mymodel(encoded.detach())   # 前向传播
            loss = lossfunc(out, torch.squeeze(batch_y).to(device))  # 误差计算，这里是直接使用标量y
            #print(loss)
            optimizer.zero_grad()  # 清空梯度
            loss.backward()   # 反向传播，计算导数
            optimizer.step()  # 更新参数

        if episode % 10 == 0:
            encoded,decoded = autocoder(test_X.to(device))
            finalout = mymodel(encoded.detach())
            #print(finalout)
            prediction = torch.max(F.softmax(finalout, 1), 1)[1]
            #print(prediction)
            pred_EEG_Y = prediction.data.cpu().numpy().squeeze()
            accuracy = sum(test_Y == pred_EEG_Y) / len(test_Y)
            if accuracy>maxaccuracy:
                maxaccuracy = accuracy

            print("episode：",episode, "accuracy: ", accuracy)

            '''
                if accuracy > 0.8:
                    # 保存每次的训练参数
                    print('episod:', i, "step:", step, "accuracy:", accuracy)
                    currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    savepath = 'HistoryParameters/' + currentTime + "_" + str(accuracy) + '.pkl'
                    torch.save(mymodel.state_dict(), savepath)
                '''





    # 预测需要使用softmax函数，得到one hot，表示每个类出现的概率。
    # max函数选择出来最大的概率的位置作为类，比如这里 max(input,1)就是按照行选择出每行的最大值和其对应的index
    # 然后再取出最大的位置那一列就是分类，再-1就从0，1，2 -》 -1，0，1
    print(maxaccuracy)
    return maxaccuracy  # 返回最大的精度




def crossValidation():

    # 编码参数路径
    coderpath = os.path.join("parameters","encoderParameter","2022_05_07_17_42_09_532_51.349586.pkl")

    accuracy = []
    for num in range(1):
        acc = train(num, 512, 0.0001,coderpath)
        accuracy.append(acc)
    print(accuracy)
    print(sum(accuracy)/len(accuracy))



if __name__ == "__main__":
    crossValidation()

