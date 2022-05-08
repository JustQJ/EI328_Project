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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 读入数据
EEG_X = sio.loadmat(PathSEED_X)['X'][0]
EEG_Y = sio.loadmat(PathSEED_Y)['Y'][0] + 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class LSTM(nn.Module):
    def __init__(self, inputsize, hiddensize, numlayers,num_class):
        super(LSTM, self).__init__()
        self.num_layers = numlayers
        self.hidden_size = hiddensize
        self.class_number = num_class
        self.lstm = nn.LSTM(inputsize, hiddensize, numlayers, batch_first=True)
        self.outlayer = nn.Linear(hiddensize,num_class)

    def forward(self,input):
        # input (batch, time_step, input_size)
        # out (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # 初始化最初的状态

        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)

        # 前向传播
        out, (hn, hc) = self.lstm(input,(h0,c0))

        # 输出所有的时刻，即从中间的时间序列维度进行拆分，然后拼接
        finalout = []
        for i in range(out.size(0)):
            finalout.append(self.outlayer(out[i,:,:]))

        return torch.stack(finalout, dim=1).view(-1,self.class_number)




def LSTMtrain(testNumber, Batchsize, learnrate):
    set_seed(0)

    # 制作数据集，每个时间序列12秒，两个序列的交叉时间为4s，即12条数据和4条数据
    '''
    把数据制作成一个个的序列
    timestep 为序列长度
    timeoverlap 两个时间序列之间交叉的时间
    segmentnumber 一个对象的数据可以制作出来的序列的数量
    '''
    timestep = 12
    timeoverlap = 4

    totaltime = EEG_X[0].shape[0]
    segmentnumber = int((totaltime-timeoverlap)/(timestep-timeoverlap))

    train_X = None
    train_Y = None
    for i in range(0, testNumber):
        if train_X is None:
            tempx = []
            tempy = []
            for j in range(segmentnumber):
                tempx.append(EEG_X[0][j*(timestep-timeoverlap):(j+1)*(timestep-timeoverlap)+timeoverlap, :])
                tempy.append(EEG_Y[0][j*(timestep-timeoverlap):(j+1)*(timestep-timeoverlap)+timeoverlap, :])
            train_X = torch.FloatTensor(tempx)
            train_Y = torch.LongTensor(tempy)
        else:
            tempx = []
            tempy = []
            for j in range(segmentnumber):
                tempx.append(EEG_X[i][j * (timestep - timeoverlap):(j + 1) * (timestep - timeoverlap) + timeoverlap, :])
                tempy.append(EEG_Y[i][j * (timestep - timeoverlap):(j + 1) * (timestep - timeoverlap) + timeoverlap, :])
            train_X = torch.cat((train_X, torch.FloatTensor(tempx)), 0)
            train_Y = torch.cat((train_Y, torch.LongTensor(tempy)), 0)


    for i in range(testNumber+1, 15):
        if train_X is None:
            tempx = []
            tempy = []
            for j in range(segmentnumber):
                tempx.append(EEG_X[testNumber+1][j * (timestep - timeoverlap):(j + 1) * (timestep - timeoverlap) + timeoverlap, :])
                tempy.append(EEG_Y[testNumber+1][j * (timestep - timeoverlap):(j + 1) * (timestep - timeoverlap) + timeoverlap, :])
            train_X = torch.FloatTensor(tempx)
            train_Y = torch.LongTensor(tempy)
        else:
            tempx = []
            tempy = []
            for j in range(segmentnumber):
                tempx.append(EEG_X[i][j * (timestep - timeoverlap):(j + 1) * (timestep - timeoverlap) + timeoverlap, :])
                tempy.append(EEG_Y[i][j * (timestep - timeoverlap):(j + 1) * (timestep - timeoverlap) + timeoverlap, :])
            train_X = torch.cat((train_X, torch.FloatTensor(tempx)), 0)
            train_Y = torch.cat((train_Y, torch.LongTensor(tempy)), 0)

    temptx = []
    testsegmentnumber = int(totaltime/timestep)
    for j in range(testsegmentnumber):
        temptx.append(
            EEG_X[testNumber][j *timestep:(j + 1) * timestep, :])

    test_X = torch.FloatTensor(temptx)
    test_Y =  EEG_Y[testNumber][:testsegmentnumber*timestep,:].squeeze()



    torch_dataset = Data.TensorDataset(train_X, train_Y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Batchsize,
        shuffle=True

    )



    # 定义网络模型
    mymodel = LSTM(inputsize=310, hiddensize=512, numlayers=1, num_class=3).to(device)

    # 不同的优化器：Momentum RMSprop, Adam
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learnrate)
    # CrossEntropyLoss 自动实现了一个激活函数，已经softmax的操作，所以可以直接将网络输出的one hot和一维target放在一起
    lossfunc = nn.CrossEntropyLoss().to(device)
    # 训练次数
    maxepisodes = 1000
    maxaccuracy = 0
    for episode in range(maxepisodes):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = mymodel(batch_x.to(device))   # 前向传播 shape (batch*timestep, 310)

            loss = lossfunc(out, batch_y.view(-1,).to(device))  # 误差计算，这里是直接使用标量y

            optimizer.zero_grad()  # 清空梯度
            loss.backward()   # 反向传播，计算导数
            optimizer.step()  # 更新参数

        if episode % 10 == 0:
            finalout = mymodel(test_X.to(device))
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





    # 加载的方法
    '''
    mymodel2 = BaseNN(input_dim=310, output_dim=3)
    loadpath = savepath
    mymodel2.load_state_dict(torch.load(loadpath))
    '''

# 做15折交叉验证来判断一个模型的平均能力，即各种参数的使用

def crossValidation():

    accuracy = []
    for num in range(15):
        acc = LSTMtrain(num, 512, 0.0001)
        accuracy.append(acc)
    print(accuracy)
    print(sum(accuracy)/len(accuracy))


if __name__ == "__main__":
    crossValidation()