
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from envs import PathSEED_X, PathSEED_Y
import os
import time
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)

# 只用原数据X进行训练
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    # 当训练时，使用decoder, 当训练好后，使用模型产生encoder
    def forward(self,input):
        encoder = self.encoder(input)
        decoder = self.decoder(encoder)

        return encoder, decoder





class BaseNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU()

        )
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(512, output_dim)

        )

    def forward(self, input):

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)

        return input





# testNumber: 指定哪一维数据作为验证数据集

def Basetrain(testNumber, Batchsize, learnrate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读入数据, 制作数据集

    EEG_X = sio.loadmat(PathSEED_X)['X'][0]
    EEG_Y = sio.loadmat(PathSEED_Y)['Y'][0]+1
    train_X = None
    train_Y = None
    for i in range(0, testNumber):
        if train_X == None:
            train_X = torch.FloatTensor(EEG_X[0])
            train_Y = torch.LongTensor(EEG_Y[0])
        else:
            train_X = torch.cat((train_X, torch.FloatTensor(EEG_X[i])), 0)
            train_Y = torch.cat((train_Y, torch.LongTensor(EEG_Y[i])), 0)

    for i in range(testNumber+1, 15):
        if train_X == None:
            train_X = torch.FloatTensor(EEG_X[testNumber+1])
            train_Y = torch.LongTensor(EEG_Y[testNumber+1])
        else:
            train_X = torch.cat((train_X, torch.FloatTensor(EEG_X[i])), 0)
            train_Y = torch.cat((train_Y, torch.LongTensor(EEG_Y[i])), 0)

    test_X = torch.FloatTensor(EEG_X[testNumber])
    test_Y =  EEG_Y[testNumber].squeeze()


    torch_dataset = Data.TensorDataset(train_X, train_Y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=Batchsize,
        shuffle=True

    )



    # 定义网络模型
    mymodel = BaseNN(input_dim=310, output_dim=3).to(device)

    # 不同的优化器：Momentum RMSprop, Adam
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=learnrate)
    # CrossEntropyLoss 自动实现了一个激活函数，已经softmax的操作，所以可以直接将网络输出的one hot和一维target放在一起
    lossfunc = nn.CrossEntropyLoss().to(device)
    # 训练次数
    maxepisodes = 1000
    maxaccuracy = 0
    for episode in range(maxepisodes):
        for step, (batch_x, batch_y) in enumerate(loader):
            out = mymodel(batch_x.to(device))   # 前向传播
            loss = lossfunc(out, torch.squeeze(batch_y).to(device))  # 误差计算，这里是直接使用标量y
            print(loss)
            optimizer.zero_grad()  # 清空梯度
            loss.backward()   # 反向传播，计算导数
            optimizer.step()  # 更新参数

        if (episode)% 10 == 0:
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
        acc = Basetrain(num, 512, 0.0001)
        accuracy.append(acc)
    print(accuracy)
    print(sum(accuracy)/len(accuracy))



if __name__ == '__main__':
    #Basetrain()
    #import time
    #print(type(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))

    crossValidation()

    '''
    一次结果
    [0.6396582203889216, 0.6187389510901591, 0.6269888037713612, 0.931349440188568, 0.6812021213906895, 0.3945197407189157, 0.6859163229228049, 0.4699469652327637, 0.8258691809074838, 0.652327637006482, 0.849145550972304, 0.5574543311726576, 0.9269298762522098, 0.4964643488509134, 0.8025928108426635]
0.6772736201139266
    '''










