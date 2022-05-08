
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

'''
trainSingleObject 函数用来训练15个模型
testSingleObjdect 可以测试15个模型
'''

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

def saveObjectModel(model,objectNumber, episode, accuracy):
    '''
    :param model: 模型
    :param objectNumber: 对象的序号
    :param episode: 训练的次数
    :param accuracy: 测试的精度
    :return: 保存下该模型的参数
    '''

    currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    filename = currentTime + "_" + str(objectNumber) + "_" + str(episode) + "_" + str(accuracy) + '.pkl'
    savepath = os.path.join("HistoryParameters", "ObjectParameters", filename)
    torch.save(model.state_dict(), savepath)


# 对每个人的数据进行单独训练一个模型
def trainSingleObject(number):
    '''

    :param number: 训练模型的序号
    :return: 保存训练的模型参数
    '''
    set_seed(0)
    batchSize = 512
    maxepisode = 1000
    learnrate = 0.001

    X_Data = torch.FloatTensor(EEG_X[number])
    Y_Data = torch.LongTensor(EEG_Y[number])
    dataset = Data.TensorDataset(X_Data, Y_Data)
    train_data, test_data = random_split(dataset,[round(0.8*X_Data.shape[0]),X_Data.shape[0]-round(0.8*X_Data.shape[0])], generator=torch.Generator().manual_seed(42))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batchSize, shuffle=True, drop_last=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=X_Data.shape[0] - round(0.8 * X_Data.shape[0]))
    singleModel = BaseNN(input_dim=310, output_dim=3).to(device)
    optimizer = torch.optim.Adam(singleModel.parameters(), lr=learnrate)
    lossfunc = nn.CrossEntropyLoss().to(device)

    flag = False
    for episode in range(maxepisode):

        for step, (batch_x, batch_y) in enumerate(train_loader):
            out = singleModel(batch_x.to(device))
            loss = lossfunc(out, torch.squeeze(batch_y).to(device))
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        if (episode+1)%10 == 0:
            for step, (test_x, test_y) in enumerate(test_loader):
                out = singleModel(test_x.to(device))
                prediction = torch.max(F.softmax(out, 1), 1)[1]
                pred_EEG_Y = prediction.data.cpu().numpy().squeeze()
                accuracy = sum(test_y.numpy().squeeze() == pred_EEG_Y) / len(pred_EEG_Y)
                print("number", number+1,"episode", episode+1, "accuracy", accuracy)

                if accuracy>0.96 and episode>200:
                    saveObjectModel(singleModel,number+1, episode+1, accuracy)
                    flag = True

        # 保存到参数后跳出循环
        if flag:
            break

# 使用得到的模型对所有数据进行测试
def testSingleObjdect(number, parameterfile):
    '''

    :param number: 需要测试的模型序号
    :param parameterfile: 对应的模型参数文件路径
    :return: 输出测试精度
    '''

    X_Data = torch.FloatTensor(EEG_X[number])
    Y_Data =EEG_Y[number].squeeze()

    testModel = BaseNN(input_dim=310, output_dim=3)
    testModel.load_state_dict(torch.load(parameterfile))
    out = testModel(X_Data)
    prediction = torch.max(F.softmax(out,dim=1),dim=1)[1]
    pred = prediction.numpy().squeeze()
    accuracy = sum(pred==Y_Data)/len(pred)
    print("number", number, "accuracy", accuracy)



# testNumber: 指定哪一维数据作为验证数据集

def Basetrain(testNumber, Batchsize, learnrate):
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
        acc = Basetrain(num, 512, 0.0001)
        accuracy.append(acc)
    print(accuracy)
    print(sum(accuracy)/len(accuracy))



if __name__ == '__main__':
    #Basetrain()
    #import time
    #print(type(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))

    #crossValidation()

    '''
    一次结果
    [0.6396582203889216, 0.6187389510901591, 0.6269888037713612, 0.931349440188568, 0.6812021213906895, 0.3945197407189157, 0.6859163229228049, 0.4699469652327637, 0.8258691809074838, 0.652327637006482, 0.849145550972304, 0.5574543311726576, 0.9269298762522098, 0.4964643488509134, 0.8025928108426635]
0.6772736201139266
    '''
    # train object

    # for i in range(15):
    #     trainSingleObject(i)

    # test object

    testSingleObjdect(0,"HistoryParameters/ObjectParameters/2022_05_08_15_13_06_9_2201.0.pkl")












