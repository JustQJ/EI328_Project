'''
训练15个对象的分类器
'''

import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import random_split
import os
import time
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pathSEED =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III"))
PathSEED_X = os.getenv('SEED_III_X', os.path.join(pathSEED,"EEG_X.mat"))
PathSEED_Y = os.getenv('SEED-III_Y', os.path.join(pathSEED,"EEG_Y.mat"))

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
            print(loss)
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
                    # 打开注释保存模型参数
                    # saveObjectModel(singleModel,number+1, episode+1, accuracy)
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


if __name__ == '__main__':
    for i in range(15):
        trainSingleObject(i)













