'''
用于训练mfsanTrain网络
'''


import math
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
from torchvision import models
'''
一个提取公共特征共享网络，暂时用NN
14个提取私有特征网络，用NN
14个分类器，用一个线性层

'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 读入数据
PathSEED_X =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_X.mat"))
PathSEED_Y =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_Y.mat"))
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



resnet_dict = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
}


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def MMD_Loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

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

'''
MFSAN 网络
'''
class MFSAN(nn.Module):
    def __init__(self, shareNet_inDim, shareNet_outDim, specific_outDim, class_number):
        super(MFSAN, self).__init__()
        self.objectNumber = 14
        self.shareNet_inDim = shareNet_inDim
        self.shareNet_outDim = shareNet_outDim
        self.specific_outDim = specific_outDim
        self.class_number = class_number
        self.build_network()

    def build_network(self):
        self.shareNet = resnet_dict['resnet18'](pretrained=True)
        self.shareNet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.shareNet.fc = nn.Linear(512, self.shareNet_outDim)
        self.shareNet.to(device)
        self.specificNetList = []
        self.classifierList = []

        for i in range(14):
            model = BaseNN(self.shareNet_outDim, self.specific_outDim).to(device)
            model1 = nn.Sequential(nn.Linear(self.specific_outDim, self.class_number)).to(device)
            self.specificNetList.append(model)
            self.classifierList.append(model1)


    def forward(self, data_source, data_target, label_source, number):
        '''
        :param data_source: 14个源域数据形成的list
        :param data_target: 目标域数据
        :param label_source: 14个源数据对应的label形成的list
        :return: 对应的三种loss
        '''
        # 通过shareNet
        target_con = self.shareNet(data_target.to(device))
        source_con = self.shareNet(data_source.to(device))

        # target通过所有的specificNet和classifier
        target_spec = []
        target_class = []
        for i in range(self.objectNumber):
            target_spec.append(self.specificNetList[i](target_con))
            target_class.append(self.classifierList[i](target_spec[i]))


        # 每个源数据经过其对应的网络
        source_spec = self.specificNetList[number](source_con)
        source_class = self.classifierList[number](source_spec)

        # 计算loss
        mmdloss = 0
        discloss = 0
        classloss = 0
        epi = 1e-8 # 用于交叉熵函数

        mmdloss += MMD_Loss(source_spec, target_spec[number])
        classloss += F.cross_entropy(source_class+epi,  torch.squeeze(label_source.to(device)))

        for j in range(self.objectNumber):
            if j!=number:
                discloss += torch.mean(torch.abs(torch.softmax(target_class[number]+epi, dim=1) - torch.softmax(target_class[j]+epi,dim=1)))

        discloss = discloss/13

        return mmdloss, discloss, classloss



    def prediction(self, data_target):
        '''
        :param data_target: 要预测的目标
        :return: 返回每个分类器的结果
        '''

        sharedata = self.shareNet(data_target.to(device))
        predList = []
        for i in range(self.objectNumber):
            specficdata = self.specificNetList[i](sharedata)
            predList.append(self.classifierList[i](specficdata))


        return predList

def train(testNumber, shareNet_lr, specificNet_lr, classifier_lr):
    set_seed(0)
    '''
    :param testNumber: 测验的对象编号
    :param shareNet_lr: 共享网络的学习率
    :param specificNet_lr: 每个映射的学习率，一个list
    :param classifier_lr: 每个分类器的学习率，一个list
    :return:保存训练好的模型
    '''


    # 基本参数
    Batchsize = 256
    Maxepisode = 400
    shareNet_inDim = 310
    shareNet_outDim = 128
    specific_outDim = 64
    class_number = 3



    # 制作数据集
    source_loader_list = []
    target_loader = None
    for i in range(15):
        if i!=testNumber:
            trainx = torch.FloatTensor(EEG_X[i].reshape(EEG_X[i].shape[0],1,62,5))
            trainy = torch.LongTensor(EEG_Y[i])
            dataset = Data.TensorDataset(trainx,trainy)
            loader = Data.DataLoader(dataset=dataset, batch_size=Batchsize, shuffle=True)
            source_loader_list.append(loader)
        else:
            targetx = torch.FloatTensor(EEG_X[i].reshape(EEG_X[i].shape[0],1,62,5))
            targety = torch.LongTensor(EEG_Y[i])
            dataset = Data.TensorDataset(targetx, targety)
            target_loader = Data.DataLoader(dataset=dataset, batch_size=Batchsize, shuffle=True)
    source_iters = []
    for i in range(14):
        source_iters.append(iter(source_loader_list[i]))
    target_iter = iter(target_loader)

    # 定义模型和优化器
    mymodel = MFSAN(shareNet_inDim, shareNet_outDim, specific_outDim, class_number)
    shareNet_opt = torch.optim.Adam(mymodel.shareNet.parameters(), lr=shareNet_lr)
    specific_opts = []
    classifier_opts = []
    for i in range(14):
        specific_opts.append(torch.optim.Adam(mymodel.specificNetList[i].parameters(), lr=specificNet_lr[i]))
        classifier_opts.append(torch.optim.Adam(mymodel.classifierList[i].parameters(), lr=classifier_lr[i]))

    # 开始训练
    maxacc = 0
    for episode in range(Maxepisode):
        # 获取数据
        source_datas = []
        source_lables = []
        for i in range(14):
            try:
                src_data, src_label = next(source_iters[i])
            except StopIteration:
                source_iters[i] = iter(source_loader_list[i])
                src_data, src_label = next(source_iters[i])
            source_datas.append(src_data)
            source_lables.append(src_label)

        try:
            target_data, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_data, _ = next(target_iter)

        for i in range(14):
            mmdloss, discloss, classloss = mymodel.forward(source_datas[i], target_data, source_lables[i], i)
            gamma = 2 / (1 + math.exp(-10 * (episode) / (Maxepisode))) - 1
            loss = classloss + gamma * (mmdloss + discloss)
            for j in range(14):
                specific_opts[j].zero_grad()
                classifier_opts[j].zero_grad()
            shareNet_opt.zero_grad()

            loss.backward()
            #print("episode: ", episode, " loss: ", loss, "three loss: ", [mmdloss.data.cpu().numpy(), discloss.data.cpu().numpy(), classloss.data.cpu().numpy()])

            shareNet_opt.step()
            for j in range(14):
                specific_opts[j].step()
                classifier_opts[j].step()


        if (episode+1) % 5 == 0:
            acc = modeltest(mymodel, testNumber, episode+1)
            if acc>maxacc:
                maxacc = acc
                # 打开注释可以保存参数
                # saveOrloadParameters(mymodel, testNumber, True, episode+1, acc)

    return maxacc

def saveOrloadParameters(model, testNumber, save, episode=None, accuracy=None):
    '''
    :param model:需要载入或者保存参数的模型
    :param testNumber: 当前要测试的组
    :param save: bool，true 保存， false 载入参数
    :return: 保存参数或者载入参数的模型
    '''

    if save:
        currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        savepath = os.path.join('parameters', str(testNumber), currentTime+"_"+str(episode)+"_"+str(accuracy))

        for i in range(14):
            torch.save(model.specificNetList[i].state_dict(), savepath+"_"+'specificNet'+str(i)+'.pkl')
            torch.save(model.classifierList[i].state_dict(), savepath + "_" + 'classifier' + str(i) + '.pkl')
        torch.save(model.shareNet.state_dict(), savepath + "_" + 'shareNet' + '.pkl')

    else:
        pathdir = os.path.join('parameters', str(testNumber))
        loadpathlist = os.listdir(pathdir)
        for path in loadpathlist:
            splitpath = path.split('_')
            if splitpath[-1][0:2] == "cl":
                number = 0
                if splitpath[-1][10:12].isdigit():
                    number = int(splitpath[-1][10:12])
                else:
                    number = int(splitpath[-1][10])

                model.classifierList[number].load_state_dict(torch.load(os.path.join(pathdir, path)))
            elif splitpath[-1][0:2] == "sp":
                number = 0
                if splitpath[-1][11:13].isdigit():
                    number = int(splitpath[-1][11:13])
                else:
                    number = int(splitpath[-1][11])

                model.specificNetList[number].load_state_dict(torch.load(os.path.join(pathdir, path)))

            else:
                model.shareNet.load_state_dict(torch.load(os.path.join(pathdir, path)))

def modeltest(model, testNumber, episode=0):
    '''
    :param model: 训练过的模型
    :param testNumber: 测试的对象
    :param episode: 训练次数
    :return: 测试精确度
    '''
    epi = 1e-8
    testX = torch.FloatTensor(EEG_X[testNumber].reshape(EEG_X[testNumber].shape[0],1,62,5))
    testY = EEG_Y[testNumber].squeeze()
    with torch.no_grad():
        outlist = model.prediction(testX)
        predlist = []
        for i in range(14):
            pred = F.softmax(outlist[i]+epi, 1)
            predlist.append(pred)

        avgpred = predlist[0]
        for i in range(1,14):
            avgpred += predlist[i]

        avgpred = avgpred/14

        avgpred_y = (torch.max(avgpred,1)[1]).data.cpu().numpy().squeeze()
        avgaccuracy = sum(testY == avgpred_y) / len(testY)

        accuracylist = []
        for i in range(14):
            pred_y =  (torch.max(predlist[i],1)[1]).data.cpu().numpy().squeeze()
            accuracy = sum(testY == pred_y)/len(testY)
            accuracylist.append(accuracy)

        print("test object: ", testNumber, " train episode: ", episode, " avgaccuracy: ", avgaccuracy, " accuracy: ", accuracylist)

    return avgaccuracy

def main():
    '''
    :return: 进行15折交叉验证训练
    '''
    shareNet_lr = 0.001
    specific_lr = [0.001 for i in range(14)]
    classifier_lr = [0.001 for i in range(14)]
    acc = []
    for i in range(15):
        accuracy = train(i, shareNet_lr, specific_lr, classifier_lr)
        acc.append(accuracy)

    print(acc)
    print(sum(acc)/len(acc))



if __name__ == "__main__":
    '''
    运行会进行模型训练，但是不会保存参数，如果要保存参数，请打开第281行注释，会保存到./parameter的各个文件夹下
    '''
    main()


