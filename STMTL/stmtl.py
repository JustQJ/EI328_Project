'''
实现了STM算法
'''

import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os
import random
# 读入数据
PathSEED_X =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_X.mat"))
PathSEED_Y =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_Y.mat"))
EEG_X = sio.loadmat(PathSEED_X)['X'][0]
EEG_Y = sio.loadmat(PathSEED_Y)['Y'][0] + 1

parameterfile = []
parameterfiles = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir,"BaseNN", "HistoryParameters", "ObjectParameters"))
files = os.listdir(parameterfiles)
for file in files:
    parameterfile.append(os.path.join(parameterfiles,file))

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

def mappingMatrix(testNumber, gammahat, betahat, destNumber, session1, session2,session3):
    '''
    :param testNumber: 测试的对象的序号
    :param gammahat: 计算矩阵参数1
    :param betahat: 计算矩阵参数2
    :param destNumber: 本次要匹配的源的对象序号
    :param session1: 测试对象的第一个session数据的范围
    :param session2:测试对象的第二个session数据的范围
    :param session3:测试对象的第三个session数据的范围
    :return: 返回mapping 矩阵 A和 b
    '''
    # 校验数据
    cal_X = []
    cal_Y = []
    choose1 = random.sample(range(0, session1), 30)
    choose2 = random.sample(range(session1+1, session2), 30)
    choose3 = random.sample(range(session2+1, session3), 30)
    cal_X.append(EEG_X[testNumber].take(choose1, 0))
    cal_X.append(EEG_X[testNumber].take(choose2, 0))
    cal_X.append(EEG_X[testNumber].take(choose3, 0))
    cal_Y.append(EEG_Y[testNumber].take(choose1, 0))
    cal_Y.append(EEG_Y[testNumber].take(choose2, 0))
    cal_Y.append(EEG_Y[testNumber].take(choose3, 0))



    mood = [[], [], []]
    for j in range(EEG_Y[destNumber].shape[0]):
        if EEG_Y[destNumber][j][0] == 2:
            mood[0].append(EEG_X[destNumber][j])
        elif EEG_Y[destNumber][j][0] == 1:
            mood[1].append(EEG_X[destNumber][j])
        else:
            mood[2].append(EEG_X[destNumber][j])

    happyk = KMeans(n_clusters=15).fit(np.array(mood[0])).cluster_centers_
    smoothk = KMeans(n_clusters=15).fit(np.array(mood[1])).cluster_centers_
    sadk = KMeans(n_clusters=15).fit(np.array(mood[2])).cluster_centers_
    cluster_X = [happyk, smoothk, sadk]


    # 计算校验数据的destination
    moods = []
    for j in range(3):
        if cal_Y[j][0][0] == 2:
            happymap = []
            for k in range(cal_X[j].shape[0]):
                distances = [np.linalg.norm(cal_X[j][k] - cluster_X[0][u]) for u in range(15)]
                minindex = distances.index(min(distances))
                happymap.append(cluster_X[0][minindex])
            happymap = np.array(happymap)
            moods.append(happymap)
        elif cal_Y[j][0][0] == 1:
            smoothmap = []
            for k in range(cal_X[j].shape[0]):
                distances = [np.linalg.norm(cal_X[j][k] - cluster_X[1][u]) for u in range(15)]
                minindex = distances.index(min(distances))
                smoothmap.append(cluster_X[1][minindex])
            smoothmap = np.array(smoothmap)
            moods.append(smoothmap)
        elif cal_Y[j][0][0] == 0:
            sadmap = []
            for k in range(cal_X[j].shape[0]):
                distances = [np.linalg.norm(cal_X[j][k] - cluster_X[2][u]) for u in range(15)]
                minindex = distances.index(min(distances))
                sadmap.append(cluster_X[2][minindex])
            sadmap = np.array(sadmap)
            moods.append(sadmap)
    Dest = np.vstack((moods[0], moods[1], moods[2]))

    # 解方程，结出7个源的A和b
    beta_hat = betahat
    gamma_hat = gammahat
    dim = 310
    Origin = np.vstack((cal_X[0], cal_X[1], cal_X[2]))
    f = [1 for i in range(Origin.shape[0])]
    beta = beta_hat / dim * np.trace(
        sum([f[j] * np.dot(Origin[j].reshape(-1, 1), Origin[j].reshape(1, -1)) for j in range(Origin.shape[0])]))
    gamma = gamma_hat * sum(f)
    f_hat = sum(f) + gamma
    origin_hat = sum([f[j] * Origin[j].reshape(-1, 1) for j in range(Origin.shape[0])])
    dest_hat = sum([f[j] * Dest[j].reshape(-1, 1) for j in range(Origin.shape[0])])
    P = sum([f[j] * np.dot(Origin[j].reshape(-1, 1), Origin[j].reshape(1, -1)) for j in
             range(Origin.shape[0])]) - origin_hat * origin_hat.transpose() / f_hat + beta * np.eye(dim)
    Q = sum([f[j] * np.dot(Dest[j].reshape(-1, 1), Origin[j].reshape(1, -1)) for j in
             range(Origin.shape[0])]) - dest_hat * origin_hat.transpose() / f_hat + beta * np.eye(dim)
    A0 = np.dot(Q, np.linalg.inv(P))
    b0 = (dest_hat - np.dot(A0, origin_hat)) / f_hat


    return A0, b0


def learning(testNumber, gamma_hat, beta_hat, choosenumber):

    session1 = 0
    session2 = 0
    session3 = 0
    first = EEG_Y[testNumber][0][0]
    for i in range(EEG_Y[testNumber].shape[0]):
        if EEG_Y[testNumber][i][0] != first:
            if session1 == 0:
                session1 = i - 1
                first = EEG_Y[testNumber][i][0]
            elif session2 == 0:
                session2 = i - 1
                first = EEG_Y[testNumber][i][0]
            elif session3 == 0:
                session3 = i - 1
                break

    # 前三个session用于排序源数据
    order_X = EEG_X[testNumber][0: session3 + 1, :]
    order_Y = EEG_Y[testNumber][0: session3 + 1, :]


    # 测试数据
    test_X = EEG_X[testNumber][session3 + 1:, :]
    test_Y = EEG_Y[testNumber][session3 + 1:, :]

    # 选择排在前面choosenumber个的分类器作为分类标准

    thisfiles = parameterfile.copy()
    thisfiles.pop(testNumber)
    classifer = []
    for i in range(14):
        classifer.append(BaseNN(input_dim=310, output_dim=3))
        classifer[i].load_state_dict(torch.load(thisfiles[i]))
    accuracylist = []
    for i in range(14):
        out = classifer[i](torch.FloatTensor(order_X))
        pred = torch.max(F.softmax(out, 1), 1)[1]
        accuracy = sum(pred.numpy().squeeze() == order_Y.squeeze()) / len(order_Y.squeeze())
        accuracylist.append(accuracy)

    accuracybackup = accuracylist.copy()
    selectedsource = []
    for i in range(choosenumber):
        maxn = max(accuracylist)
        maxindex = accuracylist.index(maxn)
        accuracylist[maxindex] = -1
        if maxindex >= testNumber:
            maxindex = maxindex + 1
        selectedsource.append(maxindex)

    selectedsource = sorted(selectedsource)

    Ab_dict = {}
    for item in selectedsource:
        A, b = mappingMatrix(testNumber, gamma_hat, beta_hat, item, session1,session2,session3)
        Ab_dict[item] = (A, b)

    chooseacuuracy = []
    for item in selectedsource:
        if item > testNumber:
            chooseacuuracy.append(accuracybackup[item - 1])
        else:
            chooseacuuracy.append(accuracybackup[item])
    weight = []
    for item in chooseacuuracy:
        weight.append(item / sum(chooseacuuracy))

    totalout = 0
    for i in range(len(selectedsource)):
        mapping_x = np.dot(Ab_dict[selectedsource[i]][0], test_X.transpose()) + Ab_dict[selectedsource[i]][1]
        out = (classifer[i](torch.FloatTensor(mapping_x.transpose()))).detach()
        totalout += weight[i] * out
    pred = torch.max(F.softmax(totalout, 1), 1)[1]
    finalacc = sum(pred.numpy().squeeze() == test_Y.squeeze()) / len(test_Y.squeeze())


    return finalacc


def crossValidation():
    accuracy = []
    for i in range(15):
        ac = learning(i, 1.2, 0.2,13)
        accuracy.append(ac)
    avgacc = sum(accuracy)/len(accuracy)
    print("the accuracy of each subject:", accuracy)
    print("the average accuracy of 15-fold validation:", avgacc)


if __name__ == '__main__':

    crossValidation()





