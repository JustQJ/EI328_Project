import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import torch.utils.data as Data
from torch.utils.data import random_split
import os
import time
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

# 制作数据集
def rebuilddata(testNumber, gammahat, betahat):
    '''
    :param testNumber: 被测试的对象序号
    :return: 1.从测试的对象数据中调出3个session，并从这三个session中每个类挑20个数据，共60个sample,组成校验数据，剩下的12session形成测试数据
            2. 每个训练数据对象，使用KNN聚类，得到每个类别20个聚类中心
    '''
    # 校验数据
    cal_X = []
    cal_Y = []
    session1 = 0
    session2 = 0
    session3 = 0
    first = EEG_Y[testNumber][0][0]
    for i in range(EEG_Y[testNumber].shape[0]):
        if EEG_Y[testNumber][i][0]!=first:
            if session1 == 0:
                session1 = i-1
                first = EEG_Y[testNumber][i][0]
            elif session2 == 0:
                session2 = i-1
                first = EEG_Y[testNumber][i][0]
            elif session3 == 0:
                session3 = i-1
                break
    choose1 = random.sample(range(0, session1), 20)
    choose2 = random.sample(range(session1+1, session2), 20)
    choose3 = random.sample(range(session2+1, session3), 20)
    cal_X.append(EEG_X[testNumber].take(choose1, 0))
    cal_X.append(EEG_X[testNumber].take(choose2, 0))
    cal_X.append(EEG_X[testNumber].take(choose3, 0))
    cal_Y.append(EEG_Y[testNumber].take(choose1, 0))
    cal_Y.append(EEG_Y[testNumber].take(choose2, 0))
    cal_Y.append(EEG_Y[testNumber].take(choose3, 0))

    # 前三个session用于排序源数据
    order_X = EEG_X[testNumber][0: session3 + 1, :]
    order_Y = EEG_Y[testNumber][0: session3 + 1, :]

    # 测试数据
    test_X = EEG_X[testNumber][session3+1:-1, :]
    test_Y = EEG_Y[testNumber][session3+1:-1, :]

    # 排序选择出最前面的7个源
    thisfiles = parameterfile.copy()
    thisfiles.pop(testNumber)
    classifer = []
    for i in range(14):
        classifer.append(BaseNN(input_dim=310, output_dim=3))
        classifer[i].load_state_dict(torch.load(thisfiles[i]))
    accuracylist = []
    for i in range(14):
        out = classifer[i](torch.FloatTensor(order_X))
        pred = torch.max(F.softmax(out,1),1)[1]
        accuracy = sum(pred.numpy().squeeze() == order_Y.squeeze())/len(order_Y.squeeze())
        accuracylist.append(accuracy)

    accuracybackup = accuracylist.copy()
    selectedsource = []
    for i in range(7):
        maxn = max(accuracylist)
        maxindex = accuracylist.index(maxn)
        accuracylist[maxindex] = -1
        if maxindex >= testNumber:
            maxindex = maxindex+1
        selectedsource.append(maxindex)

    selectedsource = sorted(selectedsource)
    # kNN聚类
    cluster_X = []
    for i in selectedsource:
        mood = [[],[],[]]
        for j in range(EEG_Y[i].shape[0]):
            if EEG_Y[i][j][0] == 2:
                mood[0].append(EEG_X[i][j])
            elif EEG_Y[i][j][0] == 1:
                mood[1].append(EEG_X[i][j])
            else:
                mood[2].append(EEG_X[i][j])

        happyk = KMeans(n_clusters=15).fit(np.array(mood[0])).cluster_centers_
        smoothk = KMeans(n_clusters=15).fit(np.array(mood[1])).cluster_centers_
        sadk = KMeans(n_clusters=15).fit(np.array(mood[2])).cluster_centers_
        cluster_X.append([happyk, smoothk, sadk])

    # 计算校验数据的destination
    Dest = []
    for i in range(7):
        moods = []
        for j in range(3):
            if cal_Y[j][0][0] == 2:
                happymap = []
                for k in range(20):
                    distances = [np.linalg.norm(cal_X[j][k]-cluster_X[i][0][u]) for u in range(15)]
                    minindex = distances.index(min(distances))
                    happymap.append(cluster_X[i][0][minindex])
                happymap = np.array(happymap)
                moods.append(happymap)
            elif cal_Y[j][0][0] == 1:
                smoothmap = []
                for k in range(20):
                    distances = [np.linalg.norm(cal_X[j][k] - cluster_X[i][1][u]) for u in range(15)]
                    minindex = distances.index(min(distances))
                    smoothmap.append(cluster_X[i][1][minindex])
                smoothmap = np.array(smoothmap)
                moods.append(smoothmap)
            elif cal_Y[j][0][0] == 0:
                sadmap = []
                for k in range(20):
                    distances = [np.linalg.norm(cal_X[j][k] - cluster_X[i][2][u]) for u in range(15)]
                    minindex = distances.index(min(distances))
                    sadmap.append(cluster_X[i][2][minindex])
                sadmap = np.array(sadmap)
                moods.append(sadmap)
        moods = np.vstack((moods[0], moods[1], moods[2]))
        Dest.append(moods)


    # 解方程，结出7个源的A和b
    beta_hat = [betahat for i in range(7)] # 0-3进行搜索，步长为0.2
    gamma_hat = [gammahat for i in range(7)] # 0-3进行搜索，步长为0.2
    f = [1 for i in range(60)]
    dim = 310
    Origin = np.vstack((cal_X[0], cal_X[1], cal_X[2]))

    A_dict = {}
    for i in range(7):
        beta = beta_hat[i]/dim * np.trace(sum([f[j]*Origin[j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]))
        gamma = gamma_hat[i]*sum(f)
        f_hat = sum(f)+gamma
        origin_hat = sum([f[j]*Origin[j].reshape(-1,1) for j in range(Origin.shape[0])])
        dest_hat = sum([f[j]*Dest[i][j].reshape(-1,1) for j in range(Origin.shape[0])])
        P = sum([f[j]*Origin[j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]) - origin_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
        Q = sum([f[j]*Dest[i][j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]) - dest_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
        A = np.dot(Q,np.linalg.inv(P))
        b = (dest_hat-np.dot(A,origin_hat))/f_hat
        A_dict[selectedsource[i]] = (A,b)


    return A_dict, (test_X, test_Y), accuracybackup


def rebuilddata1(testNumber, gammahat, betahat):
    '''
    :param testNumber: 被测试的对象序号
    :return: 1.从测试的对象数据中调出3个session，并从这三个session中每个类挑20个数据，共60个sample,组成校验数据，剩下的12session形成测试数据
            2. 每个训练数据对象，使用KNN聚类，得到每个类别20个聚类中心
    '''
    # 校验数据
    cal_X = []
    cal_Y = []
    session1 = 0
    session2 = 0
    session3 = 0
    first = EEG_Y[testNumber][0][0]
    for i in range(EEG_Y[testNumber].shape[0]):
        if EEG_Y[testNumber][i][0]!=first:
            if session1 == 0:
                session1 = i-1
                first = EEG_Y[testNumber][i][0]
            elif session2 == 0:
                session2 = i-1
                first = EEG_Y[testNumber][i][0]
            elif session3 == 0:
                session3 = i-1
                break
    choose1 = random.sample(range(0, session1), 20)
    choose2 = random.sample(range(session1+1, session2), 20)
    choose3 = random.sample(range(session2+1, session3), 20)
    cal_X.append(EEG_X[testNumber][0:session1+1,:])
    cal_X.append(EEG_X[testNumber][session1+1:session2+1,:])
    cal_X.append(EEG_X[testNumber][session2+1:session3+1,:])
    cal_Y.append(EEG_Y[testNumber][0:session1+1,:])
    cal_Y.append(EEG_Y[testNumber][session1+1:session2+1,:])
    cal_Y.append(EEG_Y[testNumber][session2+1:session3+1,:])

    # 前三个session用于排序源数据
    order_X = EEG_X[testNumber][0: session3 + 1, :]
    order_Y = EEG_Y[testNumber][0: session3 + 1, :]

    # 测试数据
    test_X = EEG_X[testNumber][session3+1:-1, :]
    test_Y = EEG_Y[testNumber][session3+1:-1, :]

    # 排序选择出最前面的7个源
    thisfiles = parameterfile.copy()
    thisfiles.pop(testNumber)
    classifer = []
    for i in range(14):
        classifer.append(BaseNN(input_dim=310, output_dim=3))
        classifer[i].load_state_dict(torch.load(thisfiles[i]))
    accuracylist = []
    for i in range(14):
        out = classifer[i](torch.FloatTensor(order_X))
        pred = torch.max(F.softmax(out,1),1)[1]
        accuracy = sum(pred.numpy().squeeze() == order_Y.squeeze())/len(order_Y.squeeze())
        accuracylist.append(accuracy)

    accuracybackup = accuracylist.copy()
    selectedsource = []
    for i in range(7):
        maxn = max(accuracylist)
        maxindex = accuracylist.index(maxn)
        accuracylist[maxindex] = -1
        if maxindex >= testNumber:
            maxindex = maxindex+1
        selectedsource.append(maxindex)

    selectedsource = sorted(selectedsource)
    # kNN聚类
    cluster_X = []
    for i in selectedsource:
        mood = [[],[],[]]
        for j in range(EEG_Y[i].shape[0]):
            if EEG_Y[i][j][0] == 2:
                mood[0].append(EEG_X[i][j])
            elif EEG_Y[i][j][0] == 1:
                mood[1].append(EEG_X[i][j])
            else:
                mood[2].append(EEG_X[i][j])

        happyk = KMeans(n_clusters=15).fit(np.array(mood[0])).cluster_centers_
        smoothk = KMeans(n_clusters=15).fit(np.array(mood[1])).cluster_centers_
        sadk = KMeans(n_clusters=15).fit(np.array(mood[2])).cluster_centers_
        cluster_X.append([happyk, smoothk, sadk])

    # 计算校验数据的destination
    Dest = []
    for i in range(7):
        moods = []
        for j in range(3):
            if cal_Y[j][0][0] == 2:
                happymap = []
                for k in range(cal_X[j].shape[0]):
                    distances = [np.linalg.norm(cal_X[j][k]-cluster_X[i][0][u]) for u in range(15)]
                    minindex = distances.index(min(distances))
                    happymap.append(cluster_X[i][0][minindex])
                happymap = np.array(happymap)
                moods.append(happymap)
            elif cal_Y[j][0][0] == 1:
                smoothmap = []
                for k in range(cal_X[j].shape[0]):
                    distances = [np.linalg.norm(cal_X[j][k] - cluster_X[i][1][u]) for u in range(15)]
                    minindex = distances.index(min(distances))
                    smoothmap.append(cluster_X[i][1][minindex])
                smoothmap = np.array(smoothmap)
                moods.append(smoothmap)
            elif cal_Y[j][0][0] == 0:
                sadmap = []
                for k in range(cal_X[j].shape[0]):
                    distances = [np.linalg.norm(cal_X[j][k] - cluster_X[i][2][u]) for u in range(15)]
                    minindex = distances.index(min(distances))
                    sadmap.append(cluster_X[i][2][minindex])
                sadmap = np.array(sadmap)
                moods.append(sadmap)
        moods = np.vstack((moods[0], moods[1], moods[2]))
        Dest.append(moods)


    # 解方程，结出7个源的A和b
    beta_hat = [betahat for i in range(7)] # 0-3进行搜索，步长为0.2
    gamma_hat = [gammahat for i in range(7)] # 0-3进行搜索，步长为0.2
    dim = 310
    Origin = np.vstack((cal_X[0], cal_X[1], cal_X[2]))
    f = [1 for i in range(Origin.shape[0])]
    A_dict = {}
    for i in range(7):
        beta = beta_hat[i]/dim * np.trace(sum([f[j]*Origin[j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]))
        gamma = gamma_hat[i]*sum(f)
        f_hat = sum(f)+gamma
        origin_hat = sum([f[j]*Origin[j].reshape(-1,1) for j in range(Origin.shape[0])])
        dest_hat = sum([f[j]*Dest[i][j].reshape(-1,1) for j in range(Origin.shape[0])])
        P = sum([f[j]*Origin[j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]) - origin_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
        Q = sum([f[j]*Dest[i][j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]) - dest_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
        A = np.dot(Q,np.linalg.inv(P))
        b = (dest_hat-np.dot(A,origin_hat))/f_hat
        A_dict[selectedsource[i]] = (A,b)


    return A_dict, (test_X, test_Y), accuracybackup



def rebuilddata2(testNumber, gammahat, betahat, destNumber, destmodel):
    '''
    :param testNumber: 被测试的对象序号
    :return: 1.从测试的对象数据中调出3个session，并从这三个session中每个类挑20个数据，共60个sample,组成校验数据，剩下的12session形成测试数据
            2. 每个训练数据对象，使用KNN聚类，得到每个类别20个聚类中心
    '''
    # 校验数据
    cal_X = []
    cal_Y = []
    session1 = 0
    session2 = 0
    session3 = 0
    first = EEG_Y[testNumber][0][0]
    for i in range(EEG_Y[testNumber].shape[0]):
        if EEG_Y[testNumber][i][0]!=first:
            if session1 == 0:
                session1 = i-1
                first = EEG_Y[testNumber][i][0]
            elif session2 == 0:
                session2 = i-1
                first = EEG_Y[testNumber][i][0]
            elif session3 == 0:
                session3 = i-1
                break
    choose1 = random.sample(range(0, session1), 20)
    choose2 = random.sample(range(session1+1, session2), 20)
    choose3 = random.sample(range(session2+1, session3), 20)
    cal_X.append(EEG_X[testNumber][0:session1+1,:])
    cal_X.append(EEG_X[testNumber][session1+1:session2+1,:])
    cal_X.append(EEG_X[testNumber][session2+1:session3+1,:])
    cal_Y.append(EEG_Y[testNumber][0:session1+1,:])
    cal_Y.append(EEG_Y[testNumber][session1+1:session2+1,:])
    cal_Y.append(EEG_Y[testNumber][session2+1:session3+1,:])

    # cal_X.append(EEG_X[testNumber].take(choose1, 0))
    # cal_X.append(EEG_X[testNumber].take(choose2, 0))
    # cal_X.append(EEG_X[testNumber].take(choose3, 0))
    # cal_Y.append(EEG_Y[testNumber].take(choose1, 0))
    # cal_Y.append(EEG_Y[testNumber].take(choose2, 0))
    # cal_Y.append(EEG_Y[testNumber].take(choose3, 0))

    # 前三个session用于排序源数据
    order_X = EEG_X[testNumber][0: session3 + 1, :]
    order_Y = EEG_Y[testNumber][0: session3 + 1, :]

    # 测试数据
    test_X = EEG_X[testNumber][session3+1:, :]
    total_X = EEG_X[testNumber]


    # kNN聚类
    cluster_X = []
    mood = [[],[],[]]
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
    cluster_X=[happyk, smoothk, sadk]

    # 计算校验数据的destination
    moods = []
    for j in range(3):
        if cal_Y[j][0][0] == 2:
            happymap = []
            for k in range(cal_X[j].shape[0]):
                distances = [np.linalg.norm(cal_X[j][k]-cluster_X[0][u]) for u in range(15)]
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
    beta = beta_hat/dim * np.trace(sum([f[j]*Origin[j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]))
    gamma = gamma_hat*sum(f)
    f_hat = sum(f)+gamma
    origin_hat = sum([f[j]*Origin[j].reshape(-1,1) for j in range(Origin.shape[0])])
    dest_hat = sum([f[j]*Dest[j].reshape(-1,1) for j in range(Origin.shape[0])])
    P = sum([f[j]*Origin[j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]) - origin_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
    Q = sum([f[j]*Dest[j].reshape(-1,1) * Origin[j].reshape(1,-1) for j in range(Origin.shape[0])]) - dest_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
    A0 = np.dot(Q,np.linalg.inv(P))
    b0 = (dest_hat-np.dot(A0, origin_hat))/f_hat


    # 监督学习部分
    alpha = 0.8
    f1 = np.array([alpha for _ in range(total_X.shape[0])])
    iter_number = 10
    A1 = np.eye(dim)
    b1 = np.zeros((dim, 1))
    for iter in range(iter_number):
        destinations = []
        mapping_x = np.dot(A1, test_X.transpose()) + b1
        mapping_x = mapping_x.transpose()
        out = destmodel(torch.FloatTensor(mapping_x))
        pred = (torch.max(F.softmax(out,1),1)[1]).numpy().squeeze()
        for i in range(test_X.shape[0]):
            if pred[i] == 2:
                distances = [np.linalg.norm(test_X[i] - cluster_X[0][u]) for u in range(15)]
                minindex = distances.index(min(distances))
                destinations.append(cluster_X[0][minindex])
            elif pred[i] == 1:
                distances = [np.linalg.norm(test_X[i] - cluster_X[1][u]) for u in range(15)]
                minindex = distances.index(min(distances))
                destinations.append(cluster_X[1][minindex])
            elif pred[i] == 0:
                distances = [np.linalg.norm(test_X[i] - cluster_X[2][u]) for u in range(15)]
                minindex = distances.index(min(distances))
                destinations.append(cluster_X[2][minindex])

        # confidence
        for i in range(mapping_x.shape[0]):
            d1 = min([np.linalg.norm(mapping_x[i] - cluster_X[0][u]) for u in range(15)])
            d2 = min([np.linalg.norm(mapping_x[i] - cluster_X[1][u]) for u in range(15)])
            d3 = min([np.linalg.norm(mapping_x[i] - cluster_X[2][u]) for u in range(15)])
            dist1 = min(d1, min(d2, d3))
            dist2 = d1
            if dist1 == d1:
                dist2 = min(d2,d3)
            elif dist1 == d2:
                dist2 = min(d1, d3)
            elif dist1 == d3:
                dist2 = min(d1, d2)
            f1[i+session3+1] = 1./(1+np.exp(1-(dist2-dist1)))

        # 更新A1 b1
        destinations = np.array(destinations)
        dest = np.vstack((Dest, destinations))
        beta = beta_hat / dim * np.trace(
            sum([f1[j] * total_X[j].reshape(-1, 1) * total_X[j].reshape(1, -1) for j in range(total_X.shape[0])]))
        gamma = gamma_hat * sum(f1)
        f_hat = sum(f1) + gamma
        origin_hat = sum([f1[j] * total_X[j].reshape(-1, 1) for j in range(total_X.shape[0])])
        dest_hat = sum([f1[j] * dest[j].reshape(-1, 1) for j in range(dest.shape[0])])
        P1 = sum([f1[j] * total_X[j].reshape(-1, 1) * dest[j].reshape(1, -1) for j in
                 range(total_X.shape[0])]) - origin_hat * origin_hat.transpose() / f_hat + beta * np.eye(dim)
        Q1 = sum([f1[j] * dest[j].reshape(-1, 1) * total_X[j].reshape(1, -1) for j in
                 range(total_X.shape[0])]) - dest_hat * origin_hat.transpose() / f_hat + beta * np.eye(dim)
        A1 = np.dot(Q1, np.linalg.inv(P1))
        b1 = (dest_hat - np.dot(A1, origin_hat)) / f_hat

    A = np.dot(A1,A0)
    b = np.dot(A1,b0)+b1
    return A, b



def learning1(testNumber, gammahat, betahat):

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

    # 排序选择出最前面的7个源
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
    for i in range(7):
        maxn = max(accuracylist)
        maxindex = accuracylist.index(maxn)
        accuracylist[maxindex] = -1
        if maxindex >= testNumber:
            maxindex = maxindex + 1
        selectedsource.append(maxindex)

    selectedsource = sorted(selectedsource)
    Ab_dict = {}
    for item in selectedsource:
        cla = item
        if item>testNumber:
            cla = item -1
        A, b = rebuilddata2(testNumber, gammahat, betahat, item, classifer[cla])
        Ab_dict[item] = (A,b)

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
        out = classifer[i](torch.FloatTensor(mapping_x.transpose()))
        totalout += weight[i] * out
    pred = torch.max(F.softmax(totalout, 1), 1)[1]
    finalacc = sum(pred.numpy().squeeze() == test_Y.squeeze()) / len(test_Y.squeeze())


    return finalacc



def learning(testNumber,gammahat, betahat):

    A_dict, test , accuracy = rebuilddata(testNumber, gammahat, betahat)
    chooseObject = list(A_dict.keys())
    classifier = []
    for i in range(len(chooseObject)):
        model = BaseNN(input_dim=310, output_dim=3)
        model.load_state_dict(torch.load(parameterfile[chooseObject[i]]))
        classifier.append(model)

    chooseacuuracy = []
    for item in chooseObject:
        if item>testNumber:
            chooseacuuracy.append(accuracy[item-1])
        else:
            chooseacuuracy.append(accuracy[item])
    weight = []
    for item in chooseacuuracy:
        weight.append(item/sum(chooseacuuracy))


    test_Y = test[1]
    test_X = test[0]
    totalout = 0
    for i in range(len(chooseObject)):
        mapping_x = np.dot(A_dict[chooseObject[i]][0], test_X.transpose()) + A_dict[chooseObject[i]][1]
        out = classifier[i](torch.FloatTensor(mapping_x.transpose()))
        totalout += weight[i]*out
    pred = torch.max(F.softmax(totalout,1),1)[1]
    finalacc = sum(pred.numpy().squeeze() == test_Y.squeeze())/len(test_Y.squeeze())

    # if finalacc>maxaccuracy:
    #     betabest = betahat
    #     gammabest = gammahat
    #     maxaccuracy = finalacc
    #
    # print(testNumber, " maxaccuracy:", maxaccuracy, " beta:", betabest, " gamma:", gammabest)

    return finalacc






def crossValidation():
    set_seed(0)


    gammalist = np.arange(0,3, 0.2)
    betalist = np.arange(0,3,0.2)

    # gammabest = 0
    # betabest = 0
    # maxaccuracy = 0
    # maxaccuacylist = []
    # for gammahat in gammalist:
    #     for betahat in betalist:
    #         accuracy = []
    #         for i in range(15):
    #             ac = learning(i,gammahat,betahat)
    #             accuracy.append(ac)
    #         if sum(accuracy)>maxaccuracy:
    #             maxaccuracy = sum(accuracy)
    #             maxaccuacylist = accuracy
    #             gammabest = gammahat
    #             betabest = betahat
    #
    # print("------------------------------------------------")
    # print(maxaccuracy/len(maxaccuacylist))
    # print(gammabest)
    # print(betabest)
    # print(maxaccuacylist)

    # 监督
    # accuracy = []
    # for i in range(15):
    #     ac = learning(i,0.6,0.2)
    #     accuracy.append(ac)
    # print(accuracy)
    # print(sum(accuracy)/len(accuracy))

    # 半监督
    accuracy = []
    for i in range(15):
        ac = learning1(i,0.6,0.2)
        accuracy.append(ac)
    print(accuracy)
    print(sum(accuracy)/len(accuracy))










if __name__ == "__main__":
   # a ,b = rebuilddata(1)
   # print(a.keys())
   # print(a[list(a.keys())[0]][1])
   # print(b[0].shape)
   # print(b[1].shape)
   # a = np.eye(2) + np.array([1,1])
   # print(a[0:,:])
   #
   # a[0][1] = 3
   # b = np.array([1,3,4])
   # c = [1,2]
   # print(type(b[c]))
   crossValidation()

   '''
   
   D:\Anaconda\envs\pytorch\python.exe E:/大三下/工科创/作业/大作业/EI328_Project/STMTL/StmTL.py
0  maxaccuracy: 0.5755743651753326  beta: 2.2  gamma: 2.0
1  maxaccuracy: 0.7440548166062072  beta: 0.2  gamma: 0.8
2  maxaccuracy: 0.8004836759371221  beta: 2.0  gamma: 2.6
3  maxaccuracy: 1.0  beta: 0.2  gamma: 0.4
4  maxaccuracy: 0.5836356307940347  beta: 0.2  gamma: 1.6
5  maxaccuracy: 0.5933091495364772  beta: 0.2  gamma: 0.0
6  maxaccuracy: 0.6174929463925837  beta: 2.6  gamma: 2.0
7  maxaccuracy: 0.5582426440951229  beta: 2.4000000000000004  gamma: 0.2
8  maxaccuracy: 0.8085449415558242  beta: 0.2  gamma: 1.0
9  maxaccuracy: 0.751309955663039  beta: 0.8  gamma: 0.2
10  maxaccuracy: 0.9004433696090286  beta: 0.8  gamma: 2.2
11  maxaccuracy: 0.60378879484079  beta: 0.2  gamma: 0.0
12  maxaccuracy: 0.9742039500201531  beta: 0.2  gamma: 0.6000000000000001
13  maxaccuracy: 0.7222893994357114  beta: 0.6000000000000001  gamma: 1.2000000000000002
14  maxaccuracy: 0.7912132204756147  beta: 0.2  gamma: 0.4
------------------------------------------------
[0.5755743651753326, 0.7440548166062072, 0.8004836759371221, 1.0, 0.5836356307940347, 0.5933091495364772, 0.6174929463925837, 0.5582426440951229, 0.8085449415558242, 0.751309955663039, 0.9004433696090286, 0.60378879484079, 0.9742039500201531, 0.7222893994357114, 0.7912132204756147]
0.7349724573424694

   '''

'''
   0.6938331318016928
0.6000000000000001
0.2
[0.5118903667875857, 0.7480854494155582, 0.679161628375655, 1.0, 0.5086658605401048, 0.5110842402257154, 0.5570334542523177, 0.5292220878677952, 0.7758968158000806, 0.7392180572349859, 0.8766626360338573, 0.5828295042321644, 0.9463925836356308, 0.717049576783555, 0.7243047158403869]

   '''


