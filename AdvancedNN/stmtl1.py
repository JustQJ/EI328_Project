import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os
import random
import cvxopt

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
    beta = beta_hat/dim * np.trace(sum([f[j]*np.dot(Origin[j].reshape(-1,1), Origin[j].reshape(1,-1)) for j in range(Origin.shape[0])]))
    gamma = gamma_hat*sum(f)
    f_hat = sum(f)+gamma
    origin_hat = sum([f[j]*Origin[j].reshape(-1,1) for j in range(Origin.shape[0])])
    dest_hat = sum([f[j]*Dest[j].reshape(-1,1) for j in range(Origin.shape[0])])
    P = sum([f[j]*np.dot(Origin[j].reshape(-1,1) , Origin[j].reshape(1,-1)) for j in range(Origin.shape[0])]) - origin_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
    Q = sum([f[j]*np.dot(Dest[j].reshape(-1,1), Origin[j].reshape(1,-1)) for j in range(Origin.shape[0])]) - dest_hat*origin_hat.transpose()/f_hat + beta*np.eye(dim)
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
        mapping_orderx = np.dot(A1, order_X.transpose()) + b1
        mapping_orderx = mapping_orderx.transpose()
        labelout = F.softmax(destmodel(torch.FloatTensor(mapping_orderx)),1)
        labeloutt = labelout.detach().numpy()
        labelpred = (torch.max(labelout, 1)[1]).numpy()
        delta = []
        for i in  range(order_X.shape[0]):
            probility = np.sort(labeloutt[i])
            delta.append(probility[2]-probility[0])
        deltamin = min(delta)
        deltamax = max(delta)
        kumber = 20
        deltawide = (deltamax-deltamin)/kumber
        ffk = [0 for i in range(kumber)]
        bin = [[] for i in range(kumber)]
        for i in range(order_X.shape[0]):
            j = min(kumber-1, int((delta[i]-deltamin)/deltawide)) #找到其对应的范围
            a = 0
            if labelpred[i] == order_Y[i][0]: #判断是否预测正确
                a = 1
            bin[j].append(a)
        for i in range(kumber):
            if len(bin[i])==0:
                ffk[i] = 0
            else:
                ffk[i] = sum(bin[i])/len(bin[i]) #为真的数量除以总数

        pk = quadprog(ffk, kumber)


        outt = F.softmax(out,1).detach().numpy()
        for i in range(mapping_x.shape[0]):
            probility = np.sort(outt[i])
            deltaa = probility[2]-probility[0]
            confidence = 0
            if deltaa<=deltamin:
                confidence = pk[0]
            elif deltaa>=deltamax:
                confidence = pk[kumber-1]
            else:
                j = min(kumber-1, int((deltaa-deltamin)/deltawide))
                confidence = pk[j][0]

            f1[i+session3+1] = confidence

        # 更新A1 b1
        destinations = np.array(destinations)
        dest = np.vstack((Dest, destinations))
        beta = beta_hat / dim * np.trace(
            sum([f1[j] * np.dot(total_X[j].reshape(-1, 1) , total_X[j].reshape(1, -1)) for j in range(total_X.shape[0])]))
        gamma = gamma_hat * sum(f1)
        f_hat = sum(f1) + gamma
        origin_hat = sum([f1[j] * total_X[j].reshape(-1, 1) for j in range(total_X.shape[0])])
        dest_hat = sum([f1[j] * dest[j].reshape(-1, 1) for j in range(dest.shape[0])])
        P1 = sum([f1[j] * np.dot(total_X[j].reshape(-1, 1) , dest[j].reshape(1, -1)) for j in
                 range(total_X.shape[0])]) - np.dot(origin_hat , origin_hat.transpose()) / f_hat + beta * np.eye(dim)
        Q1 = sum([f1[j] * np.dot(dest[j].reshape(-1, 1) , total_X[j].reshape(1, -1)) for j in
                 range(total_X.shape[0])]) - np.dot(dest_hat , origin_hat.transpose()) / f_hat + beta * np.eye(dim)
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

    # for i in range(14):
    #     out = classifer[i](torch.FloatTensor(EEG_X[i+1]))
    #     pred = torch.max(F.softmax(out,1),1)[1]
    #     ac = sum(pred.numpy().squeeze() == EEG_Y[i+1].squeeze()) / len(EEG_Y[i+1].squeeze())

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


    print("test:",finalacc)
    return finalacc


def crossValidation():
    set_seed(0)


    gammalist = np.arange(0,3, 0.2)
    betalist = np.arange(0,3,0.2)

    accuracy = []
    for i in range(15):
        ac = learning1(i,0.6,0.2)
        accuracy.append(ac)
    print(accuracy)
    print(sum(accuracy)/len(accuracy))




def quadprog(f, k):
    '''
    :param f: 一个list
    :param k: 数量
    :return: 解得的p
    '''

    P = 2*np.eye(k)
    q = -2*np.array(f).reshape(-1,1)
    G = np.zeros((k+1, k))
    for i in range(k+1):
        if i==0:
            G[i][0] = -1
        elif i==k:
            G[i][k-1] = 1
        else:
            G[i][i-1] = 1
            G[i][i] = -1
    h = np.zeros((k+1,1))
    h[k][0] = 1

    # 转换为cvxopt的矩阵格式
    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    G = cvxopt.matrix(G, tc='d')
    h = cvxopt.matrix(h, tc='d')

    sol = cvxopt.solvers.qp(P,q,G,h)

    return np.array(sol['x'])








if __name__ == '__main__':
    crossValidation()

