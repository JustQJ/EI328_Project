from sklearn.cluster import KMeans
import scipy.io as sio
import numpy as np
import os
import cvxopt

PathSEED_X =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_X.mat"))
PathSEED_Y =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_Y.mat"))
EEG_X = sio.loadmat(PathSEED_X)['X'][0]
EEG_Y = sio.loadmat(PathSEED_Y)['Y'][0] + 1


def mappingMatrix(testNumber, gammahat, betahat, destNumber, session1, session2, session3):
    '''
    :param testNumber: 测试的对象
    :param gammahat: 参数
    :param betahat: 参数
    :param destNumber: 源域对象
    :param session1: 第一个session的范围
    :param session2:  第二个session的范围
    :param session3:  第三个session的范围
    :return: 从目标到该源域的mapping参数 A, b
    '''

    # 校验数据
    cal_X = []
    cal_Y = []

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


    A1 = np.eye(dim)
    b1 = np.zeros((dim, 1))
    # 监督学习部分

    alpha = 0.8
    f1 = np.array([alpha for _ in range(total_X.shape[0])])
    iter_number = 5


    for iter in range(iter_number):
        destinations = []

        mapping_x = np.dot(A1, test_X.transpose()) + b1
        mapping_x = mapping_x.transpose()
        deltatest = []  # 测试数据x的两个距离的差值

        for i in range(mapping_x.shape[0]):
            d1 = min([np.linalg.norm(mapping_x[i] - cluster_X[0][u]) for u in range(15)])
            d2 = min([np.linalg.norm(mapping_x[i] - cluster_X[1][u]) for u in range(15)])
            d3 = min([np.linalg.norm(mapping_x[i] - cluster_X[2][u]) for u in range(15)])
            dists = [d1, d2, d3]
            label = dists.index(min(dists)) #预测标签
            distances = [np.linalg.norm(test_X[i] - cluster_X[label][u]) for u in range(15)] #在预测标签的类别中找到target
            mindex = distances.index(min(distances))
            destinations.append(cluster_X[label][mindex])
            dists1 = sorted(dists)

            deltatest.append(dists1[1]-dists1[0])


        # confidence
        mapping_orderx = np.dot(A1, order_X.transpose()) + b1
        mapping_orderx = mapping_orderx.transpose()
        predlabel = np.zeros((order_X.shape[0],)) #
        delta = []
        for i in range(mapping_orderx.shape[0]):
            d1 = min([np.linalg.norm(mapping_orderx[i] - cluster_X[0][u]) for u in range(15)])
            d2 = min([np.linalg.norm(mapping_orderx[i] - cluster_X[1][u]) for u in range(15)])
            d3 = min([np.linalg.norm(mapping_orderx[i] - cluster_X[2][u]) for u in range(15)])
            dist1 = min(d1, min(d2, d3))
            dist2 = d1
            if dist1 == d1:
                dist2 = min(d2,d3)
                predlabel[i] = 2
            elif dist1 == d2:
                dist2 = min(d1, d3)
                predlabel[i] = 1
            elif dist1 == d3:
                dist2 = min(d1, d2)
                predlabel[i] = 0

            delta.append(dist2-dist1)

        deltamin = min(delta)
        deltamax = max(delta)
        kumber = 20
        deltawide = (deltamax - deltamin) / kumber
        ffk = [0 for i in range(kumber)]
        bin = [[] for i in range(kumber)]
        for i in range(order_X.shape[0]):
            j = min(kumber - 1, int((delta[i] - deltamin) / deltawide))  # 找到其对应的范围
            a = 0
            if predlabel[i] == order_Y[i][0]:  # 判断是否预测正确
                a = 1
            bin[j].append(a)
        for i in range(kumber):
            if len(bin[i]) == 0:
                ffk[i] = 0
            else:
                ffk[i] = sum(bin[i]) / len(bin[i])  # 为真的数量除以总数

        pk = quadprog(ffk, kumber)

        for i in range(mapping_x.shape[0]):

            if deltatest[i]<=deltamin:
                f1[i + session3 + 1] = pk[0]
            elif deltatest[i]>=deltamax:
                f1[i + session3 + 1] = pk[kumber-1]
            else:
                j = min(kumber - 1, int((deltatest[i] - deltamin) / deltawide))
                f1[i + session3 + 1] = pk[j]

        # 更新A1 b1
        destinations = np.array(destinations)
        dest = np.vstack((Dest, destinations))
        beta = beta_hat / dim * np.trace(
            sum([f1[j] * np.dot(total_X[j].reshape(-1, 1), total_X[j].reshape(1, -1)) for j in
                 range(total_X.shape[0])]))
        gamma = gamma_hat * sum(f1)
        f_hat = sum(f1) + gamma
        origin_hat = sum([f1[j] * total_X[j].reshape(-1, 1) for j in range(total_X.shape[0])])
        dest_hat = sum([f1[j] * dest[j].reshape(-1, 1) for j in range(dest.shape[0])])
        P1 = sum([f1[j] * np.dot(total_X[j].reshape(-1, 1), dest[j].reshape(1, -1)) for j in
                  range(total_X.shape[0])]) - np.dot(origin_hat, origin_hat.transpose()) / f_hat + beta * np.eye(
            dim)
        Q1 = sum([f1[j] * np.dot(dest[j].reshape(-1, 1), total_X[j].reshape(1, -1)) for j in
                  range(total_X.shape[0])]) - np.dot(dest_hat, origin_hat.transpose()) / f_hat + beta * np.eye(dim)
        A1 = np.dot(Q1, np.linalg.inv(P1))
        b1 = (dest_hat - np.dot(A1, origin_hat)) / f_hat

    A = np.dot(A1, A0)
    b = np.dot(A1, b0) + b1
    return A,b, cluster_X


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

    return np.array(sol['x']).squeeze()



def learning(testNumber, gamma_hat, beta_hat):
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

    Abc_dict = {}

    for i in range(15):
        if i!=testNumber:
            A,b,cluster = mappingMatrix(testNumber, gamma_hat,beta_hat, i, session1, session2, session3)
            Abc_dict[i] = (A,b, cluster)

    # 使用已知数据来判断每个source的精度，用于计算权重
    accurancy = []
    for i in range(15):
        if i!=testNumber:
            mapping_orderx = (np.dot(Abc_dict[i][0], order_X.transpose()) + Abc_dict[i][1]).transpose()
            rightpred = 0
            for j in range(mapping_orderx.shape[0]):
                d1 = min([np.linalg.norm(mapping_orderx[j] - Abc_dict[i][2][0][u]) for u in range(15)])
                d2 = min([np.linalg.norm(mapping_orderx[j] - Abc_dict[i][2][1][u]) for u in range(15)])
                d3 = min([np.linalg.norm(mapping_orderx[j] - Abc_dict[i][2][2][u]) for u in range(15)])

                dists = [d1, d2, d3]
                if dists[2 - order_Y[j][0]] == min(dists):
                    rightpred+=1
            acc = rightpred/order_Y.shape[0]
            accurancy.append(acc)

    weights = [acc/sum(accurancy) for acc in accurancy]

    # 测试
    testaccracy = []
    pred = []
    for i in range(15):
        if i!=testNumber:
            mapping_testx = (np.dot(Abc_dict[i][0], test_X.transpose()) + Abc_dict[i][1]).transpose()
            currentpred = np.zeros((mapping_testx.shape[0],))
            for j in range(mapping_testx.shape[0]):
                d1 = min([np.linalg.norm(mapping_testx[j] - Abc_dict[i][2][0][u]) for u in range(15)])
                d2 = min([np.linalg.norm(mapping_testx[j] - Abc_dict[i][2][1][u]) for u in range(15)])
                d3 = min([np.linalg.norm(mapping_testx[j] - Abc_dict[i][2][2][u]) for u in range(15)])

                dists = [d1, d2, d3]
                indexmin = dists.index(min(dists))
                currentpred[j] = 2-indexmin


            pred.append(currentpred)

    finalpred = np.zeros((test_X.shape[0],))
    for i in range(test_X.shape[0]):
        w0 = 0
        w1 = 0
        w2 = 0
        for j in range(14):
            if pred[j][i] == 0:
                w0+=weights[j]
            elif pred[j][i] == 1:
                w1+=weights[j]
            elif pred[j][i] == 2:
                w2+=weights[j]

        ww = [w0, w1, w2]
        finalpred[i] = ww.index(max(ww))

    finalacc = sum(finalpred == test_Y.squeeze())/finalpred.shape[0]

    return finalacc

def crossValidation():

    gamma_hat = 0.2
    beta_hat = 0.6
    crossaccracy = []
    for i in range(15):
        ac = learning(i, gamma_hat, beta_hat)
        crossaccracy.append(ac)
    totalacc = sum(crossaccracy) / len(crossaccracy)
    print("the accuracy of each subject:", crossaccracy)
    print("the average accuracy of 15-fold validation:", totalacc)

if __name__ == '__main__':
    '''
    可以运行，但是效果极差，时间较长
    '''
    crossValidation()









