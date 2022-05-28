import math
import os.path

import torch.nn.functional as F
import torch
from scipy.io import loadmat
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

learning_rate = 0.0001
SEQ_SIZE = 4
BATCH_SIZE = 1000

# 添加两行数据的路径
EEG_X_Path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_X"))
EEG_Y_Path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_Y"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder_s(nn.Module):
    def __init__(self, input_nc=3, encode_dim=62, lstm_hidden_size=62,
                 seq_len=SEQ_SIZE, num_lstm_layers=1, bidirectional=False):
        super(Encoder_s, self).__init__()
        self.seq_len = seq_len
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size

        self.encoder = nn.Sequential(

        )

        # self.fc = nn.Linear(310, encode_dim)
        self.lstm = nn.LSTM(310, encode_dim, batch_first=True)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h), Variable(c)

    def forward(self, x):
        # x.shape [batchsize, seqsize, 310]
        B = x.size(0)
        x = x.view(B * SEQ_SIZE, 310)
        # [batchsize * seqsize, 310]
        # x = self.fc(x)
        # [batchsize , seqsize ,310]
        x = x.view(-1, SEQ_SIZE, x.size(1))
        h0, c0 = self.init_hidden(x)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return hn

class Decoder_s(nn.Module):
    def __init__(self, output_nc=3, encode_dim=62):
        super(Decoder_s, self).__init__()

        self.project = nn.Sequential(
            # nn.Linear(encode_dim, 62),
            # nn.ReLU(),
            nn.Linear(62, 3),
            nn.Sigmoid()
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear()
        # )

    def forward(self, x):
        decode = self.project(x)
        return decode

class net_s(nn.Module):
    def __init__(self):
        super(net_s, self).__init__()
        self.n1 = Encoder_s()
        self.n2 = Decoder_s()

    def forward(self, x):
        output = self.n1(x)
        output = self.n2(output)          # B*1
        return output

def train_s(j):
    model = net_s()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.MSELoss()

    train_x,train_y=make_train_data_s(j)

    for epoch in range(5000):
        print('epoch {}'.format(epoch + 1))
        train_loss = 0.
        train_acc = 0.
        for i in range(train_x.size(0)):
            inputs, label = Variable(train_x[i]).cuda(), Variable(train_y[i]).cuda()
            label = label.view(BATCH_SIZE)
            output = model(inputs)
            output = output[0]
            # if i == 0:
                # print(label)
                # print('333333333333333')
                # print(output)
            #label = label.view(1, -1)
            loss = loss_func(output, label) # / label.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.data.cpu().numpy()))

    torch.save(model.state_dict(), path_save)

def loadEEG(j):  # j not in
    datax = loadmat(EEG_X_Path)
    aa = np.array(datax['X'])

    datay = loadmat(EEG_Y_Path)
    bb = np.array(datay['Y'])

    x=[]
    y=[]
    prex=[]
    prey=[]

    for i in range(15):
        if i!=j:
            for k in range(3394):
                x.append(aa[0][i][k].tolist())
                y.append(bb[0][i][k][0])
        else:
            for k in range(3394):
                prex.append(aa[0][i][k].tolist())
                prey.append(bb[0][i][k][0])

    return x, y, prex, prey

def make_train_data_s(j):
    global device
    x,y,prex,prey=loadEEG(j)
    train_x=[]
    train_y=[]
    for k in range(14):
        for i in range(math.floor(3391/BATCH_SIZE)):
            x_data=np.zeros((BATCH_SIZE,4,310))
            y_data=np.zeros((BATCH_SIZE,1))
            for b in range(BATCH_SIZE):
                for ll in range(4):
                    xi=k*3394+i*BATCH_SIZE+b+ll
                    for pp in range(310):
                        x_data[b][ll][pp]=x[xi][pp]
                y_data[b][0]=y[k*3394+i*BATCH_SIZE+b]+1
            train_x.append(x_data)
            train_y.append(y_data)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).to(torch.long).to(device)
    # train_y = torch.from_numpy(train_y).float().to(device)
    return train_x, train_y

def make_test_data_s(j):
    global device
    x,y,prex,prey=loadEEG(j)
    train_x = []
    train_y = []
    for i in range(math.floor(3391/BATCH_SIZE)):
        x_data=np.zeros((BATCH_SIZE,4,310))
        y_data=np.zeros((BATCH_SIZE,1))
        for b in range(BATCH_SIZE):
            for ll in range(4):
                xi=i*BATCH_SIZE+b+ll
                for pp in range(310):
                    x_data[b][ll][pp]=prex[xi][pp]
            y_data[b][0]=prey[i*BATCH_SIZE+b]+1
        train_x.append(x_data)
        train_y.append(y_data)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).to(torch.long).to(device)
    # train_y = torch.from_numpy(train_y).float().to(device)
    return train_x, train_y

def test0(pp):
    num = pp
    model = net_s()
    model.load_state_dict(torch.load(path_save))
    model=model.cuda()
    model.eval()  # 切换为测试模式，防⽌参数更新
    train_x, train_y = make_test_data_s(pp)
    ans = 0
    ss = 0
    for i in range(train_x.size(0)):
        inputs, label = Variable(train_x[i]).cuda(), Variable(train_y[i]).cuda()
        label = label.view(BATCH_SIZE)
        output = model(inputs)
        output = output[0]
        c = 0
        for kk in range(len(output)):
            for pp in range(3):
                if output[kk][pp]==torch.max(output[kk]):
                    pre=pp
            if pre==label[kk]:
                c+=1
        acc = c/len(output)
        # print(acc)
        ans+=c
        ss+=len(output)

    return ans/ss

if __name__ == '__main__':


    '''
    该循环为训练模型
    '''
    # for number in range(15):
    #     path_save = os.path.join('model', 'lstm_model' + str(number) + '.t7')
    #     train_s(number)


    '''
    该循环为测试模型，因为已经有了参数，所以将上面的循环注释掉可直接测试， 但是效果不好
    '''
    accuracy = []
    for number in range(15):
        path_save = os.path.join('model', 'lstm_model' + str(number) + '.t7')
        acc = test0(number)
        accuracy.append(acc)
    avgacc = sum(accuracy)/len(accuracy)
    print("the accuracy of each subject:", accuracy)
    print("the average accuracy of 15-fold validation:", avgacc)
