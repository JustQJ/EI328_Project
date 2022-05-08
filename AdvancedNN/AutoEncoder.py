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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# 3层网络，输入、隐藏和输出层
class AutoCoder(nn.Module):
    def __init__(self, inputDim, hiddenDim):
        super(AutoCoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hiddenDim, inputDim),
            nn.ReLU()
        )

    def forward(self,input):
        encoder = self.encoder(input)
        decoder = self.decoder(encoder)

        return encoder, decoder


# 保存的训练参数
def saveparameter(model,epispode,loss):
        currentTime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        filename = currentTime+"_"+str(epispode)+"_"+str(loss)+'.pkl'
        savepath = os.path.join("parameters","encoderParameter",filename)
        torch.save(model.state_dict(), savepath)

# 计算KL散度
def kl_divergence(p,q):
    p = F.softmax(p,dim=1)
    q = torch.sigmoid(q)
    q = F.softmax(q,dim=1)
    s1 = torch.sum(p*torch.log(p/q))
    s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))
    return s1+s2




# 用所有数据训练编解码器
def trainCoder():

    set_seed(0)
    batchSize = 512
    learnrate = 0.001
    inputdim = 310
    hiddedim = 64

    #关于sparse的参数
    RHO = 0.05
    sparse = False
    beta = 100

    saveloss = 52 #保存的loss

    # 制作数据集，只需要X
    X = None
    for i in range(0, 15):
        if X is None:
            X = torch.FloatTensor(EEG_X[0])
        else:
            X = torch.cat((X, torch.FloatTensor(EEG_X[i])), 0)

    # 划分训练集和测试集
    dataset = Data.TensorDataset(X,X)
    train_data, eval_data = random_split(dataset,[round(0.8*X.shape[0]),X.shape[0]-round(0.8*X.shape[0])], generator=torch.Generator().manual_seed(42))
    loader = Data.DataLoader(dataset = train_data, batch_size=batchSize,shuffle = True,drop_last=False)
    test_loader = Data.DataLoader(dataset=eval_data, batch_size = X.shape[0]-round(0.8*X.shape[0]))

    # 散度自编码的度量值
    rho = torch.FloatTensor([RHO for _ in range(hiddedim)]).unsqueeze(0)

    # 定义模型
    myEncoder = AutoCoder(inputdim, hiddedim)
    optimizer = torch.optim.Adam(myEncoder.parameters(), lr=learnrate)

    for i in range(1000):
        for step, (train_x, train_y) in enumerate(loader):
            encoder, decoder = myEncoder(train_x)

            # loss计算
            MSE_Loss = ((train_y-decoder)**2).mean()
            loss = MSE_Loss
            if sparse: #使用散度
                rho_hat = torch.sum(encoder,dim=0,keepdim=True)
                sparsity_penalty = beta*kl_divergence(rho,rho_hat)
                loss = loss+sparsity_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # 测试精度
        for step, (test_x, test_y) in enumerate(test_loader):
            encoder, decoder = myEncoder(test_x)
            loss = ((test_y-decoder)**2).mean()
            print("step",i, "loss",loss)
            if loss.detach().numpy()<saveloss:
                saveparameter(myEncoder,i,loss.detach().numpy())













if __name__ == "__main__":
    trainCoder()

