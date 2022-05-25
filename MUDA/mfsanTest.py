'''
用于测试训练好的mfsan网络
'''

from mfsanTrain import set_seed
from mfsanTrain import MFSAN
from mfsanTrain import saveOrloadParameters
from mfsanTrain import modeltest


def crossValidation():

    set_seed(0)
    shareNet_inDim = 310
    shareNet_outDim = 128
    specific_outDim = 64
    class_number = 3
    totalacc = []
    for i in range(15):
        mymodel = MFSAN(shareNet_inDim, shareNet_outDim, specific_outDim, class_number)

        saveOrloadParameters(mymodel, i, False)

        acc = modeltest(mymodel, i, 3)
        totalacc.append(acc)
    print("the accuracy of each subject:",totalacc)
    print("the average accuracy of 15-fold validation:",sum(totalacc)/len(totalacc))



if __name__ == '__main__':
    crossValidation()