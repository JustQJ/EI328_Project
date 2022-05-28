from scipy.io import loadmat
import numpy as np
from sklearn.svm import SVC
import os


# 添加两行数据的路径
EEG_X_Path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_X"))
EEG_Y_Path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_Y"))

datax = loadmat(EEG_X_Path)
aa=np.array(datax['X'])



datay = loadmat(EEG_Y_Path)
bb=np.array(datay['Y'])


def svm(C, kernel):
	ratio=[]

	for j in range(15):
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


	#此处修改核函数和其他参数来调整SVM

		clf = SVC(C=C,kernel=kernel)

		clf.fit(x,y)
		predict=clf.predict(prex)


		sumright=0
		for i in range(3394):
			if predict[i]==prey[i]:
				sumright+=1
		ratio.append(sumright/3394)

	sumra=0
	for i in range(15):
		sumra+=ratio[i]

	print("the accuracy of each subject:", ratio)
	print("the average accuracy of 15-fold validation:", sumra/15)

if __name__ == '__main__':

	'''
	可以调整参数C 和 kernel, 需要运行较长时间
	'''
	c = 0.7
	kernel = 'rbf'
	# c = 0.5
	# kernel = 'poly'
	svm(c, kernel)

