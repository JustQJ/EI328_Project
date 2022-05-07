from scipy.io import loadmat
import numpy as np
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os

# x = [[0,0],[1,1],[1,0],[0,1]]
# y = [0,0,1,1]

# clf = SVC(kernel='poly')
# clf.fit(x,y)
# print(clf.predict([[0,0]]))
# 添加两行数据的路径
EEG_X_Path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_X"))
EEG_Y_Path = os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III","EEG_Y"))

datax = loadmat(EEG_X_Path)
aa=np.array(datax['X'])



datay = loadmat(EEG_Y_Path)
bb=np.array(datay['Y'])


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

	print('kk0')


#此处修改核函数和其他参数来调整SVM

	clf = SVC(C=0.9,kernel='linear')

##########


# print(clf.predict([x[3392]]))


	clf.fit(x,y)

	print('kk1')

	predict=clf.predict(prex)

	print('kk2')

	sumright=0
	for i in range(3394):
		if predict[i]==prey[i]:
			sumright+=1
	ratio.append(sumright/3394)
	print(j,sumright/3394)

print(ratio)

sumra=0
for i in range(15):
	sumra+=ratio[i]
print(sumra/15)


