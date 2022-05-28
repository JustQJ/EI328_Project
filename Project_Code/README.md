该文件夹下的的目录结构如下：

**EDlstm** 文件夹下的文件实现了PPDA算法

- **ED.py** 直接运行可以测试得到结果，但效果不佳
- **model**文件夹下是15个训练好的模型参数



**MFSAN** 文件夹下的文件实现了MFSAN算法

- **mfsanTrain.py** 直接运行可以对网络进行训练
- **mfsanTest.py** 直接运行可以测试训练好的网络，进行15折交叉验证
- **parameters** 文件夹下包括15组训练好的模型参数



**STMTL**文件夹实现了两种STM算法

- **supervisedStm.py** 是supervised STM算法实现，直接运行可以计算15折交叉验证结果
- **semiSupervisedStm.py** 是semi-supervised STM算法实现，直接运行可以计算15折交叉验证结果， 但效果很差

- **BaseNN** 文件下文件是为每个对象数据训练的一个分类器

  - **BaseNN.py** 直接运行可以训练15个对象的各自分类器


  - **HistoryParameters/ObjectParameters** 文件夹下保存训练好的15个分类器的模型参数，该参数在**supervisedStm.py** 中被使用，使用时请保证只含有一套参数




**SVM** 文件夹下的文件实现了SVM算法

- **SVM.py** 直接运行可以得到SVM算法的运行结果，但时间较长
- **imgs** 中是几次测试结果



**SEED-III** 数据集文件夹下是项目的数据集，所有代码运行均使用该文件夹下的数据文件，所以请注意数据读取路径问题

**requirement.txt** 中是项目需要的依赖库



作者：李欣然 519030910121

​			唐鹏 517020910038
