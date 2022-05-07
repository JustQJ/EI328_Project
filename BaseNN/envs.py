import os


# 数据包所在位置

pathSEED =os.path.abspath(os.path.join(os.path.abspath(__file__),os.pardir,os.pardir,"SEED-III"))
PathSEED_X = os.getenv('SEED_III_X', os.path.join(pathSEED,"EEG_X.mat"))
PathSEED_Y = os.getenv('SEED-III_Y', os.path.join(pathSEED,"EEG_Y.mat"))
