import preprocessing
import pandas as pd
import numpy as np

# testing loadDataset
filename = 'train.txt'
traindata, trainlabel = preprocessing.loadDataset(filename = filename)
print(traindata, traindata.shape[0], traindata.shape)
A = traindata.iloc[1]
kTup = ('rbf', 2)
K = preprocessing.kernelTrans(X = traindata, A = A, kTup = kTup)
os = preprocessing.optStruct(traindata, trainlabel, 1e5, 1e-3, kTup)
print(os.K)
print(os.alphas.shape, os.labelMat.shape)
print(preprocessing.calcEk(os, 1))

# testing selectsecondindex
# a = []
# for i in range(10):
#     a.append(preprocessing.selectsecondindex(1, 3))
# print(a)