import pandas as pd
import numpy as np

def loadDataset(filename):
    train_data = pd.read_table(filename, sep='\s+', header=None)  # sep='   '
    datamat, labelmat = train_data.iloc[:, :2], train_data.iloc[: ,2]
    return datamat, labelmat
# filename = 'train.txt'
# traindata, trainlabel =loadDataset(filename = filename)
# print(traindata, traindata.shape[0])
def selectsecondindex(first, max_index):
    '''

    :param first:
    :param max_index:
    :return: second number should be in the range of [0, max_index], but cannot be the first
    '''
    second = first
    while second == first:
        second = int(np.random.randint(0, max_index + 1))
    return second

def clipAlpha(a, H, L):
    '''
    To ensure a within the range of [L, H]
    :param a:
    :param H:
    :param L:
    :return:
    '''
    if a > H:
        a = H
    elif a < L:
        a = L
    return a

def kernelTrans(X, A, kTup):
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = np.matmul(X, A)
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltarow = X.iloc[j] - A
            K[j] = np.matmul(deltarow, deltarow)
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('What the hell? The input seems arbitrary')
    return K

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = np.mat(classLabels).reshape(-1,1)
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X.iloc[i], kTup)

def calcEk(os, k):
    #print(np.multiply(os.alphas, os.labelMat).shape, os.K[:,k].shape)
    fXk = float(np.multiply(os.alphas, os.labelMat).T * os.K[:, k] + os.b)
    #print(fXk)
    Ek = fXk - float(os.labelMat[k])
    return Ek


