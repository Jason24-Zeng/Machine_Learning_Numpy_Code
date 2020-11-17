import numpy as np

def accuracy_metric(actual, predicted):
    num = actual.shape[1] # 这个地方是多少呢shape[0] or shape[1]
    return float(np.array([actual.iloc[i] == predicted[i] for i in range(num)]).sum()/num)