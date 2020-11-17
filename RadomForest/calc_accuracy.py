import numpy as np

def accuracy_metric(actual, predicted):
    num = len(actual)
    return float(np.array([actual[i] == predicted[i] for i in range(num)]).sum()/num)