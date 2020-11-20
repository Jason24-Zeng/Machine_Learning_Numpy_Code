import Create_Samples
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
def Testing(num_examples, mus, vars):
    train_data, subsample = Create_Samples.create_sample(num_examples, mus, vars)
    #print(train_data)
    n_points = train_data.shape[0]
    n_clusters = mus.shape[0]
    pdfs = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        pdfs[:, i] = multivariate_normal.pdf(train_data, mus[i], np.diag(vars[i]))
    print(pdfs)
    best_pred = np.argmax(pdfs, axis = 1)
    print(best_pred)
    print(accuracy(best_pred, num_examples))
    plot_train(train_data, best_pred, num_examples)

def accuracy(best_pred, num_examples):
    scores = 0
    prev = 0
    #print(num_examples)
    for i in range(len(num_examples)):
        scores += (best_pred[prev: prev + num_examples[i]] == i).sum()
        prev += num_examples[i]
    print(scores)
    #print(best_pred[])
def plot_train(train_data, best_pred, num_examples):
    traindata = pd.DataFrame(data=train_data)
    prev = 0
    right_subsample = []
    wrong_subsample  = []
    for index, number in enumerate(num_examples):
        subsample = traindata.iloc[prev: prev+number]
        #prev = number
        right_subsample.append(subsample[best_pred[prev:prev+number] == index])
        wrong_subsample.append(subsample[best_pred[prev:prev+number] != index])
        prev += number
    plt.figure(figsize=(20, 16))
    plt.axis([-10, 15, -5, 15])
    color = ['r','b','y']
    for i in range(len(right_subsample)):
        plt.scatter(right_subsample[i].iloc[:, 0], right_subsample[i].iloc[:, 1], c = color[i], marker= 'x', s=25)
        plt.scatter(wrong_subsample[i].iloc[:, 0], wrong_subsample[i].iloc[:, 1], c = color[i], marker='o', s=25)
    plt.savefig('testing.jpg')
    plt.show()



if __name__ == '__main__':
    train_data_num =np.array([40, 60, 100])
    mus_psedu = np.array([[0.5, 0.5], [5.5, 2.5], [1, 7]])
    vars_psedu = np.array([[1, 3], [2, 2], [6, 2]])
    Testing(train_data_num, mus_psedu, vars_psedu)