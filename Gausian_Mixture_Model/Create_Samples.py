import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')
def create_sample(m_examples, mus, vars):
    #X = ([] for _ in range(len(mus)))
    total_X = np.array([])
    subsample = []
    for i in range(len(mus)):
        X = np.random.multivariate_normal(mus[i], np.diag(vars[i]), m_examples[i])
        subsample.append(X)
        if i == 0:
            total_X = X
        else:
            total_X = np.vstack((total_X, X))
    return total_X, subsample
def plot_sample(subsample):
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    for i in range(len(subsample)):
        plt.scatter(subsample[i][:, 0], subsample[i][:, 1], s=5)
    plt.show()

def initial_weight(n_cluster, total_X):
    dim, n_points = total_X.shape
    # for now, we initialize mu with formed structure
    mus = np.array([[0, -1], [6, 0], [0, 9]])
    variances = np.array([[1, 1], [1, 1], [1, 1]])
    W = np.ones((n_points, n_cluster))/n_cluster
    pi = W.sum(axis = 0) /W.sum()
    return mus, variances, W, pi
