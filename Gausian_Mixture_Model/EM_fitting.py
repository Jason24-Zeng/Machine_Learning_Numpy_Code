import numpy as np
from scipy.stats import multivariate_normal

def update_W(X, mu, var, pi):
    n_points, n_clusters = X.shape[0], mu.shape[0]
    #print(n_points, n_clusters)
    pdfs = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(X, mean= mu[i], cov = np.diag(var[i]))
    #print(pdfs.shape, pdfs.sum(axis=1).shape)
    W = pdfs/pdfs.sum(axis=1).reshape(-1,1)
    # if we don't use reshape, the pdfs.sum(axis =1) will have size (2000,), which cannot broadcast according to the siz
    # e of pdf
    return W

def update_Pi(W):
    pi = W.sum(axis=0)/W.sum()
    return pi

def update_mu(X, W):
    n_clusters = W.shape[1]
    _, dim = X.shape
    mu = np.zeros((n_clusters, dim))
    for i in range(n_clusters):
        mu[i] = np.average(X, axis=0, weights=W[:, i])
    return mu
def update_var(X, mu, W):
    n_clusters = W.shape[1]
    _, dim = X.shape
    var = np.zeros((n_clusters, dim))
    for i in range(n_clusters):
        var[i] = np.average((X-mu[i])**2, axis = 0, weights = W[:, i])
    return var

def cal_logLH(X, pi, mu, var):
    n_points = X.shape[0]
    n_clusters = mu.shape[1]
    pdfs = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(X, mu[i], np.diag(var[i]))

    return np.mean(np.log(pdfs.sum(axis=1)))
def fit(X, mu, var, pi):
    loglh = []
    for i in range(5):
        loglh.append(cal_logLH(X, pi, mu, var))
        W = update_W(X, mu, var, pi)
        pi = update_Pi(W)
        mu = update_mu(X, W)
        var = update_var(X, mu, W)
        print("log-likelihood:{}".format(loglh[-1]))
    return mu, var
