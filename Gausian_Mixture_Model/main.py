import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')
import Create_Samples
import EM_fitting


if __name__ == '__main__':
    m_examples_psedu = [400, 600, 1000]
    mus_psedu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    vars_psedu = [[1, 3], [2, 2], [6, 2]]
    X, subsample = Create_Samples.create_sample(m_examples_psedu, mus_psedu, vars_psedu)
    mus, variances, W, pi = Create_Samples.initial_weight(len(m_examples_psedu), X)
    #print(mus, variances, W, pi)
    print(X.shape)
    mu, var = EM_fitting.fit(X, mus, variances, pi)
    print(mu)

    Create_Samples.plot_sample(subsample)

