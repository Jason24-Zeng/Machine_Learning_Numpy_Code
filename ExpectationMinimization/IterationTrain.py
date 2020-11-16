from scipy.stats import binom
import numpy as np
def iterations_EM(data, theta, iter=1):
    for i in range(iter):
        print('iteration {}, theta: {}'.format(i+1, theta))
        theta = single_EM(data, theta)
    return theta
def single_EM(data, theta):
    '''
    update the theta with 1 time expectation maximization
    :param data:
    :param theta:
    :return: new_theta
    '''
    #print(data, data.shape)
    lentheta = len(theta)
    contribution = np.array([0.0 for _ in theta])
    weight = np.array([0.0 for _ in theta])
    counts = {}

    for i in range(lentheta):
        counts[i] = {'H': 0, 'T': 0}

    for i in range(data.shape[0]):
        test = data.iloc[i][:]
        #print(test)
        len_test = len(test)
        num_Hs = test.sum()
        #print(num_Hs)
        num_Ts = len_test - num_Hs
        # Expectation Step
        for index in range(lentheta):
            contribution[index] = binom.pmf(num_Hs, len_test, theta[index])
        #print(contribution)
        for index in range(lentheta):
            weight[index] = float(contribution[index]/contribution[:].sum())
        for index in range(lentheta):
            counts[index]['H'] += float(weight[index] * num_Hs)
            counts[index]['T'] += float(weight[index] * num_Ts)
        # Maximization Step

        new_theta = [float(counts[index]['H']/(counts[index]['H']+counts[index]['T'])) for index in range(lentheta)]
    return new_theta

