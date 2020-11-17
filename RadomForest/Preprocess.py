import pandas as pd
import numpy as np
def loadsample(filename):
    dataset = pd.read_csv(filename, header=None)
    # change label 'R' and 'M' to 0 and 1
    label = dataset.iloc[:, -1]
    label[label == 'M'] = 1
    label[label == 'R'] = 0
    dataset.iloc[:, -1] = label
    return dataset
def CVsplit(dataset, k_folds):
    data_split = list()
    m_examples, num_labels = dataset.shape
    shuffled_index = np.arange(m_examples)
    np.random.shuffle(shuffled_index)
    # method1: recording the index of data for future use
    # num_subsample = m_examples//k_folds
    num_index = list()
    for i in range(k_folds-1):
        num_index.append(shuffled_index[(i * m_examples)//k_folds: ((i+1)*m_examples)//k_folds])
        data_split.append(dataset.iloc[num_index[i]])
    num_index.append(shuffled_index[((k_folds-1)*m_examples)//k_folds:-1])
    data_split.append(dataset.iloc[num_index[-1]])
    return data_split