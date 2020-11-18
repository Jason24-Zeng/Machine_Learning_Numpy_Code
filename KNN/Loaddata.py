import numpy as np
import pandas as pd
def loaddata(filename):
    dataset = pd.read_table(filename, sep='\s+', header = None)
    return dataset
def split_train_and_test(dataset, ratio):
    X = dataset.iloc[:, :-1]
    newX = X/(np.tile(X.max(axis = 0) - X.min(axis = 0), (X.shape[0], 1)))
    m_examples, features = dataset.shape
    indice_shuffle = np.arange(m_examples)
    np.random.shuffle(indice_shuffle)
    #print(indice_shuffle)
    m_train = int(m_examples*ratio)
    train_X = newX.iloc[indice_shuffle[:m_train]]
    train_label = dataset.iloc[indice_shuffle[:m_train], -1]
    test_X = newX.iloc[indice_shuffle[m_train:]]
    test_label = dataset.iloc[indice_shuffle[m_train:], -1]
    return train_X, test_X, train_label, test_label