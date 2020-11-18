import Loaddata
import numpy as np
import KNN_myself
if __name__ == '__main__':
    filename = 'data.txt'
    dataset = Loaddata.loaddata(filename)
    X = dataset.iloc[:, :-1]
    newX = X/(np.tile(X.max(axis = 0) - X.min(axis = 0), (X.shape[0], 1)))
    print(X.max(axis = 0) - X.min(axis = 0))
    print(newX)