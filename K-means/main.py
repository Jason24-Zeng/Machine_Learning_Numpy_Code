import Loaddata
import numpy as np
import Kmean_myself

if __name__ == '__main__':
    filename = 'data.txt'
    dataset = Loaddata.loaddata(filename)

    train_X, test_X, train_labels, test_labels = Loaddata.split_train_and_test(dataset, ratio=0.9)

