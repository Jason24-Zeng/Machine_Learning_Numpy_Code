import Loaddata
import numpy as np
import Kmean_myself

if __name__ == '__main__':
    filename = 'data.txt'
    dataset = Loaddata.loaddata(filename)

    train_X, test_X, train_labels, test_labels = Loaddata.split_train_and_test(dataset, ratio=0.9)
    clf = Kmean_myself.Kmean(5, 2)
    print(clf.max_iteration, clf.k)
    a = np.array([]).astype(int)
    a = np.append(a, 1)
    print(a)
    center = clf.fit(train_X)
    print(clf.center)
    # num_center = 4
    # center = train_X.iloc[: num_center]
    # # print(center.shape[0])
    # train_x = train_X.iloc[0]
    # new = np.tile(train_x, (center.shape[0], 1)) - center
    # # print(new)
    # new_square = new ** (2)
    # # print(new_square)
    # dist_square = new_square.sum(axis = 1)
    # # print(dist_square)
    # cluster = int(np.argmin(dist_square))
    # a = [1,2,3]
    # print(a[cluster])

