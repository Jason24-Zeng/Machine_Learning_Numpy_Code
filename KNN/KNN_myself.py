import numpy as np
import operator
def predict_single(test_x, train_X, train_labels, k):
    # test_x is just a point, instead of matrix
    # train_labels is a pandas DataFrame structure
    matrix_diff = np.tile(test_x, (train_X.shape[0], 1)) - train_X
    matrix_diff = matrix_diff ** 2

    dist_square = matrix_diff.sum(axis = 1)
    #print(dist_square)
    dist = dist_square ** (0.5)
    sorteddistindice = dist.argsort()
    sorted_labels = train_labels.iloc[sorteddistindice]
    knn_sorted_labels = sorted_labels.iloc[:k]
    #print(np.bincount(knn_sorted_labels))
    #print(knn_sorted_labels.shape, np.bincount(knn_sorted_labels))
    return np.argmax(np.bincount(knn_sorted_labels))

def predict_examples(test_X, train_X, test_labels, train_labels, k):
    test_examples, num_feature = test_X.shape
    predict = 0
    predict_val = []
    for i in range(test_examples):
        #print(test_X.iloc[1].shape)
        #train_x = np.array(test_X.iloc[i])
        result = predict_single(test_X.iloc[i], train_X, train_labels, k)
        if result == test_labels.iloc[i]:
            predict += 1
        predict_val.append([result, test_labels.iloc[i]])
    print(predict_val)
    return float(predict/test_examples)



