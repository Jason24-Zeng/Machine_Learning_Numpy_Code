import Loaddata
import numpy as np
import KNN_myself
if __name__ == '__main__':
    filename = 'data.txt'
    dataset = Loaddata.loaddata(filename)
    #dataset.iloc[1] =
    train_X, test_X, train_labels, test_labels = Loaddata.split_train_and_test(dataset, ratio=0.9)
    #print(np.bincount(train_labels))
    #print(train_X, train_labels.shape)
    # test_x = test_X.iloc[0]
    # number = KNN_myself.predict(test_x, train_X, train_labels, k=10)
    # print(number, test_labels.iloc[0])
    accuracy = KNN_myself.predict_examples(test_X, train_X, test_labels, train_labels, k=20)
    print('The total accuracy is: {}'.format(accuracy))


