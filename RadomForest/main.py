# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import Preprocess
import calc_accuracy
import DecisionTree
import numpy as np




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #np.seed(1)
    filename = "~/PycharmProjects/RadomForest/sonar-all-data.csv"
    dataset = Preprocess.loadsample(filename=filename)
    #data_split = Preprocess.CVsplit(dataset=dataset, k_folds=5)
    subsample, test_data = DecisionTree.subsample(datasets = dataset, ratio = 0.7)
    # b = subsample.index
    # print(subsample, subsample.index)
    node = DecisionTree.build_tree(train = subsample, max_depth = 10, min_size = 2, n_features = 15)
    print(test_data)
    a = DecisionTree.predict_test(node = node, test = test_data)
    print(a)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
