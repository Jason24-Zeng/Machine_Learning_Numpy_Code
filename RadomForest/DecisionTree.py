import numpy as np
import pandas as pd
def subsample(datasets, ratio):
    m_examples, n_features = datasets.shape
    choose_sample = int(m_examples * ratio)
    train_index = np.random.randint(m_examples, size=choose_sample)
    # new_train_index = list(set(train_index))
    test_index = []
    for i in range(m_examples):
        if i not in train_index:
            test_index.append(i)
    train_sample = datasets.iloc[train_index]
    test_sample = datasets.iloc[test_index]
    return train_sample, test_sample
    # n_subsamples = int(m_examples * ratio)
    # np.random.shuffle(index)
    # shuffled_index = index[:n_subsamples]
    # test_index = index[n_subsamples:]
    # dataset_subsample = datasets.iloc[shuffled_index]
    # test_dataset = datasets.iloc[test_index]
    # return dataset_subsample, test_dataset
def d_split(index, value, dataset):
    left, right = list(), list()
    m_example, fea_and_label = dataset.shape
    for m in range(m_example):
        if dataset.iloc[m, index] < value:
            left.append(m)
        else:
            right.append(m)
    return dataset.iloc[left], dataset.iloc[right]

def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        #num_class = len(class_values)
        for group in groups:
            m_examples = group.shape[0]
            if m_examples == 0:
                continue
            proportion = group[group.iloc[:, -1] == class_value].iloc[:, -1].sum()/float(m_examples)
            gini += (proportion * (1 - proportion))
    return gini
def get_random_feature(num, num_total_feature):
    total_index = np.arange(num_total_feature)
    np.random.shuffle(total_index)
    return total_index[:num]


def get_split(dataset, n_feature): #n_feature would be the target number of feature used per tree
    class_labels = dataset.iloc[:, -1].unique()
    best_index, best_val, best_groups, best_scores = 999, 999, None, 999
    m_examples, num_total_feature = dataset.shape[0], dataset.shape[1] - 1
    feature = get_random_feature(n_feature, num_total_feature)
    for index in feature:
        for data_index in range(m_examples):
            groups = d_split(index, dataset.iloc[data_index, index], dataset)
            gini = gini_index(groups, class_labels)
            if gini < best_scores:
                best_val, best_groups, best_scores, best_index = dataset.iloc[data_index, index], groups, gini, index
    return {'groups': best_groups, 'index': best_index, 'value': best_val}

def to_terminal(group):
    return np.argmax(group.iloc[:, -1])

def split_branch(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if left.shape[0] == 0 or right.shape[0] == 0:
        node['left'] = node['right'] = to_terminal(pd.concat([left, right], axis=0))
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if left.shape[0] <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split_branch(node['left'], max_depth, min_size, n_features, depth + 1)

    if right.shape[0] <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split_branch(node['right'], max_depth, min_size, n_features, depth + 1)

def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split_branch(root, max_depth, min_size, n_features, depth = 1)
    return root
def predict(node, row): # row is a row in pandas
    if row.iloc[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def predict_test(node, test):
    m_example, num_feature = test.shape
    true_case = 0
    for i in range(m_example):
        if predict(node, test.iloc[i]) == test.iloc[i, -1]:
            true_case += 1
    return float(true_case/m_example)

def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count())

def random_forest(train, test, max_depth, min_size, ratio, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        train_sample, test_sample = subsample(datasets=train, ratio= ratio)
        tree = build_tree(train=train_sample, max_depth = max_depth, min_size = min_size, n_features = n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, test.iloc[i]) for i in test.shape[0]]
    return predictions


