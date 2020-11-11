# This is a sample Python script.
import numpy as np


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def compute_gini_complexity(groups, class_labels):
    size = sum([len(group) for group in groups])
    gini_scores = 0.0
    for group in groups:
        num = float(len(group))
        if num == 0:
            continue
        scores = 0.0
        for label in class_labels:
            proportion = (group[:, -1] == label).sum() / num
            scores += proportion * proportion
        gini_scores += (1.0 - scores) * num / size

    return gini_scores


class DecisionTree(object):
    def __init__(self, _max_depth, _min_splits):
        self.max_depth = _max_depth
        self.min_splits = _min_splits

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.train_data = np.column_stack((self.features, self.labels))
        self.build_tree()

    def build_tree(self):
        self.root = self.best_split(self.train_data)
        self.split_branch(self.root, 1)

    def split(self, index, label, data):
        data_left = np.array([]).reshape(0, self.train_data.shape[1])
        data_right = np.array([]).reshape(0, self.train_data.shape[1])
        for row in data:
            if row[index] <= label:
                data_left = np.vstack((data_left, row))
            if row[index] > label:
                data_right = np.vstack((data_right, row))
        return data_left, data_right

    def best_split(self, data):
        """
        choose the best index and row to split the previous group
        :
        """
        class_labels = np.unique(data[:, -1])
        best_scores = 999
        best_index = 999
        best_label = 999
        best_groups = None
        for index in range(data.shape[1] - 1):
            for label in class_labels:
                groups = self.split(index, label, data)
                gini_index = compute_gini_complexity(groups, class_labels)
                if gini_index < best_scores:
                    best_scores = gini_index
                    best_index = index
                    best_label = label
                    best_groups = groups
        result = {'index': best_index, 'label': best_label, 'groups': best_groups}
        return result

    def split_branch(self, node, depth):
        left_node, right_node = node['group']
        del (node['group'])
        if not isinstance(left_node, np.ndarray) or not isinstance(right_node, np.ndarray):
            node['left'] = self.terminal_node(left_node + right_node)
            node['right'] = self.terminal_node(left_node + right_node)
            return
        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return
        if len(left_node) <= self.min_splits:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'], depth + 1)
        if len(right_node) <= self.min_splits:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'], depth + 1)

    #@staticmethod
    def terminal_node(self, _group):
        class_label, count = np.unique(_group, return_counts=True)
        return class_label[np.argmax(count)]
    def _predict(self, node, row):
        if row[node['index']] < node['label']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']


    def predict(self, test_data):
        self.predicted_label = np.array([])
        for index in test_data:
            self.predicted_label = np.append(self.predicted_label, self._predict(self.root, index))
        return self.predicted_label
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('hello')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
