import numpy as np
class DecisionTree(object):
    def __init__(self, _max_length, _min_splits):
        self.max_length = _max_length
        self.min_splits = _min_splits
    def build_tree(self):
        self.root = self.best_split(self.train_data)
        self.split_branch(self.root, depth = 1)
        return self.root
    #@property
    def terminal_node(self, group):
        class_labels, count = np.unique(group, return_counts= True)
        return class_labels[np.argmax(count)]
    def split_branch(self, node, depth):
        # condition 1: if it cannot be split, create left and right leaves node with no difference, and end it
        node_left, node_right = node['group']
        # node_left, node_right will be the list group with the best split
        del(node['group'])
        if not isinstance(node_left, np.ndarray) or not isinstance(node_right, np.ndarray):
            node['left'] = self.terminal_node(node_left + node_right)
            node['right'] = self.terminal_node(node_left + node_right)
            return
        # condition 2: if depth outmax the maxdepth, terminate it
        if depth >= self.max_length:
            node['left'] = self.terminal_node(node_left)
            node['right'] = self.terminal_node(node_right)
            return
        # condition 3.a: split node_left
        # 3.a.1. if the length of left node >= min_splits
        if len(node_left) <= self.min_splits:
            node['left'] = self.terminal_node(node_left)
        # 3.a.1. if the length of left node < min_splits
        else:
            node['left'] = self.best_split(node_left)
            self.split_branch(node['left'], depth + 1)
        # 3.b.1. if the length of right node >= min_splits
        if len(node_right) <= self.min_splits:
            node['right'] = self.terminal_node(node_right)
        # 3.b.2. if the length of right node < min_splits
        else:
            node['right'] = self.best_split(node_right)
            self.split_branch(node['right'], depth + 1)
    def best_split(self, data):
        class_labels = np.unique(data[:, -1])

        best_index = 999
        best_val = 999
        best_group = None
        best_scores = 999
        # iterate all the group index
        for index in range(data.shape[1] - 1):
            # iterate all the possible label
            for row in data:
                # split by the indexed_feature and its value, when we change row, the index won't change but will change the value
                split_group = self.split(index, row[index], data)
                gini_scores = self.compute_gini_complexity(split_group, class_labels)
                if best_scores > gini_scores:
                    best_scores = gini_scores
                    best_index = index
                    best_val = row[index]
                    best_group = split_group
        result = {'index': best_index, 'group': best_group, 'val': best_val}
        return result
    def compute_gini_complexity(self, groups, labels):
        gini_scores = 0.0
        num_samples = sum([len(group) for group in groups])
        for group in groups:
            size = float(len(group))
            if size == 0: #这个边界条件实在很重要！没有它，计算答案会出很大偏差，因为分母size = 0
                continue
            score = 0.0
            for label in labels:
                proportion = (group[:,-1] == label).sum() / size
                score += proportion * proportion
            gini_scores += (1.0 - score) * (size/num_samples)
        return gini_scores

    def split(self, index, val, data):
        left_node = np.array([]).reshape(0, self.train_data.shape[1])
        right_node = np.array([]).reshape(0, self.train_data.shape[1])
        for row in data:
            if row[index] <= val:
                left_node = np.vstack((left_node, row))
            else:
                right_node = np.vstack((right_node, row))
        return left_node, right_node
    def fit(self, _feature, _labels):
        self.feature = _feature
        self.labels = _labels
        self.train_data = np.column_stack((self.feature, self.labels))
        self.build_tree()
    def _predict(self, node, row):
        """
        Recursively traverse through the tress to determine the
        class of unseen sample data point during prediction
        :param node:
        :param row:
        :return:
        """
        if row[node['index']] < node['val']:
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
        """
        predict the set of data point
        :param test_data:
        :return:
        """
        self.predicted_label = np.array([])
        for data in test_data:
            self.predicted_label = np.append(self.predicted_label, self._predict(self.root, data))

        return self.predicted_label







