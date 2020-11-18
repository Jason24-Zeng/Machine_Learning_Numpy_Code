import numpy as np
class Kmean(object):
    def __init__(self, center_num, iter):
        self.k = center_num
        self.max_iteration = iter

    def fit(self, train_data): # 能力有限，暂时用循环做
        self.train_data = train_data
        cluster_index = [np.array([]).astype(int) for _ in range(self.k)]
        self.center = np.array(self.train_data.iloc[:self.k])
        print(self.center, self.center.shape)
        m_examples, n_features = self.train_data.shape
        for i in range(m_examples):
            train_x = self.train_data.iloc[i]
            matrix = np.tile(train_x, (self.center.shape[0], 1)) - self.center
            matrix_square = matrix ** 2
            dist_square = matrix_square.sum(axis = 1)
            cluster = int(np.argmin(dist_square))
            #print(cluster, i)
            cluster_index[cluster] = np.append(cluster_index[cluster], i)
        #print(cluster_index)
        #cluster_points = []
        for i in range(self.k):
            new_cluster = self.train_data.iloc[cluster_index[i]]
            #print(new_cluster.shape, cluster_index[i])
            cluster_mean = np.mean(new_cluster, axis = 0)
            #cluster_points.append(new_cluster)
            print(self.center[i], cluster_mean)
            self.center[i] = cluster_mean
        if self.max_iteration == 1:
            return self.center
        self._fit(1)
        return self.center

    def _fit(self, recursion):
        #prev_center = self.center
        m_examples, n_features = self.train_data.shape
        cluster_index = [np.array([]).astype(int) for _ in range(self.k)]
        for i in range(m_examples):
            train_x = self.train_data.iloc[i]
            matrix = np.tile(train_x, (self.center.shape[0], 1)) - self.center
            matrix_square = matrix ** 2
            dist_square = matrix_square.sum(axis = 1)
            cluster = int(np.argmin(dist_square))
            cluster_index[cluster] = np.append(cluster_index[cluster], i)
        for i in range(self.k):
            new_cluster = self.train_data.iloc[cluster_index[i]]
            cluster_mean = np.mean(new_cluster, axis = 0)
            #cluster_points.append(new_cluster)
            self.center[i] = cluster_mean
        if recursion >= self.max_iteration:
            #self.center[:] == prev_center[:] or
            return
        self._fit(recursion + 1)
