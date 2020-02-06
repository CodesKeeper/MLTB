import numpy as np
from sklearn import neighbors
from sklearn.datasets.samples_generator import make_classification


class KNN():
    def __init__(self, n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, n_neighbors=15,
                 weights='distance'):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_redundant = n_redundant
        self.n_clusters_per_class = n_clusters_per_class
        self.n_classes = n_classes
        self.n_neighbors = n_neighbors
        self.weights = weights

    def make_data(self):
        # 训练数据，X表示训练数据的坐标，Y表示训练数据的标签
        X, Y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_redundant=self.n_redundant,
                                   n_clusters_per_class=self.n_clusters_per_class, n_classes=self.n_classes)
        # 为模型设置训练参数
        clf = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights)
        # 模型训练
        clf.fit(X, Y)
        # 根据训练集的范围大小确定测试集的范围大小
        x_min, x_max = np.around(X[:, 0].min(), decimals=0) - 1, np.around(X[:, 0].max(), decimals=0) + 1
        y_min, y_max = np.around(X[:, 1].min(), decimals=0) - 1, np.around(X[:, 1].max(), decimals=0) + 1
        # 生成测试数据坐标
        xx, yy = np.meshgrid(np.arange(x_min, x_max+0.2, 0.2),
                             np.arange(y_min, y_max+0.2, 0.2))
        grid_test = np.stack((xx.flat, yy.flat), axis=1)
        # 利用训练好的模型对测试集进行预测

        z = clf.predict(grid_test)
        z.shape = (np.size(grid_test, 0), 1)
        Q = np.hstack((grid_test, z))

        Y.shape = (self.n_samples, 1)
        P = np.hstack((X, Y))

        data = {'train_data': P.tolist(), 'test_data': Q.tolist(), 'X_min': x_min, 'X_max': x_max, 'Y_min': y_min,
                'Y_max': y_max}
        return data
