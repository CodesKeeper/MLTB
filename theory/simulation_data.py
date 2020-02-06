import numpy as np
from sklearn.datasets.samples_generator import make_regression, make_classification, make_blobs


class simulation_data():
    #  最小二乘法仿真数据生成
    def __init__(self, K=1, B=2, N_true=100, N_noise=80, mu=5, sigma=6, loop_max=1000, epsilon=1e-5, alpha=0.000001):
        self.K = K  # 直线斜率
        self.B = B  # 直线截距
        self.N_true = N_true  # 标准数据数据量
        self.N_noise = N_noise  # 噪声数据数据量
        self.mu = mu  # 正态分布期望
        self.sigma = sigma  # 正态分布标准差
        self.X_true = np.arange(N_true)  # 标准数据横坐标
        self.Y_true = K * self.X_true + B  # 标准数据纵坐标
        self.X_noise = np.arange(0, N_true, N_true / N_noise)  # 噪声数据横坐标
        self.Y_noise = K * self.X_noise + B  # 噪声数据纵坐标
        self.rand_data = np.random.normal(mu, sigma, len(self.X_noise))  # 产生正态分布的随机数【与x同维数】
        self.noise_Y = self.Y_noise + self.rand_data  # 产生带噪数据（Y值）
        self.X = np.append(self.X_true, self.X_noise)  # 数据合并
        self.Y = np.append(self.Y_true, self.noise_Y)
        self.loop_max = loop_max
        self.epsilon = epsilon
        self.alpha = alpha

    def LSM(self):
        n = len(self.X)
        sumX, sumY, sumXY, sumXX = 0, 0, 0, 0
        for i in range(0, n):
            sumX += self.X[i]
            sumY += self.Y[i]
            sumXX += self.X[i] * self.X[i]
            sumXY += self.X[i] * self.Y[i]
        self.k, self.b = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX), (sumXX * sumY - sumX * sumXY) / (
                n * sumXX - sumX * sumX)
        self.fitted_Y = self.X * self.k + self.b

    def BGD(self):
        m = len(self.X)  # 数据量
        x0 = np.full(m, 1.0)
        input_data = np.vstack([x0, self.X]).T  # 将偏置b第一个分量
        output_data = self.Y
        w = np.random.randn(2)  # 随机生成的拟合参数
        diff = 0  # 用于累加
        error = np.zeros(2)  # 用于计算两次更新参数间的偏差
        count = 0  # 循环次数
        finish = 0  # 终止标志
        while count < self.loop_max:
            count += 1
            sum_m = np.zeros(2)
            for i in range(m):
                dif = (np.dot(w, input_data[i]) - output_data[i]) * input_data[i]
                sum_m = sum_m + dif
            w = w - self.alpha * sum_m
            if np.linalg.norm(w - error) < self.epsilon:
                finish = 1
                break
            else:
                error = w
        self.k = w[1]
        self.b = w[0]
        self.fitted_Y = self.k * self.X + self.b
        return self.fitted_Y

    def SGD(self):
        m = len(self.X)  # 数据量
        x0 = np.full(m, 1.0)
        input_data = np.vstack([x0, self.X]).T  # 将偏置b第一个分量
        output_data = self.Y
        w = np.random.randn(2)  # 随机生成的拟合参数
        diff = 0  # 用于累加
        error = np.zeros(2)  # 用于计算两次更新参数间的偏差
        count = 0  # 循环次数
        finish = 0  # 终止标志
        while count < self.loop_max:
            count += 1
            # 遍历训练数据集，不断更新权值
            for i in range(m):
                diff = np.dot(w, input_data[i]) - output_data[i]  # 训练集代入,计算误差值
                w = w - self.alpha * diff * input_data[i]
            if np.linalg.norm(w - error) < self.epsilon:
                finish = 1
                break
            else:
                error = w
        self.k = w[1]
        self.b = w[0]
        self.fitted_Y = self.k * self.X + self.b
        return self.fitted_Y

    def turn_json(self, type):
        # 将数据写入到json文件---------------------------
        if type == 'LSM':
            self.LSM()
        elif type == 'BGD':
            self.BGD()
        elif type == 'SGD':
            self.SGD()
        data = {}
        data['X_true'] = self.X_true.tolist()
        data['Y_true'] = self.Y_true.tolist()
        data['X_noise'] = self.X_noise.tolist()
        data['noise_Y'] = self.noise_Y.tolist()
        data['X'] = self.X.tolist()
        data['Y'] = self.Y.tolist()
        data['fitted_Y'] = self.fitted_Y.tolist()
        data['k'] = self.k
        data['b'] = self.b

        # 将两列数据整合为一列---------------------------
        def change(x, y):
            xy = np.vstack((x, y))  # 将X与Y合并为两行多列的数组
            temp_list = [[]] * len(xy[0])  # 创建长度为180的空列表
            for i in range(0, len(xy[0])):  # 遍历数组的每一列
                temp_list[i] = xy[:, i].flatten().tolist()  # XY[:,i]是选中数组中的每列
            return temp_list

        # 将列表数据以字典类型保存-----------------------
        json_data = {}
        json_data['True_Points'] = change(data['X_true'], data['Y_true'])
        json_data['Noisy_Points'] = change(data['X_noise'], data['noise_Y'])
        json_data['Simulation_data'] = change(data['X'], data['Y'])
        json_data['Fitted_data'] = change(data['X'], data['fitted_Y'])
        json_data['Fitted_k'] = data['k']
        json_data['Fitted_b'] = data['b']
        return json_data
