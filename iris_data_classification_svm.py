import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split


# def load_data(filename):
#     data = np.genfromtxt(filename, delimiter=',')
#     x = data[:, 1:]  # 数据特征
#     y = data[:, 0].astype(int)  # 标签
#     scaler = StandardScaler()
#     x_std = scaler.fit_transform(x)  # 标准化
#     # 将数据划分为训练集和测试集，test_size=.3表示30%的测试集
#     x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.3)
#     return x_train, x_test, y_train, y_test

data_set = pd.read_csv('D:\\Michael\\Documents\\01 项目\\01 大数据平台\\14 数据分析\\智能装备\\测试数据\\鸢尾花\\BP神经网络\\MPL Classifier&Regression\\iris_training.csv', header=None)
x_train = data_set.ix[:, 0:3].values
y_train = data_set.ix[:, 4].values

data_test = pd.read_csv('D:\\Michael\\Documents\\01 项目\\01 大数据平台\\14 数据分析\\智能装备\\测试数据\\鸢尾花\\BP神经网络\\MPL Classifier&Regression\\iris_test.csv', header=None)
x_test = data_test.ix[:, 0:3].values
y_test = data_test.ix[:, 4].values

# rbf核函数，设置数据权重
svc = SVC(kernel='rbf', class_weight='balanced', )
c_range = np.logspace(-5, 15, 11, base=2)
gamma_range = np.logspace(-9, 3, 13, base=2)

# 网格搜索交叉验证的参数范围，cv=3,3折交叉
param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)

# 训练模型
clf = grid.fit(x_train, y_train)

predict_results = clf.predict(x_test)
print(predict_results)
print(y_test)

# 计算测试集精度
score = grid.score(x_test, y_test)
print('精度为%s' % score)

