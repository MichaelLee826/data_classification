from sklearn.neural_network import MLPRegressor
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import math


data = []
traffic_feature = []
traffic_target = []

# csv_file = csv.reader(open('D:\\Michael\\Documents\\01 项目\\01 大数据平台\\14 数据分析\\智能装备\\测试数据\\鸢尾花\\BP神经网络\\MPLClassifier\\iris.csv'))
# for content in csv_file:
#     content = list(map(float, content))
#     if len(content) != 0:
#         data.append(content)
#         traffic_feature.append(content[0:5])
#         traffic_target.append(content[-1])
# print('data=',data)
# print('traffic_feature=',traffic_feature)
# print('traffic_target=',traffic_target)

data_set = pd.read_csv('D:\\Michael\\Documents\\01 项目\\01 大数据平台\\14 数据分析\\智能装备\\测试数据\\鸢尾花\\BP神经网络\\MPL Classifier&Regression\\iris_training.csv', header=None)
feature_train = data_set.ix[:, 0:3].values
target_train = data_set.ix[:, 4].values

data_test = pd.read_csv('D:\\Michael\\Documents\\01 项目\\01 大数据平台\\14 数据分析\\智能装备\\测试数据\\鸢尾花\\BP神经网络\\MPL Classifier&Regression\\iris_test.csv', header=None)
feature_test = data_test.ix[:, 0:3].values
target_test = data_test.ix[:, 4].values

# scaler = StandardScaler()                               # 标准化转换
# scaler.fit(traffic_feature)                             # 训练标准化对象
# traffic_feature = scaler.transform(traffic_feature)      # 转换数据集
# feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.2,random_state=0)


# 神经网络输入为2，第一隐藏层神经元个数为5，第二隐藏层神经元个数为2，输出结果为2分类。
# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，
# SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）,SGD标识随机梯度下降。
clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 3), random_state=1)
clf.fit(feature_train, target_train)
predict_results = clf.predict(feature_test)

predict_results_array = np.around(predict_results)
predict_results_array = predict_results_array.astype(np.int64)

print(predict_results_array)
print(target_test)

# for i in range(len(predict_results)):
#     if predict_results[i] < 0 or math.fabs(0 - predict_results[i]) < 0.5:
#         predict_results[i] = 0
#     elif predict_results[i] - 1.0 < 0.5:
#         predict_results[i] = 1
#     else:
#         predict_results[i] = 2


print(accuracy_score(predict_results_array, target_test))
conf_mat = confusion_matrix(target_test, predict_results_array)
print(conf_mat)
print(classification_report(target_test, predict_results_array))

