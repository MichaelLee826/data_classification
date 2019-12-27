# data_classification
###### 使用Python实现数据分类

## 1.鸢尾花数据集
鸢尾花数据集是学习分类算法经典的数据集
##### 1.1 数据
1.`iris.csv`：全部数据
2.`iris_training.csv`：训练数据
3.`iris_test.csv`：测试数据

##### 1.2 代码
1.`1iris_data_classification__knn.py`
通过KNN算法实现4种鸢尾花的分类
2.`iris_data_classification__bpnn_V1.py`
通过BP神经网络，基于2个特征，实现2种鸢尾花的分类
3.`iris_data_classification__bpnn_V2.py`
通过BP神经网络，基于4个特征，实现3种鸢尾花的分类
4.`iris_data_cluster_sklearn.py`
通过KMeans和DBSCAN算法实现鸢尾花的聚类

## 2.自生成的圆坐标数据集
自动生成一些不同半径的圆的坐标，然后进行分类
##### 2.1 数据
运行程序生成即可。

##### 2.2 代码
1.`circle_data_generate.py`
生成圆的坐标
2.`circle_data_plot.py`
根据圆的坐标绘制散点图，便于查看数据
3.`circle_data_process.py`
将坐标转换成半径，并做格式处理（只适用于BP神经网络算法）
4.`circle_data_classification__knn.py`
通过KNN算法实现不同半径圆的分类
5.`circle_data_classification_bpnn.py`
通过BP神经网络实现不同半径圆的分类


