import pandas as pd
import matplotlib.pyplot as plt

data_set = pd.read_csv('D:\\circles.csv')
X = data_set[['X']].values
Y = data_set[['Y']].values

X1 = X[1:2400]
Y1 = Y[1:2400]

X2 = X[2401:4800]
Y2 = Y[2401:4800]

X3 = X[4801:]
Y3 = Y[4801:]

batch = data_set[['batch']].values
label = data_set[['label']].values


plt.scatter(X1, Y1, s=1, c="b", label="scatter figure")
plt.scatter(X2, Y2, s=1, c="g", label="scatter figure")
plt.scatter(X3, Y3, s=1, c="r", label="scatter figure")
plt.axis("equal")
plt.show()




