import numpy as np
import math
import random
import matplotlib.pyplot as plt
import csv


# 生成圆的坐标
def generate_circle(lower, upper):
    data_ur = np.zeros(shape=(12, 2))   # 第一象限
    data_ul = np.zeros(shape=(12, 2))   # 第二象限
    data_dl = np.zeros(shape=(12, 2))   # 第三象限
    data_dr = np.zeros(shape=(12, 2))   # 第四象限

    radius = random.randint(int(lower), int(upper))
    angles = np.arange(0, 0.5 * np.pi, 1 / 24 * np.pi)      # 每隔7.5度取一次坐标
    for i in range(12):
        temp_ur = np.zeros(2)
        temp_ul = np.zeros(2)
        temp_dl = np.zeros(2)
        temp_dr = np.zeros(2)
        x = round(radius * math.cos(angles[i]), 2)
        y = round(radius * math.sin(angles[i]), 2)
        temp_ur[0] = x
        temp_ur[1] = y
        data_ur[i] = temp_ur
        temp_ul[0] = -x
        temp_ul[1] = y
        data_ul[i] = temp_ul
        temp_dl[0] = -x
        temp_dl[1] = -y
        data_dl[i] = temp_dl
        temp_dr[0] = x
        temp_dr[1] = -y
        data_dr[i] = temp_dr

    data_up = np.append(data_ur, data_ul, axis=0)
    data_down = np.append(data_dl, data_dr, axis=0)
    data_result = np.append(data_up, data_down, axis=0)
    return data_result, label


# 画圆
def draw_circle(data):
    length = int(data.size / 2)
    x = np.empty(48)
    y = np.empty(48)
    for i in range(length):
        for j in range(2):
            x[i] = data[i][0]
            y[i] = data[i][1]
    plt.scatter(x, y, c="y")
    plt.axis("equal")
    plt.show()


# 将坐标保存到CSV文件中
def save2csv(data, batch, label):
    out = open("D:\\circles.csv", 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')

    length = int(data.size / 2)
    for i in range(length):
        string = str(data[i][0]) + ',' + str(data[i][1]) + ',' + str(batch) + ',' + str(label)
        temp = string.split(',')
        csv_write.writerow(temp)
    out.close()


if __name__ == "__main__":
    lower = [1, 10, 20]             # 半径随机值的下限
    upper = [10, 20, 30]            # 半径随机值的上限
    label = ['0', '1', '2']         # 种类的标签

    for i in range(len(label)):
        # 每类数据生成50组
        for j in range(50):
            data, label = generate_circle(lower[i], upper[i])
            batch = 50 * i + j + 1              # 数据的批次
            save2csv(data, batch, label[i])
            # draw_circle(data)

    print("完成")
