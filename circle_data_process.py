import csv
import math


def read_csv(file_name):
    # 读取的CSV
    csv_file = csv.reader(open(file_name, encoding='utf-8'))

    # 要写入的CSV
    out_file = open("D:\\circles_data.csv", 'a', newline='')
    csv_write = csv.writer(out_file, dialect='excel')

    # 将csv_file每一行的圆坐标取出，如果是同一批次的，则写入到out_file的一行中
    rows = [row for row in csv_file]
    current_batch = 'unknown'
    current_label = 'unknown'
    data_list = []
    for r in rows:
        temp_string = str(r).replace('[', '').replace(']', '').replace('\'', '')
        item = str(temp_string).split(',')
        x = float(item[0])
        y = float(item[1])
        batch = item[2]
        label = item[3]

        if current_batch == batch:
            # data_list.append(str(x) + ';' + str(y))
            distance = math.sqrt(pow(x, 2) + pow(y, 2))
            data_list.append(distance)
        else:
            if len(data_list) != 0:
                data_list.append(current_label)
                result_string = str(data_list).replace('[', '').replace(']', '').replace('\'', '').strip()
                csv_write.writerow(result_string.split(','))

            data_list.clear()
            # data_list.append(str(x) + ';' + str(y))
            distance = math.sqrt(pow(x, 2) + pow(y, 2))
            data_list.append(distance)
            current_batch = batch
            current_label = label

    # 确保最后一个批次的数据能写入
    data_list.append(current_label)
    result_string = str(data_list).replace('[', '').replace(']', '').replace('\'', '').strip()
    csv_write.writerow(result_string.split(','))

    out_file.close()


if __name__ == "__main__":
    read_csv('D:\\circles.csv')
    print('完成')
