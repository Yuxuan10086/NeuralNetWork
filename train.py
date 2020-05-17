def pro_csv(file, shape):
    # 传入路径,单幅图像的尺寸列表,仅处理灰度图
    # 返回处理好的图像列表与标签列表,标签应处于列表的首位
    # data_list为训练用数据,img_list为显示用数据
    import csv
    import numpy as np
    img_list = []
    label = []
    data_list = []
    with open(file) as f:
        imgs = csv.reader(f, delimiter=',')
        for data in imgs:
            label.append(data[0])
            del data[0]
            img = np.zeros((shape[0], shape[1]), np.uint8)
            for i in range(shape[0]):
                img[i] = data[i * shape[1] : (i + 1) * shape[1]]
            img_list.append(np.array(img))
            data_list.append(data)
        f.close()
    return img_list, data_list, label

img_list, data_list, label = pro_csv('example.csv', [28, 28])
# data_list = list(map(int, data_list))
# print(data_list[0])
import cv2
# cv2.imshow('hh', img_list[0])
# cv2.waitKey(0)
# cv2.imshow('jj', img_list[1])
# cv2.waitKey(0)
# cv2.imshow('kk', img_list[2])
# cv2.waitKey(0)
# cv2.imshow('ll', img_list[3])
# cv2.waitKey(0)
def turn_into_int_and_move(listt):
    #传入str双层图像列表,将其全部变为int并且平移缩放
    for i in range(len(listt)):
        # listt = list(map(int, listt[i]))
        for j in range(len(listt[i])):
            listt[i][j] = int(listt[i][j]) / 255 * 0.99 + 0.01
turn_into_int_and_move(data_list)
print(data_list[0])

from NeuralNetWork import NeuralNetWork
network = NeuralNetWork([784, 300, 300, 10], 0.1)
targets = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
for i in range(100):
    targets[int(label[i])] = 0.99
    print(targets)
    print(label[i])
    network.train(data_list[i], targets)
    targets = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
point = 0
import tkinter as tk
for i in range(500, 510):
    network.query(data_list[i])
    res = list(network.final_outputs)
    print(res)
    print(max(res))
    print(res.index(max(res)))
    print('ans', label[i])
    if res.index(max(res)) == int(label[i]):
        point += 1

    win = tk.Tk()
    lab = tk.Label(win, text = str(res.index(max(res))))
    lab.pack()
    win.mainloop()


    cv2.imshow('hh', img_list[i])
    cv2.waitKey(0)
print('point:', point)
