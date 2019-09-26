import tensorflow as tf
import cv2
import numpy as np
import time

#获取tensor张量
reader = tf.train.NewCheckpointReader('mode/mnist_model.ckpt')
all_variables = reader.get_variable_to_shape_map()
#print(all_variables)
w0 = reader.get_tensor("Conv2/W_conv2/W_conv2/Adam_1")
#print(w0.shape)
#print(w0)#(5, 5, 32, 64)

dst1 = np.zeros((5 * 4 * 8,  # 图片宽*图片行个数
                 5 * 8 * 8))  # 图片高*图片列个数

rows_th1 = cols_th1 = 0

for i in range(64):  # 64个权重图

    dst = np.zeros((5 * 4,  # 图片宽*图片行个数
                    5 * 8))  # 图片高*图片列个数
    rows_th = cols_th = 0

    fileName = "w" + str(i)
    pinjieimage = np.reshape((w0[:,:,:,i]), (5,5,32))
    for j in range(32):
       pinjieimage2 = np.reshape(pinjieimage[:,:,j],(5,5))
       shape = pinjieimage2.shape
       print(pinjieimage2*100000000)
       cols = shape[1]
       rows = shape[0]

       if j % 8 == 0 and j != 0:
         print("xxxxxxxxxxxxxxxxx")
         rows_th = rows_th + 1
         cols_th = 0
    # print(rows_th, cols_th, cols, rows)
       dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols] = pinjieimage2*100000000
       cols_th = cols_th + 1

    # cv2.imshow("conv2d_1", dst)
    # cv2.waitKey()

    shape = dst.shape
    cols1 = shape[1]
    rows1 = shape[0]
    print(shape)

    if i % 8 == 0 and i != 0:
       print("!!!!!!!!!!!!!!!!")
       rows_th1 = rows_th1 + 1
       cols_th1 = 0
    # print(rows_th, cols_th, cols, rows)
    dst1[rows_th1 * rows1:(rows_th1 + 1) * rows1, cols_th1 * cols1:(cols_th1 + 1) * cols1] = dst
    cols_th1 = cols_th1 + 1

    cv2.imshow("conv2d_2", dst1)
    cv2.waitKey()
    print("===========================")


    # shape = pinjieimage.shape  # 三通道的影像需把-1改成1
    # print(shape)








