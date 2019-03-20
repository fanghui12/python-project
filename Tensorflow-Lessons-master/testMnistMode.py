import tensorflow as tf
import cv2
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
(X_train, Y_train) = mnist.train.next_batch(1000)

i = random.randint(0,999) # 迭代 0 到 59999 之间的数字
fileName = "mnist-image/"+ str(Y_train[i]) + "_" + str(i) + ".jpg"

im = np.reshape(X_train[i],(28,28))
im[im == 0] = 255
cv2.imshow("image",im)
cv2.imwrite(fileName, im)
