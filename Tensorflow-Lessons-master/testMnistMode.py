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
#===============存储图片=========
#把矩阵中为0的像素点转为255
im[im == 0] = 255
#cv2.imshow("image",im)
cv2.imwrite(fileName, im)
#================================
testImage = X_train[i]
lable = Y_train[i];

save_file = 'mode'


saver = tf.train.import_meta_graph('mode/mnist_model.ckpt.meta')
gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
print(tensor_name_list)
x = gragh.get_operation_by_name('x').outputs[0]
y_ = gragh.get_operation_by_name('y').outputs[0]
keep_prob =gragh.get_operation_by_name('keep_prob').outputs[0]

init = tf.global_variables_initializer()

# Launch the graph
# 启动图
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(save_file))# 加载变量值
    print('finish loading model!')
    pred = tf.get_collection('pred_network')[0]
    image = np.reshape(testImage, (1, 784))
    lable = np.reshape(lable, (1, 10))
    y = sess.run(pred, feed_dict={x: image,keep_prob:1.0})
    print(y)
    b=np.argmax(y)
    print(b)
    print(lable)


