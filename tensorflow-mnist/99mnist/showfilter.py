import tensorflow as tf
import cv2
import numpy as np
import time

#获取tensor张量
reader = tf.train.NewCheckpointReader('mode/mnist_model.ckpt')
all_variables = reader.get_variable_to_shape_map()
print(all_variables)
w1 = reader.get_tensor("Conv1/W_conv1/W_conv1")
w2 = reader.get_tensor("Conv2/W_conv2/W_conv2")
print(w1.shape)
outImage1 = np.zeros((25, 32))
outImage1 = np.reshape(w1, (25, 32))

print(w2.shape)
outImage2 = np.zeros((800, 64))
outImage2 = np.reshape(w2, (800, 64))


for i in range(32):  # 32个权重图
    filename = "image/filterW1"+ str(i)+".jpg"
    pinjieimage = np.reshape((outImage1[:, i]), (5, 5))
    print(pinjieimage*1000)
    cv2.imwrite(filename,pinjieimage*1000)

for i in range(32):  # 32个权重图
    filename = "image/filterW2"+ str(i)+".jpg"
    pinjieimage = np.reshape((outImage1[:, i]), (5, 5))
    print(pinjieimage*1000)
    cv2.imwrite(filename,pinjieimage*1000)


#print(w0)

#获取图
saver = tf.train.import_meta_graph('mode/mnist_model.ckpt.meta')
gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
print(tensor_name_list)

x = gragh.get_operation_by_name('input/x-image/x_image').outputs[0]
keep_prob =gragh.get_operation_by_name('fc1/keep_prob/keep_prob').outputs[0]
prediction = gragh.get_operation_by_name('fc2/softmax/prediction').outputs[0]
control_dependency_1 = gragh.get_operation_by_name('train/gradients/Conv1/conv2d_1/Conv2D_grad/tuple/control_dependency_1').outputs[0]
control_dependency = gragh.get_operation_by_name('train/gradients/Conv1/conv2d_1/Conv2D_grad/tuple/control_dependency').outputs[0]
Conv2D_1 = gragh.get_operation_by_name('Conv1/conv2d_1/Conv2D').outputs[0]
Conv2D_2 = gragh.get_operation_by_name('Conv2/conv2d_2/Conv2D').outputs[0]

print(control_dependency)
print(control_dependency_1)
print(Conv2D_1)