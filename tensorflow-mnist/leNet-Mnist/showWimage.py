import tensorflow as tf
import cv2
import numpy as np
import time

#获取tensor张量
reader = tf.train.NewCheckpointReader('mode/lenet')
all_variables = reader.get_variable_to_shape_map()
print(all_variables)
# w0 = reader.get_tensor("fc2/W_fc2/W_fc2/Adam_1")
# print(type(w0))
# print(w0.shape)
#print(w0)

#获取图
saver = tf.train.import_meta_graph('mode/lenet.meta')
gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
print(tensor_name_list)

conv1_w = gragh.get_operation_by_name('conv1_w').outputs[0]
Conv2D =gragh.get_operation_by_name('Conv2D').outputs[0]
conv2_w = gragh.get_operation_by_name('conv2_w').outputs[0]
Conv2D_1 = gragh.get_operation_by_name('Conv2D_1').outputs[0]

print(conv1_w)
print(Conv2D)
print(Conv2D_1)