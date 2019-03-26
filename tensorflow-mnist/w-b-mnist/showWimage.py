import tensorflow as tf
import cv2
import numpy as np

#获取tensor张量
reader = tf.train.NewCheckpointReader('mode/train_model.ckpt')
all_variables = reader.get_variable_to_shape_map()
print(all_variables)
w = reader.get_tensor("weights")
print(w.shape)

#获取图
saver = tf.train.import_meta_graph('mode/train_model.ckpt.meta')
gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
print(tensor_name_list)

# x = gragh.get_operation_by_name('input/x-image/x_image').outputs[0]
# keep_prob =gragh.get_operation_by_name('fc1/keep_prob/keep_prob').outputs[0]
# prediction = gragh.get_operation_by_name('fc2/softmax/prediction').outputs[0]
# control_dependency_1 = gragh.get_operation_by_name('train/gradients/Conv1/conv2d_1/Conv2D_grad/tuple/control_dependency_1').outputs[0]
# control_dependency = gragh.get_operation_by_name('train/gradients/Conv1/conv2d_1/Conv2D_grad/tuple/control_dependency').outputs[0]
# Conv2D = gragh.get_operation_by_name('Conv1/conv2d_1/Conv2D').outputs[0]
#
# print(control_dependency)
# print(control_dependency_1)
# print(Conv2D)

#展示权值图
while(1):
    for i in range(10):  # 先显示12个权重图
        fileName = "w" + str(i)
        cv2.imshow(fileName, np.reshape((w[:, i]), (28, 28)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# #保存权值图
# for i in range(10):
#     fileName = "W-image/w" +str(i)+".jpg"
#     im= np.zeros((28,28))
#     im =np.reshape((w[:,i]),(28,28))
#     im[im>0] =255
#     cv2.imwrite(fileName,im)
#     print(im)