import tensorflow as tf
import cv2
import numpy as np

#获取tensor张量
reader = tf.train.NewCheckpointReader('mode/mnist_model.ckpt')
all_variables = reader.get_variable_to_shape_map()
print(all_variables)
w0 = reader.get_tensor("fc2/W_fc2/W_fc2/Adam_1")
print(type(w0))
print(w0.shape)
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
Conv2D = gragh.get_operation_by_name('Conv1/conv2d_1/Conv2D').outputs[0]

print(control_dependency)
print(control_dependency_1)
print(Conv2D)

cap = cv2.VideoCapture(0)

# Launch the graph
# 启动图
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('mode'))# 加载变量值
    print('finish loading model!')
    while (1):
        ret, frame = cap.read()
        cv2.rectangle(frame, (265, 195), (345, 275), (0, 0, 255), 2)
        cv2.imshow("capture", frame)
        roiImg = frame[200:270, 270:340]
        img = cv2.resize(roiImg, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("roi", img)
        np_img = img.astype(np.float32)
        # img = np.zeros((28, 28))
        image = np.reshape(img, (-1, 28, 28, 1))
        conv2d = sess.run(Conv2D, feed_dict={x: image, keep_prob: 1.0})#(1,28,28,32)
        outImage = np.zeros((784,32))
        outImage = np.reshape(conv2d,(784,32))
        for i in range(12):#先显示12个权重图
         fileName = "w" + str(i)
         cv2.imshow(fileName,np.reshape((outImage[:,i]),(28,28)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
