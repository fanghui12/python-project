import tensorflow as tf
import cv2
import numpy as np
import time

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
Conv2D_1 = gragh.get_operation_by_name('Conv1/conv2d_1/Conv2D').outputs[0]
Conv2D_2 = gragh.get_operation_by_name('Conv2/conv2d_2/Conv2D').outputs[0]

print(control_dependency)
print(control_dependency_1)
print(Conv2D_1)

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
        #===========================conv2d_1===============================
        image = np.reshape(img, (-1, 28, 28, 1))
        conv2d_1 = sess.run(Conv2D_1, feed_dict={x: image, keep_prob: 1.0})#(1,24,24,32)
        outImage = np.zeros((576,32))
        outImage = np.reshape(conv2d_1,(576,32))
        dst = np.zeros((24 * 4,#图片宽*图片行个数
                        24 * 8)) #图片高*图片列个数
        rows_th = cols_th = 0
        for i in range(32):#32个权重图
          fileName = "w" + str(i)
          pinjieimage = np.reshape((outImage[:, i]), (24, 24))
          shape = pinjieimage.shape  # 三通道的影像需把-1改成1
          cols = shape[1]
          rows = shape[0]

          if i % 8 == 0 and i != 0:
             rows_th = rows_th + 1
             cols_th = 0
          #print(rows_th, cols_th, cols, rows)
          dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols] = pinjieimage
          cols_th = cols_th + 1
        cv2.imshow("conv2d_1", dst)
        #===========================conv2d_2===============================
        image = np.reshape(img, (-1, 28, 28, 1))
        conv2d_2 = sess.run(Conv2D_2, feed_dict={x: image, keep_prob: 1.0})#(1,8,8,64)
        outImage = np.zeros((64,64))
        outImage = np.reshape(conv2d_2,(64,64))
        dst = np.zeros((8 * 8,#图片宽*图片行个数
                        8 * 8)) #图片高*图片列个数
        rows_th=cols_th=0
        for i in range(64):#64个权重图
          fileName = "w" + str(i)
          #cv2.imshow(fileName,np.reshape((outImage[:,i]),(8,8)))
          pinjieimage = np.reshape((outImage[:, i]), (8, 8))
          shape = pinjieimage.shape  # 三通道的影像需把-1改成1
          cols = shape[1]
          rows = shape[0]

          if i%8 == 0 and i!=0:
            rows_th =rows_th+1
            cols_th=0
          #print(rows_th,cols_th,cols,rows)
          dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols] = pinjieimage
          cols_th = cols_th + 1
        cv2.imshow("conv2d_2", dst)
        #=========================================================
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
