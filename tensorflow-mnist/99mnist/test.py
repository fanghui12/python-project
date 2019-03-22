import tensorflow as tf
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


save_file = 'mode'


saver = tf.train.import_meta_graph('mode/mnist_model.ckpt.meta')
gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
print(tensor_name_list)
x = gragh.get_operation_by_name('input/x-image/x_image').outputs[0]
keep_prob =gragh.get_operation_by_name('fc1/keep_prob/keep_prob').outputs[0]
prediction = gragh.get_operation_by_name('fc2/softmax/prediction').outputs[0]
Conv2D = gragh.get_operation_by_name('Conv1/conv2d_1/Conv2D').outputs[0]

cap = cv2.VideoCapture(0)

# Launch the graph
# 启动图
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(save_file))# 加载变量值
    print('finish loading model!')
    #pred = tf.get_collection('pred_network')[0]
    while(1):
        ret, frame = cap.read()
        cv2.rectangle(frame, (270, 200), (340, 270), (0, 0, 255), 2)
        cv2.imshow("capture", frame)
        roiImg = frame[200:270, 270:340]
        img = cv2.resize(roiImg, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("roi",img)
        np_img = img.astype(np.float32)
        #img = np.zeros((28, 28))
        image = np.reshape(img, (-1,28, 28, 1))
        #cv2.imshow('input/image',image)q
        y = sess.run(prediction, feed_dict={x: image,keep_prob:1.0})
        if(y.max() > 0.99):
           b=np.argmax(y)
           print(b)
        # print(lable)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
cap.release()
cv2.destroyAllWindows()