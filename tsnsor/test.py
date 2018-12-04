#导入依赖库
import numpy as np #这是Python的一种开源的数值计算扩展，非常强大
import tensorflow as tf  #导入tensorflow

##构造数据##
x_data=np.random.rand(100).astype(np.float32) #随机生成100个类型为float32的值
y_data=x_data*0.1+2.8  #定义方程式y=x_data*A+B

##建立TensorFlow神经计算结构##
weight=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))
y=weight*x_data+biases



saver = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:

    model_file = tf.train.latest_checkpoint('lenet/')
    saver.restore(sess, model_file)
    print(sess.run(weight), sess.run(biases))
    print(sess.run(y))
    print(y_data)

