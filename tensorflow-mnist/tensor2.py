#! /usr/bin/env python  
# -*- coding:utf-8 -*-
"""
Created on Fri Jul 28 15:27:21 2017

@author: dell
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
learning_rate = 0.001
n_input = 784  # MNIST 数据输入 (图片尺寸: 28*28)
n_classes = 10  # MNIST 总计类别 (数字 0-9)
# Features and Labels
# 特征和标签
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
# 权重和偏置项
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


import tensorflow as tf

import math
save_file = './mn/train_model.ckpt1'
saver = tf.train.Saver()
batch_size = 128
n_epochs = 100
# Launch the graph
# 加载图
with tf.Session() as sess:
    saver.restore(sess, save_file)

    print(sess.run(weights).__sizeof__())
    print(sess.run(bias))
    test_accuracy = sess.run(
            accuracy,
            feed_dict={features: mnist.test.images, labels: mnist.test.labels})

    print('Test Accuracy: {}'.format(test_accuracy))
