#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 15:18:58 2017

@author: dell
"""

import tensorflow as tf

# Remove previous Tensors and Operations
# 移除之前的  Tensors 和运算
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
n_input = 784  # MNIST 数据输入 (图片尺寸: 28*28)
n_classes = 10  # MNIST 总计类别 (数字 0-9)

# Import MNIST data
# 加载 MNIST 数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

# Define loss and optimizer
# 定义损失函数和优化器
lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(lost)

# Calculate accuracy
# 计算准确率
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


import math

save_file = 'mn/train_model.ckpt1'
batch_size = 128
n_epochs = 500

saver = tf.train.Saver()

# Launch the graph
# 启动图
with tf.Session() as sess:
    saver.restore(sess, save_file)

    summary_writer = tf.train.SummaryWriter('/tmp/logdir', sess.graph)

    # Training cycle
    # 训练循环
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples / batch_size)# 进1法整除
        # Loop over all batches
        # 遍历所有 batch
        for i in range(total_batch):
            batch_features, batch_labels = mnist.train.next_batch(batch_size)
            print(batch_labels)
            sess.run(
                optimizer,
                feed_dict={features: batch_features, labels: batch_labels})

        # Print status for every 10 epochs
        # 每运行10个 epoch 打印一次状态
        if epoch % 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features: mnist.validation.images,
                    labels: mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(epoch, valid_accuracy))

    # Save the model
    # 保存模型
    saver.save(sess, save_file)
    print('Trained Model Saved.')

'''

saver = tf.train.Saver()


# Launch the graph
# 加载图
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
'''