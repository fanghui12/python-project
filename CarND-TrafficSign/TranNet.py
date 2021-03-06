#! /usr/bin/env python  
# -*- coding:utf-8 -*-

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
# 操作完成之后，对训练数据进行随机化排序处理
from sklearn.utils import shuffle


# 在验证集上发现过拟合现象严重，增加 Dropout 层
def LeNet(x):
    mu = 0
    sigma = 0.1

    # Layer 1
    # Conv
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma),name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(6),name='conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding="VALID") + conv1_b
    # Input : 32 * 32 * 1
    # Output: 28 * 28 * 6

    # Activation
    conv1 = tf.nn.relu(conv1)

    # Pooling
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # Output: 14 * 14 * 6

    # Layer 2
    # Conv
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma),name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(16),name='conv2_b')
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding="VALID") + conv2_b
    # Output: 10 * 10 * 16

    # Activation
    conv2 = tf.nn.relu(conv2)

    # Pooling
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # Output: 5 * 5 * 16

    # Flatten
    fc0 = flatten(conv2)
    # Output: 400

    # Layer 3
    # Fully onnected
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma),name='fc1_W')
    fc1_b = tf.Variable(tf.zeros(120),name='fc1_b')
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4
    # Fully connnected
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma),name='fc2_W')
    fc2_b = tf.Variable(tf.zeros(84),name='fc2_b')
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation
    fc2 = tf.nn.relu(fc2)

    # Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 5
    # Fully connected
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma),name='fc3_W')
    fc3_b = tf.Variable(tf.zeros(n_classes),name='fc3_b')
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits



def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



# Load pickled data
# TODO: Fill this in based on where you saved the training and testing data

training_file = "data/train.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


n_classes = len(np.unique(y_train))
print("Number of classes =", n_classes)


# 在 Yann LeCun 的论文中提到，使用灰度图可以进步提高分类的准确度。那么我们首先尝试将图像转化为灰度图

# 保留一下原始格式的图像，万一后面有用呢？
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)
X_valid_rgb = X_valid
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)
X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

# 输出，转化后的图像尺寸
print("Train")
print("***********************")
print("X_train_rgb shape:", X_train_rgb.shape)
print("X_train_gry shape:", X_train_gry.shape)



# 灰度化完成，下面就要进行归一化操作
X_train_norm = (X_train_gry - 128) / 128
X_valid_norm = (X_valid_gry - 128) / 128
X_test_norm = (X_test_gry - 128) / 128

print("Normalized X_train_gry shape:", X_train_norm.shape)
print("Normalized X_valid_gry shape:", X_valid_norm.shape)
print("Normalized X_test_gry shape:", X_test_norm.shape)


EPOCHS = 5
BATCH_SIZE = 128


x = tf.placeholder(tf.float32, (None, 32,32,1),name='x')
y = tf.placeholder(tf.int32, (None),name='y')
one_hot_y = tf.one_hot(y, n_classes)
# Dropout 保留的概率，降低过拟合。
keep_prob = tf.placeholder(tf.float32)

learning_rate = 0.001

logits = LeNet(x)

# 假如需要保存y，以便在预测时使用
tf.add_to_collection('pred_network', logits)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_y,1))
with tf.name_scope('accuracy'):
   accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy_operation)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
# cost function
with tf.name_scope('loss'):
   loss_operation = tf.reduce_mean(cross_entropy)
tf.summary.scalar('loss', loss_operation)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

# combine all summaries
merged = tf.summary.merge_all()

save_file = 'lenet/train_model.ckpt'
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    num_examples = len(X_train_norm)
    #saver.restore(sess, save_file)

    test_accuracy = evaluate(X_test_norm, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    for i in range(EPOCHS):
        X_train_norm, y_train = shuffle(X_train_norm, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_norm[offset:end], y_train[offset:end]
            summary,_=sess.run([merged,training_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        writer.add_summary(summary, offset)
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, save_file)
    print("Model saved")


