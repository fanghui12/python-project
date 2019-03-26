import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load dataset
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

# define batch size
batch_size = 100
# calculate number of batches
n_batch = mnist.train.num_examples // batch_size

# define placeholders
x = tf.placeholder(tf.float32, [None, 784],name='x')
y = tf.placeholder(tf.float32, [None, 10],name='y')
keep_prob = tf.placeholder(tf.float32,name ='keep_prob')
lr = tf.Variable(0.001, dtype=tf.float32,name='lr')

# create simple NeuroNet
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1),name='W1')
b1 = tf.Variable(tf.zeros([500]) + 0.1,name='b1')
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1),name='W2')
b2 = tf.Variable(tf.zeros([300]) + 0.1,name='b2')
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1),name='W3')
b3 = tf.Variable(tf.zeros([10]) + 0.1,name='b3')
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3,name='prediction')

# cost function
with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
# train with gradient descent
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# initialize variables
init = tf.global_variables_initializer()

# find accuracy of trained model
correct_prediction = tf.equal(tf.argmax(y, 1),
                              tf.argmax(prediction, 1))  # convert a list of booleans into a single boolean value
with tf.variable_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# combine all summaries
merged = tf.summary.merge_all()


save_file = 'mode/mnist_model.ckpt'
saver = tf.train.Saver()

# 假如需要保存y，以便在预测时使用
tf.add_to_collection('pred_network', prediction)

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(51):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        writer.add_summary(summary, epoch)
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter" + str(epoch) + ", Testing Accuracy: " + str(test_acc) + ", Learning Rate: " + str(sess.run(lr)))
    # Save the model
    # 保存模型
    saver.save(sess, save_file)
    print('Trained Model Saved.')