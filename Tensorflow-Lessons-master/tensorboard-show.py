import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load dataset
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


#define batch size
batch_size = 100
#calculate number of batches
n_batch = mnist.train.num_examples // batch_size

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#name scope
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None,784],name='x-input')
    y = tf.placeholder(tf.float32, [None,10],name='y-input')

with tf.name_scope('layer'):
    #create simple NeuroNet
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)


with tf.name_scope('loss'):
    #quadratic cost function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    #train with gradient descent
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# initialize variables
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    # find accuracy of trained model
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction,
                                                                 1))  # convert a list of booleans into a single boolean of accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# combine all summaries
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(101):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})


        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ", Testing Accuracy: " + str(acc))