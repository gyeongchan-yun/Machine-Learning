from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# == one-hot vector: represents 1 on only-one dimension for label ex) label = 3 out of 0-9, one-hot = [0,0,0,1,0,0,0,0,0,0]
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# == softmax: It's feasible to get probabilistic value. Each value is between 0 and 1, the sum of all is 1.

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))  # weight
b = tf.Variable(tf.zeros([10]))  # bias

# softmax model
y = tf.nn.softmax(tf.matmul(x, W) + b)



# == train == #
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 0.5 is learning rate

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(10):
    print("train epoch: {}\n".format(i))
    batch_xs, batch_ys = mnist.train.next_batch(100)  # stochastic training
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# == test == #
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))