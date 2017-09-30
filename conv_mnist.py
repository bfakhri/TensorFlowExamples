# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
# Mnist Data
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Define variable functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# Define conv and pool functions
def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Get Data
mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True)

SIZE_X = 28
SIZE_Y = 28
NUM_CLASSES = 10 
output_steps = 100

# Placeholders for data and labels
x = tf.placeholder(tf.float32, shape=[None, SIZE_X*SIZE_Y])
y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

# Weights 
W = tf.Variable(tf.zeros([SIZE_X*SIZE_Y,NUM_CLASSES]))
b = tf.Variable(tf.zeros([NUM_CLASSES]))

# Create the session
sess = tf.Session()

# Init all weights
sess.run(tf.global_variables_initializer())

# Network Prediction
y_pred = tf.matmul(x,W) + b

# Objective Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# Define the training step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# Train 
for step in range(1000):
    if(step%output_steps == 0):
        print("Training Step: ", str(step))
    batch = mnist.train.next_batch(100)
    sess.run([train_step], feed_dict={x: batch[0], y_true: batch[1]})

correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_eval = sess.run([accuracy], feed_dict={x: mnist.test.images, y_true: mnist.test.labels})

print("Model Accuracy: " + str(accuracy_eval[0]*100) + '%')

