# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np

# Mnist Data
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Constants to eventually parameterise
BASE_LOGDIR = './logs/'
RUN = '1'
LEARN_RATE = 1e-4
BATCH_SIZE = 512
MAX_TRAIN_STEPS = 1000 
output_steps = 20
# Enable or disable GPU
SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 1})

# Define variable functions
def weight_variable(shape, name="W"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="B"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# Get Data
mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True)
# Data Params
SIZE_X = 28
SIZE_Y = 28
NUM_CLASSES = 10 

# LSTM Params
CELL_SIZE = 256     # Number of weights in the cell
NUM_CELLS = SIZE_X  # Number of cells in series

with tf.name_scope('MainGraph'):
    with tf.name_scope('Inputs'):
        # Placeholders for data and labels
        x = tf.placeholder(tf.float32, shape=[None, SIZE_X*SIZE_Y])
        y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
        # Dropout Placeholder (probability of dropping)
        keep_prob = tf.placeholder(tf.float32)

        # Reshape X to make it into a 2D image
        x_image = tf.reshape(x, [-1, SIZE_X, SIZE_Y, 1])
        tf.summary.image('sample_image', x_image, 3)

        x_sq = tf.squeeze(x_image, axis=3)

    with tf.name_scope('LSTM'):
        cell = tf.contrib.rnn.LSTMCell(CELL_SIZE, state_is_tuple=True)
        x_sq = tf.transpose(x_sq, [1,0,2])
        x_sq = tf.reshape(x_sq, [-1, SIZE_X])
        x_split = tf.split(x_sq, SIZE_Y, axis=0)   # Create a tuple of vectors
        #c_initial = tf.zeros((BATCH_SIZE, CELL_SIZE), dtype=np.float32)
        #h_initial = tf.zeros((BATCH_SIZE, CELL_SIZE), dtype=np.float32)
        #initial_state = tf.nn.rnn_cell.LSTMStateTuple(c_initial, h_initial)
        #outputs, states = tf.nn.static_rnn(cell, x_split, initial_state=initial_state)
        outputs, states = tf.nn.static_rnn(cell, x_split, dtype=tf.float32)

    with tf.name_scope('FC'):
        W_fc1 = weight_variable([CELL_SIZE, NUM_CLASSES])
        b_fc1 = bias_variable([NUM_CLASSES])
        
        # Output Layer
        y_pred = tf.matmul(outputs[-1], W_fc1) + b_fc1


    with tf.name_scope('Objective'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('Metrics'):
        # Define test metrics
        correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Create summary for accuracy
        tf.summary.scalar('accuracy', accuracy)

# Define the training step
train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(cross_entropy)

# Create the session
sess = tf.Session(config=SESS_CONFIG)

# Init all weights
sess.run(tf.global_variables_initializer())

# Merge Summaries and Create Summary Writer for TB
all_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(BASE_LOGDIR + RUN)
writer.add_graph(sess.graph) 

# Train 
with sess.as_default():
    for cur_step in range(MAX_TRAIN_STEPS):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if cur_step % output_steps == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1.0})
            print('Step: ' + str(cur_step) + '\t\tTrain Acc: ' + str(round(100*train_accuracy, 2)) + '%')
            # Calculate and write-out all summaries
            all_sums = sess.run(all_summaries, feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1.0})
            writer.add_summary(all_sums, cur_step) 
        train_step.run(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1.0}))

