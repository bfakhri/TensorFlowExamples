# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np

# Mnist Data
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Constants to eventually parameterise
## Base Dir to write logs to
BASE_LOGDIR = './logs/'
## Subdirectory for this experiment
RUN = '3'
## Learning Rate for Adam Optimizer
LEARN_RATE = 1e-4
## Number of images to push through the network at a time
BATCH_SIZE = 256 
## Number of Epochs to train for
MAX_EPOCHS = 10 
## How many training steps between outputs to screen and tensorboard
output_steps = 20
## Enable or disable GPU (0 disables GPU, 1 enables GPU)
SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 1})

# Define functions that create useful variables
def weight_variable(shape, name="W"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name="B"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 2D Convolution Func 
def conv2d(x, W, name='conv'):
    with tf.name_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max-Pooling Function - Pooling explained here: 
# http://ufldl.stanford.edu/tutorial/supervised/Pooling/
def max_pool_2x2(x, name='max_pool'):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Define a Convolutional Layer 
def conv_layer(x, fan_in, fan_out, name="convl"):
    with tf.name_scope(name):
        # Create Weight Variables
        W = weight_variable([5, 5, fan_in, fan_out], name="W")
        B = bias_variable([fan_out], name="B")
        # Convolve the input using the weights
        conv = conv2d(x, W)
        # Push input+bias through activation function
        activ = tf.nn.relu(conv + B)
        # Create histograms for visualization
        tf.summary.histogram("Weights", W)
        tf.summary.histogram("Biases", B)
        tf.summary.histogram("Activations", activ) 
        # MaxPool Output
        return max_pool_2x2(activ)


# Get Data
mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True)
MAX_TRAIN_STEPS = int(MAX_EPOCHS*mnist.train.num_examples/BATCH_SIZE)
SIZE_X = 28     # Number of pixels in x direction
SIZE_Y = 28     # Number of pixels in y direction
NUM_CLASSES = 10# Number of classes in the dataset 

# Begin Defining the Computational Graph
with tf.name_scope('MainGraph'):
    with tf.name_scope('Inputs'):
        # Placeholders for data and labels
        ## Mnist gives images as flat vectors, thus the size [None, SizeX*SizeY] 
        ## instead of the more intuitive [None, SizeX, SizeY]
        x = tf.placeholder(tf.float32, shape=[None, SIZE_X*SIZE_Y])

        ## Ground-Truth Labels, as 1-hot vectors
        y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

        ## Dropout probability. Dropout is similar to model averaging
        ## https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        keep_prob = tf.placeholder(tf.float32)

        # Reshape X to make it into a 2D image
        x_image = tf.reshape(x, [-1, SIZE_X, SIZE_Y, 1])
        tf.summary.image('sample_image', x_image, 3)
        color_og = tf.tile(x_image, [1, 1, 1, 2])
        #tf.summary.image('color_og', x_image, 3)

    # Convolution Layers
    conv1 = conv_layer(x_image, 1, 32, name='Conv1') 
    conv2 = conv_layer(conv1, 32, 64, name='Conv2') 
    for i in range(100):
        print(i, conv2.shape)
    
    # Here we implement spatial softmax
    conv2_exp = conv2
    features = tf.reshape(tf.transpose(conv2_exp, [0, 3, 1, 2]), [-1, conv2.shape[1]*conv2.shape[2]])
    conv2_ssm = tf.nn.softmax(features)
    # Reshape and transpose back to original format.

    conv2_ssm = tf.transpose(tf.reshape(conv2_ssm, [-1, 1, conv2.shape[1], conv2.shape[2]]), [0, 2, 3, 1])
    print(conv2_ssm.shape)
    #conv2_ssm =  tf.contrib.layers.spatial_softmax(conv2, name='Conv2-ssm') 
    #tf.summary.image('conv2_viz_smm1', tf.expand_dims(conv2_ssm[:,:,:,0], axis=3), 3)
    tf.summary.image('conv2_viz_smm1', conv2_ssm[:,:,:], 3)
    #tf.summary.image('conv2_viz_smm2', tf.expand_dims(conv2_ssm[:,:,:,0], axis=3), 3)

    # Fully Connected Layers
    with tf.name_scope('FC1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        
        h_pool2_flat = tf.reshape(conv2_ssm, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('FC2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        # Dropout Layer
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # FC Layer 2 - Output Layer
        y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    with tf.name_scope('Objective'):
        # Define the objective function
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
            train_accuracy, ssm_out = sess.run([accuracy, conv2_ssm], feed_dict={x: batch[0], y_true: batch[1], keep_prob: 1.0})
            print('Step: ' + str(cur_step) + '\t\tTrain Acc: ' + str(round(100*train_accuracy, 2)) + '%')
            print(sum(sum(sum(ssm_out))))
            # Calculate and write-out all summaries
            # Validate on batch from validation set
            val_batch = mnist.validation.next_batch(BATCH_SIZE)
            all_sums = sess.run(all_summaries, feed_dict={x: val_batch[0], y_true: val_batch[1], keep_prob: 1.0})
            writer.add_summary(all_sums, cur_step) 
        train_step.run(feed_dict={x: batch[0], y_true: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_true: mnist.test.labels, keep_prob: 1.0}))

