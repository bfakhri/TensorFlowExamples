# Inspired by the TF Tutorial: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
import numpy as np

# Mnist Data
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Constants to eventually parameterise
BASE_LOGDIR = './logs/'
RUN = '12'
LEARN_RATE = 1e-5
BATCH_SIZE = 1024 
MAX_EPOCHS = 1000 
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


# Define conv and pool functions
def conv2d(x, W, name='conv'):
    with tf.name_scope(name):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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
SIZE_X = 28
SIZE_Y = 28
NUM_CLASSES = 10 

with tf.name_scope('MainGraph'):
    with tf.name_scope('Inputs'):
        # Placeholders for data and labels
        x = tf.placeholder(tf.float32, shape=[None, SIZE_X*SIZE_Y])
        y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

        # Reshape X to make it into a 2D image
        x_image = tf.reshape(x, [-1, SIZE_X, SIZE_Y, 1])
        tf.summary.image('original_image', x_image, 3)


    # FC Encoder Layers
    with tf.name_scope('encoder_FC1'):
        W_fc1 = weight_variable([SIZE_X*SIZE_Y, 1024])
        b_fc1 = bias_variable([1024])
        
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    with tf.name_scope('encoder_FC2'):
        W_fc2 = weight_variable([1024, 512])
        b_fc2 = bias_variable([512])
        # FC Layer 2 - Output Layer
        latent = tf.matmul(h_fc1, W_fc2) + b_fc2
    with tf.name_scope('latent_space'):
        # Split latent variable into mean and std devs
        latent_mu, latent_sigma = tf.split(latent, [256, 256], 1)
        # Generate the noise component epsilon as a standard normal RV
        epsilon = tf.random_normal(tf.shape(latent_mu), 0, 1, dtype=tf.float32)
        z = latent_mu + latent_sigma*epsilon

    with tf.name_scope('decoder_FC1'):
        W_fc_up1 = weight_variable([256, 1024])
        b_fc_up1 = bias_variable([1024])
        # FC Layer 2 - Output Layer
        h_fc_up1 = tf.matmul(z, W_fc_up1) + b_fc_up1

    with tf.name_scope('decoder_FC2'):
        W_fc_up2 = weight_variable([1024, SIZE_X*SIZE_Y])
        b_fc_up2 = bias_variable([SIZE_X*SIZE_Y])
        # FC Layer 2 - Output Layer
        gen_vec = tf.matmul(h_fc_up1, W_fc_up2) + b_fc_up2
        gen_img = tf.reshape(gen_vec, [-1, SIZE_X, SIZE_Y, 1])
        tf.summary.image('Generated_Image', gen_img, 3)

    with tf.name_scope('Objective'):
        # Generate KL-Divergence Loss
        #nas_latent_sigma = tf.square(tf.nn.l2_normalize(latent_sigma, dim=0, epsilon=1e-12))
        nas_latent_sigma = tf.square(latent_sigma)
        #batch_kl_div = tf.log(1/nas_latent_sigma+1e-9) + (nas_latent_sigma*nas_latent_sigma + latent_mu*latent_mu)/2.0 - 0.5
        batch_kl_div = 1.0 + tf.log(tf.square(latent_sigma)+1e-6) - tf.square(latent_mu) - tf.square(latent_sigma) 

        kl_div = -0.5*tf.reduce_sum(batch_kl_div)
        tf.summary.scalar('KL-Divergence', kl_div)
        generation_loss = tf.losses.mean_squared_error(gen_vec, x)
        tf.summary.scalar('generation_loss', generation_loss)
        #total_loss = kl_div+2000*mse

	#generation_loss = -tf.reduce_sum(x * tf.log(1e-6 + gen_vec) + (1.0-x) * tf.log(1e-6 + 1.0 - gen_vec),1)
        #tf.summary.scalar('Generation_Loss', generation_loss)

	total_loss = tf.reduce_mean(kl_div + generation_loss)

        tf.summary.scalar('Total_Loss', total_loss)


# Define the training step
train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(total_loss)

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
            train_kl_div = sess.run(kl_div, feed_dict={x: batch[0], y_true: batch[1]})
            #train_kl_div = sess.run(n_latent_sigma, feed_dict={x: batch[0], y_true: batch[1]})
            print('Step: ' + str(cur_step) + '\t\tTrain kld: ' + str(train_kl_div))
            # Calculate and write-out all summaries
            # Validate on batch from validation set
            val_batch = mnist.validation.next_batch(BATCH_SIZE)
            all_sums = sess.run(all_summaries, feed_dict={x: val_batch[0], y_true: val_batch[1]})
            writer.add_summary(all_sums, cur_step) 
        train_step.run(feed_dict={x: batch[0], y_true: batch[1]})


