import tensorflow as tf

import svhn_input
import my_summarizer


FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_integer('batch_size', 128)
# tf.flags.DEFINE_boolean('use_fp16', False)
USE_FP16 = False
BATCH_SIZE = 128


# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = svhn_input.IMAGE_SIZE
NUM_CLASSES = svhn_input.NUM_CLASSES
NUM_DIGITS = svhn_input.NUM_DIGITS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = svhn_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = svhn_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.


def inference(images):
    """

    :param images: shape=(batch, height, width, channels)
    :return: Logits
    """
    with tf.variable_scope('conv1') as scope:
        channels_out1 = 16
        if USE_FP16:
            kernel = tf.get_variable('weights', shape=[5, 5, 1, channels_out1], dtype=tf.float16)
        else:
            kernel = tf.get_variable('weights', shape=[5, 5, 1, channels_out1], dtype=tf.float32)
        stride1 = 1
        conv1 = tf.nn.conv2d(input=images, filter=kernel, strides=[1, stride1, stride1, 1], padding='SAME')

        bias1 = tf.get_variable(name='bias', shape=[channels_out1], initializer=tf.constant_initializer(0))
        pre_activation1 = tf.nn.bias_add(conv1, bias1)
        activation1 = tf.nn.relu(features=pre_activation1)  # shape=batch_size, height, width, channels_out1
        pool1 = tf.nn.max_pool(value=activation1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

        # TODO read more in paper
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm') # shape=batch_size, height, width, channels_out1
        my_summarizer.activation_summary(norm1)

    with tf.variable_scope('conv2') as scope:
        channels_out2 = 32
        if USE_FP16:
            kernel = tf.get_variable('weights', shape=[5, 5, channels_out1, channels_out2], dtype=tf.float16)
        else:
            kernel = tf.get_variable('weights', shape=[5, 5, channels_out1, channels_out2], dtype=tf.float32)
        stride2 = 1
        conv2 = tf.nn.conv2d(input=norm1, filter=kernel, strides=[1, stride2, stride2, 1], padding='SAME')
        bias2 = tf.get_variable(name='bias', shape=[channels_out2], initializer=tf.constant_initializer(0))
        pre_activation2 = tf.nn.bias_add(conv2, bias2)
        activation2 = tf.nn.relu(features=pre_activation2)  # shape=batch_size, height, width, channels_out2
        pool2 = tf.nn.max_pool(value=activation2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')  # TODO read more in paper
        my_summarizer.activation_summary(norm2)

        # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(norm2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        num_weights3 = 500
        if USE_FP16:
            weights = tf.get_variable('weights', shape=[dim, num_weights3], dtype=tf.float16)
        else:
            weights = tf.get_variable('weights', shape=[dim, num_weights3], dtype=tf.float32)
        bias3 = tf.get_variable(name='bias', shape=[num_weights3], initializer=tf.constant_initializer(0))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + bias3, name=scope.name)
        my_summarizer.activation_summary(local3)
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        softmax_linears = []
        for i in range(NUM_DIGITS):
            if USE_FP16:
                weights = tf.get_variable('weights_{}'.format(i), [num_weights3, NUM_CLASSES], dtype=tf.float16)
            else:
                weights = tf.get_variable('weights_{}'.format(i), [num_weights3, NUM_CLASSES], dtype=tf.float32)

            biases = tf.get_variable('bias_{}'.format(i), shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name + str(i))
            my_summarizer.activation_summary(softmax_linear)
            softmax_linears.append(softmax_linear)
    return softmax_linears


def loss(list_logits, list_labels):
    """

    :param list_logits: get from inference(). len(list_logits) == NUM_DIGITS, shape of each item is (batch_size, num_classes)
    :param list_labels: list of 1-D tensor (batch_size) in the order of appearing in image, len(list_labels) == NUM_DIGITS, shape of each item is (batch_size), range from 0 to 10 (11 classes)
    :return:
    """
    assert len(list_logits) == len(list_labels)
    assert len(list_labels) == NUM_DIGITS
    losses = []
    for i in range(len(list_logits)):
        fucking_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=list_labels[i], logits=list_logits[i])
        loss = tf.reduce_sum(fucking_losses)
        losses.append(loss)
        if i == 0:
            tf.summary.scalar('loss_0', loss)
    total_loss = tf.reduce_sum(losses)
    tf.summary.scalar('total_loss', total_loss)
    return total_loss


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(total_loss)
    return optimizer


