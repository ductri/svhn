import tensorflow as tf
from datetime import datetime

if __name__ == '__main__':
    with tf.Graph().as_default():
        x = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[0, 0, 0], logits=[[0.99, 0.05, 0.05], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
        with tf.Session().as_default() as sess:
            x_ =sess.run(x)
            print(x_)


        print('Done')