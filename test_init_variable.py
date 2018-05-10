import tensorflow as tf
from datetime import datetime

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.variable_scope(name_or_scope='test', reuse=tf.AUTO_REUSE):
            w = tf.get_variable(name='w', shape=[2, 2], dtype=tf.float32, initializer=tf.zeros_initializer())
            w = tf.add(w, 1)
            t = tf.get_variable(name='w', shape=[2, 2], dtype=tf.float32, initializer=tf.zeros_initializer())
        init_op = tf.global_variables_initializer()
        with tf.Session().as_default() as sess:
            sess.run(init_op)
            x_, y = sess.run([w, t])
            print(x_)
            print(y)
        print('Done')
