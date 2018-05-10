import numpy as np
import tensorflow as tf
from datetime import datetime

import model

import svhn_input


IMAGE_SIZE = model.IMAGE_SIZE
NUM_DIGITS = model.NUM_DIGITS
BATCH_SIZE = model.BATCH_SIZE
CHECKPOINT_DIR = './models/'
PREFIX = 'ver1'


def run_train():
    """

    :param images_array: shape=batch_size, image_height, image_width, 1
    :param labels_array: list of 5 items whose shape=batch_size
    :return:
    """
    with tf.Graph().as_default() as graph:
        global_step = tf.train.get_or_create_global_step()

        input_train_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input_train_placeholder')

        list_labels = []
        for i in range(NUM_DIGITS):
            list_labels.append(tf.placeholder(dtype=tf.int32, shape=[None], name='labels_{}'.format(i)))

        list_logits = model.inference(images=input_train_placeholder)

        batch_loss = model.loss(list_logits, list_labels)
        batch_loss_summary = tf.summary.scalar('total_loss', batch_loss)

        optimizer = model.train(batch_loss, global_step)

        # Accuracy
        list_top_k_op = []
        for i in range(NUM_DIGITS):
            top_k_op = tf.nn.in_top_k(list_logits[i], list_labels[i], 1)
            list_top_k_op.append(tf.reduce_mean(tf.cast(top_k_op, dtype=tf.int8)))
        batch_accuracy = tf.reduce_mean(list_top_k_op)
        batch_accuracy_summary = tf.summary.scalar('batch_accuracy', batch_accuracy)

        init = tf.global_variables_initializer()
        now = str(datetime.now())
        train_writer = tf.summary.FileWriter('log/train_{}'.format(now), graph=graph)
        test_writer = tf.summary.FileWriter('log/test_{}'.format(now), graph=graph)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.Session(config=config) as sess:
            # Run the initializer
            sess.run(init)
            step = 0
            test_images, test_labels = svhn_input.get_test(size=10000)
            test_images = test_images.reshape(list(test_images.shape) + [1])
            for images, labels in svhn_input.get_batch(batch_size=BATCH_SIZE, num_epoch=500):
                images = images.reshape(list(images.shape) + [1])
                labels = np.array(labels, dtype=int)

                sess.run(optimizer, feed_dict=
                    {input_train_placeholder: images,
                     list_labels[0]: labels[:, 0],
                     list_labels[1]: labels[:, 1],
                     list_labels[2]: labels[:, 2],
                     list_labels[3]: labels[:, 3],
                     list_labels[4]: labels[:, 4]
                     })

                if step%10 == 0:
                    summary1, summary2 = sess.run([batch_loss_summary, batch_accuracy_summary], feed_dict=
                    {input_train_placeholder: images,
                     list_labels[0]: labels[:, 0],
                     list_labels[1]: labels[:, 1],
                     list_labels[2]: labels[:, 2],
                     list_labels[3]: labels[:, 3],
                     list_labels[4]: labels[:, 4]
                     })
                    train_writer.add_summary(summary1, step)
                    train_writer.add_summary(summary2, step)

                    summary1, summary2 = sess.run([batch_loss_summary, batch_accuracy_summary], feed_dict=
                    {input_train_placeholder: test_images,
                     list_labels[0]: test_labels[:, 0],
                     list_labels[1]: test_labels[:, 1],
                     list_labels[2]: test_labels[:, 2],
                     list_labels[3]: test_labels[:, 3],
                     list_labels[4]: test_labels[:, 4]
                     })
                    test_writer.add_summary(summary1, step)
                    test_writer.add_summary(summary2, step)

                    path = saver.save(sess, save_path=CHECKPOINT_DIR+PREFIX, global_step=step)
                    print('Saved model at {}'.format(path))

                step += 1

                if step % 100 == 0:
                    print('-'*50)
                    print('step', step)

                    actual_labels, predicted_logits, loss = sess.run([list_labels[0], list_logits[0], batch_loss],
                                                                     feed_dict={input_train_placeholder: images,
                     list_labels[0]: labels[:, 0],
                     list_labels[1]: labels[:, 1],
                     list_labels[2]: labels[:, 2],
                     list_labels[3]: labels[:, 3],
                     list_labels[4]: labels[:, 4]
                     })
                    print('first digits of first 10 samples', actual_labels[:10])
                    print('predict first digits of first 10 samples', np.argmax(predicted_logits[:10], axis=1))
                    print('predict first digits of first 10 samples', predicted_logits[:5])
                    print('images', images[:10, 0, 0, 0])
                    print()


def main(argv=None):  # pylint: disable=unused-argument
    svhn_input.bootstrap()
    run_train()


if __name__ == '__main__':
    tf.app.run()