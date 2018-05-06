import numpy as np
import tensorflow as tf
from datetime import datetime

import model

import svhn_input


IMAGE_SIZE = model.IMAGE_SIZE
NUM_DIGITS = model.NUM_DIGITS
BATCH_SIZE = model.BATCH_SIZE


def run_train():
    """

    :param images_array: shape=batch_size, image_height, image_width, 1
    :param labels_array: list of 5 items whose shape=batch_size
    :return:
    """
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        input_placeholder = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
        list_labels = [tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE])] * NUM_DIGITS

        list_logits = model.inference(images=input_placeholder)
        batch_loss = model.loss(list_logits, list_labels)
        optimizer = model.train(batch_loss, global_step)

        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log/{}'.format(str(datetime.now())))

        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            step = 0
            for images, labels in svhn_input.get_batch(batch_size=128, num_epoch=1):
                images = images.reshape(list(images.shape) + [1])
                labels = np.array(labels)
                _, summary = sess.run([optimizer, merged], feed_dict=
                    {input_placeholder: images,
                    list_labels[0]: labels[:, 0],
                     list_labels[1]: labels[:, 1],
                     list_labels[2]: labels[:, 2],
                     list_labels[3]: labels[:, 3],
                     list_labels[4]: labels[:, 4],
                     })

                train_writer.add_summary(summary, step)
                step += 1
                if step % 10 == 0:
                    print(step)
                    print(labels[0])


def main(argv=None):  # pylint: disable=unused-argument
    svhn_input.bootstrap()
    run_train()


if __name__ == '__main__':
    tf.app.run()