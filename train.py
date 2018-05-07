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
        list_labels = []
        for i in range(NUM_DIGITS):
            list_labels.append(tf.placeholder(dtype=tf.int32, shape=[BATCH_SIZE]))

        list_logits = model.inference(images=input_placeholder)
        batch_loss = model.loss(list_logits, list_labels)
        optimizer = model.train(batch_loss, global_step)

        init = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('log/{}'.format(str(datetime.now())))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=config) as sess:
            # Run the initializer
            sess.run(init)
            step = 0
            for images, labels in svhn_input.get_batch(batch_size=BATCH_SIZE, num_epoch=100):
                images = images.reshape(list(images.shape) + [1])
                labels = np.array(labels, dtype=int)

                summary, _ = sess.run([merged, optimizer], feed_dict=
                    {input_placeholder: images,
                     list_labels[0]: labels[:, 0],
                     list_labels[1]: labels[:, 1],
                     list_labels[2]: labels[:, 2],
                     list_labels[3]: labels[:, 3],
                     list_labels[4]: labels[:, 4]
                     })

                train_writer.add_summary(summary, step)
                step += 1
                if step % 1 == 0:
                    print('-'*50)
                    print('step', step)

                    actual_labels, predicted_logits, loss = sess.run([list_labels[0], list_logits[0], batch_loss],
                                                                     feed_dict={input_placeholder: images,
                                                                                 list_labels[0]: labels[:, 0],
                                                                                 list_labels[1]: labels[:, 1],
                                                                                 list_labels[2]: labels[:, 2],
                                                                                 list_labels[3]: labels[:, 3],
                                                                                 list_labels[4]: labels[:, 4]})
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