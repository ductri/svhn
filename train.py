import tensorflow as tf
import model

import svhn_input


IMAGE_SIZE = model.IMAGE_SIZE
NUM_DIGITS = model.NUM_DIGITS
BATCH_SIZE = model.BATCH_SIZE


def run_train(images_array, labels_array):
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
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)

            for i in range(100):
                sess.run(optimizer, feed_dict=
                    {input_placeholder: images_array,
                    list_labels[0]: labels_array[0],
                     list_labels[1]: labels_array[1],
                     list_labels[2]: labels_array[2],
                     list_labels[3]: labels_array[3],
                     list_labels[4]: labels_array[4],
                     }
                )


if __name__ == '__main__':
    svhn_input.bootstrap()
    inputs, lables = svhn_input.get_one_batch_input(BATCH_SIZE)
    inputs = inputs.reshape(list(inputs.shape) + [1])
    list_labels = [lables[:, 0], lables[:, 1], lables[:, 2], lables[:, 3], lables[:, 4]]
    optimizor = run_train(inputs, list_labels)
