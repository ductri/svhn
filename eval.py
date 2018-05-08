import tensorflow as tf
import numpy as np

import model
import train
import svhn_input


def eval_once(images, labels):
    """

    :param images: shape=(batch, height, width, channels)
    :param labels: list of 1-D tensor (batch_size) in the order of appearing in image, len(list_labels) == NUM_DIGITS, shape of each item is (batch_size), range from 0 to 10 (11 classes)
    :return:
    """

    with tf.Graph().as_default() as graph:
        checkpoint_file = tf.train.latest_checkpoint(train.CHECKPOINT_DIR)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            print('Restore model {}'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_train_placeholder = graph.get_operation_by_name("input_train_placeholder").outputs[0]
            list_labels = []
            list_softmax_linear = []
            for i in range(model.NUM_DIGITS):
                list_labels.append(graph.get_operation_by_name('labels_{}'.format(i)).outputs[0])
                list_softmax_linear.append(graph.get_operation_by_name('softmax_linear/softmax_linear{}'.format(i)).outputs[0])

            list_predictions = sess.run(list_softmax_linear, feed_dict={input_train_placeholder: images,
                                                                        list_labels[0]: labels[:, 0],
                                                                        list_labels[1]: labels[:, 1],
                                                                        list_labels[2]: labels[:, 2],
                                                                        list_labels[3]: labels[:, 3],
                                                                        list_labels[4]: labels[:, 4]
                                                                        })
            list_predictions = np.array(list_predictions)
            list_predictions = np.argmax(list_predictions, axis=2).transpose()
            counter = 0
            for i in range(list_predictions.shape[0]):
                if (list_predictions[i] == labels[i]).all():
                    counter += 1
            print('accuracy: {}/{}={:2f}'.format(counter, list_predictions.shape[0], counter*1.0 / list_predictions.shape[0]))

            n = 10
            first_10_labels = labels[:n, :]
            first_10_predictions = list_predictions[:n]

            print('first_10_labels')
            print(first_10_labels)
            print('first_10_predictions')
            print(first_10_predictions)


if __name__ == '__main__':
    svhn_input.bootstrap()
    images, labels = svhn_input.get_test(128)
    images = images.reshape(list(images.shape) + [1])
    eval_once(images, labels)
