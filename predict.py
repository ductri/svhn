import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import svhn_input

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('FROZEN_GRAPH_FILE', './frozen_models/frozen_model.pb', 'path to frozen graph file')
tf.flags.DEFINE_string('IMAGE_FILE', './input.png', 'path to frozen graph file')
tf.flags.DEFINE_integer('SOURCE', 1, '1:train, 2:test, 3:file')


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    return graph


def load_image(file_name):
    plt.figure()
    jpgfile = Image.open(file_name)
    jpgfile = jpgfile.resize([32, 32])
    gray_image = np.array(jpgfile.convert('L')) * 1.0
    normalized_image = (gray_image - 128)*1.0 / 255
    plt.imshow(normalized_image)

    return normalized_image.reshape([1] + list(normalized_image.shape) + [1])


def load_train(index):
    index = int(index)
    normalized_image = svhn_input.train_set['images'][index]
    print('label', svhn_input.train_set['labels'][index])
    plt.figure()
    plt.imshow(normalized_image)
    return normalized_image.reshape([1] + list(normalized_image.shape) + [1])


def load_test(index):
    index = int(index)
    normalized_image = svhn_input.test_set['images'][index]
    print('label', svhn_input.test_set['labels'][index])
    plt.figure()
    plt.imshow(normalized_image)
    return normalized_image.reshape([1] + list(normalized_image.shape) + [1])


if __name__ == '__main__':


    # We use our "load_graph" function
    graph = load_graph(FLAGS.FROZEN_GRAPH_FILE)

    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
    #     print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # We access the input and output nodes
    image_input = graph.get_operation_by_name('input_train_placeholder').outputs[0]

    graph_outputs = [graph.get_operation_by_name('softmax_linear/softmax_linear{}'.format(i)).outputs[0] for i in range(5)]

    # We launch a Session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(graph=graph, config=config) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        if FLAGS.SOURCE == 1:
            svhn_input.bootstrap()
            images = load_train(FLAGS.IMAGE_FILE)
        elif FLAGS.SOURCE == 2:
            svhn_input.bootstrap()
            images = load_test(FLAGS.IMAGE_FILE)
        else:
            images = load_image(FLAGS.IMAGE_FILE)
        outputs = sess.run(graph_outputs, feed_dict={
            image_input: images
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        digits = []
        for i in range(5):
            digit = np.argmax(outputs[i], axis=1)[0]
            if digit == 10:
                digits.append('-')
            elif digit == 9:
                digits.append('0')
            else:
                digits.append(str(digit+1))
        print('*'*50)
        print()
        print()
        print('MY PREDICTION:')
        print('\t' + ''.join(digits))
        plt.show()