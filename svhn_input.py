import os
# import urllib
import tarfile
import h5py
import numpy
from PIL import Image
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 32
NUM_CLASSES = 11
NUM_DIGITS = 5

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


train_set, test_set = None, None
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('ALL_DATASET_PATH', '/root/code/all_dataset', 'path to all_dataset')


def load_data():
    """
    Loads the SVHN dataset
    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.
    :return:
    """

    ALL_DATASET_PATH = FLAGS.ALL_DATASET_PATH

    SVHN_DATASET_PATH = os.path.join(ALL_DATASET_PATH, 'svhn')
    # Download the SVHN dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(SVHN_DATASET_PATH, dataset)

        if not os.path.isfile(new_path):
            origin = (
                'http://ufldl.stanford.edu/housenumbers/' + dataset
            )
            print('Downloading data from %s' % origin)
            # urllib.request.urlretrieve(origin, new_path)
        else:
            print('Files {} are already downloaded'.format(dataset))
        return new_path

    train_dataset = check_dataset('train.tar.gz')
    test_dataset = check_dataset('test.tar.gz')

    def format_data(dataset):

        my_tar = tarfile.open(os.path.join(SVHN_DATASET_PATH, dataset), 'r:gz')

        data_file_split = os.path.splitext(dataset)[0]
        data_type = os.path.splitext(data_file_split)[0]

        def check_extracted_file(folder_name):
            new_path = os.path.join(SVHN_DATASET_PATH, folder_name)
            if not os.path.exists(new_path):
                my_tar.extractall(os.path.join(SVHN_DATASET_PATH))
                process_data()

        def process_data():
            print('... processing data (should only occur when downloading for the first time)')
            # Access label information in digitStruct.mat
            new_path = os.path.join(SVHN_DATASET_PATH, data_type, 'digitStruct.mat')
            f = h5py.File(new_path, 'r')

            digitStructName = f['digitStruct']['name']
            digitStructBbox = f['digitStruct']['bbox']

            def getName(n):
                return ''.join([chr(c[0]) for c in f[digitStructName[n][0]].value])

            def bboxHelper(attr):
                if (len(attr) > 1):
                    attr = [f[attr.value[j].item()].value[0][0] for j in range(len(attr))]
                else:
                    attr = [attr.value[0][0]]
                return attr

            def getBbox(n):
                bbox = {}
                bb = digitStructBbox[n].item()
                # bbox = bboxHelper(f[bb]["label"])
                bbox['height'] = bboxHelper(f[bb]["height"])
                bbox['label'] = bboxHelper(f[bb]["label"])
                bbox['left'] = bboxHelper(f[bb]["left"])
                bbox['top'] = bboxHelper(f[bb]["top"])
                bbox['width'] = bboxHelper(f[bb]["width"])
                return bbox

            def getDigitStructure(n):
                s = getBbox(n)
                s['name'] = getName(n)
                return s

            # Process labels
            print('... creating image box bound dict for %s data' % data_type)
            image_dict = {}
            for i in range(len(digitStructName)):
                image_dict[getName(i)] = getBbox(i)
                if (i%1000 == 0):
                    print('     image dict processing: %i/%i complete' %(i,len(digitStructName)))
            print('... dict processing complete')

            # Process the data
            print('... processing image data and labels')

            names = []
            for item in os.listdir(os.path.join(SVHN_DATASET_PATH, data_type)):
                if item.endswith('.png'):
                    names.append(item)

            y = []
            x = []
            for i in range(len(names)):
                path = os.path.join(SVHN_DATASET_PATH, data_type)
                if len(image_dict[names[i]]['label']) > 5:
                    continue
                y.append(image_dict[names[i]]['label'])
                image = Image.open(path + '/' + names[i])
                left = int(min(image_dict[names[i]]['left']))
                upper = int(min(image_dict[names[i]]['top']))
                right = int(max(image_dict[names[i]]['left'])) + int(max(image_dict[names[i]]['width']))
                lower = int(max(image_dict[names[i]]['top'])) + int(max(image_dict[names[i]]['height']))
                image = image.crop(box = (left, upper, right, lower))
                image = image.resize([32, 32])
                image_array = numpy.array(image)
                x.append(image_array)
                if (i%1000 == 0):
                    print('     image processing: %i/%i complete' %(i,len(names)))
            print('... image processing complete')

            # Save data
            print('... pickling data')
            out = {}
            out['names'] = names
            out['labels'] = y
            out['images'] = x
            output_file = data_type + 'pkl.gz'
            out_path = os.path.join(SVHN_DATASET_PATH, output_file)
            p = gzip.open(out_path, 'wb')
            pickle.dump(out, p)
            p.close()

            my_tar.close()
            # clean up (delete test/train folders that were used to create the pickled data)
            # shutil.rmtree(os.path.join(SVHN_DATASET_PATH, data_type))

        check_extracted_file(data_type)

    # This check will run everytime load_data() is called

    if not os.path.isfile(os.path.join(SVHN_DATASET_PATH, 'trainpkl.gz')):
        format_data('train.tar.gz')

    f_train = gzip.open(os.path.join(SVHN_DATASET_PATH, 'trainpkl.gz'), 'rb')
    train_set = pickle.load(f_train)
    f_train.close()

    if not os.path.isfile(os.path.join(SVHN_DATASET_PATH, 'testpkl.gz')):
        format_data('test.tar.gz')

    f_test = gzip.open(os.path.join(SVHN_DATASET_PATH, 'testpkl.gz'), 'rb')
    test_set = pickle.load(f_test)

    train_set['images'] = train_set['images']
    train_set['labels'] = train_set['labels']
    print('Training size: {}'.format(len(train_set['images'])))
    print('Testing size: {}'.format(len(test_set['images'])))
    f_test.close()
    return train_set, test_set


EMPTY = 11


def normalize_labels(dataset):
    for i in range(len(dataset['labels'])):
        new_labels = dataset['labels'][i] + [EMPTY]*(5-len(dataset['labels'][i]))
        new_labels = [label-1 for label in new_labels]
        dataset['labels'][i] = new_labels
    dataset['labels'] = np.array(dataset['labels'])


def show_sample(images, labels):
    plt.figure(figsize=(12, 14))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title('-'.join([str(int(digit)) if digit != EMPTY else '-' for digit in labels[i]]))
        plt.axis('off')
    plt.show()


def standardize_value(dataset):
    for i in range(len(dataset['images'])):
        gray_image = np.array(Image.fromarray(dataset['images'][i]).convert('L')) * 1.0
        normalized_image = (gray_image - 128)/255
        dataset['images'][i] = normalized_image
    dataset['images'] = np.array(dataset['images'])


def bootstrap():
    global train_set
    global test_set
    train_set, test_set = load_data()
    normalize_labels(train_set)
    normalize_labels(test_set)
    standardize_value(train_set)
    standardize_value(test_set)


def get_one_batch_input(batch_size=128):
    index = np.random.randint(len(train_set['images']) - batch_size)
    return np.array(train_set['images'][index:index+batch_size]), np.array(train_set['labels'][index:index+batch_size])


def get_batch(batch_size=128, num_epoch=10):
    data_size = train_set['images'].shape[0]
    num_batch_per_epoch = int(data_size/batch_size) - 1
    for i in range(num_epoch):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_images = train_set['images'][shuffle_indices]
        shuffled_labels = train_set['labels'][shuffle_indices]

        for batch_idx in range(num_batch_per_epoch-1):
            yield (shuffled_images[batch_idx*batch_size:(batch_idx+1)*batch_size],
                   shuffled_labels[batch_idx*batch_size:(batch_idx+1)*batch_size])


def get_test(size=10000):
    return test_set['images'][:size], test_set['labels'][:size]


if __name__ == '__main__':
    bootstrap()
    index = 0
    for images, labels in get_batch(batch_size=128, num_epoch=1):
        print('shape', images.shape)
        print('images1', images)
        index += 1
        if index == 2:
            break

