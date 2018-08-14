import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from keras.datasets import cifar10, mnist, fashion_mnist
from keras.utils import to_categorical
import numpy as np
import random

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def load_cifar10() :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_mnist() :
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1) / 255.0
    test_data = np.expand_dims(test_data, axis=-1) / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def load_fashion() :
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    train_data = np.expand_dims(train_data, axis=-1) / 255.0
    test_data = np.expand_dims(test_data, axis=-1) / 255.0

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)


    return train_data, train_labels, test_data, test_labels

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch, img_size):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [img_size, img_size], 4)
    return batch