# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MNIST dataset.

This module will download dataset from http://yann.lecun.com/exdb/mnist/ and
parse training set and test set into paddle reader creators.
"""
import paddle.dataset.common
import subprocess
import numpy
import platform
__all__ = ['train', 'test', 'convert']

URL_PREFIX = 'http://yann.lecun.com/exdb/mnist/'
TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
TEST_IMAGE_MD5 = '9fb629c4189551a2d022fa330f9573f3'
TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
TEST_LABEL_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'
TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'


def reader_creator(image_filename, label_filename, buffer_size):
    def reader():
        if platform.system() == 'Darwin':
            zcat_cmd = 'gzcat'
        elif platform.system() == 'Linux':
            zcat_cmd = 'zcat'
        else:
            raise NotImplementedError()

        # According to http://stackoverflow.com/a/38061619/724872, we
        # cannot use standard package gzip here.
        m = subprocess.Popen([zcat_cmd, image_filename], stdout=subprocess.PIPE)
        m.stdout.read(16)  # skip some magic bytes

        l = subprocess.Popen([zcat_cmd, label_filename], stdout=subprocess.PIPE)
        l.stdout.read(8)  # skip some magic bytes

        try:  # reader could be break.
            while True:
                labels = numpy.fromfile(
                    l.stdout, 'ubyte', count=buffer_size).astype("int")

                if labels.size != buffer_size:
                    break  # numpy.fromfile returns empty slice after EOF.

                images = numpy.fromfile(
                    m.stdout, 'ubyte', count=buffer_size * 28 * 28).reshape(
                        (buffer_size, 28 * 28)).astype('float32')

                images = images / 255.0 * 2.0 - 1.0

                for i in range(buffer_size):
                    yield images[i, :], int(labels[i])
        finally:
            try:
                m.terminate()
            except:
                pass
            try:
                l.terminate()
            except:
                pass

    return reader


def train():
    """
    MNIST training set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(TRAIN_IMAGE_URL, 'mnist',
                                       TRAIN_IMAGE_MD5),
        paddle.dataset.common.download(TRAIN_LABEL_URL, 'mnist',
                                       TRAIN_LABEL_MD5), 100)


def test():
    """
    MNIST test set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator(
        paddle.dataset.common.download(TEST_IMAGE_URL, 'mnist', TEST_IMAGE_MD5),
        paddle.dataset.common.download(TEST_LABEL_URL, 'mnist', TEST_LABEL_MD5),
        100)


def fetch():
    paddle.dataset.common.download(TRAIN_IMAGE_URL, 'mnist', TRAIN_IMAGE_MD5)
    paddle.dataset.common.download(TRAIN_LABEL_URL, 'mnist', TRAIN_LABEL_MD5)
    paddle.dataset.common.download(TEST_IMAGE_URL, 'mnist', TEST_IMAGE_MD5)
    paddle.dataset.common.download(TEST_LABEL_URL, 'mnist', TEST_LABEL_MD5)


def convert(path):
    """
    Converts dataset to recordio format
    """
    paddle.dataset.common.convert(path, train(), 1000, "minist_train")
    paddle.dataset.common.convert(path, test(), 1000, "minist_test")
