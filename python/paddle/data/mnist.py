#/usr/bin/env python
# -*- coding:utf-8 -*-

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
A utility for fetching, reading mnist handwritten digit dataset.

http://yann.lecun.com/exdb/mnist/
"""

import os
from http_download import download
from logger import logger
import gzip
import hashlib
import numpy
import struct


BASE_URL = 'http://yann.lecun.com/exdb/mnist/%s.gz'
FILE_NAME = {
    'train-images-idx3-ubyte': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte': 'ec29112dd5afa0611ce80d1b7f02629c'
}


__all__ = ['train_data', 'test_data', 'fetch']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch
    for training api.
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'mnist'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    fl = []
    for index in range(len(FILE_NAME.keys())):
        fn = os.path.join(directory, '%s.gz' % FILE_NAME.keys()[index])
        if os.path.exists(fn) and calculate_md5(fn) == FILE_NAME.keys()[0]:
            return fn
        logger.info("Downloading digital handwritten digit dataset for %s " % FILE_NAME.keys()[index])
        fl.append(download(BASE_URL % FILE_NAME.keys()[index], fn))

    return fl


def preprocess(directory=None):
    """
    :param category:
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(os.path.join('~', 'paddle_data', 'mnist'))

    raw_file_list = fetch(directory)
    print raw_file_list

    for cn in raw_file_list:
        sz = cn.split('.')[0]
        print sz
        g = gzip.GzipFile(fileobj=open(cn, 'rb'))
        open(sz, 'wb').write(g.read())


def data(filename, directory=None):
    """
    :param filename:
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(os.path.join('~', 'paddle_data', 'mnist'))

    image = '-images-idx3-ubyte'
    label = '-labels-idx1-ubyte'

    if filename is 'train':
        image_file = os.path.join(directory, filename + image)
        label_file = os.path.join(directory, filename + label)
    else:
        image_file = os.path.join(directory, 't10' + image)
        label_file = os.path.join(directory, 't10' + label)

    if os.path.exists(image_file) and os.path.exists(label_file):
        print "File is exists!"
    else:
        preprocess()

    print image_file
    print label_file

    with open(image_file, "rb") as f:
        num_magic, n, num_row, num_col = struct.unpack(">IIII", f.read(16))
        images = numpy.fromfile(f, 'ubyte', count=n * num_row * num_col).\
            reshape(n, num_row, num_col).astype('float32')
        images = images / 255.0 * 2.0 - 1.0

    with open(label_file, "rb") as fn:
        num_magic, num_label = struct.unpack(">II", fn.read(8))
        labels = numpy.fromfile(fn, 'ubyte', count=num_label).astype('int')

    return images, labels


def train_data(directory=None):
    """
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(os.path.join('~', 'paddle_data', 'mnist'))

    train_images, train_labels = data('train')
    print train_images, train_labels


def test_data(directory=None):
    """
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(os.path.join('~', 'paddle_data', 'mnist'))

    test_images, test_labels = data('test')
    print test_images, test_labels


if __name__ == '__main__':
    train_data()
    #test_data()