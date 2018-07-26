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
CIFAR dataset.

This module will download dataset from
https://www.cs.toronto.edu/~kriz/cifar.html and parse train/test set into
paddle reader creators.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
with 6000 images per class. There are 50000 training images and 10000 test
images.

The CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes
containing 600 images each. There are 500 training images and 100 testing
images per class.

"""

import pickle
import itertools
import numpy
import paddle.v2.dataset.common
import tarfile

__all__ = ['train10']

URL_PREFIX = 'https://www.cs.toronto.edu/~kriz/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'


def reader_creator(filename, sub_name, batch_size=None):
    def read_batch(batch):
        data = batch['data']
        labels = batch.get('labels', batch.get('fine_labels', None))
        assert labels is not None
        for sample, label in zip(data, labels):
            yield (sample / 255.0).astype(numpy.float32), int(label)

    def reader():
        with tarfile.open(filename, mode='r') as f:
            names = (each_item.name for each_item in f
                     if sub_name in each_item.name)

            batch_count = 0
            for name in names:
                batch = pickle.load(f.extractfile(name))
                for item in read_batch(batch):
                    if isinstance(batch_size, int) and batch_count > batch_size:
                        break
                    batch_count += 1
                    yield item

    return reader


def train10(batch_size=None):
    """
    CIFAR-10 training set creator.

    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].

    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        paddle.v2.dataset.common.download(CIFAR10_URL, 'cifar', CIFAR10_MD5),
        'data_batch',
        batch_size=batch_size)
