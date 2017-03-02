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
A utility for fetching, reading CIFAR-10 dataset.

https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import cPickle
from http_download import download
from logger import logger
import hashlib
import tarfile
import numpy

BASE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-%s-python.tar.gz'
DATA = "cifar-10-batches-py"

class Categories(object):
    Ten = '10'
    Hundred = '100'

    __md5__ = dict()
    __md5__[Ten] = 'c58f30108f718f92721af3b95e74349a'
    __md5__[Hundred] = 'eb9058c3a382ffc7106e4002c42a8d85'

__all__ = ['fetch', 'Categories', 'train_data', 'test_data']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(category=None, directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch for training api.

    Args:

    Returns:
        path to untar file.
    """
    if category is None:
        category = Categories.Ten
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'cifar'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    cn = 'cifar' + category
    fn = os.path.join(directory, '%s.tar.gz' % cn)

    if os.path.exists(fn) and calculate_md5(fn) == \
            Categories.__md5__[category]:
        return fn

    logger.info("Downloading cifar dataset for %s category" % cn)
    return download(BASE_URL % category,
                    os.path.join(directory, '%s.tar.gz' % cn))


def untar(category=None, directory=None):
    """

    :param category:
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'cifar'))
    raw_file_fn = fetch(category, directory)
    #raw_file_fn = os.path.join(directory, 'cifar10.tar.gz')
    tar = tarfile.open(raw_file_fn, "r:gz")
    names = tar.getnames()
    for file in names:
        tar.extract(file, directory)
    tar.close()


def create_mean(dataset, directory=None):
    """

    :param dataset:
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'cifar'))

    if not os.path.isfile("mean.meta"):
        mean = numpy.zeros(3 * 32 * 3)
        num = 0
        for f in dataset:
            batch = numpy.load(f)
            mean += batch['data'].sum(0)
            num += len(batch['data'])
        mean /= num
        print mean.size
        data = {"mean": mean, "size": mean.size}
        cPickle.dump(
            data, open("mean.meta", 'w'), protocol=cPickle.HIGHEST_PROTOCOL)


def train_data(directory=None):
    """
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(os.path.join('~', 'paddle_data', 'cifar'))

    untar()
    datatset = [DATA + "/data_batch_%d" % (i + 1) for i in xrange(0, 5)]
    for f in datatset:
        train_set = os.path.join(directory, f)
        fo = open(train_set, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        print dict


def test_data(directory=None):
    """
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(os.path.join('~', 'paddle_data', 'cifar'))
    untar()
    test_set = os.path.join(directory, DATA + "/test_batch")
    fo = open(test_set, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    print dict


if __name__ == '__main__':
    train_data()
    #test_data()