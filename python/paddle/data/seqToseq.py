# /usr/bin/env python
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
A utility for fetching, reading sequence to sequence data set.

http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data
"""

import os
from http_download import download
from logger import logger
from base import BaseDataSet
import gzip
import json
import hashlib
import nltk
import collections
import h5py
import numpy

BASE_URL = 'http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/%s.tgz'

DATASET_LABEL = 'label'
DATASET_SENTENCE = 'sentence'


class Categories(object):
    BiTexts = "bitexts"
    DevTest = "dev+test"
    All = [BiTexts, DevTest]

    __md5__ = dict()

    __md5__[BiTexts] = '15861dbac4a52c8c75561d5027062d7d'
    __md5__[DevTest] = '7d7897317ddd8ba0ae5c5fa7248d3ff5'

__all__ = ['fetch', 'Categories', 'preprocess', 'dataset']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_data(category=None, directory=None):
    """
    Calculate each md5 value.
    :param category:
    :param directory:
    :return:
    """
    fn = os.path.join(directory, '%s.tgz' % category)
    if os.path.exists(fn) and \
                    calculate_md5(fn) == Categories.__md5__[category]:
        return fn
    logger.info("Downloading mnist handwritten digit dataset for %s category" % category)
    return download(BASE_URL % category, fn)


def fetch(category=None, directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch
    for training api.
    :param category:
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'seqToseq'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    if category is None:
        category = [category for category in Categories.All]
        fl = []  # download file list
        for index, line in range(len(category)):
            fl.append(fetch_data(line, directory))
        return fl
