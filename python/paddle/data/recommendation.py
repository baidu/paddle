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
A utility for fetching, reading MovieLens dataset.

http://files.grouplens.org/datasets/movielens
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

BASE_URL = 'http://files.grouplens.org/datasets/movielens/%s.zip'


class Categories(object):
    M1m = "ml-1m"
    M10m = "ml-10m"
    M20m = "ml-20m"
    M100k = "ml-100k"
    MLatestSmall = "ml-latest-small"
    MLatest = "ml-latest"

    __md5__ = dict()

    __md5__[M1m] = 'c4d9eecfca2ab87c1945afe126590906'
    __md5__[M10m] = 'ce571fd55effeba0271552578f2648bd'
    __md5__[M20m] = 'cd245b17a1ae2cc31bb14903e1204af3'
    __md5__[M100k] = '0e33842e24a9c977be4e0107933c0723'
    __md5__[MLatestSmall] = 'be5b02baacd9e70dd97734ea0e19528a'
    __md5__[MLatest] = '0c827eaafc7e89c455986510827662bd'


__all__ = ['fetch', 'Categories', 'preprocess']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(category=None, directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch
    for training api.
    :param category:
    :param directory:
    :return:
    """
    if category is None:
        category = Categories.M1m

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'recommendation'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    fn = os.path.join(directory, '%s.zip' % category)

    if os.path.exists(fn) and \
                    calculate_md5(fn) == Categories.__md5__[category]:
        # already download.
        return fn

    logger.info("Downloading MovieLens dataset for %s category" % category)
    return download(BASE_URL % category, fn)
