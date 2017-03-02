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
A utility for fetching, reading sentiment data set.

http://ai.stanford.edu/%7Eamaas/data/sentiment
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

BASE_URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/%s.tar.gz'


DATASET_LABEL = 'label'
DATASET_SENTENCE = 'sentence'


class Categories(object):
    AclImdb = "aclImdb_v1"

    __md5__ = dict()

    __md5__[AclImdb] = '7c2ac02c03563afcf9b574c7e56c153a'

__all__ = ['fetch', 'Categories', 'preprocess', 'dataset']


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
        category = Categories.AclImdb

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'sentiment'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    fn = os.path.join(directory, '%s.tar.gz' % category)

    if os.path.exists(fn) and \
                    calculate_md5(fn) == Categories.__md5__[category]:
        # already download.
        return fn

    logger.info("Downloading binary sentiment classification dataset for %s category" % category)
    return download(BASE_URL % category, fn)
