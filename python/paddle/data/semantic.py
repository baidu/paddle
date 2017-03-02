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
A utility for fetching, reading semantic data set.

http://www.cs.upc.edu/~srlconll
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

BASE_URL = 'http://www.cs.upc.edu/~srlconll/%s.tar.gz'

DATASET_LABEL = 'label'
DATASET_SENTENCE = 'sentence'


class Categories(object):
    Conll05test = "conll05st-tests"

    __md5__ = dict()

    __md5__[Conll05test] = '387719152ae52d60422c016e92a742fc'


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
        category = Categories.Conll05test

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'semantic'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    fn = os.path.join(directory, '%s.json.gz' % category)

    if os.path.exists(fn) and \
                    calculate_md5(fn) == Categories.__md5__[category]:
        # already download.
        return fn

    logger.info("Downloading amazon review dataset for %s category" % category)
    return download(BASE_URL % category, fn)


