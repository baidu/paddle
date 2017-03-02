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
A utility for fetching, reading amazon product review data set.

http://jmcauley.ucsd.edu/data/amazon/
"""

import os
from http_download import download
from logger import logger
import gzip
import json
import hashlib
import nltk
import collections
import h5py
import numpy
import random


BASE_URL = 'http://snap.stanford.edu/data/' \
           'amazon/productGraph/categoryFiles/reviews_%s_5.json.gz'

DATASET_LABEL = 'label'
DATASET_SENTENCE = 'sentence'

positive_threshold = 5
negative_threshold = 2



class Categories(object):
    Books = "Books"
    Electronics = "Electronics"
    MoviesAndTV = "Movies_and_TV"
    CDsAndVinyl = "CDs_and_Vinyl"
    ClothingShoesAndJewelry = "Clothing_Shoes_and_Jewelry"
    HomeAndKitchen = "Home_and_Kitchen"
    KindleStore = "Kindle_Store"
    SportsAndOutdoors = "Sports_and_Outdoors"
    CellPhonesAndAccessories = "Cell_Phones_and_Accessories"
    HealthAndPersonalCare = "Health_and_Personal_Care"
    ToysAndGames = "Toys_and_Games"
    VideoGames = "Video_Games"
    ToolsAndHomeImprovement = "Tools_and_Home_Improvement"
    Beauty = "Beauty"
    AppsForAndroid = "Apps_for_Android"
    OfficeProducts = "Office_Products"
    PetSupplies = "Pet_Supplies"
    Automotive = "Automotive"
    GroceryAndGourmetFood = "Grocery_and_Gourmet"
    PatioLawnAndGarden = "Patio_Lawn_and_Garden"
    Baby = "Baby"
    DigitalMusic = "Digital_Music"
    MusicalInstruments = "Musical_Instruments"
    AmazonInstantVideo = "Amazon_Instant_Video"

    __md5__ = dict()

    __md5__[AmazonInstantVideo] = '10812e43e99c345f63333d8ee10aef6a'
    __md5__[AppsForAndroid] = 'a7d1ae198b862eea6910fe45c842b0c6'
    __md5__[Automotive] = '757fdb1ab2c5e2fc0934047721082011'
    __md5__[Baby] = '7698a4179a1d8385e946ed9083490d22'
    __md5__[Beauty] = '5d2ccdcd86641efcfbae344317c10829'
    __md5__[Books] = 'bc1e2aa650fe51f978e9d3a7a4834bc6'
    __md5__[CDsAndVinyl] = '82bffdc956e76c32fa655b98eca9576b'
    __md5__[CellPhonesAndAccessories] = '903a19524d874970a2f0ae32a175a48f'
    __md5__[ClothingShoesAndJewelry] = 'b333fba48651ea2309288aeb51f8c6e4'
    __md5__[DigitalMusic] = '35e62f7a7475b53714f9b177d9dae3e7'
    __md5__[Electronics] = 'e4524af6c644cd044b1969bac7b62b2a'
    __md5__[GroceryAndGourmetFood] = 'd8720f98ea82c71fa5c1223f39b6e3d9'
    __md5__[HealthAndPersonalCare] = '352ea1f780a8629783220c7c9a9f7575'
    __md5__[HomeAndKitchen] = '90221797ccc4982f57e6a5652bea10fc'
    __md5__[KindleStore] = 'b608740c754287090925a1a186505353'
    __md5__[MoviesAndTV] = 'd3bb01cfcda2602c07bcdbf1c4222997'
    __md5__[MusicalInstruments] = '8035b6e3f9194844785b3f4cee296577'
    __md5__[OfficeProducts] = '1b7e64c707ecbdcdeca1efa09b716499'
    __md5__[PatioLawnAndGarden] = '4d2669abc5319d0f073ec3c3a85f18af'
    __md5__[PetSupplies] = '40568b32ca1536a4292e8410c5b9de12'
    __md5__[SportsAndOutdoors] = '1df6269552761c82aaec9667bf9a0b1d'
    __md5__[ToolsAndHomeImprovement] = '80bca79b84621d4848a88dcf37a1c34b'
    __md5__[ToysAndGames] = 'dbd07c142c47473c6ee22b535caee81f'
    __md5__[VideoGames] = '730612da2d6a93ed19f39a808b63993e'


__all__ = ['fetch', 'data', 'train_data', 'test_data']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(category=None, directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url, and return the download path to fetch
    for training api.
    :param category:
    :param directory:
    :return:
    """
    if category is None:
        category = Categories.Electronics

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'amazon'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    fn = os.path.join(directory, '%s.json.gz' % category)

    if os.path.exists(fn) and \
                    calculate_md5(fn) == Categories.__md5__[category]:
        # already download.
        return fn

    logger.info("Downloading amazon review dataset for %s category" % category)
    return download(BASE_URL % category, fn)


def preprocess(category=None, directory=None):
    """
    Download and preprocess amazon reviews data set. Save the preprocessed
    result to hdf5 file.

    In preprocess, it uses nltk to tokenize english sentence. It is slightly
    different from moses. But nltk is a pure python library, it could be
    integrated well with Paddle.

    :return: hdf5 file name.
    """
    if category is None:
        category = Categories.Electronics

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'amazon'))

    preprocess_fn = os.path.join(directory, '%s.hdf5' % category)
    raw_file_fn = fetch(category, directory)

    word_dict = collections.defaultdict(int)
    if not os.path.exists(preprocess_fn):  # already preprocessed
        with gzip.open(raw_file_fn, mode='r') as f:
            for sample_num, line in enumerate(f):
                txt = json.loads(line)['reviewText']
                try:  # automatically download nltk tokenizer data.
                    words = nltk.tokenize.word_tokenize(txt, 'english')
                except LookupError:
                    nltk.download('punkt')
                    words = nltk.tokenize.word_tokenize(txt, 'english')
                for each_word in words:
                    word_dict[each_word] += 1
            sample_num += 1

        word_dict_sorted = []
        for each in word_dict:
            word_dict_sorted.append((each, word_dict[each]))

        word_dict_sorted.sort(cmp=lambda a, b: a[1] > b[1])

        word_dict = dict()

        h5file = h5py.File(preprocess_fn, 'w')
        try:
            word_dict_h5 = h5file.create_dataset(
                'word_dict',
                shape=(len(word_dict_sorted), ),
                dtype=h5py.special_dtype(vlen=str))
            for i, each in enumerate(word_dict_sorted):
                word_dict_h5[i] = each[0]
                word_dict[each[0]] = i

            sentence = h5file.create_dataset(
                DATASET_SENTENCE,
                shape=(sample_num, ),
                dtype=h5py.special_dtype(vlen=numpy.int32))

            label = h5file.create_dataset(
                DATASET_LABEL, shape=(sample_num, 1), dtype=numpy.int8)

            with gzip.open(raw_file_fn, mode='r') as f:
                for i, line in enumerate(f):
                    obj = json.loads(line)
                    txt = obj['reviewText']
                    score = numpy.int8(obj['overall'])
                    words = nltk.tokenize.word_tokenize(txt, 'english')
                    words = numpy.array(
                        [word_dict[w] for w in words], dtype=numpy.int32)
                    sentence[i] = words
                    label[i] = score

        finally:
            h5file.close()
    return preprocess_fn


def data(batch_size, category=None, directory=None):
    """

    :param batch_size:
    :param category:
    :param directory:
    :return:
    """
    if category is None:
        category = Categories.Electronics

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'amazon'))

    fn = preprocess(category=category, directory=directory)
    datasets = h5py.File(fn, 'r')

    label = datasets[DATASET_LABEL]
    sentence = datasets[DATASET_SENTENCE]

    if label.shape[0] <= batch_size:
        lens = label.shape[0]
    else:
        lens = batch_size

    for index in range(lens):
        if label[index] >= positive_threshold:
            print (numpy.array(sentence[index]), label[index] >= positive_threshold)
        elif label[index] <= negative_threshold:
            print (numpy.array(sentence[index]), label[index] <= negative_threshold)


def test_data(batch_size, category=None, directory=None):
    """

    :param batch_size:
    :param category:
    :param directory:
    :return:
    """
    if category is None:
        category = Categories.Electronics

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'amazon'))

    fn = preprocess(category=category, directory=directory)
    datasets = h5py.File(fn, 'r')

    label = datasets[DATASET_LABEL]
    sentence = datasets[DATASET_SENTENCE]

    if label.shape[0] <= batch_size:
        lens = label.shape[0]
    else:
        lens = batch_size

    positive_idx = []
    negative_idx = []
    for i, lbl in enumerate(label):
        if label[i] >= positive_threshold:
            positive_idx.append(i)
        elif lbl <= negative_threshold:
            negative_idx.append(i)

    __test_set__ = positive_idx[:lens] + negative_idx[:lens]

    random.shuffle(__test_set__)

    for index in range(lens):
        print (numpy.array(sentence[index]), label[index] >= positive_threshold)


def train_data(batch_size, category=None, directory=None):
    """

    :param batch_size:
    :param category:
    :param directory:
    :return:
    """
    if category is None:
        category = Categories.Electronics

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'amazon'))

    fn = preprocess(category=category, directory=directory)
    datasets = h5py.File(fn, 'r')

    label = datasets[DATASET_LABEL]
    sentence = datasets[DATASET_SENTENCE]

    if label.shape[0] <= batch_size:
        lens = label.shape[0]
    else:
        lens = batch_size

    positive_idx = []
    negative_idx = []
    for i, lbl in enumerate(label):
        if label[i] >= positive_threshold:
            positive_idx.append(i)
        elif lbl <= negative_threshold:
            negative_idx.append(i)
    __train_set__ = positive_idx[lens:] + negative_idx[lens:]

    random.shuffle(__train_set__)

    for index in range(lens):
        print (numpy.array(sentence[index]), label[index] >= positive_threshold)


if __name__ == '__main__':
    data(10)

