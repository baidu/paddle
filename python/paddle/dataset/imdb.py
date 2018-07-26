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
IMDB dataset.

This module downloads IMDB dataset from
http://ai.stanford.edu/%7Eamaas/data/sentiment/. This dataset contains a set
of 25,000 highly polar movie reviews for training, and 25,000 for testing.
Besides, this module also provides API for building dictionary.
"""

import paddle.dataset.common
import collections
import tarfile
import re
import string

__all__ = ['build_dict', 'train', 'test', 'convert']

URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'


def tokenize(pattern):
    """
    Read files that match the given pattern.  Tokenize and yield each file.
    """

    with tarfile.open(paddle.dataset.common.download(URL, 'imdb', MD5)) as tarf:
        # Note that we should use tarfile.next(), which does
        # sequential access of member files, other than
        # tarfile.extractfile, which does random access and might
        # destroy hard disks.
        tf = next(tarf)
        while tf != None:
            if bool(pattern.match(tf.name)):
                # newline and punctuations removal and ad-hoc tokenization.
                yield tarf.extractfile(tf).read().rstrip("\n\r").translate(
                    None, string.punctuation).lower().split()
            tf = next(tarf)


def build_dict(pattern, cutoff):
    """
    Build a word dictionary from the corpus. Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    word_freq = collections.defaultdict(int)
    for doc in tokenize(pattern):
        for word in doc:
            word_freq[word] += 1

    # Not sure if we should prune less-frequent words here.
    word_freq = [x for x in list(word_freq.items()) if x[1] > cutoff]

    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(list(zip(words, list(range(len(words))))))
    word_idx['<unk>'] = len(words)
    return word_idx


def reader_creator(pos_pattern, neg_pattern, word_idx):
    UNK = word_idx['<unk>']
    INS = []

    def load(pattern, out, label):
        for doc in tokenize(pattern):
            out.append(([word_idx.get(w, UNK) for w in doc], label))

    load(pos_pattern, INS, 0)
    load(neg_pattern, INS, 1)

    def reader():
        for doc, label in INS:
            yield doc, label

    return reader


def train(word_idx):
    """
    IMDB training set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    sequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator(
        re.compile("aclImdb/train/pos/.*\.txt$"),
        re.compile("aclImdb/train/neg/.*\.txt$"), word_idx)


def test(word_idx):
    """
    IMDB test set creator.

    It returns a reader creator, each sample in the reader is an zero-based ID
    sequence and label in [0, 1].

    :param word_idx: word dictionary
    :type word_idx: dict
    :return: Test reader creator
    :rtype: callable
    """
    return reader_creator(
        re.compile("aclImdb/test/pos/.*\.txt$"),
        re.compile("aclImdb/test/neg/.*\.txt$"), word_idx)


def word_dict():
    """
    Build a word dictionary from the corpus.

    :return: Word dictionary
    :rtype: dict
    """
    return build_dict(
        re.compile("aclImdb/((train)|(test))/((pos)|(neg))/.*\.txt$"), 150)


def fetch():
    paddle.dataset.common.download(URL, 'imdb', MD5)


def convert(path):
    """
    Converts dataset to recordio format
    """
    w = word_dict()
    paddle.dataset.common.convert(path, lambda: train(w), 1000, "imdb_train")
    paddle.dataset.common.convert(path, lambda: test(w), 1000, "imdb_test")
