#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import tarfile
import numpy as np
import six
from six.moves import cPickle as pickle

from paddle.io import Dataset
from .utils import _check_exists_and_download


__all__ = ['Cifar']

URL_PREFIX = 'https://dataset.bj.bcebos.com/cifar/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'

MODE_FLAG_MAP = {'train10': 'data_batch', 'test10': 'test_batch', 'train100': 'train', 'test100': 'test'}


class Cifar(Dataset):
    """
    Implement of Cifar dataset

    Args:
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train100', 'test100', 'train10' or 'test10' mode. Default 'train100'.
        download(bool): whether auto download cifar dataset if
            :attr:`data_file` unset. Default
            True

    Examples:

        .. code-block:: python

            from paddle.incubate.hapi.datasets import Cifar

            cifar = Cifar(mode='train10')

            for i in range(len(cifar)):
                sample = cifar[i]
                print(sample[0].shape, sample[1])

    """

    def __init__(self,
                 data_file=None,
                 mode='train100',
                 transform=None,
                 download=True):
        assert mode.lower() in ['train10', 'test10', 'train100', 'test100'], \
            "mode should be 'train10', 'test10', 'train100' or 'test100', but got {}".format(mode)
        self.mode = mode.lower()
        self.flag = MODE_FLAG_MAP[self.mode]

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file not set and auto download disabled"
            data_url = CIFAR10_URL if self.mode in ['train10', 'test10'] else CIFAR100_URL
            data_md5 = CIFAR10_MD5 if self.mode in ['train10', 'test10'] else CIFAR100_MD5
            self.data_file = _check_exists_and_download(
                data_file, data_url, data_md5, 'cifar', download)

        self.transform = transform

        # read dataset into memory
        self._load_anno()

    def _load_anno(self):
        self.data = None
        self.labels = []
        with tarfile.open(self.data_file, mode='r') as f:
            names = (each_item.name for each_item in f
                     if self.flag in each_item.name)
            for name in names:
                if six.PY2:
                    batch = pickle.load(f.extractfile(name))
                else:
                    batch = pickle.load(
                        f.extractfile(name), encoding='bytes')
                batch_data = batch[six.b('data')]
                batch_labels = batch.get(
                    six.b('labels'), batch.get(six.b('fine_labels'), None))
                assert batch_labels is not None
                self.data = np.concatenate(self.data, batch_data) if self.data is not None else batch_data
                self.labels.extend(batch_labels)


    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)
