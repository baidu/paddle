# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import tarfile
import os

import paddle
import paddle.distributed.fleet as fleet
from paddle.fluid.log_helper import get_logger

logger = get_logger(
    "paddle", logging.INFO, fmt='%(asctime)s - %(levelname)s - %(message)s')

DATA_URL = "http://paddle-ctr-data.bj.bcebos.com/avazu_ctr_data.tgz"
DATA_MD5 = "c11df99fbd14e53cd4bfa6567344b26e"
"""
avazu_ctr_data/train.txt
avazu_ctr_data/infer.txt
avazu_ctr_data/test.txt
avazu_ctr_data/data.meta.txt
"""


def download_file():
    file_name = "avazu_ctr_data"
    path = paddle.dataset.common.download(DATA_URL, file_name, DATA_MD5)

    dir_name = os.path.dirname(path)
    text_file_dir_name = os.path.join(dir_name, file_name)

    if not os.path.exists(text_file_dir_name):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(dir_name)
    return text_file_dir_name


def load_dnn_input_record(sent):
    return list(map(int, sent.split()))


def load_lr_input_record(sent):
    res = []
    for _ in [x.split(':') for x in sent.split()]:
        res.append(int(_[0]))
    return res


class DatasetCtrReader(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def iter():
            fs = line.strip().split('\t')
            dnn_input = load_dnn_input_record(fs[0])
            lr_input = load_lr_input_record(fs[1])
            click = [int(fs[2])]
            yield ("dnn_data", dnn_input), \
                  ("lr_data", lr_input), \
                  ("click", click)

        return iter


def prepare_data():
    """
    load data meta info from path, return (dnn_input_dim, lr_input_dim)
    """
    file_dir_name = download_file()
    meta_file_path = os.path.join(file_dir_name, 'data.meta.txt')
    train_file_path = os.path.join(file_dir_name, 'train.txt')
    with open(meta_file_path, "r") as f:
        lines = f.readlines()
    err_info = "wrong meta format"
    assert len(lines) == 2, err_info
    assert 'dnn_input_dim:' in lines[0] and 'lr_input_dim:' in lines[
        1], err_info
    res = map(int, [_.split(':')[1] for _ in lines])
    res = list(res)
    dnn_input_dim = res[0]
    lr_input_dim = res[1]
    logger.info('dnn input dim: %d' % dnn_input_dim)
    logger.info('lr input dim: %d' % lr_input_dim)
    return dnn_input_dim, lr_input_dim, train_file_path


if __name__ == "__main__":
    pairwise_reader = DatasetCtrReader()
    pairwise_reader.run_from_stdin()
