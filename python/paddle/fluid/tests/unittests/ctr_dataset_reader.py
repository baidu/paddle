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

import os
import logging
import tarfile
import tempfile
import random
import warnings

import paddle
import paddle.distributed.fleet as fleet

logging.basicConfig()
logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)

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
        res.append(int(_[0]) % 10000)
    return res


class CtrReader(object):
    def __init__(self):
        pass

    def _reader_creator(self, filelist):
        def get_rand(low=0.0, high=1.0):
            return random.random()

        def reader():
            for file in filelist:
                with open(file, 'r') as f:
                    for line in f:
                        if get_rand() < 0.05:
                            fs = line.strip().split('\t')
                            dnn_input = load_dnn_input_record(fs[0])
                            lr_input = load_lr_input_record(fs[1])
                            click = [int(fs[2])]
                            yield [dnn_input] + [lr_input] + [click]

        return reader


class DatasetCtrReader(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def get_rand(low=0.0, high=1.0):
            return random.random()

        def iter():
            if get_rand() < 0.05:
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


def gen_fake_line(dnn_data_num=7,
                  dnn_data_range=1e5,
                  lr_data_num=5,
                  lr_data_range=1e5):
    line = ""

    # for deep data
    for index in range(dnn_data_num):
        data = str(random.randint(0, dnn_data_range - 1))
        if index < dnn_data_num - 1:
            data += " "
        line += data
    line += "\t"

    # for wide data
    for index in range(lr_data_num):
        data = str(random.randint(0, lr_data_range - 1)) + ":" + str(1)
        if index < lr_data_num - 1:
            data += " "
        line += data
    line += "\t"

    # for label
    line += str(random.randint(0, 1))
    line += "\n"
    return line


def prepare_fake_data(file_nums=6, file_lines=1000):
    """
    Create fake data with same type as avazu_ctr_data
    """
    file_dir = tempfile.mkdtemp()
    warnings.warn("Fake data write in {}".format(file_dir))
    for file_index in range(file_nums):
        with open(
                os.path.join(file_dir,
                             "ctr_train_data_part_{}".format(file_index)),
                'w+') as fin:
            file_str = ""
            for line_index in range(file_lines):
                file_str += gen_fake_line()
            fin.write(file_str)
            warnings.warn("Write done ctr_train_data_part_{}".format(
                file_index))

    file_list = [os.path.join(file_dir, x) for x in os.listdir(file_dir)]
    assert len(file_list) == file_nums

    return file_list


if __name__ == "__main__":
    pairwise_reader = DatasetCtrReader()
    pairwise_reader.run_from_stdin()
