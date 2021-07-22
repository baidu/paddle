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

import os
from paddle.incubate.hapi.text.glue import TSVDataset, GlueCoLA, GlueSST2, GlueMRPC, GlueSTSB, GlueQQP, GlueMNLI, GlueQNLI, GlueRTE, GlueWNLI
from paddle.incubate.hapi.text.data_utils import SimpleDataset
from paddle.dataset.common import DATA_HOME, md5file
import shutil
import unittest

DEFAULT_ROOT = os.path.join(DATA_HOME, 'glue')
SEGMENT_LIST = ['train', 'dev', 'test']


def test_md5(dataset_list):
    for i, segment in enumerate(SEGMENT_LIST):
        filename, data_hash, field_indices, num_discard_samples = dataset_list[
            i].SEGMENTS[segment]
        fullname = os.path.join(DEFAULT_ROOT, filename)
        assert data_hash == md5file(fullname)
    return fullname


class TestGlueCoLA(unittest.TestCase):
    def setUp(self):
        self.cola_train = GlueCoLA('train')
        self.cola_dev = GlueCoLA('dev')
        self.cola_test = GlueCoLA('test')
        self.datasets = [self.cola_train, self.cola_dev, self.cola_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.cola_train.get_labels() == ["0", "1"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueWNLI(unittest.TestCase):
    def setUp(self):
        self.wnli_train = GlueWNLI('train')
        self.wnli_dev = GlueWNLI('dev')
        self.wnli_test = GlueWNLI('test')
        self.datasets = [self.wnli_train, self.wnli_dev, self.wnli_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.wnli_train.get_labels() == ["0", "1"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueSST2(unittest.TestCase):
    def setUp(self):
        self.sst_train = GlueSST2('train')
        self.sst_dev = GlueSST2('dev')
        self.sst_test = GlueSST2('test')
        self.datasets = [self.sst_train, self.sst_dev, self.sst_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.sst_train.get_labels() == ["0", "1"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueMRPC(unittest.TestCase):
    def setUp(self):
        self.mrpc_train = GlueMRPC('train')
        self.mrpc_dev = GlueMRPC('dev')
        self.mrpc_test = GlueMRPC('test')
        self.datasets = [self.mrpc_train, self.mrpc_dev, self.mrpc_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.mrpc_train.get_labels() == ["0", "1"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueSTSB(unittest.TestCase):
    def setUp(self):
        self.stsb_train = GlueSTSB('train')
        self.stsb_dev = GlueSTSB('dev')
        self.stsb_test = GlueSTSB('test')
        self.datasets = [self.stsb_train, self.stsb_dev, self.stsb_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.stsb_train.get_labels() == None

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueQQP(unittest.TestCase):
    def setUp(self):
        self.qqp_train = GlueQQP('train')
        self.qqp_dev = GlueQQP('dev')
        self.qqp_test = GlueQQP('test')
        self.datasets = [self.qqp_train, self.qqp_dev, self.qqp_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.qqp_train.get_labels() == ["0", "1"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueMNLI(unittest.TestCase):
    def setUp(self):
        self.mnli_train = GlueMNLI('train')
        self.mnli_dev_matched = GlueMNLI('dev_matched')
        self.mnli_dev_mismatched = GlueMNLI('dev_mismatched')
        self.mnli_test_matched = GlueMNLI('test_matched')
        self.mnli_test_mismatched = GlueMNLI('test_mismatched')
        self.datasets = [
            self.mnli_train, self.mnli_dev_matched, self.mnli_dev_mismatched,
            self.mnli_test_matched, self.mnli_test_mismatched
        ]
        self.fullname = self.test_dataset()

    def test_dataset(self):
        segment_list = [
            'train', 'dev_matched', 'dev_mismatched', 'test_matched',
            'test_mismatched'
        ]
        for i, segment in enumerate(segment_list):
            filename, data_hash, field_indices, num_discard_samples = self.datasets[
                i].SEGMENTS[segment]
            fullname = os.path.join(DEFAULT_ROOT, filename)
            assert data_hash == md5file(fullname)
        return fullname

    def test_get_labels(self):
        assert self.mnli_train.get_labels(
        ) == ["contradiction", "entailment", "neutral"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueQNLI(unittest.TestCase):
    def setUp(self):
        self.qnli_train = GlueQNLI('train')
        self.qnli_dev = GlueQNLI('dev')
        self.qnli_test = GlueQNLI('test')
        self.datasets = [self.qnli_train, self.qnli_dev, self.qnli_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.qnli_train.get_labels() == ["entailment", "not_entailment"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestGlueRTE(unittest.TestCase):
    def setUp(self):
        self.rte_train = GlueRTE('train')
        self.rte_dev = GlueRTE('dev')
        self.rte_test = GlueRTE('test')
        self.datasets = [self.rte_train, self.rte_dev, self.rte_test]
        self.fullname = test_md5(self.datasets)

    def test_get_labels(self):
        assert self.rte_train.get_labels() == ["entailment", "not_entailment"]

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


class TestSimpleDataset(unittest.TestCase):
    def setUp(self):
        dataset = GlueCoLA('test')
        self.simple_dataset = SimpleDataset(dataset)

    def test_shuffle(self):
        shuffled_dataset = self.simple_dataset.shuffle()

    def test_sort(self):
        sorted_dataset = self.simple_dataset.sort(buffer_size=5)

    def test_filter(self):
        def predicate_func(each_data):
            return True if (len(each_data[0]) % 2) == 1 else False

        filtered_dataset = self.simple_dataset.filter(predicate_func)
        for each_data in filtered_dataset:
            assert len(each_data[0]) % 2 == 1

    def test_apply(self):
        def transform_func(each_data):
            each_data[0] = each_data[0][:500]
            return each_data

        applied_data = self.simple_dataset.apply(transform_func)
        for each_data in applied_data:
            assert len(each_data[0]) <= 500

    def test_shard(self):
        shared_data = self.simple_dataset.shard()

    def tearDown(self):
        shutil.rmtree(DEFAULT_ROOT)


if __name__ == '__main__':
    unittest.main()
