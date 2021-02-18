# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
import math
import random
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.io import Dataset, IterableDataset, TensorDataset, \
        ComposeDataset, ChainDataset, DataLoader, random_split, Subset, BufferedShuffleDataset, get_worker_info

# tests' settings
IMAGE_SIZE = 32
START = 0
END = 10
EXPECTED_RANGE = 10
BUFFER_SIZES = [5, 15]
WORKER_SETTING = [0, 1]


def shuffle_ds_init_fn(worker_id):
    random.seed(123)


def worker_init_fn(worker_id):
    worker_info = get_worker_info()

    dataset = worker_info.dataset
    start = dataset.start
    end = dataset.end
    num_per_worker = int(
        math.ceil((end - start) / float(worker_info.num_workers)))

    worker_id = worker_info.id
    dataset.start = start + worker_id * num_per_worker
    dataset.end = min(dataset.start + num_per_worker, end)


def get_dataloader(dataset, num_workers, worker_init_fn):
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=1,
        drop_last=True,
        worker_init_fn=worker_init_fn)


class RandomDataset(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return image, label


class RandomIterableDataset(IterableDataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __iter__(self):
        for i in range(self.sample_num):
            np.random.seed(i)
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            yield image, label


class TestTensorDataset(unittest.TestCase):
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            input_np = np.random.random([16, 3, 4]).astype('float32')
            input = paddle.to_tensor(input_np)
            label_np = np.random.random([16, 1]).astype('int32')
            label = paddle.to_tensor(label_np)

            dataset = TensorDataset([input, label])
            assert len(dataset) == 16
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True)

            for i, (input, label) in enumerate(dataloader()):
                assert len(input) == 1
                assert len(label) == 1
                assert input.shape == [1, 3, 4]
                assert label.shape == [1, 1]
                assert isinstance(input, paddle.Tensor)
                assert isinstance(label, paddle.Tensor)
                assert np.allclose(input.numpy(), input_np[i])
                assert np.allclose(label.numpy(), label_np[i])

    def test_main(self):
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)


class TestComposeDataset(unittest.TestCase):
    def test_main(self):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        dataset1 = RandomDataset(10)
        dataset2 = RandomDataset(10)
        dataset = ComposeDataset([dataset1, dataset2])
        assert len(dataset) == 10

        for i in range(len(dataset)):
            input1, label1, input2, label2 = dataset[i]
            input1_t, label1_t = dataset1[i]
            input2_t, label2_t = dataset2[i]
            assert np.allclose(input1, input1_t)
            assert np.allclose(label1, label1_t)
            assert np.allclose(input2, input2_t)
            assert np.allclose(label2, label2_t)


class TestRandomSplitApi(unittest.TestCase):
    def test_main(self):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        dataset1, dataset2 = paddle.io.random_split(range(5), [1, 4])

        self.assertTrue(len(dataset1) == 1)
        self.assertTrue(len(dataset2) == 4)

        elements_list = list(range(5))

        for _, val in enumerate(dataset1):
            elements_list.remove(val)

        for _, val in enumerate(dataset2):
            elements_list.remove(val)

        self.assertTrue(len(elements_list) == 0)


class TestRandomSplitError(unittest.TestCase):
    def test_errors(self):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        self.assertRaises(ValueError, paddle.io.random_split, range(5), [3, 8])
        self.assertRaises(ValueError, paddle.io.random_split, range(5), [8])
        self.assertRaises(ValueError, paddle.io.random_split, range(5), [])


class TestSubsetDataset(unittest.TestCase):
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        input_np = np.random.random([5, 3, 4]).astype('float32')
        input = paddle.to_tensor(input_np)
        label_np = np.random.random([5, 1]).astype('int32')
        label = paddle.to_tensor(label_np)

        dataset = TensorDataset([input, label])
        even_subset = paddle.io.Subset(dataset, [0, 2, 4])
        odd_subset = paddle.io.Subset(dataset, [1, 3])

        assert len(dataset) == 5

        def prepare_dataloader(dataset):
            return DataLoader(
                dataset,
                places=places,
                num_workers=num_workers,
                batch_size=1,
                drop_last=True)

        dataloader = prepare_dataloader(dataset)
        dataloader_even = prepare_dataloader(even_subset)
        dataloader_odd = prepare_dataloader(odd_subset)

        def assert_basic(input, label):
            assert len(input) == 1
            assert len(label) == 1
            assert input.shape == [1, 3, 4]
            assert label.shape == [1, 1]
            assert isinstance(input, paddle.Tensor)
            assert isinstance(label, paddle.Tensor)

        elements_list = list()
        for _, (input, label) in enumerate(dataloader()):
            assert_basic(input, label)
            elements_list.append(label)

        for _, (input, label) in enumerate(dataloader_even()):
            assert_basic(input, label)
            elements_list.remove(label)

        odd_list = list()
        for _, (input, label) in enumerate(dataloader_odd()):
            assert_basic(input, label)
            odd_list.append(label)

        self.assertEqual(odd_list, elements_list)

    def test_main(self):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)


class TestChainDataset(unittest.TestCase):
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        dataset1 = RandomIterableDataset(10)
        dataset2 = RandomIterableDataset(10)
        dataset = ChainDataset([dataset1, dataset2])

        samples = []
        for data in iter(dataset):
            samples.append(data)
        assert len(samples) == 20

        idx = 0
        for image, label in iter(dataset1):
            assert np.allclose(image, samples[idx][0])
            assert np.allclose(label, samples[idx][1])
            idx += 1
        for image, label in iter(dataset2):
            assert np.allclose(image, samples[idx][0])
            assert np.allclose(label, samples[idx][1])
            idx += 1

    def test_main(self):
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(num_workers=0, places=p)


class RangeIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(self.start, self.end):
            yield np.array([i])


class CountingIterableDataset(IterableDataset):
    def __init__(self, n):
        super(CountingIterableDataset, self).__init__()
        self.n = n

    def __iter__(self):
        return iter([np.array([i]) for i in range(self.n)])


class TestBufferShuffleDataset(unittest.TestCase):
    def run_main(self, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1

        dataset_range_ = RangeIterableDataset(start=START, end=EXPECTED_RANGE)
        dataset_count_ = CountingIterableDataset(EXPECTED_RANGE)

        for num_workers in WORKER_SETTING:
            dataloader_range_origin = get_dataloader(
                dataset_range_, num_workers, worker_init_fn)
            list_orgin = [data for data in dataloader_range_origin]
            for buffer_size in BUFFER_SIZES:
                dataloader_count_with_buffer = get_dataloader(
                    BufferedShuffleDataset(dataset_count_, buffer_size),
                    num_workers, shuffle_ds_init_fn)
                list_count = [data for data in dataloader_count_with_buffer]

                dataloader_range_with_buffer = get_dataloader(
                    BufferedShuffleDataset(dataset_range_, buffer_size),
                    num_workers, shuffle_ds_init_fn)
                list_range = [data for data in dataloader_range_with_buffer]

                list_count, list_range = sorted(list_count), sorted(list_range)
                for origin_item, count_item, range_item in zip(
                        list_orgin, list_count, list_range):
                    # self.assertEqual(origin_item, count_item)
                    # self.assertEqual(count_item, range_item)
                    # self.assertEqual(range_item, origin_item)

                    assert origin_item == count_item
                    assert count_item == range_item
                    assert range_item == origin_item
                # self.assertCountEqual(list_range, list_count)
                # self.assertCountEqual(list_count, list_orgin)
                # self.assertCountEqual(list_orgin, list_range)

    def test_main(self):
        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for p in places:
            self.run_main(places=p)


if __name__ == '__main__':
    unittest.main()
