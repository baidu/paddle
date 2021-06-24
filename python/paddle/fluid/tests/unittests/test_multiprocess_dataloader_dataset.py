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

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.io import Dataset, IterableDataset, TensorDataset, \
        ComposeDataset, ChainDataset, DataLoader, random_split, Subset

IMAGE_SIZE = 32


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


class NumpyMixTensorDataset(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('int64')
        return paddle.to_tensor(image, place=paddle.CPUPlace()), label


class TestNumpyMixTensorDataset(TestTensorDataset):
    def run_main(self, num_workers, places):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            dataset = NumpyMixTensorDataset(16)
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
                assert input.shape == [1, IMAGE_SIZE]
                assert label.shape == [1, 1]
                assert isinstance(input, paddle.Tensor)
                assert isinstance(label, paddle.Tensor)


class ComplextDataset(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        return (3.1, 'abc', paddle.to_tensor(
            np.random.random([IMAGE_SIZE]).astype('float32'),
            place=paddle.CPUPlace()),
                [1, np.random.random([2]).astype('float32')], {
                    'a': 2.0,
                    'b': np.random.random([2]).astype('float32')
                })


class TestComplextDataset(unittest.TestCase):
    def run_main(self, num_workers):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            dataset = ComplextDataset(16)
            assert len(dataset) == 16
            dataloader = DataLoader(
                dataset,
                places=place,
                num_workers=num_workers,
                batch_size=2,
                drop_last=True)

            for i, data in enumerate(dataloader()):
                assert len(data) == 5
                # data[0]: collate 3.1
                assert data[0].shape == [2]
                assert isinstance(data[1], list)
                # data[1]: collate 'abc'
                assert len(data[1]) == 2
                assert isinstance(data[1][0], str)
                assert isinstance(data[1][1], str)
                # data[2]: collate tensor
                assert data[2].shape == [2, IMAGE_SIZE]
                # data[3]: collate list
                assert isinstance(data[3], list)
                assert data[3][0].shape == [2]
                assert data[3][1].shape == [2, 2]
                # data[4]: collate dict
                assert isinstance(data[4], dict)
                assert data[4]['a'].shape == [2]
                assert data[4]['b'].shape == [2, 2]

    def test_main(self):
        for num_workers in [0, 2]:
            self.run_main(num_workers)


class SingleFieldDataset(Dataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        return np.random.random((2, 3)).astype('float32')


class TestSingleFieldDataset(unittest.TestCase):
    def init_dataset(self):
        self.sample_num = 16
        self.dataset = SingleFieldDataset(self.sample_num)

    def run_main(self, num_workers):
        paddle.static.default_startup_program().random_seed = 1
        paddle.static.default_main_program().random_seed = 1
        place = paddle.CPUPlace()
        with fluid.dygraph.guard(place):
            self.init_dataset()
            dataloader = DataLoader(
                self.dataset,
                places=place,
                num_workers=num_workers,
                batch_size=2,
                drop_last=True)

            for i, data in enumerate(dataloader()):
                assert isinstance(data, paddle.Tensor)
                assert data.shape == [2, 2, 3]

    def test_main(self):
        for num_workers in [0, 2]:
            self.run_main(num_workers)


class SingleFieldIterableDataset(IterableDataset):
    def __init__(self, sample_num):
        self.sample_num = sample_num

    def __iter__(self):
        for _ in range(self.sample_num):
            yield np.random.random((2, 3)).astype('float32')


class TestSingleFieldIterableDataset(TestSingleFieldDataset):
    def init_dataset(self):
        self.sample_num = 16
        self.dataset = SingleFieldIterableDataset(self.sample_num)


class TestDataLoaderGenerateStates(unittest.TestCase):
    def setUp(self):
        self.inputs = [(0, 1), (0, 2), (1, 3)]
        self.outputs = [[1835504127, 1731038949, 1320224556, 2330041505],
                        [2834126987, 2358157858, 1860244682, 1437227251],
                        [457190280, 2660306227, 859341110, 354512857]]

    def test_main(self):
        from paddle.fluid.dataloader.worker import _generate_states
        for inp, outp in zip(self.inputs, self.outputs):
            out = _generate_states(*inp)
            assert out == outp


if __name__ == '__main__':
    unittest.main()
