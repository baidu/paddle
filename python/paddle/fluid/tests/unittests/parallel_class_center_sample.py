# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function

import unittest

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle import framework


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)


def class_center_sample_numpy(label, classes_list, num_samples):
    unique_label = np.unique(label)
    nranks = len(classes_list)
    class_interval = np.cumsum(np.insert(classes_list, 0, 0))
    pos_class_center_per_device = []
    unique_label_per_device = []

    for i in range(nranks):
        index = np.logical_and(unique_label >= class_interval[i],
                               unique_label < class_interval[i + 1])
        pos_class_center_per_device.append(unique_label[index] - class_interval[
            i])
        unique_label_per_device.append(unique_label[index])

    num_samples_per_device = []
    for pos_class_center in pos_class_center_per_device:
        num_samples_per_device.append(max(len(pos_class_center), num_samples))
    sampled_class_interval = np.cumsum(np.insert(num_samples_per_device, 0, 0))

    remapped_dict = {}
    for i in range(nranks):
        for idx, v in enumerate(unique_label_per_device[i],
                                sampled_class_interval[i]):
            remapped_dict[v] = idx

    remapped_label = []
    for l in label:
        remapped_label.append(remapped_dict[l])

    return remapped_label, pos_class_center_per_device


class TestParallelClassCenterSampleOp(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

    def test_class_center_sample(self):

        rank_id = dist.get_rank()
        nranks = dist.get_world_size()

        seed = 1025
        set_random_seed(seed)
        paddle.seed(rank_id * 10)
        random.seed(seed)
        np.random.seed(seed)

        batch_size = 20
        num_samples = 6

        for dtype in ('int32', 'int64'):
            for _ in range(5):
                classes_list = np.random.randint(10, 15, (nranks, ))
                num_class = np.sum(classes_list)

                np_label = np.random.randint(
                    0, num_class, (batch_size, ), dtype=dtype)
                label = paddle.to_tensor(np_label, dtype=dtype)
                np_remapped_label, np_sampled_class_center_per_device = class_center_sample_numpy(
                    np_label, classes_list, num_samples)
                remapped_label, sampled_class_index = paddle.class_center_sample(
                    label, classes_list[rank_id], num_samples)
                np.testing.assert_allclose(remapped_label.numpy(),
                                           np_remapped_label)
                np_sampled_class_index = np_sampled_class_center_per_device[
                    rank_id]
                np.testing.assert_allclose(
                    sampled_class_index.numpy()[:len(np_sampled_class_index)],
                    np_sampled_class_index)


if __name__ == '__main__':
    unittest.main()
