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
from paddle.incubate.hapi.text.data_utils import SamplerHelper
import unittest
import numpy as np


class TestSamplerHelper(unittest.TestCase):
    def setUp(self):
        dataset = ["Paddle", "Baidu", "DLTP"]
        self.sampler_helper = SamplerHelper(dataset)

    def test_get_set_length(self):
        length = self.sampler_helper.length
        self.sampler_helper.length = length

    def test_apply(self):
        def fn(sampler_helper):
            buffer_size = 3
            seed = 2020
            random_generator = np.random.RandomState(seed)

            def _impl():
                buf = []
                for idx in iter(sampler_helper):
                    buf.append(idx)
                    if buffer_size > 0 and len(buf) >= buffer_size:
                        random_generator.shuffle(buf)
                        for b in buf:
                            yield b
                        buf = []
                if len(buf) > 0:
                    random_generator.shuffle(buf)
                    for b in buf:
                        yield b

            return _impl

        applied_sampler = self.sampler_helper.apply(fn)
        for idx in iter(applied_sampler):
            break

    def test_shuffle(self):
        shuffled_sampler = self.sampler_helper.shuffle()
        for idx in iter(shuffled_sampler):
            break

    def test_sort(self):
        sorted_sampler = self.sampler_helper.sort(buffer_size=-1)
        for idx in iter(sorted_sampler):
            break
            print(idx)

    def test_batch(self):
        batch_size = 2
        batched_sampler = self.sampler_helper.batch(batch_size, drop_last=True)
        for idx in iter(batched_sampler):
            break

    def test_shard(self):
        shared_sampler = self.sampler_helper.shard()
        for idx in iter(shared_sampler):
            break

    def test_list(self):
        listed_sampler = self.sampler_helper.list()
        iterator = listed_sampler.iterable()
        for idx in iterator:
            break


if __name__ == '__main__':
    unittest.main()
