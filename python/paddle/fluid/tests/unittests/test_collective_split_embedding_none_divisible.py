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
import unittest
import numpy as np
import paddle
from paddle.distributed import fleet

paddle.enable_static()


class TestCollectiveSplitAssert(unittest.TestCase):
    def network(self):
        fleet.init()
        data = paddle.static.data(
            name='tindata', shape=[10, 1000], dtype="float32")
        emb_out = paddle.distributed.split(
            data, (7, 8), operation="embedding", num_partitions=2)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            self.network()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
