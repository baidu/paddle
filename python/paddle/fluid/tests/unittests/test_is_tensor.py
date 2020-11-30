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
import paddle

DELTA = 0.00001


class TestIsTensorApi(unittest.TestCase):
    def test_is_tensor_real(self, dtype="float32"):
        """Test is_tensor api with a real tensor
        """
        paddle.disable_static()
        x = paddle.rand([3, 2, 4], dtype=dtype)
        self.assertTrue(paddle.is_tensor(x))

    def test_is_tensor_list(self, dtype="float32"):
        """Test is_tensor api with a list
        """
        paddle.disable_static()
        x = [1, 2, 3]
        self.assertFalse(paddle.is_tensor(x))

    def test_is_tensor_number(self, dtype="float32"):
        """Test is_tensor api with a number
        """
        paddle.disable_static()
        x = 5
        self.assertFalse(paddle.is_tensor(x))


if __name__ == '__main__':
    unittest.main()
