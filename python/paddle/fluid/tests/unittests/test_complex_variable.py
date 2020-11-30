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

import unittest
import numpy as np
import paddle
import paddle.fluid.dygraph as dg


class TestComplexVariable(unittest.TestCase):
    def compare(self):
        a = np.array([[1.0 + 1.0j, 2.0 + 1.0j],
                      [3.0 + 1.0j, 4.0 + 1.0j]]).astype(self._dtype)
        b = np.array([[1.0 + 1.0j, 1.0 + 1.0j]]).astype(self._dtype)

        with dg.guard():
            x = dg.to_variable(a, "x")
            y = dg.to_variable(b)
            out = paddle.complex.elementwise_add(x, y)
            self.assertIsNotNone("{}".format(out))

        self.assertTrue(np.allclose(out.numpy(), a + b))
        self.assertEqual(out.numpy().dtype, self._dtype)
        self.assertEqual(out.shape, x.shape)

    def test_attrs(self):
        self._dtype = "complex64"
        self.compare()
        self._dtype = "complex128"
        self.compare()


if __name__ == '__main__':
    unittest.main()
