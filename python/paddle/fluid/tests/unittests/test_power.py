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

from __future__ import print_function
import paddle
import paddle.tensor as tensor
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import numpy as np
import unittest

def _run_power(mode, x, y):
    # dynamic mode
    if mode=='dynamic':
        paddle.enable_imperative()
        # y is scalar
        if isinstance(y, (int, long, float)):
            x_ = paddle.imperative.to_variable(x)
            y_ = y
            res = paddle.power(x_, y_)
            return res.numpy()
        # y is tensor
        else:
            x_ = paddle.imperative.to_variable(x)
            y_ = paddle.imperative.to_variable(y)
            res = paddle.power(x_, y_)
            return res.numpy()
    # static mode
    elif mode=='static':
        paddle.disable_imperative()
        # y is scalar
        if isinstance(y, (int, long, float)):
            with program_guard(Program(), Program()):
                x_ = paddle.nn.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = y
                res = paddle.power(x_, y_)
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                outs = exe.run(feed={'x':x}, fetch_list=[res])
                return outs[0]
        # y is tensor
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.nn.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = paddle.nn.data(name="y", shape=y.shape, dtype=y.dtype)
                res = paddle.power(x_, y_)
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                outs = exe.run(feed={'x':x, 'y': y}, fetch_list=[res])
                return outs[0]

class TestPowerAPI(unittest.TestCase):
    """TestPowerAPI."""

    def test_power(self):
        """test_power."""
        np.random.seed(7)
        # test 1-d float tensor ** float scalar
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = np.random.rand() * 10
        res = _run_power('dynamic', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))
        res = _run_power('static', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))

        # test 1-d float tensor ** int scalar
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = int(np.random.rand() * 10)
        res = _run_power('dynamic', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))
        res = _run_power('static', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))

#         
        x = (np.random.rand(*dims) * 10).astype(np.int64)
        y = int(np.random.rand() * 10)
        res = _run_power('dynamic', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))
        res = _run_power('static', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))

        # test 1-d float tensor ** 1-d float tensor
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(*dims) * 10).astype(np.float64)
        res = _run_power('dynamic', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))
        res = _run_power('static', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))

#        # test 1-d float tensor ** 1-d int tensor
#        dims = (np.random.randint(200, 300),)
#        x = (np.random.rand(*dims) * 10).astype(np.float64)
#        y = (np.random.rand(*dims) * 10).astype(np.int64)
#        res = _run_power('dynamic', x, y)
#        self.assertTrue(np.allclose(res, np.power(x, y)))
#        res = _run_power('static', x, y)
#        self.assertTrue(np.allclose(res, np.power(x, y)))

#        # test 1-d int tensor ** 1-d float tensor
#        dims = (np.random.randint(200, 300),)
#        x = (np.random.rand(*dims) * 10).astype(np.int64)
#        y = (np.random.rand(*dims) * 10).astype(np.float64)
#        res = _run_power('dynamic', x, y)
#        self.assertTrue(np.allclose(res, np.power(x, y)))
#        res = _run_power('static', x, y)
#        self.assertTrue(np.allclose(res, np.power(x, y)))

        # test 1-d int tensor ** 1-d int tensor
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.int64)
        y = (np.random.rand(*dims) * 10).astype(np.int64)
        res = _run_power('dynamic', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))
        res = _run_power('static', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))

        # test broadcast
        dims = (np.random.randint(1, 10), np.random.randint(5, 10), np.random.randint(5, 10))
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1]) * 10).astype(np.float64)
        res = _run_power('dynamic', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))
        res = _run_power('static', x, y)
        self.assertTrue(np.allclose(res, np.power(x, y)))


class TestPowerError(unittest.TestCase):
    """TestPowerError."""

    def test_errors(self):
        """test_errors."""
        np.random.seed(7)

        # test dynamic computation graph: inputs must be broadcastable
        dims = (np.random.randint(1, 10), np.random.randint(5, 10), np.random.randint(5, 10))
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.float64)
        self.assertRaises(fluid.core.EnforceNotMet, _run_power, 'dynamic', x, y)
        self.assertRaises(fluid.core.EnforceNotMet, _run_power, 'static', x, y)


if __name__ == '__main__':
    unittest.main()