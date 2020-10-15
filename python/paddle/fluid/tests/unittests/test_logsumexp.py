#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import unittest
import numpy as np
from op_test import OpTest


def ref_logsumexp(x, axis=None, keepdim=False, reduce_all=False):
    if isinstance(axis, int):
        axis = (axis, )
    elif isinstance(axis, list):
        axis = tuple(axis)
    if reduce_all:
        axis = None
    out = np.log(np.exp(x).sum(axis=axis, keepdims=keepdim))
    return out


class TestLogsumexp(OpTest):
    def setUp(self):
        self.op_type = 'logsumexp'
        self.shape = [2, 3, 4, 5]
        self.dtype = 'float64'
        self.axis = [-1]
        self.keepdim = False
        self.reduce_all = False
        self.set_attrs()

        np.random.seed(10)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = ref_logsumexp(x, self.axis, self.keepdim, self.reduce_all)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {
            'axis': self.axis,
            'keepdim': self.keepdim,
            'reduce_all': self.reduce_all
        }

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['Out'])


class TestLogsumexp_shape(TestLogsumexp):
    def set_attrs(self):
        self.shape = [4, 5, 6]


class TestLogsumexp_axis(TestLogsumexp):
    def set_attrs(self):
        self.axis = [0, -1]


class TestLogsumexp_axis_all(TestLogsumexp):
    def set_attrs(self):
        self.axis = [0, 1, 2, 3]


class TestLogsumexp_keepdim(TestLogsumexp):
    def set_attrs(self):
        self.keepdim = True


class TestLogsumexp_reduce_all(TestLogsumexp):
    def set_attrs(self):
        self.reduce_all = True


class TestLogsumexpError(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            self.assertRaises(TypeError, paddle.logsumexp, 1)
            x1 = paddle.fluid.data(name='x1', shape=[120], dtype="int32")
            self.assertRaises(TypeError, paddle.logsumexp, x1)


class TestLogsumexpAPI(unittest.TestCase):
    def setUp(self):
        self.shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        self.place = paddle.CUDAPlace(0) if paddle.fluid.core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def api_case(self, axis=None, keepdim=False):
        out_ref = ref_logsumexp(self.x, axis, keepdim)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape)
            out = paddle.logsumexp(x, axis, keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        self.assertTrue(np.allclose(res[0], out_ref))

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out = paddle.logsumexp(x, axis, keepdim)
        self.assertTrue(np.allclose(out.numpy(), out_ref))
        paddle.enable_static()

    def test_api(self):
        self.api_case()
        self.api_case(2)
        self.api_case([-1])
        self.api_case([2, -3])
        self.api_case((0, 1, -1))
        self.api_case(keepdim=True)

    def test_alias(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out1 = paddle.logsumexp(x)
        out2 = paddle.tensor.logsumexp(x)
        out3 = paddle.tensor.math.logsumexp(x)
        out_ref = ref_logsumexp(self.x)
        for out in [out1, out2, out3]:
            self.assertTrue(np.allclose(out.numpy(), out_ref))
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
