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


def ref_var(x, axis=None, unbiased=True, keepdim=False):
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis, )
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=ddof, keepdims=keepdim)


class TestVarAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place=paddle.CUDAPlace(0) \
            if paddle.fluid.core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def set_attrs(self):
        pass

    def static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape, self.dtype)
            out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
        paddle.enable_static()
        return out.numpy()

    def test_api(self):
        out_ref = ref_var(self.x, self.axis, self.unbiased, self.keepdim)
        out_dygraph = self.dygraph()
        out_static = self.static()
        for out in [out_dygraph, out_static]:
            self.assertTrue(np.allclose(out_ref, out))
            self.assertTrue(np.equal(out_ref.shape, out.shape).all())


class TestVarAPI_dtype(TestVarAPI):
    def set_attrs(self):
        self.dtype = 'float32'


class TestVarAPI_axis_int(TestVarAPI):
    def set_attrs(self):
        self.axis = 2


class TestVarAPI_axis_list(TestVarAPI):
    def set_attrs(self):
        self.axis = [1, 2]


class TestVarAPI_axis_tuple(TestVarAPI):
    def set_attrs(self):
        self.axis = (1, 3)


class TestVarAPI_keepdim(TestVarAPI):
    def set_attrs(self):
        self.keepdim = False


class TestVarAPI_unbiased(TestVarAPI):
    def set_attrs(self):
        self.unbiased = False


class TestVarAPI_alias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(np.array([10, 12], 'float32'))
        out1 = paddle.var(x).numpy()
        out2 = paddle.tensor.var(x).numpy()
        out3 = paddle.tensor.stat.var(x).numpy()
        self.assertTrue(np.allclose(out1, out2))
        self.assertTrue(np.allclose(out1, out3))
        paddle.enable_static()


class TestVarError(unittest.TestCase):
    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [2, 3, 4], 'int32')
            self.assertRaises(TypeError, paddle.var, x)


if __name__ == '__main__':
    unittest.main()
