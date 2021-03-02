#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle
import paddle.nn.functional as F

np.random.seed(10)
paddle.enable_static()


def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def ref_softmax(x, axis=None, dtype=None):
    x_t = x.copy()
    if dtype is not None:
        x_t = x_t.astype(dtype)
    if axis is None:
        axis = -1
    return np.apply_along_axis(stable_softmax, axis, x_t)


class TestSoftmaxOp(OpTest):
    def get_x_shape(self):
        return [10, 10]

    def get_axis(self):
        return -1

    def setUp(self):
        self.op_type = "softmax"
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float64
        self.init_kernel_type()
        self.shape = self.get_x_shape()
        self.axis = self.get_axis()

        np.random.seed(0)
        x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        out = np.apply_along_axis(stable_softmax, self.axis, x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {
            'axis': self.axis,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn
        }

    def init_kernel_type(self):
        pass

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place, atol=1e-5, check_dygraph=(self.use_mkldnn == False))
        else:
            self.check_output(check_dygraph=(self.use_mkldnn == False))

    def test_check_grad(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.use_cudnn or self.dtype == np.float16:
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place, ["X"],
                    "Out",
                    max_relative_error=0.01,
                    check_dygraph=(self.use_mkldnn == False))
        else:
            self.check_grad(
                ["X"],
                "Out",
                max_relative_error=0.01,
                check_dygraph=(self.use_mkldnn == False))


class TestSoftmaxOp2(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]


class TestSoftmaxOp3(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 0


class TestSoftmaxOp4(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 1


class TestSoftmaxOp5(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 2


class TestSoftmaxOp6(TestSoftmaxOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 3


#TODO(joejiong): temporary helper function,
#it will be removed after rocm platform fix double type related functionality.
def rocm_plugin(self):
    if core.is_compiled_with_rocm():
        self.dtype = np.float32
    else:
        self.dtype = np.float64


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp(TestSoftmaxOp):
    def init_kernel_type(self):
        self.use_cudnn = True
        rocm_plugin(self)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp2(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5]


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp3(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5]

    def get_axis(self):
        return 0


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp4(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5]

    def get_axis(self):
        return 1


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp5(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5]

    def get_axis(self):
        return 2


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp6(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5]

    def get_axis(self):
        return 3


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp7(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5, 6]


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp8(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5, 6]

    def get_axis(self):
        return 0


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp9(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5, 6]

    def get_axis(self):
        return 1


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp10(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5, 6]

    def get_axis(self):
        return 2


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp11(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5, 6]

    def get_axis(self):
        return 3


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxCUDNNOp12(TestSoftmaxCUDNNOp):
    def get_x_shape(self):
        rocm_plugin(self)
        return [2, 3, 4, 5, 6]

    def get_axis(self):
        return 4


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxFP16Op(TestSoftmaxOp):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)

    # FIXME: If the x_shape is [10, 10], gradient failed.
    def test_check_grad(self):
        pass


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxFP16Op2(TestSoftmaxFP16Op):
    def get_x_shape(self):
        return [2, 3, 4, 10]


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxFP16CUDNNOp(TestSoftmaxOp):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestSoftmaxFP16CUDNNOp2(TestSoftmaxFP16CUDNNOp):
    def get_x_shape(self):
        return [2, 3, 4, 5]


class TestSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        self.x_np = np.random.uniform(-1., 1., [2, 3, 4, 5]).astype('float32')
        self.out_ref = np.apply_along_axis(stable_softmax, -1, self.x_np)
        self.executed_api()

    def executed_api(self):
        self.softmax = F.softmax

    def test_static_check(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, 'float32')
            out1 = self.softmax(x)
            m = paddle.nn.Softmax()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_softmax(self.x_np, axis=-1, dtype=None)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_check(self):
        paddle.disable_static(self.place)

        x = paddle.to_tensor(self.x_np)
        out1 = self.softmax(x)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.Softmax()
        out2 = m(x)
        out_ref = ref_softmax(self.x_np, axis=-1, dtype=None)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = self.softmax(x, axis=0)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.Softmax(axis=0)
        out2 = m(x)
        out_ref = ref_softmax(self.x_np, axis=0, dtype=None)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        platform_init_dtype = np.float64
        if core.is_compiled_with_rocm():
            platform_init_dtype = np.float32
        out = self.softmax(x, dtype=platform_init_dtype)
        out_ref = ref_softmax(self.x_np, axis=-1, dtype=platform_init_dtype)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        paddle.enable_static()

    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, self.softmax, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[2, 3], dtype='int32')
            self.assertRaises(TypeError, self.softmax, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[2, 3], dtype='float16')
            self.softmax(x_fp16)


class TestSoftmaxInplaceAPI(TestSoftmaxAPI):
    def executed_api(self):
        self.softmax = F.softmax_


if __name__ == "__main__":
    unittest.main()
