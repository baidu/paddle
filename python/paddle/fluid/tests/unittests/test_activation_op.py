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
from scipy.special import expit, erf

from op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


class TestSqrtOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of sqrt op must be Variable or numpy.ndarray.
            in1 = 1
            self.assertRaises(TypeError, fluid.layers.sqrt, in1)
            # The input dtype of sqrt op must be float16, float32, float64.
            in2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.sqrt, in2)

            in3 = fluid.layers.data(
                name='input3', shape=[12, 10], dtype="float16")
            fluid.layers.sqrt(x=in3)


class TestActivation(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.init_dtype()
        self.init_kernel_type()

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def init_dtype(self):
        self.dtype = np.float64

    def init_kernel_type(self):
        pass


class TestExpm1(TestActivation):
    def setUp(self):
        self.op_type = "expm1"
        self.init_dtype()

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.expm1(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestExpm1API(unittest.TestCase):
    def init_dtype(self):
        self.dtype = 'float64'
        self.shape = [11, 17]

    def setUp(self):
        self.init_dtype()
        self.x = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        self.out_ref = np.expm1(self.x)

        self.place = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))

    def test_static_api(self):
        paddle.enable_static()

        def run(place):
            with paddle.static.program_guard(paddle.static.Program()):
                X = paddle.fluid.data('X', self.shape, dtype=self.dtype)
                out = paddle.expm1(X)
                exe = paddle.static.Executor(place)
                res = exe.run(feed={'X': self.x})
            for r in res:
                self.assertEqual(np.allclose(self.out_ref, r), True)

        for place in self.place:
            run(place)

    def test_dygraph_api(self):
        def run(place):
            paddle.disable_static(place)
            X = paddle.to_tensor(self.x)
            out = paddle.expm1(X)
            self.assertEqual(np.allclose(self.out_ref, out.numpy()), True)
            paddle.enable_static()

        for place in self.place:
            run(place)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            X = paddle.fluid.data('X', self.shape, dtype='int32')
            self.assertRaises(TypeError, paddle.expm1, X)
        # The input dtype must be float16, float32, float64.


class TestParameter(object):
    def test_out_name(self):
        with fluid.program_guard(fluid.Program()):
            np_x = np.array([0.1])
            data = fluid.layers.data(name="X", shape=[1])
            out = eval("paddle.%s(data, name='Y')" % self.op_type)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"X": np_x}, fetch_list=[out])
            expected = eval("np.%s(np_x)" % self.op_type)
            self.assertEqual(result, expected)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = eval("paddle.%s(x).numpy()" % self.op_type)
            z_expected = eval("np.%s(np_x)" % self.op_type)
            # ROCM platform will fail in assertEqual
            if core.is_compiled_with_rocm():
                self.assertTrue(np.allclose(z, z_expected))
            else:
                self.assertEqual(z, z_expected)


class TestSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "sigmoid"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


class TestSilu(TestActivation):
    def setUp(self):
        self.op_type = "silu"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = x / (np.exp(-x) + 1)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSiluAPI(unittest.TestCase):
    # test paddle.nn.Silu, paddle.nn.functional.silu
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype('float32')
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [11, 17])
            out1 = F.silu(x)
            m = paddle.nn.Silu()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.silu(x)
        m = paddle.nn.Silu()
        out2 = m(x)
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.silu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[11, 17], dtype='int32')
            self.assertRaises(TypeError, F.silu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[11, 17], dtype='float16')
            F.silu(x_fp16)


class TestLogSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "logsigmoid"
        self.init_dtype()

        np.random.seed(2048)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.log(1 / (1 + np.exp(-x)))

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.008)


class TestLogSigmoidAPI(unittest.TestCase):
    # test paddle.nn.LogSigmoid, paddle.nn.functional.log_sigmoid
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [11, 17])
            out1 = F.log_sigmoid(x)
            m = paddle.nn.LogSigmoid()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = np.log(1 / (1 + np.exp(-self.x_np)))
        for r in res:
            self.assertTrue(np.allclose(out_ref, r))

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.log_sigmoid(x)
        m = paddle.nn.LogSigmoid()
        out2 = m(x)
        out_ref = np.log(1 / (1 + np.exp(-self.x_np)))
        for r in [out1, out2]:
            self.assertTrue(np.allclose(out_ref, r.numpy()))
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [11, 17])
            out = paddle.fluid.layers.logsigmoid(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = np.log(1 / (1 + np.exp(-self.x_np)))
        self.assertTrue(np.allclose(out_ref, res[0]))

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.log_sigmoid, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[11, 17], dtype='int32')
            self.assertRaises(TypeError, F.log_sigmoid, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[11, 17], dtype='float16')
            F.log_sigmoid(x_fp16)


class TestTanh(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "tanh"
        self.init_dtype()
        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.tanh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def init_dtype(self):
        #TODO If dtype is float64, the output (Out) has diff at CPUPlace
        # when using and not using inplace. Therefore, set dtype as float32
        # for now.
        self.dtype = np.float32


class TestTanhAPI(unittest.TestCase):
    # test paddle.tanh, paddle.nn.tanh, paddle.nn.functional.tanh
    def setUp(self):
        self.dtype = 'float32'
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.executed_api()

    def executed_api(self):
        self.tanh = F.tanh

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12], self.dtype)
            out1 = self.tanh(x)
            th = paddle.nn.Tanh()
            out2 = th(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = np.tanh(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.tanh(x)
        out2 = paddle.tanh(x)
        th = paddle.nn.Tanh()
        out3 = th(x)
        out_ref = np.tanh(self.x_np)
        for r in [out1, out2, out3]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', [10, 12], self.dtype)
            out = fluid.layers.tanh(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = np.tanh(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, self.tanh, 1)
            # The input dtype must be float16, float32.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, self.tanh, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            self.tanh(x_fp16)


class TestTanhInplaceAPI(TestTanhAPI):
    # test paddle.tanh_
    def executed_api(self):
        self.tanh = paddle.tanh_


class TestAtan(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "atan"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.arctan(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_out_name(self):
        with fluid.program_guard(fluid.Program()):
            np_x = np.array([0.1])
            data = fluid.layers.data(name="X", shape=[1])
            out = paddle.atan(data, name='Y')
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result, = exe.run(feed={"X": np_x}, fetch_list=[out])
            expected = np.arctan(np_x)
            self.assertEqual(result, expected)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = paddle.atan(x).numpy()
            z_expected = np.arctan(np_x)
            self.assertEqual(z, z_expected)


class TestSinh(TestActivation):
    def setUp(self):
        self.op_type = "sinh"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sinh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = fluid.layers.sinh(x).numpy()
            z_expected = np.sinh(np_x)
            self.assertTrue(np.allclose(z, z_expected))

    def test_api(self):
        test_data_shape = [11, 17]
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_x = np.random.uniform(0.1, 1,
                                        test_data_shape).astype("float32")
            data_x = fluid.layers.data(
                name="data_x",
                shape=test_data_shape,
                append_batch_size=False,
                dtype="float32")

            pd_sinh_out = fluid.layers.sinh(data_x)
            exe = fluid.Executor(place=fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            np_sinh_res = exe.run(fluid.default_main_program(),
                                  feed={"data_x": input_x},
                                  fetch_list=[pd_sinh_out])

        expected_res = np.sinh(input_x)
        self.assertTrue(np.allclose(np_sinh_res, expected_res))

    def test_backward(self):
        test_data_shape = [11, 17]
        with fluid.dygraph.guard():
            input_x = np.random.uniform(0.1, 1,
                                        test_data_shape).astype("float32")
            var = fluid.dygraph.to_variable(input_x)
            var.stop_gradient = False
            loss = fluid.layers.sinh(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, input_x.shape)


class TestSinhOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.sinh, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.sinh, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.sinh(x_fp16)


class TestCosh(TestActivation):
    def setUp(self):
        self.op_type = "cosh"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.cosh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = fluid.layers.cosh(x).numpy()
            z_expected = np.cosh(np_x)
            self.assertTrue(np.allclose(z, z_expected))

    def test_api(self):
        test_data_shape = [11, 17]
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_x = np.random.uniform(0.1, 1,
                                        test_data_shape).astype("float32")
            data_x = fluid.layers.data(
                name="data_x",
                shape=test_data_shape,
                append_batch_size=False,
                dtype="float32")

            pd_cosh_out = paddle.cosh(data_x)
            exe = fluid.Executor(place=fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            np_cosh_res = exe.run(fluid.default_main_program(),
                                  feed={"data_x": input_x},
                                  fetch_list=[pd_cosh_out])

        expected_res = np.cosh(input_x)
        self.assertTrue(np.allclose(np_cosh_res, expected_res))

    def test_backward(self):
        test_data_shape = [11, 17]
        with fluid.dygraph.guard():
            input_x = np.random.uniform(0.1, 1,
                                        test_data_shape).astype("float32")
            var = fluid.dygraph.to_variable(input_x)
            var.stop_gradient = False
            loss = fluid.layers.cosh(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, input_x.shape)


class TestCoshOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.cosh, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.cosh, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.cosh(x_fp16)


def ref_tanhshrink(x):
    out = x - np.tanh(x)
    return out


class TestTanhshrink(TestActivation):
    def setUp(self):
        self.op_type = "tanh_shrink"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(10, 20, [10, 17]).astype(self.dtype)
        out = ref_tanhshrink(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestTanhshrinkAPI(unittest.TestCase):
    # test paddle.nn.Tanhshrink, paddle.nn.functional.tanhshrink
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(10, 20, [10, 17]).astype(np.float64)
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.tanhshrink(x)
            tanhshrink = paddle.nn.Tanhshrink()
            out2 = tanhshrink(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_tanhshrink(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.tanhshrink(x)
        tanhshrink = paddle.nn.Tanhshrink()
        out2 = tanhshrink(x)
        out_ref = ref_tanhshrink(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.tanh_shrink(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_tanhshrink(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.tanhshrink, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.tanhshrink, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.tanhshrink(x_fp16)


def ref_hardshrink(x, threshold):
    out = np.copy(x)
    out[(out >= -threshold) & (out <= threshold)] = 0
    return out


class TestHardShrink(TestActivation):
    def setUp(self):
        self.op_type = "hard_shrink"
        self.init_dtype()

        self.threshold = 0.5
        self.set_attrs()
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype) * 10
        out = ref_hardshrink(x, self.threshold)

        self.attrs = {'threshold': self.threshold}
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def set_attrs(self):
        pass

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestHardShrink_threshold_negative(TestHardShrink):
    def set_attrs(self):
        self.threshold = -0.1


class TestHardShrinkAPI(unittest.TestCase):
    # test paddle.nn.Hardshrink, paddle.nn.functional.hardshrink
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out1 = F.hardshrink(x)
            hd = paddle.nn.Hardshrink()
            out2 = hd(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardshrink(self.x_np, 0.5)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.hardshrink(x)
        hd = paddle.nn.Hardshrink()
        out2 = hd(x)
        out_ref = ref_hardshrink(self.x_np, 0.5)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = F.hardshrink(x, 0.6)
        hd = paddle.nn.Hardshrink(0.6)
        out2 = hd(x)
        out_ref = ref_hardshrink(self.x_np, 0.6)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', [10, 12])
            out = fluid.layers.hard_shrink(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_hardshrink(self.x_np, 0.5)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.hardshrink, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.hardshrink, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.hardshrink(x_fp16)


def ref_hardtanh(x, min=-1.0, max=1.0):
    out = np.copy(x)
    out[np.abs(x - min) < 0.005] = min + 0.02
    out[np.abs(x - max) < 0.005] = max + 0.02
    out = np.minimum(np.maximum(x, min), max)
    return out


class TestHardtanhAPI(unittest.TestCase):
    # test paddle.nn.Hardtanh, paddle.nn.functional.hardtanh
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out1 = F.hardtanh(x)
            m = paddle.nn.Hardtanh()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardtanh(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.hardtanh(x)
        m = paddle.nn.Hardtanh()
        out2 = m(x)
        out_ref = ref_hardtanh(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = F.hardtanh(x, -2.0, 2.0)
        m = paddle.nn.Hardtanh(-2.0, 2.0)
        out2 = m(x)
        out_ref = ref_hardtanh(self.x_np, -2.0, 2.0)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.hardtanh, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.hardtanh, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.hardtanh(x_fp16)


def ref_softshrink(x, threshold=0.5):
    out = np.copy(x)
    out = (out < -threshold) * (out + threshold) + (out > threshold) * (
        out - threshold)
    return out


class TestSoftshrink(TestActivation):
    def setUp(self):
        self.op_type = "softshrink"
        self.init_dtype()

        threshold = 0.8

        np.random.seed(1023)
        x = np.random.uniform(0.25, 10, [10, 12]).astype(self.dtype)
        out = ref_softshrink(x, threshold)
        self.inputs = {'X': x}
        self.attrs = {"lambda": threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSoftshrinkAPI(unittest.TestCase):
    # test paddle.nn.Softshrink, paddle.nn.functional.softshrink
    def setUp(self):
        self.threshold = 0.8
        np.random.seed(1024)
        self.x_np = np.random.uniform(0.25, 10, [10, 12]).astype(np.float64)
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.softshrink(x, self.threshold)
            softshrink = paddle.nn.Softshrink(self.threshold)
            out2 = softshrink(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_softshrink(self.x_np, self.threshold)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.softshrink(x, self.threshold)
        softshrink = paddle.nn.Softshrink(self.threshold)
        out2 = softshrink(x)
        out_ref = ref_softshrink(self.x_np, self.threshold)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.softshrink(x, self.threshold)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_softshrink(self.x_np, self.threshold)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.softshrink, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.softshrink, x_int32)
            # The threshold must be no less than zero
            x_fp32 = paddle.fluid.data(
                name='x_fp32', shape=[12, 10], dtype='float32')
            self.assertRaises(ValueError, F.softshrink, x_fp32, -1.0)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.softshrink(x_fp16)


class TestSqrt(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "sqrt"
        self.init_dtype()

        np.random.seed(1023)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestRsqrt(TestActivation):
    def setUp(self):
        self.op_type = "rsqrt"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [10, 12]).astype(self.dtype) * 10
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.0005)


class TestAbs(TestActivation):
    def setUp(self):
        self.op_type = "abs"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [4, 25]).astype(self.dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is inaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestCeil(TestActivation):
    def setUp(self):
        self.op_type = "ceil"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.ceil(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    # The same reason with TestFloor
    def test_check_grad(self):
        pass


class TestFloor(TestActivation):
    def setUp(self):
        self.op_type = "floor"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.floor(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    # the gradient on floor, ceil, round is undefined.
    # we return zero as gradient, but the numpy return nan
    # The same reason with TestFloor
    def test_check_grad(self):
        pass


class TestCos(TestActivation):
    def setUp(self):
        self.op_type = "cos"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.cos(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestTan(TestActivation):
    def setUp(self):
        np.random.seed(1024)
        self.op_type = "tan"
        self.init_dtype()
        self.dtype = 'float32'
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

        out = np.tan(self.x_np)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x_np)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out_test = paddle.tan(x)
        out_ref = np.tan(self.x_np)
        self.assertTrue(np.allclose(out_ref, out_test.numpy()))
        paddle.enable_static()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [10, 12], self.dtype)
            out = paddle.tan(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = np.tan(self.x_np)
        self.assertTrue(np.allclose(out_ref, res[0]))

    def test_backward(self):
        test_data_shape = [11, 17]
        with fluid.dygraph.guard():
            input_x = np.random.uniform(0.1, 1,
                                        test_data_shape).astype("float32")
            var = paddle.to_tensor(input_x)
            var.stop_gradient = False
            loss = paddle.tan(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, input_x.shape)


class TestAcos(TestActivation):
    def setUp(self):
        self.op_type = "acos"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-0.95, 0.95, [10, 12]).astype(self.dtype)
        out = np.arccos(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSin(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "sin"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestAsin(TestActivation):
    def setUp(self):
        self.op_type = "asin"
        self.init_dtype()

        np.random.seed(2048)
        x = np.random.uniform(-0.95, 0.95, [10, 12]).astype(self.dtype)
        out = np.arcsin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestRound(TestActivation):
    def setUp(self):
        self.op_type = "round"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.round(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        pass


class TestRelu(TestActivation):
    def setUp(self):
        self.op_type = "relu"
        self.init_dtype()

        np.random.seed(1024)
        if self.dtype == np.uint16:
            x = np.random.uniform(-1, 1, [11, 17]).astype(np.float32)
            # The same reason with TestAbs
            x[np.abs(x) < 0.005] = 0.02
            out = convert_float_to_uint16(np.maximum(x, 0))
            self.inputs = {'X': convert_float_to_uint16(x)}
        else:
            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            # The same reason with TestAbs
            x[np.abs(x) < 0.005] = 0.02
            out = np.maximum(x, 0)
            self.inputs = {'X': x}

        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestReluAPI(unittest.TestCase):
    # test paddle.nn.ReLU, paddle.nn.functional.relu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.executed_api()

    def executed_api(self):
        self.relu = F.relu

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out1 = self.relu(x)
            m = paddle.nn.ReLU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = np.maximum(self.x_np, 0)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.ReLU()
        out1 = m(x)
        out2 = self.relu(x)
        out_ref = np.maximum(self.x_np, 0)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, self.relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[10, 12], dtype='int32')
            self.assertRaises(TypeError, self.relu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[10, 12], dtype='float16')
            self.relu(x_fp16)


class TestReluInplaceAPI(TestReluAPI):
    # test paddle.nn.functional.relu_
    def executed_api(self):
        self.relu = F.relu_


def ref_leaky_relu(x, alpha=0.01):
    out = np.copy(x)
    out[out < 0] *= alpha
    return out


class TestLeakyRelu(TestActivation):
    def get_alpha(self):
        return 0.02

    def setUp(self):
        self.op_type = "leaky_relu"
        self.init_dtype()
        alpha = self.get_alpha()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.05
        out = ref_leaky_relu(x, alpha)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'alpha': alpha}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestLeakyReluAlpha1(TestLeakyRelu):
    def get_alpha(self):
        return 2


class TestLeakyReluAlpha2(TestLeakyRelu):
    def get_alpha(self):
        return -0.01


class TestLeakyReluAlpha3(TestLeakyRelu):
    def get_alpha(self):
        return -2.0


class TestLeakyReluAPI(unittest.TestCase):
    # test paddle.nn.LeakyReLU, paddle.nn.functional.leaky_relu,
    # fluid.layers.leaky_relu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out1 = F.leaky_relu(x)
            m = paddle.nn.LeakyReLU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_leaky_relu(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.leaky_relu(x)
        m = paddle.nn.LeakyReLU()
        out2 = m(x)
        out_ref = ref_leaky_relu(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = F.leaky_relu(x, 0.6)
        m = paddle.nn.LeakyReLU(0.6)
        out2 = m(x)
        out_ref = ref_leaky_relu(self.x_np, 0.6)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', [10, 12])
            out = fluid.layers.leaky_relu(x, 0.01)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_leaky_relu(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.leaky_relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.leaky_relu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.leaky_relu(x_fp16)


def gelu(x, approximate):
    if approximate:
        y_ref = 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


class TestGeluApproximate(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.init_dtype()
        approximate = True
        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {"approximate": approximate}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestGelu(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.init_dtype()
        approximate = False
        np.random.seed(2048)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {"approximate": approximate}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestGELUAPI(unittest.TestCase):
    # test paddle.nn.GELU, paddle.nn.functional.gelu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [11, 17])
            out1 = F.gelu(x)
            m = paddle.nn.GELU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = gelu(self.x_np, False)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.gelu(x)
        m = paddle.nn.GELU()
        out2 = m(x)
        out_ref = gelu(self.x_np, False)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = F.gelu(x, True)
        m = paddle.nn.GELU(True)
        out2 = m(x)
        out_ref = gelu(self.x_np, True)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.gelu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[11, 17], dtype='int32')
            self.assertRaises(TypeError, F.gelu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[11, 17], dtype='float16')
            F.gelu(x_fp16)


class TestBRelu(TestActivation):
    def setUp(self):
        self.op_type = "brelu"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-5, 10, [10, 12]).astype(self.dtype)
        t_min = 1.0
        t_max = 4.0
        # The same with TestAbs
        x[np.abs(x - t_min) < 0.005] = t_min + 0.02
        x[np.abs(x - t_max) < 0.005] = t_max + 0.02
        t = np.copy(x)
        t[t < t_min] = t_min
        t[t > t_max] = t_max

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'t_min': t_min, 't_max': t_max}
        self.outputs = {'Out': t}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestBreluAPI(unittest.TestCase):
    # test paddle.fluid.layers.brelu
    def setUp(self):
        np.random.seed(1024)
        self.t_min = 0.
        self.t_max = 24.
        self.x_np = np.random.uniform(-1, 30, [10, 12]).astype('float32')
        self.out_ref = np.copy(self.x_np)
        self.out_ref[self.out_ref < self.t_min] = self.t_min
        self.out_ref[self.out_ref > self.t_max] = self.t_max
        self.out_ref = self.out_ref.astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_fluid_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [10, 12])
            out = paddle.fluid.layers.brelu(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
            self.assertTrue(np.allclose(self.out_ref, res[0]))

            paddle.disable_static(self.place)
            x = paddle.to_tensor(self.x_np)
            out = paddle.fluid.layers.brelu(x)
            self.assertTrue(np.allclose(self.out_ref, out.numpy()))
            paddle.enable_static()

    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.brelu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.brelu, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.layers.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.brelu(x_fp16)


def ref_relu6(x, threshold=6.0):
    out = np.copy(x)
    out[np.abs(x - threshold) < 0.005] = threshold + 0.02
    out = np.minimum(np.maximum(x, 0), threshold)
    return out


class TestRelu6(TestActivation):
    def setUp(self):
        self.op_type = "relu6"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 10, [10, 12]).astype(self.dtype)
        x[np.abs(x) < 0.005] = 0.02
        out = ref_relu6(x)

        self.inputs = {'X': x}
        self.attrs = {'threshold': 6.0}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestRelu6API(unittest.TestCase):
    # test paddle.nn.ReLU6, paddle.nn.functional.relu6
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 10, [10, 12]).astype(np.float64)
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.relu6(x)
            relu6 = paddle.nn.ReLU6()
            out2 = relu6(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_relu6(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.relu6(x)
        relu6 = paddle.nn.ReLU6()
        out2 = relu6(x)
        out_ref = ref_relu6(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.relu6(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_relu6(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.relu6, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.relu6, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.relu6(x_fp16)


def ref_hardswish(x, threshold=6.0, scale=6.0, offset=3.0):
    return (x * np.minimum(np.maximum(x + offset, 0.), threshold) /
            scale).astype(x.dtype)


class TestHardSwish(TestActivation):
    def setUp(self):
        self.op_type = 'hard_swish'
        self.init_dtype()

        from op_test import skip_check_grad_ci
        skip_check_grad_ci(reason="not implemented yet")

        np.random.seed(1024)
        x = np.random.uniform(-6, 6, [10, 12]).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        #the same with TestAbs
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02
        out = ref_hardswish(x, threshold, scale, offset)

        self.inputs = {'X': x}
        self.attrs = {'threshold': threshold, 'scale': scale, 'offset': offset}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        return  # not implemented yet
        self.check_grad(['X'], 'Out')


class TestHardswishAPI(unittest.TestCase):
    # test paddle.nn.Hardswish, paddle.nn.functional.hardswish
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.hardswish(x)
            m = paddle.nn.Hardswish()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardswish(self.x_np)
        for r in res:
            self.assertTrue(np.allclose(out_ref, r))

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.hardswish(x)
        m = paddle.nn.Hardswish()
        out2 = m(x)
        out_ref = ref_hardswish(self.x_np)
        for r in [out1, out2]:
            self.assertTrue(np.allclose(out_ref, r.numpy()))
        paddle.enable_static()

    def test_fluid_api(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.hard_swish(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_hardswish(self.x_np)
        self.assertTrue(np.allclose(out_ref, res[0]))

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.fluid.layers.hard_swish(x)
        self.assertTrue(np.allclose(out_ref, out.numpy()))
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.hardswish, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.hardswish, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.hardswish(x_fp16)


class TestSoftRelu(TestActivation):
    def setUp(self):
        self.op_type = "soft_relu"
        self.init_dtype()

        np.random.seed(4096)
        x = np.random.uniform(-3, 3, [4, 4]).astype(self.dtype)
        threshold = 2.0
        # The same reason with TestAbs
        x[np.abs(x - threshold) < 0.005] = threshold + 0.02
        x[np.abs(x + threshold) < 0.005] = -threshold - 0.02
        t = np.copy(x)
        t[t < -threshold] = -threshold
        t[t > threshold] = threshold
        out = np.log((np.exp(t) + 1))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'threshold': threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)


class TestSoftReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.soft_relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.soft_relu, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.soft_relu(x_fp16)


def elu(x, alpha):
    out_ref = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))
    return out_ref.astype(x.dtype)


class TestELU(TestActivation):
    def setUp(self):
        self.op_type = "elu"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-3, 3, [10, 12]).astype(self.dtype)
        alpha = 1.
        out = elu(x, alpha)
        # Note: unlike other Relu extensions, point 0 on standard ELU function (i.e. alpha = 1)
        # is differentiable, so we can skip modifications like x[np.abs(x) < 0.005] = 0.02 here
        self.inputs = {'X': x}
        self.attrs = {'alpha': alpha}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestELUAPI(unittest.TestCase):
    # test paddle.nn.ELU, paddle.nn.functional.elu
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-3, 3, [10, 12]).astype('float32')
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()
        self.executed_api()

    def executed_api(self):
        self.elu = F.elu

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out1 = self.elu(x)
            m = paddle.nn.ELU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = elu(self.x_np, 1.0)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = self.elu(x)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.ELU()
        out2 = m(x)
        out_ref = elu(self.x_np, 1.0)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = self.elu(x, 0.2)
        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.ELU(0.2)
        out2 = m(x)
        out_ref = elu(self.x_np, 0.2)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, self.elu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[10, 12], dtype='int32')
            self.assertRaises(TypeError, self.elu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[10, 12], dtype='float16')
            self.elu(x_fp16)


class TestELUInplaceAPI(TestELUAPI):
    # test paddle.nn.functional.elu_
    def executed_api(self):
        self.elu = F.elu_


class TestReciprocal(TestActivation):
    def setUp(self):
        self.op_type = "reciprocal"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.reciprocal(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


class TestLog(TestActivation):
    def setUp(self):
        self.op_type = "log"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_error(self):
        in1 = fluid.layers.data(
            name="in1", shape=[11, 17], append_batch_size=False, dtype="int32")
        in2 = fluid.layers.data(
            name="in2", shape=[11, 17], append_batch_size=False, dtype="int64")

        self.assertRaises(TypeError, fluid.layers.log, in1)
        self.assertRaises(TypeError, fluid.layers.log, in2)


class TestLog2(TestActivation):
    def setUp(self):
        self.op_type = "log2"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log2(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_error(self):
        in1 = paddle.static.data(name="in1", shape=[11, 17], dtype="int32")
        in2 = paddle.static.data(name="in2", shape=[11, 17], dtype="int64")

        self.assertRaises(TypeError, paddle.log2, in1)
        self.assertRaises(TypeError, paddle.log2, in2)

    def test_api(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = paddle.static.data(
                name="data_x", shape=[11, 17], dtype="float64")

            out1 = paddle.log2(data_x)
            exe = paddle.static.Executor(place=fluid.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            res1 = exe.run(paddle.static.default_main_program(),
                           feed={"data_x": input_x},
                           fetch_list=[out1])
        expected_res = np.log2(input_x)
        self.assertTrue(np.allclose(res1, expected_res))

        # dygraph
        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = paddle.to_tensor(np_x)
            z = paddle.log2(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log2(np_x))
        self.assertTrue(np.allclose(np_z, z_expected))


class TestLog10(TestActivation):
    def setUp(self):
        self.op_type = "log10"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log10(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_error(self):
        in1 = paddle.static.data(name="in1", shape=[11, 17], dtype="int32")
        in2 = paddle.static.data(name="in2", shape=[11, 17], dtype="int64")

        self.assertRaises(TypeError, paddle.log10, in1)
        self.assertRaises(TypeError, paddle.log10, in2)

    def test_api(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = paddle.static.data(
                name="data_x", shape=[11, 17], dtype="float64")

            out1 = paddle.log10(data_x)
            exe = paddle.static.Executor(place=paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            res1 = exe.run(paddle.static.default_main_program(),
                           feed={"data_x": input_x},
                           fetch_list=[out1])
        expected_res = np.log10(input_x)
        self.assertTrue(np.allclose(res1, expected_res))

        # dygraph
        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = paddle.to_tensor(np_x)
            z = paddle.log10(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log10(np_x))
        self.assertTrue(np.allclose(np_z, z_expected))


class TestLog1p(TestActivation):
    def setUp(self):
        self.op_type = "log1p"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log1p(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_api(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = fluid.layers.data(
                name="data_x",
                shape=[11, 17],
                append_batch_size=False,
                dtype="float64")

            out1 = paddle.log1p(data_x)
            exe = fluid.Executor(place=fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            res1 = exe.run(fluid.default_main_program(),
                           feed={"data_x": input_x},
                           fetch_list=[out1])
        expected_res = np.log1p(input_x)
        self.assertTrue(np.allclose(res1, expected_res))

        # dygraph
        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = fluid.dygraph.to_variable(np_x)
            z = paddle.log1p(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log1p(np_x))
        self.assertTrue(np.allclose(np_z, z_expected))


class TestSquare(TestActivation):
    def setUp(self):
        self.op_type = "square"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.square(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestPow(TestActivation):
    def setUp(self):
        self.op_type = "pow"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'factor': 3.0}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestPow_factor_tensor(TestActivation):
    def setUp(self):
        self.op_type = "pow"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'FactorTensor': np.array([3.0]).astype("float32")
        }

        self.attrs = {}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_api(self):
        input = np.random.uniform(1, 2, [11, 17]).astype("float32")
        x = fluid.layers.data(
            name="x", shape=[11, 17], append_batch_size=False, dtype="float32")
        res = fluid.layers.data(
            name="res",
            shape=[11, 17],
            append_batch_size=False,
            dtype="float32")

        factor_1 = 2.0
        factor_2 = fluid.layers.fill_constant([1], "float32", 3.0)
        out_1 = fluid.layers.pow(x, factor=factor_1)
        out_2 = fluid.layers.pow(x, factor=factor_2)
        out_4 = paddle.pow(x, factor_1, name='pow_res')
        out_6 = paddle.pow(x, factor_2)
        self.assertEqual(('pow_res' in out_4.name), True)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res, res_6 = exe.run(
            fluid.default_main_program(),
            feed={"x": input},
            fetch_list=[out_1, out_2, res, out_6])

        assert np.allclose(res_1, np.power(input, 2))
        assert np.allclose(res_2, np.power(input, 3))
        assert np.allclose(res_6, np.power(input, 3))

    def test_error(self):
        in1 = fluid.layers.data(
            name="in1", shape=[11, 17], append_batch_size=False, dtype="int32")
        in2 = fluid.layers.data(
            name="in2", shape=[11, 17], append_batch_size=False, dtype="int64")
        in3 = fluid.layers.data(
            name="in3",
            shape=[11, 17],
            append_batch_size=False,
            dtype="float32")
        in4 = fluid.layers.data(
            name="in4",
            shape=[11, 17],
            append_batch_size=False,
            dtype="float64")

        factor_1 = fluid.layers.fill_constant([1], "float64", 3.0)

        self.assertRaises(TypeError, fluid.layers.pow, x=in1, factor=factor_1)
        self.assertRaises(TypeError, fluid.layers.pow, x=in2, factor=factor_1)
        self.assertRaises(TypeError, fluid.layers.pow, x=in3, factor=factor_1)
        self.assertRaises(TypeError, fluid.layers.pow, x=in4, factor=factor_1)


def ref_stanh(x, scale_a=0.67, scale_b=1.7159):
    out = scale_b * np.tanh(x * scale_a)
    return out


class TestSTanh(TestActivation):
    def get_scale_a(self):
        return 0.67

    def get_scale_b(self):
        return 1.7159

    def setUp(self):
        self.op_type = "stanh"
        self.init_dtype()
        scale_a = self.get_scale_a()
        scale_b = self.get_scale_b()

        np.random.seed(1024)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        out = ref_stanh(x, scale_a, scale_b)

        self.inputs = {'X': x}
        self.attrs = {'scale_a': scale_a, 'scale_b': scale_b}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSTanhScaleA(TestSTanh):
    def get_scale_a(self):
        return 2.0


class TestSTanhScaleB(TestSTanh):
    def get_scale_b(self):
        return 0.5


class TestSTanhAPI(unittest.TestCase):
    # test paddle.nn.stanh
    def get_scale_a(self):
        return 0.67

    def get_scale_b(self):
        return 1.7159

    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.scale_a = self.get_scale_a()
        self.scale_b = self.get_scale_b()
        self.place=paddle.CUDAPlace(0) if core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out = paddle.stanh(x, self.scale_a, self.scale_b)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_stanh(self.x_np, self.scale_a, self.scale_b)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.stanh(x, self.scale_a, self.scale_b)
        out_ref = ref_stanh(self.x_np, self.scale_a, self.scale_b)
        for r in [out]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', [10, 12])
            out = fluid.layers.stanh(x, self.scale_a, self.scale_b)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_stanh(self.x_np, self.scale_a, self.scale_b)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, paddle.stanh, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, paddle.stanh, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            paddle.stanh(x_fp16)


class TestSTanhAPIScaleA(TestSTanhAPI):
    def get_scale_a(self):
        return 2.0


class TestSTanhAPIScaleB(TestSTanhAPI):
    def get_scale_b(self):
        return 0.5


def ref_softplus(x, beta=1, threshold=20):
    x_beta = beta * x
    out = np.select([x_beta <= threshold, x_beta > threshold],
                    [np.log(1 + np.exp(x_beta)) / beta, x])
    return out


class TestSoftplus(TestActivation):
    def setUp(self):
        self.op_type = "softplus"
        self.init_dtype()

        beta = 2
        threshold = 15

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = ref_softplus(x, beta, threshold)
        self.inputs = {'X': x}
        self.attrs = {'beta': beta, "threshold": threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSoftplusAPI(unittest.TestCase):
    # test paddle.nn.Softplus, paddle.nn.functional.softplus
    def setUp(self):
        self.beta = 2
        self.threshold = 15
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.softplus(x, self.beta, self.threshold)
            softplus = paddle.nn.Softplus(self.beta, self.threshold)
            out2 = softplus(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_softplus(self.x_np, self.beta, self.threshold)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.softplus(x, self.beta, self.threshold)
        softplus = paddle.nn.Softplus(self.beta, self.threshold)
        out2 = softplus(x)
        out_ref = ref_softplus(self.x_np, self.beta, self.threshold)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.softplus(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_softplus(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.softplus, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.softplus, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.softplus(x_fp16)


def ref_softsign(x):
    out = np.divide(x, 1 + np.abs(x))
    return out


class TestSoftsign(TestActivation):
    def setUp(self):
        self.op_type = "softsign"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = ref_softsign(x)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSoftsignAPI(unittest.TestCase):
    # test paddle.nn.Softsign, paddle.nn.functional.softsign
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.softsign(x)
            softsign = paddle.nn.Softsign()
            out2 = softsign(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_softsign(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.softsign(x)
        softsign = paddle.nn.Softsign()
        out2 = softsign(x)
        out_ref = ref_softsign(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.softsign(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_softsign(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.softsign, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.softsign, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.softsign(x_fp16)


def ref_thresholded_relu(x, threshold=1.0):
    out = (x > threshold) * x
    return out


class TestThresholdedRelu(TestActivation):
    def setUp(self):
        self.op_type = "thresholded_relu"
        self.init_dtype()

        threshold = 15

        np.random.seed(1024)
        x = np.random.uniform(-20, 20, [10, 12]).astype(self.dtype)
        x[np.abs(x) < 0.005] = 0.02
        out = ref_thresholded_relu(x, threshold)
        self.inputs = {'X': x}
        self.attrs = {"threshold": threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestThresholdedReluAPI(unittest.TestCase):
    # test paddle.nn.ThresholdedReLU, paddle.nn.functional.thresholded_relu
    def setUp(self):
        self.threshold = 15
        np.random.seed(1024)
        self.x_np = np.random.uniform(-20, 20, [10, 12]).astype(np.float64)
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.thresholded_relu(x, self.threshold)
            thresholded_relu = paddle.nn.ThresholdedReLU(self.threshold)
            out2 = thresholded_relu(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_thresholded_relu(self.x_np, self.threshold)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.thresholded_relu(x, self.threshold)
        thresholded_relu = paddle.nn.ThresholdedReLU(self.threshold)
        out2 = thresholded_relu(x)
        out_ref = ref_thresholded_relu(self.x_np, self.threshold)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.thresholded_relu(x, self.threshold)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_thresholded_relu(self.x_np, self.threshold)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.thresholded_relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.thresholded_relu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.thresholded_relu(x_fp16)


def ref_hardsigmoid(x, slope=0.166666666666667, offset=0.5):
    return np.maximum(np.minimum(x * slope + offset, 1.), 0.).astype(x.dtype)


class TestHardSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "hard_sigmoid"
        self.dtype = 'float64'
        self.slope = 0.166666666666667
        self.offset = 0.5
        self.set_attrs()

        x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
        lower_threshold = -self.offset / self.slope
        upper_threshold = (1. - self.offset) / self.slope

        # Same reason as TestAbs
        delta = 0.005
        x[np.abs(x - lower_threshold) < delta] = lower_threshold - 0.02
        x[np.abs(x - upper_threshold) < delta] = upper_threshold - 0.02

        out = ref_hardsigmoid(x, self.slope, self.offset)

        self.attrs = {'slope': self.slope, 'offset': self.offset}
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def set_attrs(self):
        pass


class TestHardSigmoidFP32(TestHardSigmoid):
    def set_attrs(self):
        self.dtype = 'float32'


class TestHardSigmoidSlopeOffset(TestHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.4


class TestHardsigmoidAPI(unittest.TestCase):
    # test paddle.nn.Hardsigmoid, paddle.nn.functional.hardsigmoid
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.hardsigmoid(x)
            m = paddle.nn.Hardsigmoid()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardsigmoid(self.x_np)
        for r in res:
            self.assertTrue(np.allclose(out_ref, r))

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.hardsigmoid(x)
        m = paddle.nn.Hardsigmoid()
        out2 = m(x)
        out_ref = ref_hardsigmoid(self.x_np)
        for r in [out1, out2]:
            self.assertTrue(np.allclose(out_ref, r.numpy()))
        paddle.enable_static()

    def test_fluid_api(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.hard_sigmoid(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_hardsigmoid(self.x_np, 0.2, 0.5)
        self.assertTrue(np.allclose(out_ref, res[0]))

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.fluid.layers.hard_sigmoid(x)
        self.assertTrue(np.allclose(out_ref, out.numpy()))
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.hardsigmoid, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.hardsigmoid, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.hardsigmoid(x_fp16)


def ref_swish(x):
    out = x * expit(x)
    return out


class TestSwish(TestActivation):
    def setUp(self):
        self.op_type = "swish"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = ref_swish(x)
        self.inputs = {'X': x}
        self.attrs = {'beta': 1.0}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSwishAPI(unittest.TestCase):
    # test paddle.nn.Swish, paddle.nn.functional.swish
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float64)
        self.place=paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.swish(x)
            swish = paddle.nn.Swish()
            out2 = swish(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_swish(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.swish(x)
        swish = paddle.nn.Swish()
        out2 = swish(x)
        out_ref = ref_swish(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.swish(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_swish(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.swish, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.swish, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.swish(x_fp16)


#------------------ Test Error Activation----------------------
def create_test_error_class(op_type):
    class TestOpErrors(unittest.TestCase):
        def test_errors(self):
            with program_guard(Program(), Program()):
                op = getattr(fluid.layers, op_type)
                # The input dtype of op_type must be float32, float64.
                in1 = fluid.layers.data(
                    name='input2', shape=[12, 10], dtype="int32")
                in2 = fluid.layers.data(
                    name='input3', shape=[12, 10], dtype="int64")
                self.assertRaises(TypeError, op, in1)
                self.assertRaises(TypeError, op, in2)

    cls_name = "{0}_{1}".format(op_type, "test_errors")
    TestOpErrors.__name__ = cls_name
    globals()[cls_name] = TestOpErrors


create_test_error_class('acos')
create_test_error_class('asin')
create_test_error_class('atan')
create_test_error_class('ceil')
create_test_error_class('cos')
create_test_error_class('floor')
create_test_error_class('reciprocal')
create_test_error_class('round')
create_test_error_class('rsqrt')
create_test_error_class('sin')
create_test_error_class('sqrt')
create_test_error_class('tanh')
create_test_error_class('tan')


#------------------ Test Cudnn Activation----------------------
def create_test_act_cudnn_class(parent, atol=1e-3, grad_atol=1e-3):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestActCudnn(parent):
        def init_kernel_type(self):
            self.attrs = {"use_cudnn": True}

    cls_name = "{0}_{1}".format(parent.__name__, "cudnn")
    TestActCudnn.__name__ = cls_name
    globals()[cls_name] = TestActCudnn


create_test_act_cudnn_class(TestRelu)
create_test_act_cudnn_class(TestRelu6)
create_test_act_cudnn_class(TestSigmoid)
create_test_act_cudnn_class(TestTanh)


#------------------ Test Fp16 ----------------------
def create_test_act_fp16_class(parent,
                               atol=1e-3,
                               grad_check=True,
                               grad_atol=0.80):
    @unittest.skipIf(not paddle.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestActFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16:
                self.check_output_with_place(place, atol=atol)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16 and grad_check:
                self.check_grad_with_place(
                    place, ['X'], 'Out', max_relative_error=grad_atol)

    cls_name = "{0}_{1}".format(parent.__name__, "fp16")
    TestActFp16.__name__ = cls_name
    globals()[cls_name] = TestActFp16


create_test_act_fp16_class(TestActivation)
create_test_act_fp16_class(TestExpm1)
create_test_act_fp16_class(TestSigmoid)
create_test_act_fp16_class(TestSilu)
create_test_act_fp16_class(TestLogSigmoid)
create_test_act_fp16_class(TestTanh)
create_test_act_fp16_class(TestTanhshrink)
create_test_act_fp16_class(TestHardShrink)
create_test_act_fp16_class(TestSoftshrink)
create_test_act_fp16_class(TestSqrt)
create_test_act_fp16_class(TestAbs)
create_test_act_fp16_class(TestCeil, grad_check=False)
create_test_act_fp16_class(TestFloor, grad_check=False)
create_test_act_fp16_class(TestCos, grad_atol=0.85)
create_test_act_fp16_class(TestTan, grad_atol=0.85)
create_test_act_fp16_class(TestCosh, grad_atol=0.85)
create_test_act_fp16_class(TestAcos, grad_atol=0.85)
create_test_act_fp16_class(TestSin)
create_test_act_fp16_class(TestSinh)
create_test_act_fp16_class(TestAsin)
create_test_act_fp16_class(TestAtan)
create_test_act_fp16_class(TestRound, grad_check=False)
create_test_act_fp16_class(TestRelu)
create_test_act_fp16_class(TestGelu)
create_test_act_fp16_class(TestBRelu)
create_test_act_fp16_class(TestRelu6)
create_test_act_fp16_class(TestSoftRelu, grad_atol=0.85)
create_test_act_fp16_class(TestELU)
create_test_act_fp16_class(TestReciprocal)
create_test_act_fp16_class(TestLog)
if core.is_compiled_with_rocm():
    create_test_act_fp16_class(TestLog2, atol=5e-2, grad_atol=0.85)
else:
    create_test_act_fp16_class(TestLog2, atol=5e-2)
create_test_act_fp16_class(TestLog10, atol=5e-2)
create_test_act_fp16_class(TestLog1p, grad_atol=0.9)
create_test_act_fp16_class(TestSquare)
create_test_act_fp16_class(TestPow, atol=5e-2)
create_test_act_fp16_class(TestPow_factor_tensor, atol=5e-2)
create_test_act_fp16_class(TestSTanh, grad_atol=0.9)
create_test_act_fp16_class(TestSoftplus)
create_test_act_fp16_class(TestSoftsign)
create_test_act_fp16_class(TestThresholdedRelu)
create_test_act_fp16_class(TestHardSigmoid)
create_test_act_fp16_class(TestSwish, grad_atol=0.85)
create_test_act_fp16_class(TestHardSwish)


def create_test_act_bf16_class(parent,
                               atol=1e-2,
                               grad_check=True,
                               grad_atol=0.80):
    @unittest.skipIf(not paddle.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestActBF16(parent):
        def init_dtype(self):
            self.dtype = np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=atol)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['X'], 'Out', max_relative_error=grad_atol)

    cls_name = "{0}_{1}".format(parent.__name__, "bf16")
    TestActBF16.__name__ = cls_name
    globals()[cls_name] = TestActBF16


create_test_act_bf16_class(TestRelu)

if __name__ == "__main__":
    unittest.main()
