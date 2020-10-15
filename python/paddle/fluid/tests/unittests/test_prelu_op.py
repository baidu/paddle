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
import paddle.fluid as fluid
import six
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.nn.functional as F


def ref_prelu(x, weight):
    x_t = x.copy()
    weight = weight.reshape(1, -1, 1, 1)
    neg_indices = x <= 0
    assert x.shape == neg_indices.shape
    x_t[neg_indices] = (x_t * weight)[neg_indices]
    return (x_t, )


def ref_prelu_nn(x, num_parameters, init):
    weight_np = np.full((num_parameters), init)
    return ref_prelu(x, weight_np)


class TestFunctionalPReluAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        self.x_np = np.random.uniform(-1., 1., [1, 2, 3, 4]).astype('float32')
        self.weight_np_0 = np.random.randn(1).astype('float32')
        self.weight_np_1 = np.random.randn(self.x_np.shape[1]).astype('float32')

    def static_check(self, weight_np):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, 'float32')
            weight = paddle.fluid.data('Alpha', weight_np.shape, 'float32')
            out = F.prelu(x, weight)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np,
                                'Alpha': weight_np},
                          fetch_list=[out])
        out_ref = ref_prelu(self.x_np, weight_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def dygraph_check(self, weight_np):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        weight = paddle.to_tensor(weight_np)
        out = F.prelu(x, weight)
        out_ref = ref_prelu(self.x_np, weight_np)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)
        paddle.enable_static()

    def test_static_api(self):
        self.static_check(self.weight_np_0)
        self.static_check(self.weight_np_1)

    def test_dygraph_api(self):
        self.dygraph_check(self.weight_np_0)
        self.dygraph_check(self.weight_np_1)

    def test_error(self):
        with paddle.static.program_guard(paddle.static.Program()):
            weight_fp32 = paddle.fluid.data(
                name='weight_fp32', shape=[1], dtype='float32')
            # The input type must be Variable.
            self.assertRaises(TypeError, F.prelu, x=1, weight=weight_fp32)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(name='x_int32', shape=[2, 3], dtype='int32')
            self.assertRaises(TypeError, F.prelu, x=x_int32, weight=weight_fp32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(name='x_fp16', shape=[2, 3], dtype='float16')
            F.prelu(x=x_fp16, weight=weight_fp32)


class TestNNPReluAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        self.x_np = np.ones([1, 2, 3, 4]).astype('float32')

    def test_static_api(self):
        startup_program = paddle.static.Program()
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            x = paddle.fluid.data(name='X', shape=self.x_np.shape, dtype='float32')
            m = paddle.nn.PReLU()
            out = m(x)
            exe = paddle.static.Executor(self.place)
            exe.run(startup_program)
            res = exe.run(train_program,
                          feed={'X': self.x_np},
                          fetch_list=[out])
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU()
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(num_parameters=self.x_np.shape[1])
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, self.x_np.shape[1], 0.25)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(init=0.5)
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(weight_attr=fluid.ParamAttr(name="weight"))
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.25)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        x = paddle.to_tensor(self.x_np)
        m = paddle.nn.PReLU(weight_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(0.5)))
        out = m(x)
        out_ref = ref_prelu_nn(self.x_np, 1, 0.5)
        self.assertEqual(np.allclose(out_ref, out.numpy()), True)

        paddle.enable_static()


class PReluTest(OpTest):
    def setUp(self):
        self.init_input_shape()
        self.init_attr()
        self.op_type = "prelu"

        x_np = np.random.uniform(-1, 1, self.x_shape)
        # Since zero point in prelu is not differentiable, avoid randomize
        # zero.
        x_np[np.abs(x_np) < 0.005] = 0.02

        if self.attrs == {'mode': "all"}:
            alpha_np = np.random.uniform(-1, -0.5, (1))
        elif self.attrs == {'mode': "channel"}:
            alpha_np = np.random.uniform(-1, -0.5, [1, self.x_shape[1], 1, 1])
        else:
            alpha_np = np.random.uniform(-1, -0.5, [1] + self.x_shape[1:])

        self.inputs = {'X': x_np, 'Alpha': alpha_np}

        # NOTE(zhiqu): reshape inputs['Alpha'] from [1, 100, 1, 1] to [1, 100] + [1]*len(x.shape[2:])
        # since np operands could not be broadcast together with shapes (1,100,2,2,2,3) (1,100,1,1) 	
        reshaped_alpha = self.inputs['Alpha']
        if self.attrs == {'mode': "channel"}:
            reshaped_alpha = np.reshape(
                self.inputs['Alpha'],
                [1, self.x_shape[1]] + [1] * len(self.x_shape[2:]))

        out_np = np.maximum(self.inputs['X'], 0.)
        out_np = out_np + np.minimum(self.inputs['X'], 0.) * reshaped_alpha
        assert out_np is not self.inputs['X']
        self.outputs = {'Out': out_np}

    def init_input_shape(self):
        self.x_shape = [2, 100, 3, 4]

    def init_attr(self):
        self.attrs = {'mode': "channel"}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Alpha'], 'Out')


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAll(PReluTest):
    def init_input_shape(self):
        self.x_shape = [2, 3, 4, 5]

    def init_attr(self):
        self.attrs = {'mode': "all"}


class TestModeElt(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 2, 5, 10]

    def init_attr(self):
        self.attrs = {'mode': "element"}


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAllRank3(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        self.attrs = {'mode': "all"}


@skip_check_grad_ci(
    reason="[skip shape check] Input(Alpha) must be 1-D and only has one data in 'all' mode"
)
class TestModeAllRank6(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 2, 3, 4, 5, 6]

    def init_attr(self):
        self.attrs = {'mode': "all"}


class TestModeChannelRank3(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 200, 3]

    def init_attr(self):
        self.attrs = {'mode': "channel"}


class TestModeChannelRank6(PReluTest):
    def init_input_shape(self):
        self.x_shape = [1, 100, 2, 2, 2, 2]

    def init_attr(self):
        self.attrs = {'mode': "channel"}


class TestModeElementRank3(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 10, 10]

    def init_attr(self):
        self.attrs = {'mode': "element"}


class TestModeElementRank6(PReluTest):
    def init_input_shape(self):
        self.x_shape = [3, 2, 2, 4, 5, 2]

    def init_attr(self):
        self.attrs = {'mode': "element"}


def prelu_t(x, mode, param_attr=None, name=None):
    helper = fluid.layer_helper.LayerHelper('prelu', **locals())
    alpha_shape = [1, x.shape[1], 1, 1]
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype='float32',
        is_bias=False,
        default_initializer=fluid.initializer.ConstantInitializer(0.25))
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x,
                'Alpha': alpha},
        attrs={"mode": mode},
        outputs={"Out": out})
    return out


# error message test if mode is not one of 'all', 'channel', 'element'
class TestModeError(unittest.TestCase):
    def test_mode_error(self):
        main_program = Program()
        with fluid.program_guard(main_program, Program()):
            x = fluid.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, 'any')
            except Exception as e:
                assert (e.args[0].find('InvalidArgumentError') != -1)


if __name__ == "__main__":
    unittest.main()
