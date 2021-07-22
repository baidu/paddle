#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.incubate as incubate

paddle.enable_static()


class TestDropoutBiasFuseOp(OpTest):
    def setUp(self):
        self.op_type = "fused_dropout_bias"
        self.inputs = {
            'X': np.random.random((2, 128)).astype("float64"),
            'Bias': np.random.random((1, 128)).astype("float64")
        }
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'] + self.inputs['Bias'],
            'Mask': np.ones((2, 128)).astype('uint8')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Bias'], 'Out')


class TestDropoutBiasFuseOp5(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        paddle.seed(102)
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_x = fluid.data(name="x", shape=[40, 40], dtype="float32")
            input_bias = fluid.data(name="bias", shape=[1, 40], dtype="float32")
            res0 = incubate.dropout_bias_fuse(
                x=input_x,
                bias=input_bias,
                dropout_prob=0.,
                dropout_implementation='upscale_in_train')
            res1 = incubate.dropout_bias_fuse(
                x=input_x, bias=input_bias, dropout_prob=1., is_test=True)

            x_in_np = np.random.random([40, 40]).astype("float32")
            bias_in_np = np.random.random([1, 40]).astype("float32")
            res_np = x_in_np + bias_in_np
            res_np2 = np.zeros_like(x_in_np)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"x": x_in_np,
                                    "bias": bias_in_np},
                              fetch_list=[res0])
            self.assertTrue(np.allclose(fetches[0], res_np))
            fetches2 = exe.run(fluid.default_main_program(),
                               feed={"x": x_in_np,
                                     "bias": bias_in_np},
                               fetch_list=[res1])
            self.assertTrue(np.allclose(fetches2[0], res_np2))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                x_in_np = np.random.random([40, 40]).astype("float32")
                bias_in_np = np.random.random([1, 40]).astype("float32")
                res_np = x_in_np + bias_in_np
                res_np2 = np.zeros_like(x_in_np)
                input_x = fluid.dygraph.to_variable(x_in_np)
                input_bias = fluid.dygraph.to_variable(bias_in_np)

                res0 = incubate.dropout_bias_fuse(
                    x=input_x,
                    bias=input_bias,
                    dropout_prob=0.,
                    dropout_implementation='upscale_in_train',
                    is_test=None)
                res1 = incubate.dropout_bias_fuse(
                    x=input_x, bias=input_bias, dropout_prob=1., is_test=True)

            self.assertTrue(np.allclose(res0.numpy(), res_np))
            self.assertTrue(np.allclose(res1.numpy(), res_np2))


if __name__ == '__main__':
    unittest.main()
