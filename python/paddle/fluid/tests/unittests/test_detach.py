# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph.base import to_variable

import unittest


class Test_Detach(unittest.TestCase):
    def generate_Data(self):
        data = np.array(
            [[1, 8, 3, 9], [7, 20, 9, 6], [4, 6, 8, 10]]).astype('float32')
        return data

    def no_detach_multi(self):
        data = self.generate_Data()
        with fluid.dygraph.guard():
            linear_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(5.0))
            linear_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(6.0))
            linear = Linear(
                4,
                10,
                param_attr=linear_w_param_attrs,
                bias_attr=linear_b_param_attrs)
            linear1_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(7.0))
            linear1_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(8.0))
            linear1 = Linear(
                10,
                1,
                param_attr=linear1_w_param_attrs,
                bias_attr=linear1_b_param_attrs)
            linear2_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(9.0))
            linear2_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(10.0))
            linear2 = Linear(
                10,
                1,
                param_attr=linear2_w_param_attrs,
                bias_attr=linear2_b_param_attrs)
            data = to_variable(data)
            x = linear(data)
            x1 = linear1(x)
            x2 = linear2(x)
            loss = x1 + x2
            # print(loss, loss.shape)
            loss.backward()
            return x.gradient()

    def no_detach_single(self):
        data = self.generate_Data()
        with fluid.dygraph.guard():
            linear_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(5.0))
            linear_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(6.0))
            linear = Linear(
                4,
                10,
                param_attr=linear_w_param_attrs,
                bias_attr=linear_b_param_attrs)
            linear1_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(7.0))
            linear1_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(8.0))
            linear1 = Linear(
                10,
                1,
                param_attr=linear1_w_param_attrs,
                bias_attr=linear1_b_param_attrs)
            data = to_variable(data)
            x = linear(data)
            x1 = linear1(x)
            loss = x1
            # print(loss, loss.shape)
            loss.backward()
            return x.gradient()

    def detach_multi(self):
        data = self.generate_Data()
        with fluid.dygraph.guard():
            linear_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(5.0))
            linear_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(6.0))
            linear = Linear(
                4,
                10,
                param_attr=linear_w_param_attrs,
                bias_attr=linear_b_param_attrs)
            linear1_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(7.0))
            linear1_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(8.0))
            linear1 = Linear(
                10,
                1,
                param_attr=linear1_w_param_attrs,
                bias_attr=linear1_b_param_attrs)
            linear2_w_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(9.0))
            linear2_b_param_attrs = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(10.0))
            linear2 = Linear(
                10,
                1,
                param_attr=linear2_w_param_attrs,
                bias_attr=linear2_b_param_attrs)
            data = to_variable(data)
            x = linear(data)
            x_detach = x.detach()
            x1 = linear1(x)
            x2 = linear2(x_detach)
            loss = x1 + x2
            # print(loss, loss.shape)
            loss.backward()
            return x.gradient()

    def test_NoDetachMulti_DetachMulti(self):
        array_no_detach_multi = self.no_detach_multi()
        array_detach_multi = self.detach_multi()

        assert not np.array_equal(array_no_detach_multi, array_detach_multi)

    def test_NoDetachSingle_DetachMulti(self):
        array_no_detach_single = self.no_detach_single()
        array_detach_multi = self.detach_multi()
        assert np.array_equal(array_no_detach_single, array_detach_multi)


class TestInplace(unittest.TestCase):
    def test_forward_version(self):
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
            self.assertEqual(var.inplace_version, 0)
            detach_var_1 = var.detach()
            self.assertEqual(detach_var_1.inplace_version, 0)

            var[0] = 1.1
            self.assertEqual(var.inplace_version, 1)

            detach_var_2 = var.detach()
            self.assertEqual(detach_var_2.inplace_version, 1)

            var[0] = 3
            self.assertEqual(detach_var_1.inplace_version, 2)
            self.assertEqual(detach_var_2.inplace_version, 2)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            detach_var_b = var_b.detach()
            detach_var_b[1:2] = 3.3  # var_b is modified inplace

            var_d = var_b**2

            loss = paddle.nn.functional.relu(var_c + var_d)
            with self.assertRaisesRegexp(
                    RuntimeError,
                    "received tensor_version:{} != wrapper_version_snapshot:{}".
                    format(1, 0)):
                loss.backward()


if __name__ == '__main__':
    unittest.main()
