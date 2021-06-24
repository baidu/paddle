# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import six
import paddle
import unittest
import numpy as np

from paddle.static import InputSpec


class MySub(paddle.nn.Layer):
    def __init__(self):
        super(MySub, self).__init__()

    def forward(self, x, y, name=None):
        return paddle.subtract(x, y, name)


class NetWithOpAttr(paddle.nn.Layer):
    def __init__(self, in_num, out_num):
        super(NetWithOpAttr, self).__init__()

        self.linear = paddle.nn.Linear(in_num, out_num)
        self.bn = paddle.nn.BatchNorm(out_num)
        self.sub = MySub()

    def forward(self, x):
        out = self.linear(x)
        out = self.sub(out, x)
        out = self.bn(out)
        return out

    @paddle.jit.to_static(input_spec=[InputSpec([10, 16])])
    def with_cond(self, x):
        if paddle.mean(x) > 0.:
            out = self.linear(x)
        else:
            out = self.sub(x, x)
        out = self.bn(out)
        return out


class CheckOpAttr(unittest.TestCase):
    def setUp(self):
        self.in_num = 16
        self.out_num = 16
        self.x = paddle.randn([10, self.in_num])
        self.expected_results()

    def expected_results(self):
        self.fc_attrs = {
            "int_val": 10,
            "int_vals": [10, 20],
            "float_val": 3.8,
            "float_vals": [3.8, -0.2]
        }
        self.bn_attrs = {"bool_val": True, "bool_vals": [True, False]}
        self.sub_attrs = {"int_vals": [10, 20], "bool_vals": [True, False]}

        self.infos = {
            'matmul': self.fc_attrs,
            'elementwise_add': self.fc_attrs,
            'batch_norm': self.bn_attrs,
            'tanh': self.bn_attrs,
            'elementwise_sub': self.sub_attrs
        }

    def test_set_op_attrs(self):
        net = NetWithOpAttr(self.in_num, self.out_num)
        # set attrs
        net.linear._set_op_attrs(self.fc_attrs)
        net.bn._set_op_attrs({"bool_val": False})  # test overwrite behavior
        net.bn._set_op_attrs(self.bn_attrs)
        net.sub._set_op_attrs(self.sub_attrs)
        # assert hooks exist.
        self.assertEqual(len(net.linear._forward_pre_hooks), 1)
        self.assertEqual(len(net.linear._forward_post_hooks), 1)
        # to_static
        net = paddle.jit.to_static(
            net, input_spec=[InputSpec.from_tensor(self.x)])

        # assert attrs have be set.
        self.check_op_attrs(net.forward.concrete_program.main_program)

        # assert hooks have be clean.
        self.assertEqual(len(net.linear._forward_pre_hooks), 0)
        self.assertEqual(len(net.linear._forward_post_hooks), 0)

    def check_op_attrs(self, main_program):
        for cur_block in main_program.blocks:
            ops = cur_block.ops
            for op in ops:
                if op.type not in self.infos: continue
                for attr_name, expect_vals in six.iteritems(self.infos[
                        op.type]):
                    op_vals = op.desc.attr(attr_name)
                    if not isinstance(expect_vals, list):
                        expect_vals = [expect_vals]
                        op_vals = [op_vals]

                    for (op_val, expect_val) in zip(op_vals, expect_vals):
                        if isinstance(op_val, float):
                            # C++ vs python: 3.799999952316284 ~= 3.8
                            self.assertAlmostEqual(op_val, expect_val)
                        else:
                            self.assertEqual(op_val, expect_val)

    def test_set_op_attrs_with_sub_block(self):
        net = NetWithOpAttr(self.in_num, self.out_num)
        # set attrs
        net.linear._set_op_attrs({
            "int_vals": [0, 0]
        })  # test overwrite behavior
        net.linear._set_op_attrs(self.fc_attrs)
        net.bn._set_op_attrs(self.bn_attrs)
        net.sub._set_op_attrs(self.sub_attrs)
        # assert hooks exist.
        self.assertEqual(len(net.linear._forward_pre_hooks), 1)
        self.assertEqual(len(net.linear._forward_post_hooks), 1)

        # assert attrs have be set.
        self.check_op_attrs(net.with_cond.concrete_program.main_program)

        # assert hooks have be clean.
        self.assertEqual(len(net.linear._forward_pre_hooks), 0)
        self.assertEqual(len(net.linear._forward_post_hooks), 0)

    def test_type_error(self):
        net = NetWithOpAttr(self.in_num, self.out_num)
        # attrs should be dict
        with self.assertRaises(TypeError):
            net.linear._set_op_attrs([self.fc_attrs])


if __name__ == '__main__':
    unittest.main()
