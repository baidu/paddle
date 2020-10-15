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

import contextlib
import unittest
import numpy as np
from collections import OrderedDict

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.dygraph.parallel import DataParallel
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.parallel import _coalesce_tensors, _split_tensors, _reshape_inplace


class MyLayer(fluid.Layer):
    def __init__(self, name_scope):
        super(MyLayer, self).__init__(name_scope)

    def forward(self, inputs):
        x = fluid.layers.relu(inputs)
        x = fluid.layers.elementwise_mul(x, x)
        x = fluid.layers.reduce_sum(x)
        return [x]


class TestImperativeParallelCoalesceSplit(unittest.TestCase):
    def test_coalesce_split(self):
        with fluid.dygraph.guard():
            test_layer = MyLayer("test_layer")
            strategy = core.ParallelStrategy()
            test_layer = DataParallel(test_layer, strategy)

            # test variables prepare
            vars = []
            vars.append(to_variable(np.random.random([2, 3]).astype("float32")))
            vars.append(to_variable(np.random.random([4, 9]).astype("float32")))
            vars.append(
                to_variable(np.random.random([10, 1]).astype("float32")))
            var_groups = OrderedDict()
            var_groups.setdefault(0, vars)

            # record shapes
            orig_var_shapes = []
            for var in vars:
                orig_var_shapes.append(var.shape)

            # execute interface
            coalesced_vars = _coalesce_tensors(var_groups)
            _split_tensors(coalesced_vars)

            # compare
            for orig_var_shape, var in zip(orig_var_shapes, vars):
                self.assertEqual(orig_var_shape, var.shape)

    def test_reshape_inplace(self):
        with fluid.dygraph.guard():
            test_layer = MyLayer("test_layer")
            strategy = core.ParallelStrategy()
            test_layer = DataParallel(test_layer, strategy)

            ori_shape = [2, 25]
            new_shape = [5, 10]
            x_data = np.random.random(ori_shape).astype("float32")
            x = to_variable(x_data)
            _reshape_inplace(x, new_shape)
            self.assertEqual(x.shape, new_shape)


if __name__ == '__main__':
    unittest.main()
