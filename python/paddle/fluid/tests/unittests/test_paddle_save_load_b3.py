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

from __future__ import print_function

import unittest
import numpy as np
import os
import sys
import six

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.fluid as fluid
from paddle.fluid.optimizer import Adam
import paddle.fluid.framework as framework
from test_imperative_base import new_program_scope

IMAGE_SIZE = 784


class TestSaveLoadBinaryFormat(unittest.TestCase):
    def setUp(self):
        # enable static graph mode
        paddle.enable_static()

    def set_zero(self, prog, place, scope=None):
        if scope is None:
            scope = fluid.global_scope()
        for var in prog.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                ten = scope.find_var(var.name).get_tensor()
                if ten is not None:
                    ten.set(np.zeros_like(np.array(ten)), place)
                    new_t = np.array(scope.find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

    def replace_save_vars(self, program, dirname):
        def predicate(var):
            return var.persistable

        vars = filter(predicate, program.list_vars())
        for var in vars:
            paddle.save(
                var.get_value(),
                os.path.join(dirname, var.name),
                use_binary_format=True)

    def replace_load_vars(self, program, dirname):
        def predicate(var):
            return var.persistable

        var_list = list(filter(predicate, program.list_vars()))
        for var in var_list:
            var_load = paddle.load(os.path.join(dirname, var.name))
            # set var_load to scope
            var.set_value(var_load)

    def test_replace_save_load_vars(self):
        paddle.enable_static()
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="x", shape=[None, IMAGE_SIZE], dtype='float32')
            z = paddle.static.nn.fc(x, 10, bias_attr=False)
            z = paddle.static.nn.fc(z, 128, bias_attr=False)
            loss = fluid.layers.reduce_mean(z)
            place = fluid.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    # make sure all the paramerter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t
            # test for replace_save_vars/io.load_vars
            path_vars1 = 'test_replace_save_load_vars_binary1/model'
            self.replace_save_vars(prog, path_vars1)
            # set var to zero
            self.set_zero(prog, place)
            var_list = list(
                filter(lambda var: var.persistable, prog.list_vars()))
            fluid.io.load_vars(
                exe, path_vars1, main_program=prog, vars=var_list)

            for var in prog.list_vars():
                if var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]

                    self.assertTrue(np.array_equal(new_t, base_t))
            # test for io.save_vars/replace_load_vars
            path_vars2 = 'test_replace_save_load_vars_binary2/model/'
            fluid.io.save_vars(
                exe, path_vars2, main_program=prog, vars=var_list)
            self.set_zero(prog, place)
            self.replace_load_vars(prog, path_vars2)
            for var in prog.list_vars():
                if var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]

                    self.assertTrue(np.array_equal(new_t, base_t))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
