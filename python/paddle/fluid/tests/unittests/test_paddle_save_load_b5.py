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

    def test_save_load_selected_rows(self):
        paddle.enable_static()
        place = fluid.CPUPlace()
        height = 10
        rows = [0, 4, 7]
        row_numel = 12
        selected_rows = fluid.core.SelectedRows(rows, height)
        path = 'test_paddle_save_load_selected_rows/sr.pdsr'

        with self.assertRaises(ValueError):
            paddle.save(selected_rows, path, use_binary_format=True)

        np_array = np.random.randn(len(rows), row_numel).astype("float32")
        tensor = selected_rows.get_tensor()
        tensor.set(np_array, place)

        paddle.save(selected_rows, path, use_binary_format=True)
        load_sr = paddle.load(path)

        self.assertTrue(isinstance(load_sr, fluid.core.SelectedRows))
        self.assertTrue(list(load_sr.rows()) == rows)
        self.assertTrue(load_sr.height() == height)
        self.assertTrue(
            np.array_equal(np.array(load_sr.get_tensor()), np_array))

        with self.assertRaises(RuntimeError):
            fluid.core._save_selected_rows(
                selected_rows,
                'test_paddle_save_load_selected_rows_not_exist_file/temp')
        with self.assertRaises(RuntimeError):
            fluid.core._load_selected_rows(
                selected_rows,
                'test_paddle_save_load_selected_rows_not_exist_file/temp')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
