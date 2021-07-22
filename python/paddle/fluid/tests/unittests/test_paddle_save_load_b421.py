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

    def test_save_load_lod_tensor(self):
        paddle.enable_static()
        OUTPUT_NUM = 32
        with new_program_scope():
            x = fluid.data(name="x", shape=[None, IMAGE_SIZE], dtype='float32')
            y = fluid.layers.fc(
                x,
                OUTPUT_NUM,
                name='fc_vars', )
            prog = fluid.default_main_program()
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            prog = paddle.static.default_main_program()
            exe.run(fluid.default_startup_program())

            dirname = 'test_save_load_lod_tensor1/tensor_'
            for var in prog.list_vars():
                if var.persistable and list(
                        var.shape) == [IMAGE_SIZE, OUTPUT_NUM]:
                    tensor = var.get_value()
                    paddle.save(
                        tensor, dirname + 'fc_vars.w_0', use_binary_format=True)
                    break

            origin = np.array(var.get_value())
            var.set_value(np.zeros_like(origin))
            is_zeros = np.array(var.get_value())

            # loaded_tensor = paddle.load(dirname + 'fc_vars.w_0')
            # self.assertTrue(isinstance(loaded_tensor, fluid.core.LoDTensor))
            # self.assertTrue(
            #     list(loaded_tensor.shape()) == [IMAGE_SIZE, OUTPUT_NUM])
            # to_array = np.array(loaded_tensor)
            # self.assertTrue(np.array_equal(origin, to_array))

        with self.assertRaises(NotImplementedError):
            path = 'test_save_load_error/temp'
            paddle.save({}, path, use_binary_format=True)

        # with self.assertRaises(ValueError):
        #     path = 'test_save_load_error/temp'
        #     with open(path, "w") as f:
        #         f.write('\0')
        #     paddle.load(path)

        with self.assertRaises(ValueError):
            temp_lod = fluid.core.LoDTensor()
            paddle.save(temp_lod, path, use_binary_format=True)

        with self.assertRaises(RuntimeError):
            fluid.core._save_lod_tensor(
                temp_lod, 'test_save_load_error_not_exist_file/not_exist_file')

        with self.assertRaises(RuntimeError):
            fluid.core._load_lod_tensor(
                temp_lod, 'test_save_load_error_not_exist_file/not_exist_file')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
