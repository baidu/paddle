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
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from test_imperative_base import new_program_scope

import numpy as np
import pickle
import os

LARGE_PARAM = 2**26


class TestStaticSaveLoadLargeParameters(unittest.TestCase):
    def test_large_parameters_static_save(self):
        # enable static mode
        paddle.enable_static()
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="static_save_load_large_x",
                shape=[None, 10],
                dtype='float32')
            z = paddle.static.nn.fc(x, LARGE_PARAM, bias_attr=False)
            place = paddle.CPUPlace()
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

            path = os.path.join("test_static_save_load_large_param",
                                "static_save")
            protocol = 4
            paddle.fluid.save(prog, path, pickle_protocol=protocol)
            # set var to zero
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            paddle.fluid.load(prog, path)

            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            # set var to zero
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    ten = fluid.global_scope().find_var(var.name).get_tensor()
                    ten.set(np.zeros_like(np.array(ten)), place)

                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

            program_state = fluid.load_program_state(path)
            fluid.set_program_state(prog, program_state)
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))


if __name__ == '__main__':
    unittest.main()
