#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import os
import shutil
import paddle.fluid.core as core
import unittest
from paddle.fluid.layers.nn import _pull_box_sparse
from paddle.fluid.transpiler import collective


class TestTranspile(unittest.TestCase):
    """  TestCases for BoxPS Preload """

    def get_transpile(self, mode, trainers="127.0.0.1:6174"):
        config = fluid.DistributeTranspilerConfig()
        config.mode = 'collective'
        config.collective_mode = mode
        t = fluid.DistributeTranspiler(config=config)
        return t

    def test_transpile(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        t = self.get_transpile("single_process_multi_thread")
        t.transpile(
            trainer_id=0,
            startup_program=startup_program,
            trainers="127.0.0.1:6174",
            program=main_program)
        t = self.get_transpile("grad_allreduce")
        try:
            t.transpile(
                trainer_id=0,
                startup_program=startup_program,
                trainers="127.0.0.1:6174",
                program=main_program)
        except ValueError as e:
            print(e)

    def test_single_trainers(self):
        transpiler = collective.GradAllReduce(0)
        try:
            transpiler.transpile(
                startup_program=fluid.Program(),
                main_program=fluid.Program(),
                rank=1,
                endpoints="127.0.0.1:6174",
                current_endpoint="127.0.0.1:6174",
                wait_port="6174")
        except ValueError as e:
            print(e)
        transpiler = collective.LocalSGD(0)
        try:
            transpiler.transpile(
                startup_program=fluid.Program(),
                main_program=fluid.Program(),
                rank=1,
                endpoints="127.0.0.1:6174",
                current_endpoint="127.0.0.1:6174",
                wait_port="6174")
        except ValueError as e:
            print(e)


class TestRunCmd(unittest.TestCase):
    """ TestCases for run_cmd"""

    def test_run_cmd(self):
        ret1 = int(core.run_cmd("ls; echo $?").strip().split('\n')[-1])
        ret2 = int(core.run_cmd("ls; echo $?", -1, -1).strip().split('\n')[-1])
        self.assertTrue(ret1 == 0)
        self.assertTrue(ret2 == 0)


class TestPullBoxSparseOP(unittest.TestCase):
    """ TestCases for _pull_box_sparse op"""

    def test_pull_box_sparse_op(self):
        paddle.enable_static()
        program = fluid.Program()
        with fluid.program_guard(program):
            x = fluid.layers.data(
                name='x', shape=[1], dtype='int64', lod_level=0)
            y = fluid.layers.data(
                name='y', shape=[1], dtype='int64', lod_level=0)
            emb_x, emb_y = _pull_box_sparse([x, y], size=1)


if __name__ == '__main__':
    unittest.main()
