#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os
import unittest
import numpy as np
import tempfile
import shutil
from op_test import OpTest, randomize_probability
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.distributed.fleet import fleet
from test_dist_sparse_tensor_load_sgd import TestSparseLoadProgram


class TestSparseLoadProgramRmsprop(TestSparseLoadProgram):
    """ 
    Test Sparse load operator.
    """

    def test_server_init(self):
        scope, train_program, startup_program, loss = self.net()
        with fluid.scope_guard(scope):
            with fluid.program_guard(train_program, startup_program):
                optimizer = fluid.optimizer.RMSProp(1e-3)
                optimizer = fleet.distributed_optimizer(optimizer,
                                                        self.strategy)
                optimizer.minimize(loss)
                fleet.init_server()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
