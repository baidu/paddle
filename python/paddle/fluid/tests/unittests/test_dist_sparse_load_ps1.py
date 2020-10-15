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
from test_dist_sparse_load_ps0 import SparseLoadOp


class TestSparseLoadOpCase2(SparseLoadOp):
    def test_2ps_0_load(self):
        # init No.1 server env
        env = {}
        env["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        env["PADDLE_TRAINERS_NUM"] = str(2)
        env["TRAINING_ROLE"] = "PSERVER"
        env["PADDLE_PORT"] = "4002"
        env["POD_IP"] = "127.0.0.1"
        for k, v in env.items():
            os.environ[k] = str(v)
        """
        array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])
        """
        emb_array = np.arange(0, 1, 0.1).repeat(10).reshape(10, 10)
        fc_array = np.arange(0, 1, 0.1).repeat(10).reshape(10, 10)
        model_path = self.save_origin_model(emb_array, fc_array)

        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        loss = self.net(emb_array, fc_array)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        optimizer = fluid.optimizer.Adam(1e-3)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)
        fleet.init_server(model_path)
        emb = np.array(fluid.global_scope().find_var("embedding.block1")
                       .get_tensor())
        assert emb.all() == emb_array[1::2].all()
        shutil.rmtree(model_path)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
