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


class SparseLoadOp(unittest.TestCase):
    """ Test load operator.
    """

    def net(self, emb_array, fc_array):
        with fluid.unique_name.guard():
            dense_input = fluid.data('input', shape=[None, 1], dtype="int64")

            emb = fluid.layers.embedding(
                input=dense_input,
                is_sparse=True,
                size=[10, 10],
                param_attr=fluid.ParamAttr(
                    name="embedding",
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        emb_array)), )

            fc1 = fluid.layers.fc(
                input=emb,
                size=10,
                act="relu",
                param_attr=fluid.ParamAttr(
                    name='fc',
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        fc_array)))
            loss = fluid.layers.reduce_mean(fc1)
        return loss

    def save_origin_model(self, emb_array, fc_array):
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()
        with fluid.framework.program_guard(test_program, startup_program):
            with fluid.unique_name.guard():
                loss = self.net(emb_array, fc_array)
                optimizer = fluid.optimizer.Adam(1e-3)
                optimizer.minimize(loss)

                exe = fluid.Executor(fluid.CPUPlace())
                exe.run(startup_program)
                model_path = tempfile.mkdtemp()
                fluid.io.save_persistables(executor=exe, dirname=model_path)
        return model_path


class TestSparseLoadOpCase1(SparseLoadOp):
    def test_2ps_0_load(self):
        # init No.0 server env
        env = {}
        env["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:4001,127.0.0.1:4002"
        env["PADDLE_TRAINERS_NUM"] = str(2)
        env["TRAINING_ROLE"] = "PSERVER"
        env["PADDLE_PORT"] = "4001"
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

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        loss = self.net(emb_array, fc_array)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        optimizer = fluid.optimizer.Adam(1e-3)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)
        fleet.init_server(model_path)

        fc_w = np.array(fluid.global_scope().find_var("fc").get_tensor())

        emb = np.array(fluid.global_scope().find_var("embedding.block0")
                       .get_tensor())

        assert fc_w.all() == fc_array.all()
        assert emb.all() == emb_array[::2].all()
        shutil.rmtree(model_path)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
