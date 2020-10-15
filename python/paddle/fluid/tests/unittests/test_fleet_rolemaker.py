#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""Test cloud role maker."""

from __future__ import print_function
import os
import unittest
import paddle.fluid.incubate.fleet.base.role_maker as role_maker


class TestCloudRoleMaker(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMaker.
    """

    def setUp(self):
        """Set up, set envs."""
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001,127.0.0.2:36001"

    def test_tr_rolemaker(self):
        """Test tr rolenamer."""
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"

        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        ro.generate_role()
        self.assertTrue(ro.is_worker())
        self.assertFalse(ro.is_server())
        self.assertEqual(ro.worker_num(), 2)

    def test_ps_rolemaker(self):
        """Test ps rolemaker."""
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        ro.generate_role()
        self.assertFalse(ro.is_worker())
        self.assertTrue(ro.is_server())
        self.assertEqual(ro.worker_num(), 2)

    def test_traing_role(self):
        """Test training role."""
        os.environ["TRAINING_ROLE"] = "TEST"
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro.generate_role)

    def test_pslib_1(self):
        """Test cases for pslib."""
        import paddle.fluid as fluid
        from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
        from paddle.fluid.incubate.fleet.parameter_server.pslib import PSLib
        from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker

        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        role_maker = GeneralRoleMaker()
        #print("init rolemaker")
        #role_maker.generate_role()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        #fleet.init(role_maker)
        train_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(train_program, startup_program):
            show = fluid.layers.data(name="show", shape=[-1, 1], \
                dtype="float32", lod_level=1, append_batch_size=False)
            fc = fluid.layers.fc(input=show, size=1, act=None)
            label = fluid.layers.data(name="click", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            label_cast = fluid.layers.cast(label, dtype='float32')
            cost = fluid.layers.log_loss(fc, label_cast)
        try:
            adam = fluid.optimizer.Adam(learning_rate=0.000005)
            adam = fleet.distributed_optimizer(adam)
            adam.minimize([cost], [scope])
            fleet.run_server()
        except:
            print("do not support pslib test, skip")
            return
        fleet.clear_one_table(0)
        from paddle.fluid.incubate.fleet.base.role_maker import \
            MPISymetricRoleMaker
        try:
            role = MPISymetricRoleMaker()
            role._all_reduce([1], [2])
        except:
            print("catch expected error of not inited")
        try:
            role = MPISymetricRoleMaker()
            role._all_reduce([1], [2], "min")
        except:
            print("catch expected error of not inited")
        try:
            role = MPISymetricRoleMaker()
            role._all_reduce([1], [2], "max")
        except:
            print("catch expected error of not inited")
        try:
            role = MPISymetricRoleMaker()
            role._all_reduce([1], [2], "unknown")
        except:
            print("catch expected error of unknown type")


if __name__ == "__main__":
    unittest.main()
