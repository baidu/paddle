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
"""Test cases for role makers."""

from __future__ import print_function
import paddle
import os
import unittest

import paddle.fluid.incubate.fleet.base.role_maker as role_maker


class TestCloudRoleMaker2(unittest.TestCase):
    """
    Test cases for paddle cloud role makers.
    """

    def setUp(self):
        """Set up, set envs."""
        pass

    def test_pslib_2(self):
        """Test cases for pslib."""
        import paddle.fluid as fluid
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
        from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker
        from paddle.fluid.incubate.fleet.base.role_maker import RoleMakerBase

        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        try:
            fleet.init(None)
        except:
            print("no mpi4py, skip test_pslib_2")
            return
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
        os.environ["TRAINING_ROLE"] = "wrong"
        try:
            role1 = GeneralRoleMaker(path="./test_gloo_1")
            role1.generate_role()
        except:
            print("catch expected error of wrong TRAINING_ROLE")
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        role2 = GeneralRoleMaker(path="./test_gloo_2")
        role2._finalize()
        role2._all_gather(1)
        role2._all_gather(1)
        role2._barrier_server()
        role2._all_gather(1)
        role3 = GeneralRoleMaker(path="./test_gloo_3")
        role3._worker_gather(1)
        role3._worker_gather(1)
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        role4 = GeneralRoleMaker(path="./test_gloo_4")
        role4._worker_gather(1)
        role4._get_rank()
        role4._get_size()
        role4._all_comm.init()
        role5 = GeneralRoleMaker(path="./test_gloo_5")
        role5.get_local_endpoint()
        role5.get_local_endpoint()
        role6 = GeneralRoleMaker(path="./test_gloo_6")
        role6.get_trainer_endpoints()
        role6.get_trainer_endpoints()
        role7 = GeneralRoleMaker(path="./test_gloo_7")
        role7.get_pserver_endpoints()
        role7.get_pserver_endpoints()
        role8 = GeneralRoleMaker(path="./test_gloo_8")
        role8.is_worker()
        role8.is_worker()
        role9 = GeneralRoleMaker(path="./test_gloo_9")
        role9.is_server()
        role9.is_server()
        role10 = GeneralRoleMaker(path="./test_gloo_10")
        role10.is_first_worker()
        role10.is_first_worker()
        role11 = GeneralRoleMaker(path="./test_gloo_11")
        role11.worker_index()
        role11.worker_index()
        role12 = GeneralRoleMaker(path="./test_gloo_12")
        role12.server_index()
        role12.server_index()
        role13 = GeneralRoleMaker(path="./test_gloo_13")
        role13.worker_num()
        role13.worker_num()
        role14 = GeneralRoleMaker(path="./test_gloo_14")
        role14.server_num()
        role14.server_num()
        role15 = GeneralRoleMaker(path="./test_gloo_15")
        role15._barrier_worker()
        role15._barrier_worker()
        role16 = GeneralRoleMaker(path="./test_gloo_16")
        role16._barrier_all()
        role16._barrier_all()
        role17 = GeneralRoleMaker(path="./test_gloo_17")
        role17._barrier_server()
        role17._barrier_server()
        role18 = GeneralRoleMaker(path="./test_gloo_18")
        role18._worker_num()
        role18._worker_num()
        role19 = GeneralRoleMaker(path="./test_gloo_19")
        role19._server_num()
        role19._server_num()
        role20 = GeneralRoleMaker(path="./test_gloo_20")
        a = [1]
        b = [0]
        role20._all_reduce(a, b)
        role21 = GeneralRoleMaker(path="./test_gloo_21")
        role21.all_reduce_worker([], [])
        role21.all_reduce_worker([], [])
        role21.barrier_worker()
        role21.barrier_all()
        role22 = GeneralRoleMaker(path="./test_gloo_22")
        role22._get_rank()
        role22._get_rank()
        os.environ["PADDLE_PSERVER_ID"] = "0"
        role23 = GeneralRoleMaker(path="./test_gloo_23")
        role23._get_size()
        role23._get_size()
        with open("test_fleet_gloo_role_maker_1.txt", "w") as f:
            data = "1 1 1 1\n"
            f.write(data)

        dataset = paddle.distributed.InMemoryDataset()
        dataset.set_filelist(["test_fleet_gloo_role_maker_1.txt"])
        dataset._set_use_var([show, label])
        dataset.load_into_memory()
        dataset.get_memory_data_size(fleet)
        dataset.get_shuffle_data_size(fleet)
        os.remove("./test_fleet_gloo_role_maker_1.txt")

        class TmpClass():
            """
            dummy tmp class
            """

            def __init__(self):
                pass

            def all_reduce_worker(self, input, output):
                """
                dummy all reduce worker

                Args:
                    input(None): fake input
                    output(None): fale output
                """
                pass

            def barrier_worker(self):
                """
                dummy barrier worker
                """
                pass

        from paddle.fluid.incubate.fleet.base.fleet_base import Fleet

        class TmpFleet(Fleet):
            """
            dummy tmp fleet
            """

            def __init__(self):
                super(Fleet, self).__init__()
                self._role_maker = None

            def init_worker(self):
                """
                dummy init worker
                """
                pass

            def init_server(self, model_dir=None):
                """
                dummy init server

                Args:
                    model_dir(None): fake model_dir
                """
                pass

            def run_server(self):
                """
                dummy run server
                """
                pass

            def stop_worker(self):
                """
                dummy stop worker
                """
                pass

            def distributed_optimizer(self, optimizer, strategy=None):
                """
                dummy distributed optimizer

                Args:
                    optimizer(None): fake optimizer
                    strategy(None): fake strategy
                """
                pass

            def save_inference_model(self):
                """
                dummy save inference model
                """
                pass

            def save_persistables(self):
                """
                dummy save persistables
                """
                pass

        os.environ["TRAINING_ROLE"] = "TRAINER"
        tmp = TmpFleet()
        tmp._role_maker = TmpClass()
        tmp.all_reduce_worker([], [])
        tmp.barrier_worker()
        from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker
        tmp = RoleMakerBase()
        tmp.all_gather(1)
        tmp.all_reduce_worker([], [])
        tmp.barrier_worker()
        tmp.barrier_all()
        from paddle.fluid.incubate.fleet.base.role_maker import \
            MPISymetricRoleMaker
        tmp1 = MPISymetricRoleMaker()
        tmp1.all_gather(1)
        tmp1.all_gather(1)
        tmp2 = MPISymetricRoleMaker()
        tmp2.all_reduce_worker([], [])
        tmp3 = MPISymetricRoleMaker()
        tmp3.barrier_worker()
        tmp3.barrier_worker()
        tmp4 = MPISymetricRoleMaker()
        tmp4.barrier_all()
        tmp4.barrier_all()


if __name__ == "__main__":
    unittest.main()
