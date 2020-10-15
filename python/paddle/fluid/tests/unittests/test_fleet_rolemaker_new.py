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
import platform
import shutil
import tempfile
import unittest
import paddle
import paddle.distributed.fleet.base.role_maker as role_maker


class TestRoleMakerBase(unittest.TestCase):
    """
    Test cases for RoleMakerBase
    """

    def test_rolemaker_base(self):
        role = role_maker.RoleMakerBase()
        self.assertRaises(Exception, role._is_worker)
        self.assertRaises(Exception, role._is_server)
        self.assertRaises(Exception, role._is_first_worker)
        self.assertRaises(Exception, role._worker_num)
        self.assertRaises(Exception, role._server_num)
        self.assertRaises(Exception, role._worker_index)
        self.assertRaises(Exception, role._server_index)
        self.assertRaises(Exception, role._role_id)
        self.assertRaises(Exception, role._node_num)

        trainer_endpoints = role._get_trainer_endpoints()
        self.assertTrue(len(trainer_endpoints) == 0)
        pserver_endpoints = role._get_pserver_endpoints()
        self.assertTrue(len(pserver_endpoints) == 0)

        print(role.to_string())
        self.assertTrue(role._all_gather(1, "worker") is None)
        self.assertTrue(role._all_reduce(1, "sum", "worker") is None)
        role._barrier("worker")


class TestCloudRoleMaker(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMaker.
    """

    def setUp(self):
        """Set up, set envs."""
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ[
            "PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001,127.0.0.2:36001"
        os.environ[
            "PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001,127.0.0.2:36001"
        os.environ["POD_IP"] = "127.0.0.1"

    def test_tr_rolemaker(self):
        """Test tr rolenamer."""
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"

        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertTrue(ro._is_worker())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertFalse(ro._is_server())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._worker_num(), 2)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertTrue(ro._is_first_worker())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        worker_endpoints = ro._get_trainer_endpoints()
        self.assertEqual(worker_endpoints[0], '127.0.0.1:36001')
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._role_id(), 0)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._node_num(), 2)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertFalse(ro._is_non_distributed())
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertEqual(ro._heter_worker_num(), 0)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertFalse(ro._is_heter_worker())

    def test_tr_rolemaker_collective(self):
        ro = role_maker.PaddleCloudRoleMaker(is_collective=True)
        self.assertEqual(ro._worker_num(), 2)
        ro = role_maker.PaddleCloudRoleMaker(is_collective=True)
        self.assertEqual(ro._node_num(), 2)

    def test_ps_rolemaker(self):
        """Test ps rolemaker."""
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        ro = role_maker.PaddleCloudRoleMaker(
            is_collective=False, init_gloo=False)
        self.assertEqual(ro._server_index(), 0)
        self.assertFalse(ro._is_worker())
        self.assertTrue(ro._is_server())
        self.assertEqual(ro._server_num(), 2)
        pserver_endpoints = ro._get_pserver_endpoints()
        self.assertEqual(pserver_endpoints[0], '127.0.0.1:36001')

        self.assertEqual(ro._all_gather(1, "worker"), 1)
        self.assertEqual(ro._all_reduce(1, "sum", "worker"), 1)

    def test_traing_role(self):
        """Test training role."""
        os.environ["TRAINING_ROLE"] = "TEST"

        ro = role_maker.PaddleCloudRoleMaker(is_collective=False)
        self.assertRaises(ValueError, ro._generate_role)


class TestUserDefinedRoleMaker(unittest.TestCase):
    """
    Test cases for UserDefinedRoleMaker.
    """

    def setUp(self):
        pass

    def test_ps_rolemaker(self):

        ro = role_maker.UserDefinedRoleMaker(
            is_collective=False,
            init_gloo=False,
            server_endpoints=["127.0.0.1:36001", "127.0.0.1:36001"],
            role=role_maker.Role.SERVER,
            current_id=0,
            worker_num=2)
        self.assertEqual(ro._server_num(), 2)
        ro._generate_role()
        self.assertTrue(ro._is_server())
        self.assertEqual(ro._role_id(), 0)

    def test_tr_rolemaker(self):
        ro = role_maker.UserDefinedRoleMaker(
            is_collective=False,
            init_gloo=False,
            server_endpoints=["127.0.0.1:36001", "127.0.0.1:36001"],
            role=role_maker.Role.WORKER,
            current_id=0,
            worker_num=2)

        self.assertIn("127.0.0.1:36001", ro._get_pserver_endpoints())
        self.assertTrue(ro._is_worker())
        self.assertEqual(ro._role_id(), 0)


class TestGlooWithCloudRoleMaker(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

    def case(self, role, comm_world):
        role._barrier(comm_world)

        gather = role._all_gather(1, comm_world)
        self.assertEqual(gather[0], 1)

        all_reduce = role._all_reduce(1, "sum", comm_world)
        self.assertEqual(1, all_reduce)

    def mkdir(self):
        tmp = tempfile.mkdtemp()
        return tmp

    def clean(self, tmp):
        shutil.rmtree(tmp)

    def test_hdfs_gloo(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "worker")
        self.clean(tmp)

    def test_fs_gloo(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "worker")
        self.clean(tmp)

    def test_fs_gloo2(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.clean(tmp)

    def test_fs_gloo3(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.clean(tmp)

    def test_fs_gloo4(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "3"
        os.environ["PADDLE_GLOO_HTTP_ENDPOINT"] = "127.0.0.1:30019"

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        import time
        time.sleep(3)

    def test_fs_gloo5(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "2"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.case(role, "all")
        self.clean(tmp)

    def test_fs_gloo6(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"

        os.environ["PADDLE_WITH_GLOO"] = "2"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.case(role, "all")
        self.clean(tmp)

    def test_fs_gloo7(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"

        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "5"

        role = role_maker.PaddleCloudRoleMaker()
        self.assertRaises(ValueError, role._generate_role)

    def test_fs_gloo8(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"

        os.environ["PADDLE_WITH_GLOO"] = "2"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        def net():
            x = paddle.fluid.layers.data(name='x', shape=[13], dtype='float32')
            y_predict = paddle.fluid.layers.fc(input=x, size=1, act=None)
            y = paddle.fluid.layers.data(name='y', shape=[1], dtype='float32')
            cost = paddle.fluid.layers.square_error_cost(
                input=y_predict, label=y)
            avg_cost = paddle.fluid.layers.mean(cost)
            return avg_cost

        from paddle.distributed import fleet

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        avg_cost = net()

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False

        optimizer = paddle.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        comm_world = "server"
        fleet.util.barrier(comm_world)

        gather = fleet.util.all_gather(1, comm_world)
        self.assertEqual(gather[0], 1)

        all_reduce = fleet.util.all_reduce(1, "sum", comm_world)
        self.assertEqual(1, all_reduce)

        self.clean(tmp)


class TestGlooWithCloudRoleMaker(unittest.TestCase):
    def setUp(self):
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_TRAINER_ID"] = "0"

    def case(self, role, comm_world):
        role._barrier(comm_world)

        gather = role._all_gather(1, comm_world)
        self.assertEqual(gather[0], 1)

        all_reduce = role._all_reduce(1, "sum", comm_world)
        self.assertEqual(1, all_reduce)

    def mkdir(self):
        tmp = tempfile.mkdtemp()
        return tmp

    def clean(self, tmp):
        shutil.rmtree(tmp)

    def test_hdfs_gloo(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "worker")
        self.clean(tmp)

    def test_fs_gloo(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "worker")
        self.clean(tmp)

    def test_fs_gloo2(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.clean(tmp)

    def test_fs_gloo3(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.clean(tmp)

    def test_fs_gloo4(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "3"
        os.environ["PADDLE_GLOO_HTTP_ENDPOINT"] = "127.0.0.1:30019"

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        import time
        time.sleep(3)

    def test_fs_gloo5(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "2"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.case(role, "all")
        self.clean(tmp)

    def test_fs_gloo6(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"

        os.environ["PADDLE_WITH_GLOO"] = "2"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        role = role_maker.PaddleCloudRoleMaker()
        role._generate_role()
        self.case(role, "server")
        self.case(role, "all")
        self.clean(tmp)

    def test_fs_gloo7(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"

        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "5"

        role = role_maker.PaddleCloudRoleMaker()
        self.assertRaises(ValueError, role._generate_role)

    def test_hdfs_gloo_v2(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = ""
        os.environ["PADDLE_GLOO_FS_UGI"] = ""
        os.environ["PADDLE_GLOO_FS_PATH"] = ""

        role = role_maker.PaddleCloudRoleMaker()
        self.assertRaises(ValueError, role._generate_role)

    def test_fs_gloo_v2(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "2"
        os.environ["PADDLE_GLOO_FS_PATH"] = ""

        role = role_maker.PaddleCloudRoleMaker()
        self.assertRaises(ValueError, role._generate_role)

    def test_http_gloo_v2(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"
        os.environ["PADDLE_WITH_GLOO"] = "1"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "3"
        os.environ["PADDLE_GLOO_HTTP_ENDPOINT"] = ""

        role = role_maker.PaddleCloudRoleMaker()

    def test_fs_gloo8(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip gloo UT on MacOS/Win")
            return

        tmp = self.mkdir()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "0"

        os.environ["SYS_JOB_ID"] = "gloo_for_cluster"

        os.environ["PADDLE_WITH_GLOO"] = "2"
        os.environ["PADDLE_GLOO_RENDEZVOUS"] = "1"
        os.environ["PADDLE_GLOO_FS_NAME"] = "NULL"
        os.environ["PADDLE_GLOO_FS_UGI"] = "NULL"
        os.environ["PADDLE_GLOO_FS_PATH"] = tmp

        def net():
            x = paddle.fluid.layers.data(name='x', shape=[13], dtype='float32')
            y_predict = paddle.fluid.layers.fc(input=x, size=1, act=None)
            y = paddle.fluid.layers.data(name='y', shape=[1], dtype='float32')
            cost = paddle.fluid.layers.square_error_cost(
                input=y_predict, label=y)
            avg_cost = paddle.fluid.layers.mean(cost)
            return avg_cost

        from paddle.distributed import fleet

        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        avg_cost = net()

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False

        optimizer = paddle.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        comm_world = "server"
        fleet.util.barrier(comm_world)

        gather = fleet.util.all_gather(1, comm_world)
        self.assertEqual(gather[0], 1)

        all_reduce = fleet.util.all_reduce(1, "sum", comm_world)
        self.assertEqual(1, all_reduce)

        self.clean(tmp)


if __name__ == "__main__":
    unittest.main()
