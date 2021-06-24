# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet
import os
import sys

from paddle.distributed.fleet.utils.fs import LocalFS, HDFSClient
import paddle.fluid.incubate.checkpoint.auto_checkpoint as acp
from paddle.fluid.incubate.checkpoint.checkpoint_saver import PaddleModel
from paddle.fluid.framework import program_guard
from paddle.fluid import unique_name

import numpy as np
from paddle.io import Dataset, BatchSampler, DataLoader

from paddle.fluid.tests.unittests.auto_checkpoint_utils import AutoCheckpointBase, get_logger

paddle.enable_static()
logger = get_logger()


class AutoCheckPointACLBase(AutoCheckpointBase):
    def setUp(self):
        get_logger()
        logger.info("enter tests")

        self._old_environ = dict(os.environ)
        proc_env = {
            "PADDLE_RUNNING_ENV": "PADDLE_EDL_AUTO_CHECKPOINT",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": "test_job_auto",
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_CHECKPOINT_PATH": "auto_checkpoint",
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1",
            "PADDLE_EDL_FS_CACHE": ".auto_checkpoint_test",
            "PADDLE_EDL_SAVE_CHECKPOINT_INTER": "0"
        }
        os.environ.update(proc_env)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_environ)

        file_name = os.path.basename(__file__)
        base_name = os.path.splitext(file_name)[0]
        print("runnng name:", base_name)

    def _run_normal(self):
        exe, main_prog, startup_prog = self._generate()

        save_dir = "./run_save_model"
        fs = LocalFS()

        fs.delete(save_dir)
        logger.info("begin _run_normal")

        compiled, data_loader, optimizer, loss, image, label = self._init_env(
            exe, main_prog, startup_prog)
        for i in range(3):
            self.assertEqual(acp._get_train_epoch_range(), None)
            self.assertEqual(acp.g_acp_type, None)
            for data in data_loader():
                self.assertEqual(acp.g_acp_type, None)
                self.assertEqual(acp._get_train_epoch_range(), None)
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        self.assertEqual(acp.g_acp_type, None)
        self.assertEqual(acp._get_train_epoch_range(), None)

        m1 = PaddleModel(exe, compiled)
        m1.serialize(save_dir)

        m2 = PaddleModel(exe, compiled)
        m2.deserialize(save_dir)

        logger.info("end _run_normal")
        fs.delete(save_dir)

    def _not_use_train(self):
        logger.info("begin _not_use_train")
        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        epochs = []
        for i in acp.train_epoch_range(3, 0):
            epochs.append(i)
            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        self.assertEqual(epochs, [0, 1, 2])
        logger.info("end _not_use_train")

    def _run_save_0(self, break_epoch_no=None):
        logger.info("begin _run_save_0")
        fs = LocalFS()
        save_dir = "./run_save_0"
        fs.delete(save_dir)

        exe, main_prog, startup_prog = self._generate()

        compiled, data_loader, optimizer, loss, image, label = \
            self._init_env(exe, main_prog, startup_prog)

        o = None
        i = 0
        name = None
        for i in acp.train_epoch_range(3, 0):
            o = acp._get_train_epoch_range()
            name = o.name

            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

            self.assertEqual(len(o._exe_status), 1)

            if break_epoch_no is not None:
                if i == break_epoch_no:
                    break

        o = acp._get_train_epoch_range()
        assert o == None, "now train epoch must not exits now"
        if break_epoch_no is None:
            self.assertEqual(i, 2)
        else:
            self.assertEqual(i, break_epoch_no)

        fs.delete(save_dir)
        logger.info("end _run_save_0")

    def _run_load_0(self, break_epoch_no=None):
        logger.info("begin _run_load_0")
        exe, main_prog, startup_prog = self._generate()

        fs = LocalFS()
        save_dir = "./run_load_0"
        fs.delete(save_dir)

        compiled, data_loader, optimizer, loss, image, label = self._init_env(
            exe, main_prog, startup_prog)

        o = None
        i = 0
        check = False

        epochs = []
        for i in acp.train_epoch_range(3, 0):
            epochs.append(i)

            for data in data_loader():
                fetch = exe.run(compiled, feed=data, fetch_list=[loss])

        o = acp._get_train_epoch_range()
        self.assertTrue(o == None, "now train epoch must not exits now")
        self.assertEqual(i, 2)

        if break_epoch_no is not None:
            if break_epoch_no == 0:
                self.assertEqual(epochs, [0, 1, 2])
            elif break_epoch_no == 1:
                self.assertEqual(epochs, [1, 2])
            elif break_epoch_no == 2:
                self.assertEqual(epochs, [2])
        else:
            self.assertEqual(epochs, [2])

        fs.delete(save_dir)
        logger.info("begin _run_load_0")

    def _test_corner_epoch_no(self, break_epoch_no):
        logger.info("begin test_corener_epoch_no")
        checker = acp._get_checker()
        fs = HDFSClient(checker.hdfs_home, None)

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_0(break_epoch_no=break_epoch_no)
        self._reset_generator()
        self._run_load_0(break_epoch_no=break_epoch_no)

        fs.delete(checker.hdfs_checkpoint_path)
        logger.info("end test_corener_epoch_no")


class AutoCheckpointTest(AutoCheckPointACLBase):
    def setUp(self):
        get_logger()
        logger.info("enter tests")

        self._old_environ = dict(os.environ)
        proc_env = {
            "PADDLE_RUNNING_ENV": "PADDLE_EDL_AUTO_CHECKPOINT",
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_RUNNING_PLATFORM": "PADDLE_CLOUD",
            "PADDLE_JOB_ID": "test_job_auto_0",
            "PADDLE_EDL_HDFS_HOME": "/usr/local/hadoop-2.7.7",
            "PADDLE_EDL_HDFS_NAME": "",
            "PADDLE_EDL_HDFS_UGI": "",
            "PADDLE_EDL_HDFS_CHECKPOINT_PATH": "auto_checkpoint_0",
            "PADDLE_EDL_ONLY_FOR_CE_TEST": "1",
            "PADDLE_EDL_FS_CACHE": ".auto_checkpoint_test_0",
            "PADDLE_EDL_SAVE_CHECKPOINT_INTER": "0"
        }
        os.environ.update(proc_env)

    def test_normal(self):
        logger.info("begin test_normal")
        checker = acp._get_checker()

        fs = HDFSClient(checker.hdfs_home, None)

        fs.delete(checker.hdfs_checkpoint_path)
        self._clear_envs()
        self._reset_generator()
        self._run_normal()
        self._readd_envs()
        logger.info("end test_normal")

    def test_basic(self):
        logger.info("begin test_basic")
        checker = acp._get_checker()
        self.assertEqual(checker.run_env, "PADDLE_EDL_AUTO_CHECKPOINT")
        self.assertEqual(checker.platform, "PADDLE_CLOUD")
        self.assertEqual(checker.save_checkpoint_inter, 0)
        print(checker)

        fs = HDFSClient(checker.hdfs_home, None)

        fs.delete(checker.hdfs_checkpoint_path)
        self._reset_generator()
        self._run_save_0()

        self._reset_generator()
        self._run_load_0()

        logger.info("end test_basic")

    def test_not_use(self):
        logger.info("begin test_not_use")

        self._clear_envs()
        self._reset_generator()
        self._not_use_train()
        self._readd_envs()

        logger.info("end test_not_use")

    def test_checker(self):
        os.environ.pop("PADDLE_JOB_ID", None)
        try:
            checker = acp.AutoCheckpointChecker()
            self.assertFalse(True)
        except Exception as e:
            pass
        os.environ["PADDLE_JOB_ID"] = "test_job_auto_1"


if __name__ == '__main__':
    unittest.main()
