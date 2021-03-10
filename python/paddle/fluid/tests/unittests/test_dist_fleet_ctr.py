# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
from test_dist_fleet_base import TestFleetBase


class TestDistMnistAsync2x2(TestFleetBase):
    def _setup_config(self):
        self._mode = "async"
        self._reader = "pyreader"

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "CPU_NUM": "2"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_ctr.py", delta=1e-5, check_error_log=False)


class TestDistCtrHalfAsync2x2(TestFleetBase):
    def _setup_config(self):
        self._mode = "async"
        self._reader = "pyreader"

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "30000",  # 5sec to fail fast
            "http_proxy": "",
            "FLAGS_communicator_send_queue_size": "2",
            "FLAGS_communicator_max_merge_var_num": "2",
            "CPU_NUM": "2",
            "SAVE_MODEL": "0"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        self.check_with_place(
            "dist_fleet_ctr.py", delta=1e-5, check_error_log=False)
        # test
        self.assertEqual(1, 2)


if __name__ == "__main__":
    unittest.main()
