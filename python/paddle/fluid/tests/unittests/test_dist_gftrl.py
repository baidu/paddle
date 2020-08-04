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

from __future__ import print_function
import os
import unittest
from test_dist_base import TestDistBase

import os
flag_name = os.path.splitext(__file__)[0]


class TestDistGFtrlSync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._enforce_place = "CPU"

    def test_gftrl(self):
        self.check_with_place(
            "dist_gftrl.py",
            delta=100,
            check_error_log=True,
            log_name=flag_name)


class TestDistGFtrlAsync(TestDistBase):
    def _setup_config(self):
        self._sync_mode = False
        self._enforce_place = "CPU"

    def test_gftrl(self):
        self.check_with_place(
            "dist_gftrl.py",
            delta=100,
            check_error_log=True,
            log_name=flag_name)


if __name__ == "__main__":
    unittest.main()
