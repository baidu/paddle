# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import unicode_literals
from __future__ import print_function

import unittest
import os
import sys
import subprocess
import paddle

paddle.enable_static()


class TestNanInf(unittest.TestCase):
    def setUp(self):
        self._python_interp = sys.executable
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            self._python_interp += " -m coverage run --branch -p"

        self.env = os.environ.copy()

    def check_nan_inf(self):
        cmd = self._python_interp

        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env)

        out, err = proc.communicate()
        returncode = proc.returncode

        print(out)
        print(err)

        assert returncode == 0
        # in python3, type(out+err) is 'bytes', need use encode
        assert (out + err
                ).find('There are `nan` or `inf` in tensor'.encode()) != -1

    def test_nan_inf_in_static_mode(self):
        self._python_interp += " check_nan_inf_base.py"
        self.check_nan_inf()

    def test_nan_inf_in_dynamic_mode(self):
        self._python_interp += " check_nan_inf_base_dygraph.py"
        self.check_nan_inf()


class TestNanInfEnv(TestNanInf):
    def setUp(self):
        super(TestNanInfEnv, self).setUp()
        # windows python have some bug with env, so need use str to pass ci
        # otherwise, "TypeError: environment can only contain strings"
        self.env[str("PADDLE_INF_NAN_SKIP_OP")] = str("mul")
        self.env[str("PADDLE_INF_NAN_SKIP_ROLE")] = str("loss")
        self.env[str("PADDLE_INF_NAN_SKIP_VAR")] = str(
            "elementwise_add:fc_0.tmp_1")


if __name__ == '__main__':
    unittest.main()
