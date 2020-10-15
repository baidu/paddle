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

from __future__ import unicode_literals
from __future__ import print_function

import unittest
import os
import sys
import subprocess
import re


class TestFlagsUseMkldnn(unittest.TestCase):
    def setUp(self):
        self._python_interp = sys.executable
        self._python_interp += " check_flags_mkldnn_ops_on_off.py"

        self.env = os.environ.copy()
        self.env[str("DNNL_VERBOSE")] = str("1")
        self.env[str("FLAGS_use_mkldnn")] = str("1")

        self.relu_regex = b"^dnnl_verbose,exec,cpu,eltwise,.+alg:eltwise_relu alpha:0 beta:0,10x20x20"
        self.ew_add_regex = b"^dnnl_verbose,exec,cpu,binary.+alg:binary_add,10x20x30:10x20x30 10x20x30"
        self.matmul_regex = b"^dnnl_verbose,exec,cpu,matmul,.*b10m20n20k30"

    def flags_use_mkl_dnn_common(self, e):
        cmd = self._python_interp
        env = dict(self.env, **e)
        proc = subprocess.Popen(
            cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env)

        out, err = proc.communicate()
        returncode = proc.returncode

        assert returncode == 0
        return out

    def found(self, regex, out):
        return re.search(regex, out, re.MULTILINE)

    def test_flags_use_mkl_dnn_on_empty_off_empty(self):
        out = self.flags_use_mkl_dnn_common({})
        assert self.found(self.relu_regex, out)
        assert self.found(self.ew_add_regex, out)
        assert self.found(self.matmul_regex, out)

    def test_flags_use_mkl_dnn_on(self):
        env = {str("FLAGS_tracer_mkldnn_ops_on"): str("relu")}
        out = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out)
        assert not self.found(self.ew_add_regex, out)
        assert not self.found(self.matmul_regex, out)

    def test_flags_use_mkl_dnn_on_multiple(self):
        env = {str("FLAGS_tracer_mkldnn_ops_on"): str("relu,elementwise_add")}
        out = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out)
        assert self.found(self.ew_add_regex, out)
        assert not self.found(self.matmul_regex, out)

    def test_flags_use_mkl_dnn_off(self):
        env = {str("FLAGS_tracer_mkldnn_ops_off"): str("matmul")}
        out = self.flags_use_mkl_dnn_common(env)
        assert self.found(self.relu_regex, out)
        assert self.found(self.ew_add_regex, out)
        assert not self.found(self.matmul_regex, out)

    def test_flags_use_mkl_dnn_off_multiple(self):
        env = {str("FLAGS_tracer_mkldnn_ops_off"): str("matmul,relu")}
        out = self.flags_use_mkl_dnn_common(env)
        assert not self.found(self.relu_regex, out)
        assert self.found(self.ew_add_regex, out)
        assert not self.found(self.matmul_regex, out)

    def test_flags_use_mkl_dnn_on_off(self):
        env = {
            str("FLAGS_tracer_mkldnn_ops_on"): str("elementwise_add"),
            str("FLAGS_tracer_mkldnn_ops_off"): str("matmul")
        }
        out = self.flags_use_mkl_dnn_common(env)
        assert not self.found(self.relu_regex, out)
        assert self.found(self.ew_add_regex, out)
        assert not self.found(self.matmul_regex, out)


if __name__ == '__main__':
    unittest.main()
