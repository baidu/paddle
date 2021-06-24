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
import unittest
from test_dist_base import TestDistBase

import os
import paddle

paddle.enable_static()
flag_name = os.path.splitext(__file__)[0]


class TestPipeline(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._pipeline_mode = True
        self._nccl_comm_num = 1

    def test_dist_train(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            # TODO (sandyhouse) fix the delta value.
            # Now pipeline only gets the loss value of the last
            # microbatch, so it is not consistable with the
            # non-pipeline one.
            self.check_with_place(
                "pipeline_mnist.py",
                delta=1e0,
                check_error_log=True,
                log_name=flag_name)

    def test_dist_train_multi_device(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "pipeline_mnist_multi_device.py",
                check_error_log=True,
                delta=1e0,
                log_name=flag_name)

    def test_dist_train_one_device(self):
        import paddle.fluid as fluid
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "pipeline_mnist_one_device.py",
                check_error_log=True,
                log_name=flag_name)


if __name__ == '__main__':
    unittest.main()
