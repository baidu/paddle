#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestTruncatedNormal(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        scope = paddle.fluid.core.Scope()

        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)
        paddle.seed(SEED)

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                weight_attr = paddle.framework.ParamAttr(
                    name="linear_weight",
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0, std=2.0))
                linear = paddle.nn.Linear(
                    2, 2, weight_attr=weight_attr, bias_attr=False)

            if run_npu:
                place = paddle.NPUPlace(0)
            else:
                place = paddle.CPUPlace()

            exe = paddle.static.Executor(place)
            w = exe.run(startup_prog, fetch_list=['linear_weight'])
            return w

    def test_npu(self):
        cpu_w = self._test(False)
        npu_w = self._test(True)

        self.assertTrue(np.allclose(npu_w, cpu_w))


if __name__ == '__main__':
    unittest.main()
