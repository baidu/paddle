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
class TestTrunctedGaussianRandom(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "truncated_gaussian_random"
        self.place = paddle.NPUPlace(0)
        self.inputs = {}
        self.attrs = {
            "shape": [10],
            "mean": .0,
            "std": 1.,
            "seed": 10,
        }

        self.outputs = {"Out": np.random.random(10).astype(self.dtype)}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    #def test_npu(self):
    #    self.gaussian_random_test(place=paddle.NPUPlace(0))

    #def gaussian_random_test(self, place):

    #    program = fluid.Program()
    #    block = program.global_block()
    #    vout = block.create_var(name="Out")
    #    op = block.append_op(
    #        type=self.op_type, outputs={"Out": vout}, attrs=self.attrs)

    #    op.desc.infer_var_type(block.desc)
    #    op.desc.infer_shape(block.desc)

    #    fetch_list = []
    #    for var_name in self.outputs:
    #        fetch_list.append(block.var(var_name))

    #    exe = Executor(place)
    #    outs = exe.run(program, fetch_list=fetch_list)
    #    tensor = outs[0]
    #    self.assertAlmostEqual(numpy.mean(tensor), .0, delta=0.1)
    #    self.assertAlmostEqual(numpy.var(tensor), 0.773, delta=0.1)

    # TODO(ascendrc): Add grad test
    # def test_check_grad(self):
    #     if self.dtype == np.float16:
    #         return
    #     self.check_grad(['X'], 'Out')
    #


if __name__ == '__main__':
    unittest.main()
