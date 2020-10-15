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
import numpy as np
from op_test import OpTest, randomize_probability
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestLoadOpXpu(unittest.TestCase):
    """ Test load operator.
    """

    def setUp(self):
        self.ones = np.ones((4, 4)).astype('float32')
        main_prog = fluid.Program()
        start_prog = fluid.Program()
        with fluid.program_guard(main_prog, start_prog):
            input = fluid.data('input', shape=[-1, 4], dtype='float32')
            output = layers.fc(
                input,
                4,
                param_attr=fluid.ParamAttr(
                    name='w',
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        self.ones)))
        exe = fluid.Executor(fluid.XPUPlace(0))
        exe.run(start_prog)
        fluid.io.save_persistables(
            exe, dirname="./model", main_program=main_prog)

    def test_load_xpu(self):
        main_prog = fluid.Program()
        start_prog = fluid.Program()
        with fluid.program_guard(main_prog, start_prog):
            var = layers.create_tensor(dtype='float32')
            layers.load(var, file_path='./model/w')

        exe = fluid.Executor(fluid.XPUPlace(0))
        exe.run(start_prog)
        ret = exe.run(main_prog, fetch_list=[var.name])
        self.assertTrue(np.array_equal(self.ones, ret[0]))


if __name__ == "__main__":
    unittest.main()
