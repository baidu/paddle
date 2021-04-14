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
from paddle.fluid import core

paddle.enable_static()
SEED = 2021

NPUPlace = 5


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestIncrement(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(NPUPlace)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            'X':
            OpTest.np_dtype_to_fluid_dtype(np.array([1]).astype(self.dtype)),
        }

        self.attrs = {"Step": 1}
        self.outputs = {'Out': np.array([2])}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.int64

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestIncrementFP16(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(NPUPlace)
        self.op_type = "increment"
        self.init_dtype()

        self.inputs = {
            'X':
            OpTest.np_dtype_to_fluid_dtype(np.array([1]).astype(self.dtype)),
        }
        self.pre_input_id = id(self.inputs['X'])

        self.attrs = {"Step": 1}
        self.outputs = {'Out': np.array([2])}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestIncrementInplace(unittest.TestCase):
    def test_npu(self):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.array([1]).astype('float32')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[1], dtype='float32')
            b = fluid.layers.increment(a)

        place = paddle.NPUPlace(NPUPlace)

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        b_value = exe.run(main_prog, feed={"a": a_np, }, fetch_list=[b])

        print('input a id is : {}'.format(id(a)))
        print('input b id is : {}'.format(id(b)))

        self.assertEqual(id(a), id(b))
        self.assertEqual(b_value[0], 2)


if __name__ == '__main__':
    unittest.main()
