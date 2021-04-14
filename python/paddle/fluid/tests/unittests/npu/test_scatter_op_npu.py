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

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCast1(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "scatter"
        self.place = paddle.NPUPlace(0)

        ref_np = np.ones((3, 2)).astype("float32")
        index_np = np.array([1]).astype("int32")
        updates_np = np.random.random((1, 2)).astype("float32")

        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}
        self.attrs = {'overwrite': True}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestCast2(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "scatter"
        self.place = paddle.NPUPlace(0)

        ref_np = np.ones((3, 2)).astype("int32")
        index_np = np.array([1]).astype("int32")
        updates_np = np.zeros((1, 2)).astype("int32")

        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}
        self.attrs = {'overwrite': True}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestCast3(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "scatter"
        self.place = paddle.NPUPlace(0)

        ref_np = np.ones((3, 2)).astype("float32")
        index_np = np.array([1]).astype("int32")
        updates_np = np.random.random((1, 2)).astype("float32")

        output_np = np.copy(ref_np)
        output_np[index_np] += updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}
        self.attrs = {'overwrite': False}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestCast4(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "scatter"
        self.place = paddle.NPUPlace(0)

        ref_np = np.ones((3, 2)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 2)).astype("float32")

        output_np = np.copy(ref_np)
        output_np[1] = updates_np[0]
        output_np[2] = updates_np[1]
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}
        self.attrs = {'overwrite': True}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
