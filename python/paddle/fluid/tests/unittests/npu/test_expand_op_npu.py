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

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestExpand(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "expand"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.randn(3, 1, 7).astype(self.dtype)
        out = np.tile(x, [1, 10, 1])

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'expand_times': [1, 10, 1]}
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    # TODO(ascendrc): Add grad test
    # def test_check_grad(self):
    #     if self.dtype == np.float16:
    #         return
    #     self.check_grad(['X'], 'Out')
    #


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestExpandV2(TestExpand):
    def setUp(self):
        self.set_npu()
        self.op_type = "expand"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.randn(3, 1, 7).astype(self.dtype)
        out = np.tile(x, [1, 10, 1])
        expand_times = np.array([1, 10, 1]).astype(np.int32)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'ExpandTimes': OpTest.np_dtype_to_fluid_dtype(expand_times)
        }
        self.attrs = {}
        self.outputs = {'Out': out}


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestExpandFp16(TestExpand):
    no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestExpandNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 1)).astype('float32')
        label_np = np.random.randint(2, size=(32, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 1], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[32, 1], dtype='int64')

            res = paddle.fluid.layers.expand(a, [1, 32])
            loss = res.sum()
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        for epoch in range(100):

            loss_res = exe.run(main_prog,
                               feed={"a": a_np,
                                     "label": label_np},
                               fetch_list=[loss])
            if epoch % 10 == 0:
                print("Epoch {} | Loss: {}".format(epoch, loss))

        return loss_res

    def test_npu(self):
        cpu_loss = self._test(False)
        npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == '__main__':
    unittest.main()
