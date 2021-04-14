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


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSoftmax(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "softmax"
        self.init_dtype()

        x = np.random.random([3, 3]).astype(self.dtype)
        np_out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        self.inputs = {'X': x}

        self.attrs = {}
        self.outputs = {'Out': np_out}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSoftmaxNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(4, 32)).astype('float32')
        b_np = np.random.random(size=(4, 32)).astype('float32')
        label_np = np.random.randint(2, size=(4, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[4, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[4, 32], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[4, 1], dtype='int64')

            c = paddle.multiply(a, b)
            d = paddle.sqrt(c)

            # 4 x 128
            fc_1 = fluid.layers.fc(input=d, size=128)
            # 4 x 2
            prediction = fluid.layers.fc(input=fc_1, size=2)

            # 4 x 2
            prob = fluid.layers.softmax(prediction, axis=1)

            cost = fluid.layers.cross_entropy(input=prob, label=label)
            loss = fluid.layers.mean(cost)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res = exe.run(
                main_prog,
                feed={"a": a_np,
                      "b": b_np,
                      "label": label_np},
                fetch_list=[prediction, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res))

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred, rtol=1e-2))
        self.assertTrue(np.allclose(npu_loss, cpu_loss, rtol=1e-2))


if __name__ == '__main__':
    unittest.main()
