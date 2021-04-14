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
class TestElementwiseDiv(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_div"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ['X', 'Y'],
            'Out',
            max_relative_error=0.007,
            check_dygraph=False)

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'],
            'Out',
            max_relative_error=0.007,
            no_grad_set=set("X"),
            check_dygraph=False)

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', no_grad_set=set("Y"), check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestElementwiseDivFp16(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "elementwise_div"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        y = np.random.uniform(1, 2, [3, 4]).astype(self.dtype)
        out = np.divide(x, y)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.attrs = {}
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False, atol=1e-5)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestElementwiseDivNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.uniform(1, 2, [32, 32]).astype('float32')
        b_np = np.random.uniform(1, 2, [32, 32]).astype('float32')
        c_np = np.random.uniform(1, 2, [32, 32]).astype('float32')
        d_np = np.random.uniform(1, 2, [32, 32]).astype('float32')
        label_np = np.random.randint(2, size=(32, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
            c = paddle.static.data(name="c", shape=[32, 32], dtype='float32')
            d = paddle.static.data(name="d", shape=[32, 32], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[32, 1], dtype='int64')

            e = paddle.multiply(a, b)
            f = paddle.multiply(c, d)
            f.stop_gradient = True
            g = fluid.layers.elementwise_div(e, f)

            fc_1 = fluid.layers.fc(input=g, size=128)
            prediction = fluid.layers.fc(input=fc_1, size=2, act='softmax')

            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            loss = fluid.layers.reduce_mean(cost)
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

            pred_res, loss_res = exe.run(main_prog,
                                         feed={
                                             "a": a_np,
                                             "b": b_np,
                                             "c": c_np,
                                             "d": d_np,
                                             "label": label_np
                                         },
                                         fetch_list=[prediction, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res))

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == '__main__':
    unittest.main()
