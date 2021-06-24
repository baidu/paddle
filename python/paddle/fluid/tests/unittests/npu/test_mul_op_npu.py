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
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


class TestMul(OpTest):
    # case 1: (32, 5) * (5, 100) -> (32, 100)
    def config(self):
        self.x_shape = (32, 5)
        self.y_shape = (5, 100)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype)
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False, atol=1e-5)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place, ['X', 'Y'],
            'Out',
            max_relative_error=0.0065,
            check_dygraph=False)

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(
            self.place, ['Y'],
            'Out',
            no_grad_set=set("X"),
            max_relative_error=0.0065,
            check_dygraph=False)

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(
            self.place, ['X'],
            'Out',
            no_grad_set=set("Y"),
            max_relative_error=0.0065,
            check_dygraph=False)


@skip_check_grad_ci(
    reason="Don't support grad checking for NPU OP with FP16 data type.")
class TestMulFP16(TestMul):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestMul2(TestMul):
    # case 2: (20, 2, 5) * (10, 50) -> (20, 50), x_num_col_dims = 1
    def config(self):
        self.x_shape = (20, 2, 5)
        self.y_shape = (10, 50)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype)
        }
        self.outputs = {
            'Out': np.dot(self.inputs['X'].reshape(20, 10), self.inputs['Y'])
        }


@skip_check_grad_ci(
    reason="Don't support grad checking for NPU OP with FP16 data type.")
class TestMul2FP16(TestMul2):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestMul3(TestMul):
    # case 3: (20, 3, 4) * (4, 50) -> (20, 3, 50), x_num_col_dims = 2

    def config(self):
        self.x_shape = (20, 3, 4)
        self.y_shape = (4, 50)

    def setUp(self):
        self.set_npu()
        self.op_type = "mul"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        self.inputs = {
            'X': np.random.random(self.x_shape).astype(self.dtype),
            'Y': np.random.random(self.y_shape).astype(self.dtype)
        }
        self.attrs = {"x_num_col_dims": 2}
        self.outputs = {'Out': np.matmul(self.inputs['X'], self.inputs['Y'])}


@skip_check_grad_ci(
    reason="Don't support grad checking for NPU OP with FP16 data type.")
class TestMul3FP16(TestMul3):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMulNet(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.float32

    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3)).astype(self.dtype)
        b_np = np.random.random(size=(2, 3)).astype(self.dtype)
        c_np = np.random.random(size=(3, 2)).astype(self.dtype)
        d_np = np.random.random(size=(3, 2)).astype(self.dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3], dtype=self.dtype)
            b = paddle.static.data(name="b", shape=[2, 3], dtype=self.dtype)
            c = paddle.static.data(name="c", shape=[3, 2], dtype=self.dtype)
            d = paddle.static.data(name="d", shape=[3, 2], dtype=self.dtype)
            label = paddle.static.data(
                name="label", shape=[2, 1], dtype='int64')

            sum_1 = paddle.add(a, b)
            sum_2 = paddle.add(c, d)
            result = paddle.fluid.layers.mul(sum_1, sum_2)

            fc_1 = fluid.layers.fc(input=result, size=8)
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

        print("TestMulNet Start run on {} . ".format(place))
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
        self.init_dtype()
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMulNet3_2(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.float32

    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        b_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        c_np = np.random.random(size=(12, 5)).astype(self.dtype)
        d_np = np.random.random(size=(12, 5)).astype(self.dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype=self.dtype)
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype=self.dtype)
            c = paddle.static.data(name="c", shape=[12, 5], dtype=self.dtype)
            d = paddle.static.data(name="d", shape=[12, 5], dtype=self.dtype)
            label = paddle.static.data(
                name="label", shape=[2, 1], dtype='int64')

            sum_1 = paddle.add(a, b)
            sum_2 = paddle.add(c, d)
            result = paddle.fluid.layers.mul(sum_1, sum_2)

            fc_1 = fluid.layers.fc(input=result, size=8)
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

        print("testMulNet3_2 tart run on {}".format(place))
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
        self.init_dtype()
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(
            npu_pred, cpu_pred, atol=1e-5))  # atol needed on cann 20.3
        self.assertTrue(np.allclose(npu_loss, cpu_loss, atol=1e-5))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMulNet3_2_xc2(unittest.TestCase):
    def init_dtype(self):
        self.dtype = np.float32

    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        b_np = np.random.random(size=(2, 3, 4)).astype(self.dtype)
        c_np = np.random.random(size=(4, 5)).astype(self.dtype)
        d_np = np.random.random(size=(4, 5)).astype(self.dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype=self.dtype)
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype=self.dtype)
            c = paddle.static.data(name="c", shape=[4, 5], dtype=self.dtype)
            d = paddle.static.data(name="d", shape=[4, 5], dtype=self.dtype)
            label = paddle.static.data(
                name="label", shape=[2, 1], dtype='int64')

            sum_1 = paddle.add(a, b)
            sum_2 = paddle.add(c, d)
            result = paddle.fluid.layers.mul(sum_1, sum_2, x_num_col_dims=2)
            result_re = paddle.reshape(result, shape=[2, 15])

            fc_1 = fluid.layers.fc(input=result_re, size=8)
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

        print("TestMulNet3_2_xc2. Start run on {}".format(place))
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
        self.init_dtype()
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == '__main__':
    unittest.main()
