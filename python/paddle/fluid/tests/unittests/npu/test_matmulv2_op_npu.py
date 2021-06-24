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
def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size, ))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = [i for i in range(len(X.shape))]
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size, ))
        else:
            dim = [i for i in range(len(Y.shape))]
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float64")
    return Out


class TestMatMul(OpTest):
    def config(self):
        self.x_shape = (100, 24)
        self.y_shape = (24, 100)
        self.trans_x = False
        self.trans_y = False

    def setUp(self):
        self.set_npu()
        self.op_type = "matmul_v2"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.config()
        np.random.seed(SEED)
        x = np.random.random(self.x_shape).astype(self.dtype)
        y = np.random.random(self.y_shape).astype(self.dtype)
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x, self.trans_y)
        result = result.astype(self.dtype)
        self.inputs = {
            'X': x,
            'Y': y,
        }
        self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
        self.outputs = {'Out': result}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False, atol=1e-5)


    # TODO(ascendrc): Add grad test
    # def test_check_grad(self):
    #     if self.dtype == np.float16:
    #         return
    #     self.check_grad(['X'], 'Out')
    #
class TestMatMul2(TestMatMul):
    """
    case 2
    """

    def config(self):
        self.x_shape = (32, 24)
        self.y_shape = (32, 24)
        self.trans_x = False
        self.trans_y = True


class TestMatMul3(TestMatMul):
    """
    case 3
    """

    def init_dtype(self):
        self.dtype = np.float16


class TestMatMul4(TestMatMul):
    """
    case 4 dim=3
    """

    def config(self):
        self.x_shape = (2, 3, 4)
        self.y_shape = (2, 4, 3)
        self.trans_x = False
        self.trans_y = False


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMatMulNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3)).astype('float32')
        b_np = np.random.random(size=(2, 3)).astype('float32')
        c_np = np.random.random(size=(3, 2)).astype('float32')
        d_np = np.random.random(size=(3, 2)).astype('float32')
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3], dtype='float32')
            b = paddle.static.data(name="b", shape=[2, 3], dtype='float32')
            c = paddle.static.data(name="c", shape=[3, 2], dtype='float32')
            d = paddle.static.data(name="d", shape=[3, 2], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[2, 1], dtype='int64')

            sum_1 = paddle.add(a, b)
            sum_2 = paddle.add(c, d)
            result = paddle.matmul(sum_1, sum_2)

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


# The precision is aligned in NPU and GPU separately, which is only used for the usage method.
@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestMatMulNet3_2(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)
        self._dtype = "float32"

        a_np = np.random.random(size=(2, 1, 3)).astype(self._dtype)
        b_np = np.random.random(size=(2, 1, 3)).astype(self._dtype)
        c_np = np.random.random(size=(3, 2)).astype(self._dtype)
        d_np = np.random.random(size=(3, 2)).astype(self._dtype)
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 1, 3], dtype=self._dtype)
            b = paddle.static.data(name="b", shape=[2, 1, 3], dtype=self._dtype)
            c = paddle.static.data(name="c", shape=[3, 2], dtype=self._dtype)
            d = paddle.static.data(name="d", shape=[3, 2], dtype=self._dtype)
            label = paddle.static.data(
                name="label", shape=[2, 1], dtype='int64')

            sum_1 = paddle.add(a, b)
            sum_2 = paddle.add(c, d)
            sum_1 = paddle.cast(sum_1, 'float16')
            sum_2 = paddle.cast(sum_2, 'float16')
            if not run_npu:
                sum_1 = paddle.cast(sum_1, 'float32')
                sum_2 = paddle.cast(sum_2, 'float32')

            result = paddle.matmul(sum_1, sum_2)
            if run_npu:
                result = paddle.cast(result, 'float32')

            result = paddle.reshape(result, shape=[2, 2])
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

        self.assertTrue(np.allclose(npu_pred, cpu_pred, atol=1e-4))
        self.assertTrue(np.allclose(npu_loss, cpu_loss, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
