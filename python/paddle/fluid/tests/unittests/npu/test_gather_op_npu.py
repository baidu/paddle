#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.framework import core

paddle.enable_static()
SEED = 2021


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGatherOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "gather"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {
            'X': xnp,
            'Index': np.array(self.index).astype(self.index_type)
        }
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['X'],
            'Out',
            max_relative_error=0.006,
            check_dygraph=False)

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCase1(TestGatherOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (100)
        self.x_type = "float32"
        self.index = [1, 3, 5]
        self.index_type = "int32"


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class API_TestGather(unittest.TestCase):
    def test_out1(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data('data1', shape=[-1, 2], dtype='float32')
            index = fluid.layers.data('index', shape=[-1, 1], dtype='int32')
            out = paddle.fluid.layers.gather(data1, index)
            place = paddle.NPUPlace(0)
            exe = fluid.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]])
            index_1 = np.array([1, 2])
            result, = exe.run(feed={"data1": input,
                                    "index": index_1},
                              fetch_list=[out])
            expected_output = np.array([[3, 4], [5, 6]])
        self.assertTrue(np.allclose(result, expected_output))

    def test_out2(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x = paddle.fluid.data('x', shape=[-1, 2], dtype='float32')
            index = paddle.fluid.data('index', shape=[-1, 1], dtype='int32')
            out = paddle.gather(x, index)
            place = paddle.NPUPlace(0)
            exe = paddle.static.Executor(place)
            x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype('float32')
            index_np = np.array([1, 1]).astype('int32')
            result, = exe.run(feed={"x": x_np,
                                    "index": index_np},
                              fetch_list=[out])
            expected_output = gather_numpy(x_np, index_np, axis=0)
        self.assertTrue(np.allclose(result, expected_output))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestGatherGrad(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(8192, 768)).astype('float32')
        index_np = np.random.randint(0, 8192, size=(1232, 1)).astype('int32')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[8192, 768], dtype='float32')
            index = paddle.static.data(
                name="index", shape=[1232, 1], dtype='int32')
            a.stop_gradient = False
            b = paddle.gather(a, index)

            loss = fluid.layers.reduce_mean(b)
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
                                         feed={"a": a_np,
                                               "index": index_np},
                                         fetch_list=[b, loss])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}".format(
                    epoch, pred_res[0], loss_res[0]))

        return pred_res, loss_res

    def test_npu(self):
        npu_pred, npu_loss = self._test(True)
        cpu_pred, cpu_loss = self._test(False)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == "__main__":
    unittest.main()
