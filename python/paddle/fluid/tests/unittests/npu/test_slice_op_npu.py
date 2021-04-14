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

paddle.enable_static()

SEED = 2021
EPOCH = 100


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSliceOp(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.set_npu()
        self.init_dtype()
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def init_dtype(self):
        self.dtype = np.float32

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad_normal(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(
            self.place, ['Input'], 'Out', check_dygraph=False)


class TestSliceOp2(TestSliceOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype(self.dtype)
        self.starts = [1, 0, -3]
        self.ends = [3, 3, -1]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, -3:-1, :]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSliceOpFp16(TestSliceOp):
    def init_dtype(self):
        self.dtype = np.float16

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True
        self.place = paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSliceNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        batch_size = 32
        data_shape = (32, 32)
        a_np = np.random.random(size=data_shape).astype('float32')
        b_np = np.random.random(size=data_shape).astype('float32')
        label_np = np.random.randint(2, size=(batch_size, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=data_shape, dtype='float32')
            b = paddle.static.data(name="b", shape=data_shape, dtype='float32')
            label = paddle.static.data(
                name="label", shape=[batch_size, 1], dtype='int64')

            sum = paddle.add(a, b)
            z = paddle.slice(sum, axes=[0, 1], starts=[0, 0], ends=[33, 2])

            prediction = paddle.static.nn.fc(z, size=2, activation='softmax')

            cost = paddle.nn.functional.cross_entropy(
                input=prediction, label=label)
            loss = paddle.mean(cost)
            sgd = paddle.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)
        print("Start run on {}".format(place))
        for epoch in range(EPOCH):

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

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


if __name__ == '__main__':
    unittest.main()
