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
from test_softmax_op import stable_softmax
from test_softmax_with_cross_entropy_op import cross_entropy

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestSoftmaxWithCrossEntropyOp(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def initParams(self):
        self.set_npu()
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = False
        self.place = paddle.NPUPlace(0)
        self.soft_label = False
        self.init_dtype()
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]
        np.random.seed(SEED)

    def setUp(self):
        self.initParams()

        logits = getattr(
            self, "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype))
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        if self.soft_label:
            labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            labels /= np.sum(labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis,
                             self.ignore_index)

        one_hot_label = np.eye(axis_dim)[labels.reshape(-1)]

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Backprop": (softmax - one_hot_label).astype(self.dtype),
            "Softmax": softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype)
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "soft_label": self.soft_label,
            "ignore_index": self.ignore_index,
        }

        if self.axis != -1:
            self.attrs['axis'] = self.axis

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        # fp32 has low precision, cpu and npu both need to relax the max_relative_error if using fp32
        self.check_grad_with_place(
            self.place, ['Logits'],
            'Loss',
            check_dygraph=False,
            numeric_grad_delta=0.001,
            max_relative_error=0.5)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestPowNet(unittest.TestCase):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(32, 32)).astype('float32')
        b_np = np.random.random(size=(32, 32)).astype('float32')
        label_np = np.random.randint(2, size=(32, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[32, 1], dtype='int64')

            sum = paddle.add(a, b)
            z = paddle.pow(sum, 2.0)

            fc_1 = fluid.layers.fc(input=z, size=128)
            prediction = fluid.layers.fc(input=fc_1, size=2)

            cost = fluid.layers.softmax_with_cross_entropy(prediction, label)
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
