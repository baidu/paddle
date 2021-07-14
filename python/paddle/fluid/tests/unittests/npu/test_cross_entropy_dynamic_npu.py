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
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestForwardNetDynamic(unittest.TestCase):
    def _test(self, run_npu=True):
        if not paddle.in_dynamic_mode():
            paddle.disable_static()
        if run_npu:
            paddle.set_device('npu:0')
        else:
            paddle.set_device('cpu')

        np.random.seed(2021)
        paddle.seed(2021)

        b_np = np.random.random(size=(32, 32)).astype('float32')
        label_np = np.random.randint(2, size=(32, 1)).astype('int64')

        class MLP(paddle.nn.Layer):
            def __init__(self):
                super(MLP, self).__init__()
                self.linear1 = paddle.nn.Linear(in_features=32, out_features=2)

            def forward(self, x):
                x = paddle.flatten(x, start_axis=1, stop_axis=-1)
                x = self.linear1(x)
                return x

        def train(model):
            model.train()
            epochs = 20
            optim = paddle.optimizer.Adam(
                learning_rate=0.001, parameters=model.parameters())
            for epoch in range(epochs):
                x_data = paddle.to_tensor(b_np)
                y_data = paddle.to_tensor(label_np)
                predicts = model(x_data)
                loss = F.cross_entropy(predicts, y_data)
                acc = paddle.metric.accuracy(predicts, y_data)
                loss.backward()
                print("epoch: {}, loss is: {}, acc is: {}".format(
                    epoch, loss.numpy(), acc.numpy()))
                optim.step()
                optim.clear_grad()

        model = MLP()
        train(model)

        def test(model):
            model.eval()
            acc_set = []
            avg_loss_set = []
            x_data = paddle.to_tensor(b_np)
            y_data = paddle.to_tensor(label_np)
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            avg_loss = paddle.mean(loss)
            avg_loss_set.append(float(avg_loss.numpy()))
            acc = paddle.metric.accuracy(predicts, y_data)
            acc_set.append(float(acc.numpy()))
            acc_val_mean = np.array(acc_set).mean()
            avg_loss_val_mean = np.array(avg_loss_set).mean()
            print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))
            return predicts, avg_loss_val_mean, acc_val_mean

        predicts, loss, acc = test(model)
        return predicts, loss, acc

    def test_npu(self):
        cpu_pred, cpu_loss, cpu_acc = self._test(False)
        npu_pred, npu_loss, npu_acc = self._test(True)

        # relax the rtol=1e-05 to 5e-5
        self.assertTrue(np.allclose(npu_pred, cpu_pred, rtol=5e-5))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))
        self.assertTrue(np.allclose(npu_acc, cpu_acc))


if __name__ == '__main__':
    unittest.main()
