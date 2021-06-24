# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import unicode_literals
from __future__ import print_function

import os
import sys
import time
import numpy as np

os.environ[str("FLAGS_check_nan_inf")] = str("1")
os.environ[str("GLOG_vmodule")] = str("nan_inf_utils_detail=10")

import paddle
import paddle.nn as nn

np.random.seed(0)


def generator():
    batch_size = 5
    for i in range(5):
        curr_train_x = np.random.randint(
            batch_size, size=(batch_size, 3)).astype("float32")
        if i >= 2:
            curr_train_x[0, :] = np.nan
            curr_train_x[-1, :] = np.inf
        res = []
        for i in range(batch_size):
            y = i % 3
            res.append([y])
        y_label = np.array(res).astype('int64')
        yield [curr_train_x, y_label]


class TestLayer(nn.Layer):
    def __init__(self):
        super(TestLayer, self).__init__()
        self.linear1 = nn.Linear(3, 400)
        self.linear2 = nn.Linear(400, 400)
        self.linear3 = nn.Linear(400, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.sigmoid(x)
        x = self.linear2(x)
        x = nn.functional.sigmoid(x)
        x = self.linear3(x)
        x = nn.functional.softmax(x)

        return x


def check(use_cuda):
    paddle.set_device('gpu' if use_cuda else 'cpu')

    net = TestLayer()
    sgd = paddle.optimizer.SGD(learning_rate=0.05, parameters=net.parameters())

    for step, (x, y) in enumerate(generator()):
        x = paddle.to_tensor(x)
        y = paddle.to_tensor(y)

        zero = paddle.zeros(shape=[1], dtype='int64')
        fp16_zero = paddle.cast(zero, dtype='float16')

        y = y + zero

        y_pred = net(x)

        cost = nn.functional.cross_entropy(y_pred, y, use_softmax=False)
        avg_cost = paddle.mean(cost)

        acc_top1 = paddle.metric.accuracy(input=y_pred, label=y, k=1)

        print('iter={:.0f}, cost={}, acc1={}'.format(
            step, avg_cost.numpy(), acc_top1.numpy()))

        sgd.step()
        sgd.clear_grad()


if __name__ == '__main__':
    if paddle.is_compiled_with_cuda():
        try:
            check(use_cuda=True)
            assert False
        except Exception as e:
            print(e)
            print(type(e))
            # Note. Enforce in cuda kernel may not catch in paddle, and
            # Exception type will be RuntimeError
            assert type(e) == OSError or type(e) == RuntimeError
    try:
        check(use_cuda=False)
        assert False
    except Exception as e:
        print(e)
        print(type(e))
        assert type(e) == RuntimeError
