# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest
from op_test import OpTest


def nll_loss_1d(logs, targets, weight=None, reduction='mean',
                ignore_index=-100):
    input_shape = logs.shape
    N = input_shape[0]
    C = input_shape[1]
    out = np.zeros_like(targets).astype(np.float64)
    total_weight = 0
    for i in range(N):
        cur_target = targets[i]
        if cur_target == ignore_index:
            out[i] = 0
            continue
        cur_weight = weight[cur_target] if weight is not None else 1
        total_weight += cur_weight
        out[i] = -logs[i][cur_target] * cur_weight
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        return out.sum() / total_weight, np.array(
            [total_weight]).astype('float64')
    elif reduction == 'none':
        return out


def nll_loss_2d(logs, targets, weight=None, reduction='mean',
                ignore_index=-100):
    input_shape = logs.shape
    N = input_shape[0]
    H = input_shape[2]
    W = input_shape[3]
    out = np.zeros_like(targets).astype(np.float64)
    total_weight = 0
    for i in range(N):
        for h in range(H):
            for w in range(W):
                cur_target = targets[i][h][w]
                if cur_target == ignore_index:
                    out[i][h][w] = 0
                    continue
                cur_weight = weight[cur_target] if weight is not None else 1
                total_weight += cur_weight
                out[i][h][w] = -logs[i][cur_target][h][w] * cur_weight
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        return out.sum() / total_weight, np.array(
            [total_weight]).astype('float64')
    elif reduction == 'none':
        return out


class TestNLLLoss(unittest.TestCase):
    def test_NLLLoss_1D_mean(self):
        input_np = np.random.random(size=(10, 10)).astype(np.float64)
        label_np = np.random.randint(0, 10, size=(10, )).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[10, 10], dtype='float64')
            label = fluid.data(name='label', shape=[10], dtype='int64')
            nll_loss = paddle.nn.loss.NLLLoss()
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss()
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = nll_loss_1d(input_np, label_np)[0]
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_1D_sum(self):
        input_np = np.random.random(size=(10, 10)).astype(np.float64)
        label_np = np.random.randint(0, 10, size=(10, )).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[10, 10], dtype='float64')
            label = fluid.data(name='label', shape=[10], dtype='int64')
            nll_loss = paddle.nn.loss.NLLLoss(reduction='sum')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(reduction='sum')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = nll_loss_1d(input_np, label_np, reduction='sum')[0]
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_1D_with_weight_mean(self):
        input_np = np.random.random(size=(10, 10)).astype(np.float64)
        label_np = np.random.randint(0, 10, size=(10, )).astype(np.int64)
        weight_np = np.random.random(size=(10, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        # place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[10, 10], dtype='float64')
            label = fluid.data(name='label', shape=[10], dtype='int64')
            weight = fluid.data(name='weight', shape=[10], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight)
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()
        expected = nll_loss_1d(input_np, label_np, weight=weight_np)[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_1D_with_weight_sum(self):
        input_np = np.random.random(size=(10, 10)).astype(np.float64)
        label_np = np.random.randint(0, 10, size=(10, )).astype(np.int64)
        weight_np = np.random.random(size=(10, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        # place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[10, 10], dtype='float64')
            label = fluid.data(name='label', shape=[10], dtype='int64')
            weight = fluid.data(name='weight', shape=[10], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight, reduction='sum')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='sum')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()
        expected = nll_loss_1d(
            input_np, label_np, weight=weight_np, reduction='sum')[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_1D_with_weight_mean_cpu(self):
        input_np = np.random.random(size=(10, 10)).astype(np.float64)
        label_np = np.random.randint(0, 10, size=(10, )).astype(np.int64)
        weight_np = np.random.random(size=(10, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[10, 10], dtype='float64')
            label = fluid.data(name='label', shape=[10], dtype='int64')
            weight = fluid.data(name='weight', shape=[10], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight)
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()
        expected = nll_loss_1d(input_np, label_np, weight=weight_np)[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_1D_with_weight_no_reduce_cpu(self):
        input_np = np.random.random(size=(10, 10)).astype(np.float64)
        label_np = np.random.randint(0, 10, size=(10, )).astype(np.int64)
        weight_np = np.random.random(size=(10, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[10, 10], dtype='float64')
            label = fluid.data(name='label', shape=[10], dtype='int64')
            weight = fluid.data(name='weight', shape=[10], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight, reduction='none')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='none')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()
        expected = nll_loss_1d(
            input_np, label_np, weight=weight_np, reduction='none')

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_2D_mean(self):
        input_np = np.random.random(size=(5, 3, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5)).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5], dtype='int64')
            nll_loss = paddle.nn.loss.NLLLoss()
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss()
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = nll_loss_2d(input_np, label_np)[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_2D_sum(self):
        input_np = np.random.random(size=(5, 3, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5)).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5], dtype='int64')
            nll_loss = paddle.nn.loss.NLLLoss(reduction='sum')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(reduction='sum')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = nll_loss_2d(input_np, label_np, reduction='sum')[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_2D_with_weight_mean(self):
        input_np = np.random.random(size=(5, 3, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5)).astype(np.int64)
        weight_np = np.random.random(size=(3, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')

            nll_loss = paddle.nn.loss.NLLLoss(weight=weight)
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = nll_loss_2d(input_np, label_np, weight=weight_np)[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_2D_with_weight_mean_cpu(self):
        input_np = np.random.random(size=(5, 3, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5)).astype(np.int64)
        weight_np = np.random.random(size=(3, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')

            nll_loss = paddle.nn.loss.NLLLoss(weight=weight)
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = nll_loss_2d(input_np, label_np, weight=weight_np)[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_2D_with_weight_sum(self):
        input_np = np.random.random(size=(5, 3, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5)).astype(np.int64)
        weight_np = np.random.random(size=(3, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')

            nll_loss = paddle.nn.loss.NLLLoss(weight=weight, reduction='sum')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='sum')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        expected = nll_loss_2d(
            input_np, label_np, weight=weight_np, reduction='sum')[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_in_dims_not_2or4_mean(self):
        input_np = np.random.random(size=(5, 3, 5, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5, 5)).astype(np.int64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5, 5], dtype='int64')
            nll_loss = paddle.nn.loss.NLLLoss()
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(
                prog,
                feed={"input": input_np,
                      "label": label_np},
                fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss()
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        input_shape = input_np.shape
        label_shape = label_np.shape
        input_np_reshape = np.reshape(input_np,
                                      (input_shape[0], input_shape[1], 1, -1))
        label_np_reshape = np.reshape(label_np, (label_shape[0], 1, -1))
        expected = nll_loss_2d(input_np_reshape, label_np_reshape)[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_in_dims_not_2or4_with_weight_mean(self):
        input_np = np.random.random(size=(5, 3, 5, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5, 5)).astype(np.int64)
        weight_np = np.random.random(size=(3, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5, 5], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight)
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np))
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        input_shape = input_np.shape
        label_shape = label_np.shape
        input_np_reshape = np.reshape(input_np,
                                      (input_shape[0], input_shape[1], 1, -1))
        label_np_reshape = np.reshape(label_np, (label_shape[0], 1, -1))
        expected = nll_loss_2d(
            input_np_reshape, label_np_reshape, weight=weight_np)[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_in_dims_not_2or4_with_weight_sum(self):
        input_np = np.random.random(size=(5, 3, 5, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5, 5)).astype(np.int64)
        weight_np = np.random.random(size=(3, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5, 5], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight, reduction='sum')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='sum')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        input_shape = input_np.shape
        label_shape = label_np.shape
        input_np_reshape = np.reshape(input_np,
                                      (input_shape[0], input_shape[1], 1, -1))
        label_np_reshape = np.reshape(label_np, (label_shape[0], 1, -1))
        expected = nll_loss_2d(
            input_np_reshape,
            label_np_reshape,
            weight=weight_np,
            reduction='sum')[0]

        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_in_dims_not_2or4_with_weight_no_reduce(self):
        input_np = np.random.random(size=(5, 3, 5, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5, 5)).astype(np.int64)
        weight_np = np.random.random(size=(3, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        #place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5, 5], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight, reduction='none')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='none')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        input_shape = input_np.shape
        label_shape = label_np.shape
        out_shape = (input_shape[0], ) + input_shape[2:]
        input_np_reshape = np.reshape(input_np,
                                      (input_shape[0], input_shape[1], 1, -1))
        label_np_reshape = np.reshape(label_np, (label_shape[0], 1, -1))
        expected = nll_loss_2d(
            input_np_reshape,
            label_np_reshape,
            weight=weight_np,
            reduction='none')
        expected = np.reshape(expected, out_shape)
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))

    def test_NLLLoss_in_dims_not_2or4_with_weight_no_reduce_cpu(self):
        input_np = np.random.random(size=(5, 3, 5, 5, 5)).astype(np.float64)
        label_np = np.random.randint(0, 3, size=(5, 5, 5, 5)).astype(np.int64)
        weight_np = np.random.random(size=(3, )).astype(np.float64)
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[5, 3, 5, 5, 5], dtype='float64')
            label = fluid.data(name='label', shape=[5, 5, 5, 5], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')
            nll_loss = paddle.nn.loss.NLLLoss(weight=weight, reduction='none')
            res = nll_loss(input, label)

            exe = fluid.Executor(place)
            static_result = exe.run(prog,
                                    feed={
                                        "input": input_np,
                                        "label": label_np,
                                        "weight": weight_np
                                    },
                                    fetch_list=[res])

        with fluid.dygraph.guard():
            nll_loss = paddle.nn.loss.NLLLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='none')
            dy_res = nll_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_result = dy_res.numpy()

        input_shape = input_np.shape
        label_shape = label_np.shape
        out_shape = (input_shape[0], ) + input_shape[2:]
        input_np_reshape = np.reshape(input_np,
                                      (input_shape[0], input_shape[1], 1, -1))
        label_np_reshape = np.reshape(label_np, (label_shape[0], 1, -1))
        expected = nll_loss_2d(
            input_np_reshape,
            label_np_reshape,
            weight=weight_np,
            reduction='none')
        expected = np.reshape(expected, out_shape)
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))


class TestNLLLossOp1DWithReduce(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "nll_loss"
        self.with_weight = False
        input_np = np.random.uniform(0.1, 0.8,
                                     self.input_shape).astype("float64")
        label_np = np.random.randint(0, self.input_shape[1],
                                     self.label_shape).astype("int64")
        output_np, total_weight_np = nll_loss_1d(input_np, label_np)
        self.inputs = {'X': input_np, 'Label': label_np}
        if self.with_weight:
            weight_np = np.random.uniform(0.1, 0.8,
                                          self.input_shape[1]).astype("float64")
            output_np, total_weight_np = nll_loss_1d(
                input_np, label_np, weight=weight_np)
            self.inputs['Weight'] = weight_np

        self.outputs = {'Out': output_np, 'Total_weight': total_weight_np}
        self.attrs = {'reduction': 'mean', 'ignore_index': -100}

    def test_check_output(self):
        self.check_output()

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output()

    def test_check_grad(self):
        self.with_weight = True
        place = fluid.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out')
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')

    def init_test_case(self):
        self.input_shape = [10, 10]
        self.label_shape = [10]


class TestNLLLossOp1DNoReduce(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "nll_loss"
        self.with_weight = False
        input_np = np.random.uniform(0.1, 0.8,
                                     self.input_shape).astype("float64")
        label_np = np.random.randint(0, self.input_shape[1],
                                     self.label_shape).astype("int64")
        output_np = nll_loss_1d(input_np, label_np, reduction='none')
        total_weight_np = np.array([0]).astype('float64')
        self.inputs = {'X': input_np, 'Label': label_np}
        if self.with_weight:
            weight_np = np.random.uniform(0.1, 0.8,
                                          self.input_shape[1]).astype("float64")
            output_np, total_weight_np = nll_loss_1d(
                input_np, label_np, weight=weight_np, reduction='none')
            self.inputs['Weight'] = weight_np

        self.outputs = {'Out': output_np, 'Total_weight': total_weight_np}
        self.attrs = {'reduction': 'none', 'ignore_index': -100}

    def test_check_output(self):
        self.check_output()

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output()

    def test_check_grad(self):
        self.with_weight = True
        place = fluid.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out')
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')

    def init_test_case(self):
        self.input_shape = [10, 10]
        self.label_shape = [10]


class TestNLLLossOp2DWithReduce(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "nll_loss"
        self.with_weight = False
        input_np = np.random.uniform(0.1, 0.8,
                                     self.input_shape).astype("float64")
        label_np = np.random.randint(0, self.input_shape[1],
                                     self.label_shape).astype("int64")
        output_np, total_weight_np = nll_loss_2d(input_np, label_np)
        self.inputs = {'X': input_np, 'Label': label_np}
        if self.with_weight:
            weight_np = np.random.uniform(0.1, 0.8,
                                          self.input_shape[1]).astype("float64")
            output_np, total_weight_np = nll_loss_2d(
                input_np, label_np, weight=weight_np)
            self.inputs['Weight'] = weight_np

        self.outputs = {'Out': output_np, 'Total_weight': total_weight_np}
        self.attrs = {'reduction': 'mean', 'ignore_index': -100}

    def test_check_output(self):
        self.check_output()

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output()

    def test_check_grad(self):
        self.with_weight = True
        place = fluid.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out')
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')

    def init_test_case(self):
        self.input_shape = [5, 3, 5, 5]
        self.label_shape = [5, 5, 5]


class TestNLLLossOp2DNoReduce(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "nll_loss"
        self.with_weight = False
        input_np = np.random.uniform(0.1, 0.8,
                                     self.input_shape).astype("float64")
        label_np = np.random.randint(0, self.input_shape[1],
                                     self.label_shape).astype("int64")
        output_np = nll_loss_2d(input_np, label_np, reduction='none')
        total_weight_np = np.array([0]).astype('float64')
        self.inputs = {'X': input_np, 'Label': label_np}
        if self.with_weight:
            weight_np = np.random.uniform(0.1, 0.8,
                                          self.input_shape[1]).astype("float64")
            output_np, total_weight_np = nll_loss_2d(
                input_np, label_np, weight=weight_np, reduction='none')
            self.inputs['Weight'] = weight_np

        self.outputs = {'Out': output_np, 'Total_weight': total_weight_np}
        self.attrs = {'reduction': 'none', 'ignore_index': -100}

    def test_check_output(self):
        self.check_output()

    def test_check_output_with_weight(self):
        self.with_weight = True
        self.check_output()

    def test_check_grad(self):
        self.with_weight = True
        place = fluid.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out')
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')

    def init_test_case(self):
        self.input_shape = [5, 3, 5, 5]
        self.label_shape = [5, 5, 5]


class TestNLLLossName(unittest.TestCase):
    def test_name(self):
        prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        place = paddle.CPUPlace()
        with paddle.static.program_guard(prog, startup_prog):
            x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float64')
            label = paddle.fluid.data(name='label', shape=[10], dtype='int64')
            nll_loss = paddle.nn.loss.NLLLoss(name='nll_loss')
            res = nll_loss(x, label)
            self.assertTrue(res.name.startswith('nll_loss'))


class TestNLLLossInvalidArgs(unittest.TestCase):
    def test_x_dim_value_error(self):
        def test_x_dim_lt_2():
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                x = paddle.fluid.data(name='x', shape=[10, ], dtype='float64')
                label = paddle.fluid.data(name='label', shape=[10, ], dtype='float64')
                nll_loss = paddle.nn.loss.NLLLoss()
                res = nll_loss(x, label)

        self.assertRaises(ValueError, test_x_dim_lt_2)

        def test_x_dim_imperative_lt_2():
            with fluid.dygraph.guard():
                x_np = np.random.random(size=(5, )).astype(np.float64)
                label_np = np.random.randint(0, 10, size=(5, )).astype(np.int64)
                x = paddle.to_tensor(x_np)
                label = paddle.to_tensor(label_np)
                nll_loss = paddle.nn.loss.NLLLoss()
                res = nll_loss(x, label)

        self.assertRaises(ValueError, test_x_dim_imperative_lt_2)

    def test_reduction_value_error(self):
        def test_NLLLoss_reduction_not_sum_mean_none():
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float64')
                label = paddle.fluid.data(name='label', shape=[10], dtype='int64')
                nll_loss = paddle.nn.loss.NLLLoss(reduction='')
                res = nll_loss(x, label)

        self.assertRaises(ValueError, test_NLLLoss_reduction_not_sum_mean_none)

        def test_NLLLoss_reduction_imperative_not_sum_mean_none():
            with fluid.dygraph.guard():
                x_np = np.random.random(size=(5, 3)).astype(np.float64)
                label_np = np.random.randint(0, 3, size=(5, )).astype(np.int64)
                x = paddle.to_tensor(x_np)
                label = paddle.to_tensor(label_np)
                nll_loss = paddle.nn.loss.NLLLoss(reduction='')
                res = nll_loss(x, label)

        self.assertRaises(ValueError,
                          test_NLLLoss_reduction_imperative_not_sum_mean_none)

        def test_nll_loss_function_reduction_not_sum_mean_none():
            prog = paddle.static.Program()
            startup_prog = paddle.static.Program()
            place = paddle.CPUPlace()
            with paddle.static.program_guard(prog, startup_prog):
                x = paddle.fluid.data(name='x', shape=[10, 10], dtype='float64')
                label = paddle.fluid.data(name='label', shape=[10], dtype='int64')
                res = paddle.nn.functional.nll_loss(x, label, reduction='')

        self.assertRaises(ValueError,
                          test_nll_loss_function_reduction_not_sum_mean_none)

        def test_nll_loss_function_reduction_imperative_not_sum_mean_none():
            with fluid.dygraph.guard():
                x_np = np.random.random(size=(5, 3)).astype(np.float64)
                label_np = np.random.randint(0, 3, size=(5, )).astype(np.int64)
                x = paddle.to_tensor(x_np)
                label = paddle.to_tensor(label_np)
                res = paddle.nn.functional.nll_loss(x, label, reduction='')

        self.assertRaises(
            ValueError,
            test_nll_loss_function_reduction_imperative_not_sum_mean_none)


if __name__ == "__main__":
    unittest.main()
