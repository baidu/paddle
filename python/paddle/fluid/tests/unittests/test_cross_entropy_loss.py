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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import numpy as np
import unittest


def stable_softmax(x):
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def log_softmax(x, axis=-1):
    softmax_out = np.apply_along_axis(stable_softmax, axis, x)
    return np.log(softmax_out)


def cross_entropy_loss_1d(input,
                          label,
                          weight=None,
                          reduction='mean',
                          ignore_index=-100):
    log_softmax_out = log_softmax(input)
    input_shape = log_softmax_out.shape
    N = input_shape[0]
    C = input_shape[1]
    out = np.zeros_like(label).astype(np.float64)
    total_weight = 0
    for i in range(N):
        cur_target = label[i]
        if cur_target == ignore_index:
            out[i] = 0
            continue
        cur_weight = weight[cur_target] if weight is not None else 1
        total_weight += cur_weight
        out[i] = -log_softmax_out[i][cur_target] * cur_weight
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        return out.sum() / total_weight, np.array(
            [total_weight]).astype('float64')
    elif reduction == 'none':
        return out


def cross_entropy_loss_2d(input,
                          label,
                          weight=None,
                          reduction='mean',
                          ignore_index=-100):
    log_softmax_out = log_softmax(input)
    input_shape = log_softmax_out.shape
    N = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]

    out = np.zeros_like(label).astype(np.float64)
    total_weight = 0
    for i in range(N):
        for h in range(H):
            for w in range(W):
                cur_target = label[i][h][w]
                if cur_target == ignore_index:
                    out[i][h][w] = 0
                    continue
                cur_weight = weight[cur_target] if weight is not None else 1
                total_weight += cur_weight
                out[i][h][w] = -log_softmax_out[i][h][w][
                    cur_target] * cur_weight
    if reduction == 'sum':
        return np.sum(out), np.array([total_weight]).astype('float64')
    elif reduction == 'mean':
        return out.sum() / total_weight, np.array(
            [total_weight]).astype('float64')
    elif reduction == 'none':
        return out


class CrossEntropyLoss(unittest.TestCase):
    def test_cross_entropy_loss_1d_with_weight_mean(self):
        input_np = np.random.random([2, 4]).astype(np.float64)
        label_np = np.random.randint(0, 4, size=(2)).astype(np.int64)
        weight_np = np.random.random([4]).astype(np.float64)  #shape:C
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[2, 4], dtype='float64')
            label = fluid.data(name='label', shape=[2], dtype='int64')
            weight = fluid.data(
                name='weight', shape=[4],
                dtype='float64')  #weight for each class
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight)
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np)[0]

        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), axis=1)
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np)[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_1d_with_weight_sum(self):
        input_np = np.random.random([100, 200]).astype(np.float64)  #N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  #N,1
        weight_np = np.random.random([200]).astype(np.float64)  #C
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            weight = fluid.data(name='weight', shape=[200], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='sum')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='sum')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np, reduction='sum')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_1d_with_weight_none(self):
        input_np = np.random.random([100, 200]).astype(np.float64)  #N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  #N,1
        weight_np = np.random.random([200]).astype(np.float64)  #C
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            weight = fluid.data(name='weight', shape=[200], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='none')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='none')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(
            input_np, label_np, weight=weight_np, reduction='none')
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_1d_mean(self):
        input_np = np.random.random([100, 200]).astype(np.float64)  #N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  #N,1
        weight_np = np.random.random([200]).astype(np.float64)  #C
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            weight = fluid.data(name='weight', shape=[100], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss()
            ret = cross_entropy_loss(input, label)
            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={'input': input_np,
                                       'label': label_np},
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss()
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np)[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_1d_sum(self):
        input_np = np.random.random([100, 200]).astype(np.float64)  #N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  #N,1
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum')
            ret = cross_entropy_loss(input, label)
            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={'input': input_np,
                                       'label': label_np},
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np, reduction='sum')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_1d_none(self):
        input_np = np.random.random([100, 200]).astype(np.float64)  #N,C
        label_np = np.random.randint(0, 100, size=(100)).astype(np.int64)  #N,1
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(name='input', shape=[100, 200], dtype='float64')
            label = fluid.data(name='label', shape=[100], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none')
            ret = cross_entropy_loss(input, label)
            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={'input': input_np,
                                       'label': label_np},
                                 fetch_list=[ret])
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_1d(input_np, label_np, reduction='none')
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_2d_with_weight_none(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(np.float64)  #NHWC
        label_np = np.random.randint(
            0, 3, size=(2, 2, 2)).astype(np.int64)  #NHW1
        weight_np = np.random.random(size=(3, )).astype(np.float64)  #C

        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[2, 2, 2, 3], dtype='float64')
            label = fluid.data(name='label', shape=[2, 2, 2], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='none')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='none')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, weight=weight_np, reduction='none')
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_2d_with_weight_mean(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(np.float64)  #NHWC
        label_np = np.random.randint(
            0, 3, size=(2, 2, 2)).astype(np.int64)  #NHW
        weight_np = np.random.random(size=(3, )).astype(np.float64)  #C
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[2, 2, 2, 3], dtype='float64')
            label = fluid.data(name='label', shape=[2, 2, 2], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='mean')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='mean')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, weight=weight_np, reduction='mean')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_2d_with_weight_sum(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(np.float64)  #NHWC
        label_np = np.random.randint(
            0, 3, size=(2, 2, 2)).astype(np.int64)  #NHW
        weight_np = np.random.random(size=(3, )).astype(np.float64)  #C
        paddle.enable_static()

        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[2, 2, 2, 3], dtype='float64')
            label = fluid.data(name='label', shape=[2, 2, 2], dtype='int64')
            weight = fluid.data(name='weight', shape=[3], dtype='float64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=weight, reduction='sum')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                     "weight": weight_np
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                weight=fluid.dygraph.to_variable(weight_np), reduction='sum')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, weight=weight_np, reduction='sum')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_2d_none(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(np.float64)  #NHWC
        label_np = np.random.randint(
            0, 3, size=(2, 2, 2)).astype(np.int64)  #NHW
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[2, 2, 2, 3], dtype='float64')
            label = fluid.data(name='label', shape=[2, 2, 2], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none')
            ret = cross_entropy_loss(input, label)
            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                 },
                                 fetch_list=[ret])
            static_ret = np.squeeze(static_ret)
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='none')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            dy_ret_value = np.squeeze(dy_ret_value)
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(input_np, label_np, reduction='none')
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_2d_mean(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(np.float64)  #NHWC
        label_np = np.random.randint(
            0, 3, size=(2, 2, 2)).astype(np.int64)  #NHW
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[2, 2, 2, 3], dtype='float64')
            label = fluid.data(name='label', shape=[2, 2, 2], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='mean')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='mean')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(
            input_np, label_np, reduction='mean')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))

    def test_cross_entropy_loss_2d_sum(self):
        input_np = np.random.random(size=(2, 2, 2, 3)).astype(np.float64)  #NHWC
        label_np = np.random.randint(
            0, 3, size=(2, 2, 2)).astype(np.int64)  #NHW
        paddle.enable_static()
        prog = fluid.Program()
        startup_prog = fluid.Program()
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.program_guard(prog, startup_prog):
            input = fluid.data(
                name='input', shape=[2, 2, 2, 3], dtype='float64')
            label = fluid.data(name='label', shape=[2, 2, 2], dtype='int64')
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum')
            ret = cross_entropy_loss(input, label)

            exe = fluid.Executor(place)
            static_ret = exe.run(prog,
                                 feed={
                                     'input': input_np,
                                     'label': label_np,
                                 },
                                 fetch_list=[ret])
            self.assertIsNotNone(static_ret)
        with fluid.dygraph.guard():
            cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
                reduction='sum')
            dy_ret = cross_entropy_loss(
                fluid.dygraph.to_variable(input_np),
                fluid.dygraph.to_variable(label_np))
            dy_ret_value = dy_ret.numpy()
            self.assertIsNotNone(dy_ret_value)
        expected = cross_entropy_loss_2d(input_np, label_np, reduction='sum')[0]
        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, expected))
        self.assertTrue(np.allclose(dy_ret_value, expected))


if __name__ == "__main__":
    unittest.main()
