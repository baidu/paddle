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


def call_bce_layer(logit, label, weight=None, reduction='mean',
                   pos_weight=None):
    bce_logit_loss = paddle.nn.loss.BCEWithLogitsLoss(
        weight=weight, reduction=reduction, pos_weight=pos_weight)
    res = bce_logit_loss(logit, label)
    return res


def call_bce_functional(logit,
                        label,
                        weight=None,
                        reduction='mean',
                        pos_weight=None):
    res = paddle.nn.functional.binary_cross_entropy_with_logits(
        logit, label, weight=weight, reduction=reduction, pos_weight=pos_weight)
    return res


def test_static(place,
                logit_np,
                label_np,
                weight_np=None,
                reduction='mean',
                pos_weight_np=None,
                functional=False):
    paddle.enable_static()
    prog = paddle.static.Program()
    startup_prog = paddle.static.Program()
    with paddle.static.program_guard(prog, startup_prog):
        logit = paddle.fluid.data(name='logit', shape=logit_np.shape, dtype='float64')
        label = paddle.fluid.data(name='label', shape=label_np.shape, dtype='float64')
        feed_dict = {"logit": logit_np, "label": label_np}

        pos_weight = None
        weight = None
        if pos_weight_np is not None:
            pos_weight = paddle.fluid.data(
                name='pos_weight', shape=pos_weight_np.shape, dtype='float64')
            feed_dict["pos_weight"] = pos_weight_np
        if weight_np is not None:
            weight = paddle.fluid.data(
                name='weight', shape=weight_np.shape, dtype='float64')
            feed_dict["weight"] = weight_np
        if functional:
            res = call_bce_functional(logit, label, weight, reduction,
                                      pos_weight)
        else:
            res = call_bce_layer(logit, label, weight, reduction, pos_weight)
        exe = paddle.static.Executor(place)
        static_result = exe.run(prog, feed=feed_dict, fetch_list=[res])
    return static_result


def test_dygraph(place,
                 logit_np,
                 label_np,
                 weight_np=None,
                 reduction='mean',
                 pos_weight_np=None,
                 functional=False):
    paddle.disable_static()
    logit = paddle.to_tensor(logit_np)
    label = paddle.to_tensor(label_np)
    weight = None
    pos_weight = None
    if weight_np is not None:
        weight = paddle.to_tensor(weight_np)
    if pos_weight_np is not None:
        pos_weight = paddle.to_tensor(pos_weight_np)
    if functional:
        dy_res = call_bce_functional(logit, label, weight, reduction,
                                     pos_weight)
    else:
        dy_res = call_bce_layer(logit, label, weight, reduction, pos_weight)
    dy_result = dy_res.numpy()
    paddle.enable_static()
    return dy_result


def calc_bce_with_logits_loss(logit_np,
                              label_np,
                              reduction='mean',
                              weight_np=None,
                              pos_weight=None):
    expected = np.maximum(
        logit_np,
        0) - logit_np * label_np + np.log(1 + np.exp(-np.abs(logit_np)))
    if pos_weight is not None:
        expected = expected * ((pos_weight - 1) * label_np + 1)
    if weight_np is not None:
        expected = weight_np * expected

    if reduction == 'mean':
        expected = np.mean(expected)
    elif reduction == 'sum':
        expected = np.sum(expected)
    else:
        expected = expected

    return expected


class TestBCEWithLogitsLoss(unittest.TestCase):
    def test_BCEWithLogitsLoss(self):
        logit_np = np.random.uniform(0.1, 0.8, size=(20, 30)).astype(np.float64)
        label_np = np.random.randint(0, 2, size=(20, 30)).astype(np.float64)
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        reductions = ['sum', 'mean', 'none']
        for place in places:
            for reduction in reductions:
                static_result = test_static(
                    place, logit_np, label_np, reduction=reduction)
                dy_result = test_dygraph(
                    place, logit_np, label_np, reduction=reduction)
                expected = calc_bce_with_logits_loss(logit_np, label_np,
                                                     reduction)
                self.assertTrue(np.allclose(static_result, expected))
                self.assertTrue(np.allclose(static_result, dy_result))
                self.assertTrue(np.allclose(dy_result, expected))
                static_functional = test_static(
                    place,
                    logit_np,
                    label_np,
                    reduction=reduction,
                    functional=True)
                dy_functional = test_dygraph(
                    place,
                    logit_np,
                    label_np,
                    reduction=reduction,
                    functional=True)
                self.assertTrue(np.allclose(static_functional, expected))
                self.assertTrue(np.allclose(static_functional, dy_functional))
                self.assertTrue(np.allclose(dy_functional, expected))

    def test_BCEWithLogitsLoss_weight(self):
        logit_np = np.random.uniform(
            0.1, 0.8, size=(2, 3, 4, 10)).astype(np.float64)
        label_np = np.random.randint(
            0, 2, size=(2, 3, 4, 10)).astype(np.float64)
        weight_np = np.random.random(size=(2, 3, 4, 10)).astype(np.float64)
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        for reduction in ['sum', 'mean', 'none']:
            static_result = test_static(
                place,
                logit_np,
                label_np,
                weight_np=weight_np,
                reduction=reduction)
            dy_result = test_dygraph(
                place,
                logit_np,
                label_np,
                weight_np=weight_np,
                reduction=reduction)
            expected = calc_bce_with_logits_loss(
                logit_np, label_np, reduction, weight_np=weight_np)
            self.assertTrue(np.allclose(static_result, expected))
            self.assertTrue(np.allclose(static_result, dy_result))
            self.assertTrue(np.allclose(dy_result, expected))
            static_functional = test_static(
                place,
                logit_np,
                label_np,
                weight_np=weight_np,
                reduction=reduction,
                functional=True)
            dy_functional = test_dygraph(
                place,
                logit_np,
                label_np,
                weight_np=weight_np,
                reduction=reduction,
                functional=True)
            self.assertTrue(np.allclose(static_functional, expected))
            self.assertTrue(np.allclose(static_functional, dy_functional))
            self.assertTrue(np.allclose(dy_functional, expected))

    def test_BCEWithLogitsLoss_pos_weight(self):
        logit_np = np.random.uniform(
            0.1, 0.8, size=(2, 3, 4, 10)).astype(np.float64)
        label_np = np.random.randint(
            0, 2, size=(2, 3, 4, 10)).astype(np.float64)
        pos_weight_np = np.random.random(size=(3, 4, 10)).astype(np.float64)
        weight_np = np.random.random(size=(2, 3, 4, 10)).astype(np.float64)
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        reduction = "mean"
        static_result = test_static(place, logit_np, label_np, weight_np,
                                    reduction, pos_weight_np)
        dy_result = test_dygraph(place, logit_np, label_np, weight_np,
                                 reduction, pos_weight_np)
        expected = calc_bce_with_logits_loss(logit_np, label_np, reduction,
                                             weight_np, pos_weight_np)
        self.assertTrue(np.allclose(static_result, expected))
        self.assertTrue(np.allclose(static_result, dy_result))
        self.assertTrue(np.allclose(dy_result, expected))
        static_functional = test_static(
            place,
            logit_np,
            label_np,
            weight_np,
            reduction,
            pos_weight_np,
            functional=True)
        dy_functional = test_dygraph(
            place,
            logit_np,
            label_np,
            weight_np,
            reduction,
            pos_weight_np,
            functional=True)
        self.assertTrue(np.allclose(static_functional, expected))
        self.assertTrue(np.allclose(static_functional, dy_functional))
        self.assertTrue(np.allclose(dy_functional, expected))

    def test_BCEWithLogitsLoss_error(self):
        paddle.disable_static()
        self.assertRaises(
            ValueError,
            paddle.nn.BCEWithLogitsLoss,
            reduction="unsupport reduction")
        logit = paddle.to_tensor([[0.1, 0.3]], dtype='float32')
        label = paddle.to_tensor([[0.0, 1.0]], dtype='float32')
        self.assertRaises(
            ValueError,
            paddle.nn.functional.binary_cross_entropy_with_logits,
            logit=logit,
            label=label,
            reduction="unsupport reduction")
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
