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

from __future__ import division
from __future__ import print_function

import unittest

import paddle
import numpy as np
import random
import paddle.distributed as dist
import paddle.fluid as fluid
import paddle.distributed.fleet as fleet
from paddle import framework


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)


class TestParallelMarginSoftmaxCrossEntropyOp(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

    def test_parallel_margin_softmax_cross_entropy(self):
        margin1s = [1.0, 1.0, 1.35]
        margin2s = [0.5, 0.0, 0.0]
        margin3s = [0.0, 0.35, 0.0]
        scales = [64.0, 64.0, 64.0]

        rank_id = dist.get_rank()
        num_trainer = dist.get_world_size()
        batch_size = 2
        feature_length = 4
        seed = 1025
        set_random_seed(seed)
        paddle.seed(rank_id * 10)
        random.seed(seed)
        np.random.seed(seed)

        check_group = dist.new_group(list(range(num_trainer)))
        for dtype in ('float32', 'float64'):

            num_class_per_cards = [[4, 8], [2, 2], [4, 2], [3, 9]]
            for num_class_per_card in num_class_per_cards:

                num_class = np.sum(num_class_per_card)
                for margin1, margin2, margin3, scale in zip(margin1s, margin2s,
                                                            margin3s, scales):

                    for _ in range(5):
                        np_label = np.random.randint(0, num_class,
                                                     (batch_size, ))
                        label = paddle.to_tensor(np_label, dtype="int64")

                        input = paddle.randn(
                            shape=[batch_size, feature_length], dtype=dtype)
                        input.stop_gradient = False
                        input_l2 = paddle.sqrt(
                            paddle.sum(
                                paddle.square(input), axis=1, keepdim=True))
                        norm_input = paddle.divide(input, input_l2)

                        weight = paddle.randn(
                            shape=[
                                feature_length, num_class_per_card[rank_id]
                            ],
                            dtype=dtype)
                        weight.stop_gradient = False
                        weight_l2 = paddle.sqrt(
                            paddle.sum(
                                paddle.square(weight), axis=0, keepdim=True))
                        norm_weight = paddle.divide(weight, weight_l2)

                        data = paddle.matmul(norm_input, norm_weight)
                        data.stop_gradient = False

                        sta = np.sum(
                            num_class_per_card[:rank_id]) if rank_id > 0 else 0
                        end = np.sum(num_class_per_card[:rank_id + 1])

                        integral_data = np.zeros(
                            (batch_size, num_class), dtype=dtype)
                        integral_data[:, sta:end] = data.clone().detach().numpy(
                        )
                        integral_data = paddle.to_tensor(
                            integral_data, dtype=dtype)

                        paddle.distributed.all_reduce(
                            integral_data,
                            op=paddle.distributed.ReduceOp.SUM,
                            group=check_group)
                        integral_data = integral_data.detach().clone()
                        integral_data.stop_gradient = False

                        # add arcface margin to logit
                        theta = paddle.acos(integral_data)
                        one_hot_label = paddle.nn.functional.one_hot(
                            label, num_classes=num_class)
                        one_hot_label.stop_gradient = False

                        if margin1 != 1.0:
                            theta = margin1 * theta
                        if margin2 != 0.0:
                            theta = theta + margin2
                        margin_cos = paddle.cos(theta)
                        if margin3 != 0.0:
                            margin_cos = margin_cos - margin3
                        diff = one_hot_label * (margin_cos - integral_data)
                        arc_data = (integral_data + diff) * scale

                        loss_a, softmax_a = paddle.nn.functional.margin_softmax_with_cross_entropy(
                            data,
                            label,
                            margin1=margin1,
                            margin2=margin2,
                            margin3=margin3,
                            scale=scale,
                            group=check_group,
                            return_softmax=True,
                            reduction=None)
                        loss_b, softmax_b = paddle.nn.functional.softmax_with_cross_entropy(
                            logits=arc_data,
                            label=paddle.reshape(label, (-1, 1)),
                            return_softmax=True)

                        np.testing.assert_allclose(
                            loss_a.numpy(), loss_b.numpy(), rtol=1e-5)

                        integral_prob = np.zeros(
                            (batch_size, num_class), dtype=dtype)
                        integral_prob[:, sta:end] = softmax_a.clone().detach(
                        ).numpy()
                        integral_prob = paddle.to_tensor(
                            integral_prob, dtype=dtype)
                        paddle.distributed.all_reduce(
                            integral_prob,
                            op=paddle.distributed.ReduceOp.SUM,
                            group=check_group)
                        integral_prob = integral_prob.detach().clone()
                        integral_prob.stop_gradient = False

                        np.testing.assert_allclose(
                            integral_prob.numpy(),
                            softmax_b.numpy(),
                            rtol=1e-5,
                            atol=1e-6)

                        loss_a = loss_a.sum() / batch_size
                        loss_b = loss_b.sum() / batch_size
                        loss_a.backward()
                        loss_b.backward()

                        integral_grad = np.zeros(
                            (batch_size, num_class), dtype=dtype)
                        integral_grad[:, sta:end] = data.grad.clone().detach()
                        integral_grad = paddle.to_tensor(
                            integral_grad, dtype=dtype)
                        paddle.distributed.all_reduce(
                            integral_grad,
                            op=paddle.distributed.ReduceOp.SUM,
                            group=check_group)

                        np.testing.assert_allclose(
                            integral_data.grad.numpy(),
                            integral_grad.numpy(),
                            rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
