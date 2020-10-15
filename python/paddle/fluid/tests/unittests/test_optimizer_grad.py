#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from collections import defaultdict

import paddle.fluid as fluid
import paddle.fluid.optimizer as optimizer
from paddle.fluid.backward import _append_grad_suffix_

import paddle
paddle.enable_static()

np.random.seed(10)

SHAPE = [16, 10]


class SimpleNetWithCond(object):
    """
    Build net with conditional Block and useless layers.
    """

    def __init__(self, test_optimizer, param_lr=1.0, y_no_grad=False):
        self.optimizer = test_optimizer
        self.param_lr = param_lr
        self.shape = SHAPE
        self.y_no_grad = y_no_grad
        self._init_param()

    def _init_param(self):
        self.x = np.ones(self.shape).astype('float32')
        self.y = np.ones(self.shape).astype('float32') * 2.
        self.z = np.ones(self.shape).astype('float32') * 3.

    def _calc_gradient(self, cond_i):
        """
        Calculate grads of params
        """
        grads = []
        d_out_val = np.ones_like(self.x).astype("float32") / np.prod(self.shape)
        grads.append(d_out_val)  # x_grad
        if cond_i > 1:
            y_grad_ratio, z_grad_ratio = 0 if self.y_no_grad else 3, 1
        else:
            y_grad_ratio, z_grad_ratio = 3, 0
        if not self.y_no_grad:
            grads.append(d_out_val * y_grad_ratio)  # y_grad
        grads.append(d_out_val * z_grad_ratio)  # z_grad

        return grads

    def build_net(self, cond_i):
        """
        pseudo code:
            sum_xy = x + y
            sub_yz = y - z
            if i > 1:
                internal = y + z
                sum_cond = internal + z
            else:
                sum_cond = y + z
            sum_all = sum_xy + sum_yz + sum_cond
            mean_out = mean(sum_all)
            optimizer.minimize(mean_out)
        """
        param_x = fluid.layers.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=fluid.ParamAttr(
                learning_rate=self.param_lr, name="param_x"),
            default_initializer=fluid.initializer.NumpyArrayInitializer(self.x))

        param_y = fluid.layers.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=fluid.ParamAttr(
                learning_rate=self.param_lr, name="param_y"),
            default_initializer=fluid.initializer.NumpyArrayInitializer(self.y))
        param_z = fluid.layers.create_parameter(
            dtype="float32",
            shape=self.shape,
            attr=fluid.ParamAttr(
                learning_rate=self.param_lr, name="param_z"),
            default_initializer=fluid.initializer.NumpyArrayInitializer(self.z))

        sum_xy = fluid.layers.elementwise_add(param_x, param_y, name='sum_xy')
        sub_yz = fluid.layers.elementwise_sub(param_y, param_z, name='sub_yz')
        useless = fluid.layers.fc(param_x, size=1, name='fc_useless')

        def cond_true():
            cond_yz = fluid.layers.elementwise_add(
                param_y, param_z, name='sum_cond_yz')
            # param_y will not be updated
            param_y.stop_gradient = self.y_no_grad
            cond_res = fluid.layers.elementwise_add(
                cond_yz, param_z, name='sum_cond_true')
            cond_useless = fluid.layers.elementwise_mul(param_x, param_y)
            return cond_res

        def cond_false():
            cond_res = fluid.layers.elementwise_add(
                param_y, param_z, name='sum_cond_false')
            cond_useless = fluid.layers.elementwise_mul(param_z, param_z)
            return cond_res

        cond_i = fluid.layers.assign(np.array([cond_i], dtype='float32'))
        sum_cond = fluid.layers.cond(cond_i > 1.0, cond_true, cond_false)
        sum_all = fluid.layers.sum([sum_xy, sub_yz, sum_cond])
        mean_out = fluid.layers.mean(sum_all)
        self.optimizer.minimize(mean_out)

        fetch_list = ["param_x", "param_z"] if self.y_no_grad else [
            "param_x", "param_y", "param_z"
        ]
        fetch_list += [_append_grad_suffix_(param) for param in fetch_list]
        return fetch_list


class TestOptimizer(unittest.TestCase):
    """
    TestOptimizer BaseClass to be inherited to test other Optimizer.
    And only need to implement two functions:
        setUp(): to set config info of optimizer, including Optimizer and its hyper-parameter.
        _apply_gradient(): to implement the way of updating grad.
    """

    def setUp(self):
        self._init_config()
        self.optimizer = optimizer.SGDOptimizer(learning_rate=0.001)
        self.attr = {}

    def _init_config(self):
        self.NetClass = SimpleNetWithCond
        self.param_lr = [1.0, 2.0]
        self.cond_i = [0.1, 3]
        self.y_no_grad = [True, False]

    def test_optimizer(self):
        self._check_grads()

    def _apply_gradient(self, param, grad, name):
        """
        The way of updating grad in optimizer.(such as SGD)
        This method should be override.
        """
        return param - self.attr['lr'] * grad

    def _apply_optimize(self, net, grads):
        """
        apply to update all params in the net.
        """
        net.x = self._apply_gradient(net.x, grads[0], 'x')
        if len(grads) == 2:
            net.z = self._apply_gradient(net.z, grads[1], 'z')
            res = [net.x, net.z]
        else:
            net.y = self._apply_gradient(net.y, grads[1], 'y')
            net.z = self._apply_gradient(net.z, grads[2], 'z')
            res = [net.x, net.y, net.z]

        return res

    def _init_param_attr(self):
        self.param_attr = {}
        for key in ['x', 'y', 'z']:
            self.param_attr[key] = self.attr.copy()

    def _check_grads(self):
        """
        main logic code to check the validity of apply_optimize.
        """
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        # test on CPU and GPU
        for place in places:
            for param_lr in self.param_lr:
                for cond_i in self.cond_i:
                    for y_no_grad in self.y_no_grad:
                        self.attr[
                            'lr'] = param_lr * self.optimizer._learning_rate
                        self._init_param_attr()

                        main_program = fluid.Program()
                        init_program = fluid.Program()
                        with fluid.program_guard(main_program, init_program):
                            # reset optimizer._accumulators to avoid duplicate name in loop.
                            self.optimizer._accumulators = defaultdict(
                                lambda: dict())
                            test_net = self.NetClass(self.optimizer, param_lr,
                                                     y_no_grad)
                            fetch_list = test_net.build_net(cond_i)

                            exe = fluid.Executor(place)
                            exe.run(init_program)
                            # Train 2 steps to check validity
                            for batch_i in range(2):

                                res = exe.run(main_program,
                                              fetch_list=fetch_list)
                                gt_grads = test_net._calc_gradient(cond_i)
                                gt_params = self._apply_optimize(test_net,
                                                                 gt_grads)
                                param_grads = gt_params + gt_grads
                                for i in range(len(res)):
                                    np.testing.assert_allclose(res[i],
                                                               param_grads[i])


class TestAdamOptimizer(TestOptimizer):
    """
    inherit TestOptimizer and shall override two functions as follows:
        setUp(): to set config info of optimizer, including Optimizer and its hyper-parameter.
        _apply_gradient(): to implement the way of updating grad.
    """

    def setUp(self):
        self._init_config()
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        self.optimizer = optimizer.AdamOptimizer(
            learning_rate=0.01, beta1=beta1, beta2=beta2, epsilon=epsilon)
        self.attr = {
            "beta1": beta1,
            "beta2": beta2,
            "beta1_pow": beta1,
            "beta2_pow": beta2,
            "moment1": np.zeros(SHAPE).astype("float32"),
            "moment2": np.zeros(SHAPE).astype("float32"),
            "epsilon": epsilon
        }

    def _apply_gradient(self, param, grad, name):
        """
        The way of updating grad in AdamOptimizer
        """
        attr = self.param_attr[name]
        beta1, beta2 = attr["beta1"], attr["beta2"]
        moment1, moment2 = attr['moment1'], attr['moment2']
        beta1_pow, beta2_pow = attr['beta1_pow'], attr['beta2_pow']
        epsilon = attr['epsilon']

        moment1_out = beta1 * moment1 + (1. - beta1) * grad
        moment2_out = beta2 * moment2 + (1. - beta2) * np.square(grad)

        lr = attr['lr'] * np.sqrt(1. - beta2_pow) / (1. - beta1_pow)
        param_out = param - lr * (moment1_out / (np.sqrt(moment2_out) + epsilon
                                                 * np.sqrt(1 - beta2_pow)))

        # update hyper-parameter of optimizer
        self.param_attr[name]['beta1_pow'] = beta1_pow * beta1
        self.param_attr[name]['beta2_pow'] = beta2_pow * beta2
        self.param_attr[name]['moment1'] = moment1_out
        self.param_attr[name]['moment2'] = moment2_out

        return param_out


if __name__ == '__main__':
    unittest.main()
