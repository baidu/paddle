# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import copy
import math
import numpy as np
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
import paddle.fluid.core as core


def reduce_lr_on_plateau(decay_rate, threshold, cooldown, patience, m, n, loss,
                         var_list):
    def is_better(current, best, m, n):
        if m == 'min' and n == 'rel':
            return current < best - best * threshold
        elif m == 'min' and n == 'abs':
            return current < best - threshold
        elif m == 'max' and n == 'rel':
            return current > best + best * threshold
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return current > best + threshold

    if var_list[2] > 0:
        var_list[2] -= 1
        return var_list[1]

    if is_better(loss, var_list[0], m, n):
        var_list[0] = loss
        var_list[3] = 0
    else:
        var_list[3] += 1
        if var_list[3] > patience:
            var_list[2] = cooldown
            var_list[3] = 0
            new_lr = var_list[1] * decay_rate
            var_list[1] = new_lr if var_list[1] - new_lr > 1e-8 else var_list[1]

    return var_list[1]


class TestReduceOnPlateauDecay(object):
    def test_ReduceLR(self):
        # the decay rate must be less than 1.0
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1.0, factor=2.0)
        # the mode must be "min" or "max"
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate=1.0, mode="test")
        # the threshold_mode must be "rel" or "abs"
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.ReduceOnPlateau(
                learning_rate=1.0, threshold_mode="test")
        with self.assertRaises(TypeError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate="test")
        with self.assertRaises(TypeError):
            paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.5).step("test")

        places = [paddle.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        for place in places:
            for m, n in zip(['min', 'max', 'min', 'max'],
                            ['rel', 'rel', 'abs', 'abs']):
                kwargs = {
                    'learning_rate': 1.0,
                    'mode': m,
                    'factor': 0.5,
                    'patience': 3,
                    'threshold': 1e-4,
                    'threshold_mode': n,
                    'cooldown': 1,
                    'min_lr': 0,
                    'epsilon': 1e-8,
                    'verbose': False,
                }
                paddle.enable_static()
                self._test_static(place, kwargs)
                paddle.disable_static(place)
                self._test_dygraph(place, kwargs)
                paddle.enable_static()

    def _test_static(self, place, kwargs):
        paddle.enable_static()

        best = float("-10000") if kwargs['mode'] == "max" else float("10000")
        current_lr = 1.0
        cooldown_counter = 0
        num_bad_epochs = 0
        var_list = [best, current_lr, cooldown_counter, num_bad_epochs]

        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            x = fluid.layers.create_global_var(
                [1], 1, 'float32', persistable=True)
            paddle.increment(x)
            loss = paddle.sin(x)
            scheduler = paddle.optimizer.lr.ReduceOnPlateau(**kwargs)
            adam = paddle.optimizer.Adam(learning_rate=scheduler)
            adam.minimize(loss)
            lr_var = adam._global_learning_rate()
            test_prog = main_prog.clone()

        exe = paddle.static.Executor(place)
        exe.run(start_prog)

        for epoch in range(20):
            for batch_id in range(1):
                out, actual_lr = exe.run(main_prog,
                                         fetch_list=[loss.name, lr_var.name])
                expected_lr = reduce_lr_on_plateau(
                    kwargs['factor'], kwargs['threshold'], kwargs['cooldown'],
                    kwargs['patience'], kwargs['mode'],
                    kwargs['threshold_mode'], out[0], var_list)

            scheduler.step(out[0])
            actual_lr = scheduler()
            self.assertEqual(actual_lr, np.array(expected_lr))

        for epoch in range(10):
            for batch_id in range(1):
                out, actual_lr = exe.run(test_prog,
                                         fetch_list=[loss.name, lr_var.name])
                expected_lr = reduce_lr_on_plateau(
                    kwargs['factor'], kwargs['threshold'], kwargs['cooldown'],
                    kwargs['patience'], kwargs['mode'],
                    kwargs['threshold_mode'], out[0], var_list)
            scheduler.step(out[0])
            actual_lr = scheduler()
            self.assertEqual(actual_lr, np.array(expected_lr))

    def _test_dygraph(self, place, kwargs):
        paddle.disable_static(place)

        best = float("-10000") if kwargs['mode'] == "max" else float("10000")
        current_lr = 1.0
        cooldown_counter = 0
        num_bad_epochs = 0
        var_list = [best, current_lr, cooldown_counter, num_bad_epochs]

        linear = paddle.nn.Linear(10, 10)
        scheduler = paddle.optimizer.lr.ReduceOnPlateau(**kwargs)
        adam = paddle.optimizer.Adam(
            learning_rate=scheduler, parameters=linear.parameters())

        for epoch in range(20):
            for batch_id in range(1):
                x = paddle.to_tensor(epoch).astype('float32')
                loss = paddle.sin(x)
                loss.backward()
                adam.step()
                adam.clear_grad()

            scheduler.step(loss)
            # get lr from paddle
            current_lr = adam.get_lr()
            # get lr form python
            expected_lr = reduce_lr_on_plateau(
                kwargs['factor'], kwargs['threshold'], kwargs['cooldown'],
                kwargs['patience'], kwargs['mode'], kwargs['threshold_mode'],
                loss, var_list)
            self.assertEqual(current_lr, expected_lr)
        state_dict = adam.state_dict()
        scheduler1 = paddle.optimizer.lr.ReduceOnPlateau(**kwargs)
        adam1 = paddle.optimizer.Adam(
            learning_rate=scheduler1, parameters=linear.parameters())
        adam1.set_state_dict(state_dict)
        self.assertEqual(scheduler.cooldown_counter,
                         scheduler1.cooldown_counter)
        self.assertEqual(scheduler.best.numpy()[0], scheduler1.best)
        self.assertEqual(scheduler.num_bad_epochs, scheduler1.num_bad_epochs)
        self.assertEqual(scheduler.last_epoch, scheduler1.last_epoch)
        self.assertEqual(scheduler.last_lr, scheduler1.last_lr)


def noam_lr(epoch_num, d_model, warmup_steps, learning_rate=1.0, verbose=False):
    if epoch_num == 0:
        a = 1
    else:
        a = math.pow(epoch_num, -0.5)
    b = math.pow(warmup_steps, -1.5) * epoch_num
    return learning_rate * math.pow(d_model, -0.5) * min(a, b)


def lambda_lr(epoch_num, learning_rate, lr_lambda, verbose=False):
    return learning_rate * lr_lambda(epoch_num)


def piecewise_lr(epoch_num, boundaries, values, verbose=False):
    assert len(boundaries) + 1 == len(values)
    for i in range(len(boundaries)):
        if epoch_num < boundaries[i]:
            return values[i]
    return values[len(values) - 1]


def exponential_lr(epoch_num, learning_rate, gamma, verbose=False):
    return learning_rate * gamma**epoch_num


def natural_exp_lr(epoch_num, learning_rate, gamma, verbose=False):
    return learning_rate * math.exp(-1 * gamma * epoch_num)


def inverse_time_lr(epoch_num, learning_rate, gamma, verbose=False):
    return learning_rate / (1 + gamma * epoch_num)


def polynomial_lr(epoch_num,
                  learning_rate,
                  decay_steps,
                  end_lr=0.0001,
                  power=1.0,
                  cycle=False,
                  verbose=False):

    if cycle:
        div = math.ceil(epoch_num / float(decay_steps))
        if epoch_num == 0:
            div = 1
        decay_steps = decay_steps * div
    else:
        epoch_num = min(epoch_num, decay_steps)
    return (learning_rate - end_lr) * (
        (1 - float(epoch_num) / float(decay_steps))**power) + end_lr

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (
            1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (
                self.last_lr - self.eta_min) + self.eta_min


cosine_annealing_lr_current = None


def cosine_annealing_lr(epoch_num,
                        learning_rate,
                        T_max,
                        eta_min=0,
                        verbose=False):
    global cosine_annealing_lr_current
    if epoch_num == 0:
        cosine_annealing_lr_current = learning_rate
    elif (epoch_num - 1 - T_max) % (2 * T_max) == 0:
        cosine_annealing_lr_current = cosine_annealing_lr_current + (
            learning_rate - eta_min) * (1 - math.cos(math.pi / float(T_max))
                                        ) / 2
    else:
        cosine_annealing_lr_current = (1 + math.cos(
            math.pi * epoch_num / float(T_max))) / (1 + math.cos(math.pi * (
                epoch_num - 1) / float(T_max))) * (cosine_annealing_lr_current -
                                                   eta_min) + eta_min
    return cosine_annealing_lr_current


def linear_warmup_lr(epoch_num,
                     learning_rate,
                     warmup_steps,
                     start_lr,
                     end_lr,
                     verbose=False):
    if epoch_num < warmup_steps:
        return start_lr + (end_lr - start_lr) * (float(epoch_num) /
                                                 float(warmup_steps))
    else:
        return learning_rate


def multi_step_lr(epoch_num,
                  learning_rate,
                  milestones,
                  gamma=0.1,
                  verbose=False):
    for i in range(len(milestones)):
        if epoch_num < milestones[i]:
            return learning_rate * (gamma**i)
    return learning_rate * (gamma**len(milestones))


def step_lr(epoch_num, learning_rate, step_size, gamma=0.1, verbose=False):
    return learning_rate * math.pow(gamma, epoch_num // step_size)


class TestLRScheduler(unittest.TestCase):
    def _test_static(self, python_func, paddle_api, kwarg, place):
        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            x = paddle.static.data(name='x', shape=[3, 4, 5])
            y = paddle.static.data(name='y', shape=[3, 4, 5])
            z = paddle.static.nn.fc(x, 100)
            loss = paddle.mean(z)
            scheduler = paddle_api(**kwarg)
            adam = paddle.optimizer.Adam(learning_rate=scheduler)
            adam.minimize(loss)
            lr_var = adam._global_learning_rate()
            test_prog = main_prog.clone()

        num = 0
        exe = paddle.static.Executor(place)
        exe.run(start_prog)
        for epoch in range(5):
            for batch_id in range(2):
                out = exe.run(
                    main_prog,
                    feed={
                        'x': np.random.randn(3, 4, 5).astype('float32'),
                        'y': np.random.randn(3, 4, 5).astype('float32')
                    },
                    fetch_list=lr_var.name)
            self.assertEqual(out, np.array(python_func(num, **kwarg)))
            scheduler.step()
            num += 1

        for epoch in range(5):
            for batch_id in range(2):
                out = exe.run(
                    test_prog,
                    feed={
                        'x': np.random.randn(3, 4, 5).astype('float32'),
                        'y': np.random.randn(3, 4, 5).astype('float32')
                    },
                    fetch_list=lr_var.name)
            self.assertEqual(out, np.array(python_func(num, **kwarg)))
            scheduler.step()
            num += 1

        if isinstance(place, paddle.CPUPlace):
            compiled_train_prog = paddle.static.CompiledProgram(
                main_prog).with_data_parallel(
                    loss_name=loss.name, places=fluid.cpu_places(4))
            for epoch in range(5):
                python_result = python_func(num, **kwarg)
                for batch_id in range(2):
                    _ = exe.run(
                        compiled_train_prog,
                        feed={
                            'x': np.random.randn(12, 4, 5).astype('float32'),
                            'y': np.random.randn(12, 4, 5).astype('float32')
                        },
                        fetch_list=lr_var.name)
                scopes = compiled_train_prog._executor.local_scopes()
                out = np.array(scopes[0].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[1].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[2].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[3].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                scheduler.step()
                num += 1

            compiled_test_prog = paddle.static.CompiledProgram(
                test_prog).with_data_parallel(
                    loss_name=loss.name,
                    share_vars_from=compiled_train_prog,
                    places=fluid.cpu_places(4))
            for epoch in range(5):
                python_result = python_func(num, **kwarg)
                for batch_id in range(2):
                    _ = exe.run(
                        compiled_test_prog,
                        feed={
                            'x': np.random.randn(12, 4, 5).astype('float32'),
                            'y': np.random.randn(12, 4, 5).astype('float32')
                        },
                        fetch_list=lr_var.name)
                scopes = compiled_test_prog._executor.local_scopes()
                out = np.array(scopes[0].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[1].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[2].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                out = np.array(scopes[3].var(lr_var.name).get_tensor())
                self.assertEqual(out, np.array(python_result))
                scheduler.step()
                num += 1

    def _test_dygraph(self, python_func, paddle_api, kwarg, place):
        paddle.disable_static(place)
        x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
        linear = paddle.nn.Linear(10, 10)
        scheduler = paddle_api(**kwarg)
        adam = paddle.optimizer.Adam(
            learning_rate=scheduler, parameters=linear.parameters())
        for epoch in range(20):
            for batch_id in range(2):
                x = paddle.to_tensor(x)
                out = linear(x)
                loss = paddle.mean(out)
                loss.backward()
                adam.step()
                adam.clear_grad()
            current_lr = adam.get_lr()
            expected_lr = python_func(epoch, **kwarg)
            if paddle_api.__name__ != "CosineAnnealingDecay":
                self.assertEqual(current_lr, expected_lr)
                scheduler.step()
            else:
                self.assertAlmostEqual(current_lr, expected_lr)
                scheduler.step(epoch + 1)

    def test_scheduler(self):
        with self.assertRaises(NotImplementedError):
            paddle.optimizer.lr.LRScheduler().step()
        with self.assertRaises(TypeError):
            paddle.optimizer.lr.MultiStepDecay(
                learning_rate="test", milestones=[1, 2, 3])
        with self.assertRaises(TypeError):
            paddle.optimizer.lr.MultiStepDecay(
                learning_rate=0.5, milestones='test')
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.MultiStepDecay(
                learning_rate=0.5, milestones=[3, 2, 1])
        with self.assertRaises(ValueError):
            paddle.optimizer.lr.MultiStepDecay(
                learning_rate=0.5, milestones=[1, 2, 3], gamma=2)

        func_api_kwargs = [(noam_lr, paddle.optimizer.lr.NoamDecay, {
            "d_model": 0.01,
            "warmup_steps": 100,
            "verbose": False
        }), (piecewise_lr, paddle.optimizer.lr.PiecewiseDecay, {
            "boundaries": [3, 6, 9, 15, 20],
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "verbose": False
        }), (natural_exp_lr, paddle.optimizer.lr.NaturalExpDecay, {
            "learning_rate": 0.5,
            "gamma": 0.1,
            "verbose": True
        }), (inverse_time_lr, paddle.optimizer.lr.InverseTimeDecay, {
            "learning_rate": 0.5,
            "gamma": 0.1,
            "verbose": False
        }), (polynomial_lr, paddle.optimizer.lr.PolynomialDecay, {
            "learning_rate": 0.5,
            "decay_steps": 20,
            "end_lr": 0,
            "power": 1.0,
            "cycle": False,
            "verbose": True
        }), (polynomial_lr, paddle.optimizer.lr.PolynomialDecay, {
            "learning_rate": 0.5,
            "decay_steps": 20,
            "end_lr": 0,
            "power": 1.0,
            "cycle": True,
            "verbose": False
        }), (linear_warmup_lr, paddle.optimizer.lr.LinearWarmup, {
            'learning_rate': 0.5,
            'warmup_steps': 20,
            'start_lr': 0,
            'end_lr': 0.5,
            "verbose": True
        }), (exponential_lr, paddle.optimizer.lr.ExponentialDecay, {
            "learning_rate": 0.5,
            "gamma": 0.9,
            "verbose": False
        }), (multi_step_lr, paddle.optimizer.lr.MultiStepDecay, {
            "learning_rate": 0.5,
            "milestones": [3, 6, 9, 15, 20],
            "gamma": 0.8,
            "verbose": True
        }), (step_lr, paddle.optimizer.lr.StepDecay, {
            "learning_rate": 0.5,
            "step_size": 2,
            "gamma": 0.8,
            "verbose": False
        }), (lambda_lr, paddle.optimizer.lr.LambdaDecay, {
            "learning_rate": 0.5,
            "lr_lambda": lambda x: 0.95**x,
            "verbose": True
        }), (cosine_annealing_lr, paddle.optimizer.lr.CosineAnnealingDecay, {
            "learning_rate": 0.5,
            "T_max": 10,
            "verbose": False
        })]

        for python_func, paddle_api, kwarg in func_api_kwargs:
            places = [paddle.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(paddle.CUDAPlace(0))

            for place in places:
                paddle.enable_static()
                #self._test_static(python_func, paddle_api, kwarg, place)
                paddle.disable_static(place)
                self._test_dygraph(python_func, paddle_api, kwarg, place)
                paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
