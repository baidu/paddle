#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import paddle
from paddle import fluid
from paddle.fluid import layers
from paddle.distribution import *
import math


class DistributionNumpy():
    def sample(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def kl_divergence(self, other):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def probs(self, value):
        raise NotImplementedError


class UniformNumpy(DistributionNumpy):
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)
        if str(self.low.dtype) not in ['float32', 'float64']:
            self.low = self.low.astype('float32')
            self.high = self.high.astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.low + self.high).shape
        return self.low + (np.random.uniform(size=shape) *
                           (self.high - self.low))

    def log_prob(self, value):
        lb = np.less(self.low, value).astype(self.low.dtype)
        ub = np.less(value, self.high).astype(self.low.dtype)
        return np.log(lb * ub) - np.log(self.high - self.low)

    def probs(self, value):
        lb = np.less(self.low, value).astype(self.low.dtype)
        ub = np.less(value, self.high).astype(self.low.dtype)
        return (lb * ub) / (self.high - self.low)

    def entropy(self):
        return np.log(self.high - self.low)


class UniformTest(unittest.TestCase):
    def setUp(self, use_gpu=False, batch_size=5, dims=6):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = fluid.CUDAPlace(0)
            self.gpu_id = 0

        self.init_numpy_data(batch_size, dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(batch_size, dims)

        paddle.enable_static()
        self.test_program = fluid.Program()
        self.executor = fluid.Executor(self.place)
        self.init_static_data(batch_size, dims)

    def init_numpy_data(self, batch_size, dims):
        # low ans high are 'float'
        self.low_np = np.random.uniform(-2, 1)
        self.high_np = np.random.uniform(2, 4)
        self.values_np = np.array([1.0]).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = self.low_np
        self.dynamic_high = self.high_np
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[], dtype='float32')

    def compare_with_numpy(self, fetch_list, sample_shape=7, tolerance=1e-6):
        sample, entropy, log_prob, probs = fetch_list

        np_uniform = UniformNumpy(self.low_np, self.high_np)
        np_sample = np_uniform.sample([sample_shape])
        np_entropy = np_uniform.entropy()
        np_lp = np_uniform.log_prob(self.values_np)
        np_p = np_uniform.probs(self.values_np)

        np.testing.assert_equal(sample.shape, np_sample.shape)
        np.testing.assert_allclose(
            entropy, np_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            log_prob, np_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(probs, np_p, rtol=tolerance, atol=tolerance)

    def test_uniform_distribution_dygraph(self, sample_shape=7, tolerance=1e-6):
        paddle.disable_static(self.place)
        uniform = Uniform(self.dynamic_low, self.dynamic_high)
        sample = uniform.sample([sample_shape]).numpy()
        entropy = uniform.entropy().numpy()
        log_prob = uniform.log_prob(self.dynamic_values).numpy()
        probs = uniform.probs(self.dynamic_values).numpy()
        fetch_list = [sample, entropy, log_prob, probs]

        self.compare_with_numpy(fetch_list)

    def test_uniform_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        with fluid.program_guard(self.test_program):
            uniform = Uniform(self.static_low, self.static_high)
            sample = uniform.sample([sample_shape])
            entropy = uniform.entropy()
            log_prob = uniform.log_prob(self.static_values)
            probs = uniform.probs(self.static_values)
            fetch_list = [sample, entropy, log_prob, probs]

        feed_vars = {
            'low': self.low_np,
            'high': self.high_np,
            'values': self.values_np
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(program=self.test_program,
                                       feed=feed_vars,
                                       fetch_list=fetch_list)

        self.compare_with_numpy(fetch_list)


class UniformTest2(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low ans high are 'int'
        self.low_np = int(np.random.uniform(-2, 1))
        self.high_np = int(np.random.uniform(2, 4))
        self.values_np = np.array([1.0]).astype('float32')


class UniformTest3(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # test broadcast: low is float, high is numpy.ndarray with dtype 'float32'.
        self.low_np = np.random.uniform(-2, 1)
        self.high_np = np.random.uniform(5.0, 15.0,
                                         (batch_size, dims)).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class UniformTest4(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(5.0, 15.0,
                                         (batch_size, dims)).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class UniformTest5(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float64'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(5.0, 15.0,
                                         (batch_size, dims)).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = self.low_np
        self.dynamic_high = self.high_np
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')


class UniformTest6(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(5.0, 15.0,
                                         (batch_size, dims)).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np)
        self.dynamic_high = paddle.to_tensor(self.high_np)
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float32')
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float32')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class UniformTest7(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP64'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(5.0, 15.0,
                                         (batch_size, dims)).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np, dtype='float64')
        self.dynamic_high = paddle.to_tensor(self.high_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float64')
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')


class UniformTest8(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP64'. value's dtype is 'VarType.FP32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(5.0, 15.0,
                                         (batch_size, dims)).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np, dtype='float64')
        self.dynamic_high = paddle.to_tensor(self.high_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float32')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float64')
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class UniformTest9(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float32'.
        # high < low.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(-10.0, -5.0,
                                         (batch_size, dims)).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class NormalNumpy(DistributionNumpy):
    def __init__(self, loc, scale):
        self.loc = np.array(loc)
        self.scale = np.array(scale)
        if str(self.loc.dtype) not in ['float32', 'float64']:
            self.loc = self.loc.astype('float32')
            self.scale = self.scale.astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.loc + self.scale).shape
        return self.loc + (np.random.randn(*shape) * self.scale)

    def log_prob(self, value):
        var = self.scale * self.scale
        log_scale = np.log(self.scale)
        return -((value - self.loc) * (value - self.loc)) / (
            2. * var) - log_scale - math.log(math.sqrt(2. * math.pi))

    def probs(self, value):
        var = self.scale * self.scale
        return np.exp(-1. * ((value - self.loc) * (value - self.loc)) /
                      (2. * var)) / (math.sqrt(2 * math.pi) * self.scale)

    def entropy(self):
        return 0.5 + 0.5 * np.log(
            np.array(2. * math.pi).astype(self.loc.dtype)) + np.log(self.scale)

    def kl_divergence(self, other):
        var_ratio = (self.scale / other.scale)
        var_ratio = var_ratio * var_ratio
        t1 = ((self.loc - other.loc) / other.scale)
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


class NormalTest(unittest.TestCase):
    def setUp(self, use_gpu=False, batch_size=2, dims=3):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = fluid.CUDAPlace(0)
            self.gpu_id = 0

        self.init_numpy_data(batch_size, dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(batch_size, dims)

        paddle.enable_static()
        self.test_program = fluid.Program()
        self.executor = fluid.Executor(self.place)
        self.init_static_data(batch_size, dims)

    def init_numpy_data(self, batch_size, dims):
        # loc ans scale are 'float'
        self.loc_np = (np.random.ranf() - 0.5) * 4
        self.scale_np = (np.random.ranf() - 0.5) * 4
        while self.scale_np < 0:
            self.scale_np = (np.random.ranf() - 0.5) * 4
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = (np.random.ranf() - 0.5) * 4
        self.other_scale_np = (np.random.ranf() - 0.5) * 4
        while self.other_scale_np < 0:
            self.other_scale_np = (np.random.ranf() - 0.5) * 4
        self.values_np = np.random.ranf(1).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = self.loc_np
        self.dynamic_scale = self.scale_np
        self.dynamic_other_loc = self.other_loc_np
        self.dynamic_other_scale = self.other_scale_np
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[], dtype='float32')

    def compare_with_numpy(self, fetch_list, sample_shape=7, tolerance=1e-6):
        sample, entropy, log_prob, probs, kl = fetch_list

        np_normal = NormalNumpy(self.loc_np, self.scale_np)
        np_sample = np_normal.sample([sample_shape])
        np_entropy = np_normal.entropy()
        np_lp = np_normal.log_prob(self.values_np)
        np_p = np_normal.probs(self.values_np)
        np_other_normal = NormalNumpy(self.other_loc_np, self.other_scale_np)
        np_kl = np_normal.kl_divergence(np_other_normal)

        # Because assign op does not support the input of numpy.ndarray whose dtype is FP64.
        # When loc and scale are FP64 numpy.ndarray, we need to use assign op to convert it
        #  to FP32 Tensor. And then use cast op to convert it to a FP64 Tensor.
        # There is a loss of accuracy in this conversion.
        # So set the tolerance from 1e-6 to 1e-4.
        log_tolerance = 1e-4

        np.testing.assert_equal(sample.shape, np_sample.shape)
        np.testing.assert_allclose(
            entropy, np_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            log_prob, np_lp, rtol=log_tolerance, atol=log_tolerance)
        np.testing.assert_allclose(
            probs, np_p, rtol=log_tolerance, atol=log_tolerance)
        np.testing.assert_allclose(
            kl, np_kl, rtol=log_tolerance, atol=log_tolerance)

    def test_normal_distribution_dygraph(self, sample_shape=7, tolerance=1e-6):
        paddle.disable_static(self.place)
        normal = Normal(self.dynamic_loc, self.dynamic_scale)

        sample = normal.sample([sample_shape]).numpy()
        entropy = normal.entropy().numpy()
        log_prob = normal.log_prob(self.dynamic_values).numpy()
        probs = normal.probs(self.dynamic_values).numpy()
        other_normal = Normal(self.dynamic_other_loc, self.dynamic_other_scale)
        kl = normal.kl_divergence(other_normal).numpy()

        fetch_list = [sample, entropy, log_prob, probs, kl]
        self.compare_with_numpy(fetch_list)

    def test_normal_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        with fluid.program_guard(self.test_program):
            normal = Normal(self.static_loc, self.static_scale)

            sample = normal.sample([sample_shape])
            entropy = normal.entropy()
            log_prob = normal.log_prob(self.static_values)
            probs = normal.probs(self.static_values)
            other_normal = Normal(self.static_other_loc,
                                  self.static_other_scale)
            kl = normal.kl_divergence(other_normal)

            fetch_list = [sample, entropy, log_prob, probs, kl]

        feed_vars = {
            'loc': self.loc_np,
            'scale': self.scale_np,
            'values': self.values_np,
            'other_loc': self.other_loc_np,
            'other_scale': self.other_scale_np
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(program=self.test_program,
                                       feed=feed_vars,
                                       fetch_list=fetch_list)

        self.compare_with_numpy(fetch_list)


class NormalTest2(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc ans scale are 'int'
        self.loc_np = int((np.random.ranf() - 0.5) * 8)
        self.scale_np = int((np.random.ranf() - 0.5) * 8)
        while self.scale_np < 0:
            self.scale_np = int((np.random.ranf() - 0.5) * 8)
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = int((np.random.ranf() - 0.5) * 8)
        self.other_scale_np = int((np.random.ranf() - 0.5) * 8)
        while self.other_scale_np < 0:
            self.other_scale_np = int((np.random.ranf() - 0.5) * 8)
        self.values_np = np.random.ranf(1).astype('float32')


class NormalTest3(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # test broadcast: loc is float, scale is numpy.ndarray with dtype 'float32'.
        self.loc_np = (np.random.ranf() - 0.5) * 4
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = (np.random.ranf() - 0.5) * 4
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class NormalTest4(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'float32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class NormalTest5(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'float64'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = self.loc_np
        self.dynamic_scale = self.scale_np
        self.dynamic_other_loc = self.other_loc_np
        self.dynamic_other_scale = self.other_scale_np
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')


class NormalTest6(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np)
        self.dynamic_scale = paddle.to_tensor(self.scale_np)
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(self.other_loc_np)
        self.dynamic_other_scale = paddle.to_tensor(self.other_scale_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_loc = layers.data(
                name='loc', shape=[dims], dtype='float32')
            self.static_scale = layers.data(
                name='scale', shape=[dims], dtype='float32')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')
            self.static_other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float32')
            self.static_other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float32')


class NormalTest7(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP64'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='float64')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='float64')
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_loc = layers.data(
                name='loc', shape=[dims], dtype='float64')
            self.static_scale = layers.data(
                name='scale', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')
            self.static_other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float64')
            self.static_other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float64')


class NormalTest8(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP64'. value's dtype is 'VarType.FP32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='float64')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='float64')
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_loc = layers.data(
                name='loc', shape=[dims], dtype='float64')
            self.static_scale = layers.data(
                name='scale', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')
            self.static_other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float64')
            self.static_other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float64')


class CategoricalNumpy(DistributionNumpy):
    def __init__(self, logits):
        self.logits = np.array(logits).astype('float32')

    def entropy(self):
        logits = self.logits - np.max(self.logits, axis=-1, keepdims=True)
        e_logits = np.exp(logits)
        z = np.sum(e_logits, axis=-1, keepdims=True)
        prob = e_logits / z
        return -1. * np.sum(prob * (logits - np.log(z)), axis=-1, keepdims=True)

    def kl_divergence(self, other):
        logits = self.logits - np.max(self.logits, axis=-1, keepdims=True)
        other_logits = other.logits - np.max(
            other.logits, axis=-1, keepdims=True)
        e_logits = np.exp(logits)
        other_e_logits = np.exp(other_logits)
        z = np.sum(e_logits, axis=-1, keepdims=True)
        other_z = np.sum(other_e_logits, axis=-1, keepdims=True)
        prob = e_logits / z
        return np.sum(prob * (logits - np.log(z) - other_logits \
            + np.log(other_z)), axis=-1, keepdims=True)


class CategoricalTest(unittest.TestCase):
    def setUp(self, use_gpu=False, batch_size=3, dims=5):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = fluid.CUDAPlace(0)
            self.gpu_id = 0

        self.batch_size = batch_size
        self.dims = dims
        self.init_numpy_data(batch_size, dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(batch_size, dims)

        paddle.enable_static()
        self.test_program = fluid.Program()
        self.executor = fluid.Executor(self.place)
        self.init_static_data(batch_size, dims)

    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float32')
        self.other_logits_np = np.random.rand(batch_size,
                                              dims).astype('float32')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        # dist_shape = logits_shape[:-1], it represents the number of 
        #  different distributions.
        self.dist_shape = [batch_size]
        # sample shape represents the number of samples
        self.sample_shape = [2, 4]
        # value used in probs and log_prob method
        # If value is 1-D and logits is 2-D or higher dimension, value will be
        #  broadcasted to have the same number of distributions with logits.
        # If value is 2-D or higher dimentsion, it should have the same number 
        #  of distributions with logtis. ``value[:-1] = logits[:-1]
        self.value_shape = [3]

    def init_dynamic_data(self, batch_size, dims):
        self.logits = paddle.to_tensor(self.logits_np)
        self.other_logits = paddle.to_tensor(self.other_logits_np)
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = fluid.data(
                name='logits', shape=self.logits_shape, dtype='float32')
            self.other_logits_static = fluid.data(
                name='other_logits', shape=self.logits_shape, dtype='float32')
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64')

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.dist_shape + self.value_shape)
        for i in range(self.batch_size):
            for j in range(3):
                np_probs[i][j] = probability[i][self.value_np[j]]
        return np_probs

    def compare_with_numpy(self, fetch_list, tolerance=1e-6):
        sample, entropy, kl, probs, log_prob = fetch_list
        log_tolerance = 1e-4

        np.testing.assert_equal(sample.shape,
                                self.sample_shape + self.dist_shape)

        np_categorical = CategoricalNumpy(self.logits_np)
        np_other_categorical = CategoricalNumpy(self.other_logits_np)
        np_entropy = np_categorical.entropy()
        np_kl = np_categorical.kl_divergence(np_other_categorical)

        np.testing.assert_allclose(
            entropy, np_entropy, rtol=log_tolerance, atol=log_tolerance)
        np.testing.assert_allclose(
            kl, np_kl, rtol=log_tolerance, atol=log_tolerance)

        sum_dist = np.sum(self.logits_np, axis=-1, keepdims=True)
        probability = self.logits_np / sum_dist
        np_probs = self.get_numpy_selected_probs(probability)
        np_log_prob = np.log(np_probs)

        np.testing.assert_allclose(
            probs, np_probs, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            log_prob, np_log_prob, rtol=tolerance, atol=tolerance)

    def test_categorical_distribution_dygraph(self, tolerance=1e-6):
        paddle.disable_static(self.place)
        categorical = Categorical(self.logits)
        other_categorical = Categorical(self.other_logits)

        sample = categorical.sample(self.sample_shape).numpy()
        entropy = categorical.entropy().numpy()
        kl = categorical.kl_divergence(other_categorical).numpy()
        probs = categorical.probs(self.value).numpy()
        log_prob = categorical.log_prob(self.value).numpy()

        fetch_list = [sample, entropy, kl, probs, log_prob]
        self.compare_with_numpy(fetch_list)

    def test_categorical_distribution_static(self, tolerance=1e-6):
        paddle.enable_static()
        with fluid.program_guard(self.test_program):
            categorical = Categorical(self.logits_static)
            other_categorical = Categorical(self.other_logits_static)

            sample = categorical.sample(self.sample_shape)
            entropy = categorical.entropy()
            kl = categorical.kl_divergence(other_categorical)
            probs = categorical.probs(self.value_static)
            log_prob = categorical.log_prob(self.value_static)

            fetch_list = [sample, entropy, kl, probs, log_prob]

        feed_vars = {
            'logits': self.logits_np,
            'other_logits': self.other_logits_np,
            'value': self.value_np
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(program=self.test_program,
                                       feed=feed_vars,
                                       fetch_list=fetch_list)

        self.compare_with_numpy(fetch_list)


class CategoricalTest2(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor with dtype Float64
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float64')
        self.other_logits_np = np.random.rand(batch_size,
                                              dims).astype('float64')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        self.dist_shape = [batch_size]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = fluid.data(
                name='logits', shape=self.logits_shape, dtype='float64')
            self.other_logits_static = fluid.data(
                name='other_logits', shape=self.logits_shape, dtype='float64')
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64')


class CategoricalTest3(CategoricalTest):
    def init_dynamic_data(self, batch_size, dims):
        # input logtis is 2-D numpy.ndarray with dtype Float32
        # value used in probs and log_prob method is 1-D Tensor
        self.logits = self.logits_np
        self.other_logits = self.other_logits_np
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = self.logits_np
            self.other_logits_static = self.other_logits_np
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64')


class CategoricalTest4(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D numpy.ndarray with dtype Float64
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float64')
        self.other_logits_np = np.random.rand(batch_size,
                                              dims).astype('float64')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        self.dist_shape = [batch_size]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def init_dynamic_data(self, batch_size, dims):
        self.logits = self.logits_np
        self.other_logits = self.other_logits_np
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = self.logits_np
            self.other_logits_static = self.other_logits_np
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64')


# test shape of logits and value used in probs and log_prob method
class CategoricalTest5(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 1-D Tensor
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(dims).astype('float32')
        self.other_logits_np = np.random.rand(dims).astype('float32')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [dims]
        self.dist_shape = []
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.value_shape)
        for i in range(3):
            np_probs[i] = probability[self.value_np[i]]
        return np_probs


class CategoricalTest6(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor
        # value used in probs and log_prob method has the same number of batches with input
        self.logits_np = np.random.rand(3, 5).astype('float32')
        self.other_logits_np = np.random.rand(3, 5).astype('float32')
        self.value_np = np.array([[2, 1], [0, 3], [2, 3]]).astype('int64')

        self.logits_shape = [3, 5]
        self.dist_shape = [3]
        self.sample_shape = [2, 4]
        self.value_shape = [3, 2]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.value_shape)
        for i in range(3):
            for j in range(2):
                np_probs[i][j] = probability[i][self.value_np[i][j]]
        return np_probs


class CategoricalTest7(CategoricalTest):
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 3-D Tensor
        # value used in probs and log_prob method has the same number of distribuions with input
        self.logits_np = np.random.rand(3, 2, 5).astype('float32')
        self.other_logits_np = np.random.rand(3, 2, 5).astype('float32')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [3, 2, 5]
        self.dist_shape = [3, 2]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.dist_shape + self.value_shape)
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    np_probs[i][j][k] = probability[i][j][self.value_np[k]]
        return np_probs


class DistributionTestError(unittest.TestCase):
    def test_distribution_error(self):
        distribution = Distribution()

        self.assertRaises(NotImplementedError, distribution.sample)
        self.assertRaises(NotImplementedError, distribution.entropy)

        normal = Normal(0.0, 1.0)
        self.assertRaises(NotImplementedError, distribution.kl_divergence,
                          normal)

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        self.assertRaises(NotImplementedError, distribution.log_prob,
                          value_tensor)
        self.assertRaises(NotImplementedError, distribution.probs, value_tensor)

    def test_normal_error(self):
        paddle.enable_static()
        normal = Normal(0.0, 1.0)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, normal.log_prob, value)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, normal.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, normal.sample, shape)

        seed = 1.0
        # type of seed must be int
        self.assertRaises(TypeError, normal.sample, [2, 3], seed)

        normal_other = Uniform(1.0, 2.0)
        # type of other must be an instance of Normal
        self.assertRaises(TypeError, normal.kl_divergence, normal_other)

    def test_uniform_error(self):
        paddle.enable_static()
        uniform = Uniform(0.0, 1.0)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, uniform.log_prob, value)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, uniform.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, uniform.sample, shape)

        seed = 1.0
        # type of seed must be int
        self.assertRaises(TypeError, uniform.sample, [2, 3], seed)

    def test_categorical_error(self):
        paddle.enable_static()

        categorical = Categorical([0.4, 0.6])

        value = [1, 0]
        # type of value must be variable
        self.assertRaises(AttributeError, categorical.log_prob, value)

        value = [1, 0]
        # type of value must be variable
        self.assertRaises(AttributeError, categorical.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, categorical.sample, shape)

        categorical_other = Uniform(1.0, 2.0)
        # type of other must be an instance of Categorical
        self.assertRaises(TypeError, categorical.kl_divergence,
                          categorical_other)

        def test_shape_not_match_error():
            # shape of value must match shape of logits
            # value_shape[:-1] == logits_shape[:-1]
            paddle.disable_static()
            logits = paddle.rand([3, 5])
            cat = Categorical(logits)
            value = paddle.to_tensor([[2, 1, 3], [3, 2, 1]], dtype='int64')
            cat.log_prob(value)

        self.assertRaises(ValueError, test_shape_not_match_error)


class DistributionTestName(unittest.TestCase):
    def get_prefix(self, string):
        return (string.split('.')[0])

    def test_normal_name(self):
        name = 'test_normal'
        normal1 = Normal(0.0, 1.0, name=name)
        self.assertEqual(normal1.name, name)

        normal2 = Normal(0.0, 1.0)
        self.assertEqual(normal2.name, 'Normal')

        paddle.enable_static()

        sample = normal1.sample([2])
        self.assertEqual(self.get_prefix(sample.name), name + '_sample')

        entropy = normal1.entropy()
        self.assertEqual(self.get_prefix(entropy.name), name + '_entropy')

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        layers.assign(value_npdata, value_tensor)

        lp = normal1.log_prob(value_tensor)
        self.assertEqual(self.get_prefix(lp.name), name + '_log_prob')

        p = normal1.probs(value_tensor)
        self.assertEqual(self.get_prefix(p.name), name + '_probs')

        kl = normal1.kl_divergence(normal2)
        self.assertEqual(self.get_prefix(kl.name), name + '_kl_divergence')

    def test_uniform_name(self):
        name = 'test_uniform'
        uniform1 = Uniform(0.0, 1.0, name=name)
        self.assertEqual(uniform1.name, name)

        uniform2 = Uniform(0.0, 1.0)
        self.assertEqual(uniform2.name, 'Uniform')

        paddle.enable_static()

        sample = uniform1.sample([2])
        self.assertEqual(self.get_prefix(sample.name), name + '_sample')

        entropy = uniform1.entropy()
        self.assertEqual(self.get_prefix(entropy.name), name + '_entropy')

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        layers.assign(value_npdata, value_tensor)

        lp = uniform1.log_prob(value_tensor)
        self.assertEqual(self.get_prefix(lp.name), name + '_log_prob')

        p = uniform1.probs(value_tensor)
        self.assertEqual(self.get_prefix(p.name), name + '_probs')

    def test_categorical_name(self):
        name = 'test_categorical'
        categorical1 = Categorical([0.4, 0.6], name=name)
        self.assertEqual(categorical1.name, name)

        categorical2 = Categorical([0.5, 0.5])
        self.assertEqual(categorical2.name, 'Categorical')

        paddle.enable_static()

        sample = categorical1.sample([2])
        self.assertEqual(self.get_prefix(sample.name), name + '_sample')

        entropy = categorical1.entropy()
        self.assertEqual(self.get_prefix(entropy.name), name + '_entropy')

        kl = categorical1.kl_divergence(categorical2)
        self.assertEqual(self.get_prefix(kl.name), name + '_kl_divergence')

        value_npdata = np.array([0], dtype="int64")
        value_tensor = layers.create_tensor(dtype="int64")
        layers.assign(value_npdata, value_tensor)

        p = categorical1.probs(value_tensor)
        self.assertEqual(self.get_prefix(p.name), name + '_probs')

        lp = categorical1.log_prob(value_tensor)
        self.assertEqual(self.get_prefix(lp.name), name + '_log_prob')


if __name__ == '__main__':
    unittest.main()
