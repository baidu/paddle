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
        self.low = np.array(low).astype('float32')
        self.high = np.array(high).astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.low + self.high).shape
        return self.low + (np.random.uniform(size=shape) *
                           (self.high - self.low))

    def log_prob(self, value):
        lb = np.less(self.low, value).astype('float32')
        ub = np.less(value, self.high).astype('float32')
        return np.log(lb * ub) - np.log(self.high - self.low)

    def probs(self, value):
        return np.exp(self.log_prob(value))

    def entropy(self):
        return np.log(self.high - self.low)


class NormalNumpy(DistributionNumpy):
    def __init__(self, loc, scale):
        self.loc = np.array(loc).astype('float32')
        self.scale = np.array(scale).astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.loc + self.scale).shape
        return self.loc + (np.random.randn(*shape) * self.scale)

    def log_prob(self, value):
        var = self.scale * self.scale
        log_scale = np.log(self.scale)
        return -((value - self.loc) * (value - self.loc)) / (
            2. * var) - log_scale - math.log(math.sqrt(2. * math.pi))

    def probs(self, value):
        return np.exp(self.log_prob(value))

    def entropy(self):
        return 0.5 + 0.5 * np.log(np.array(2. * math.pi).astype(
            'float32')) + np.log(self.scale)

    def kl_divergence(self, other):
        var_ratio = (self.scale / other.scale)
        var_ratio = var_ratio * var_ratio
        t1 = ((self.loc - other.loc) / other.scale)
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


class DistributionTest(unittest.TestCase):
    def setUp(self, use_gpu=False):
        self.use_gpu = use_gpu
        if not use_gpu:
            place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            place = fluid.CUDAPlace(0)
            self.gpu_id = 0
        self.executor = fluid.Executor(place)

    def build_normal_program(self, test_program, batch_size, dims, loc_float,
                             scale_float, other_loc_float, other_scale_float,
                             scale_np, other_scale_np, loc_np, other_loc_np,
                             values_np):
        with fluid.program_guard(test_program):
            loc = layers.data(name='loc', shape=[dims], dtype='float32')
            scale = layers.data(name='scale', shape=[dims], dtype='float32')

            other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float32')
            other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float32')

            values = layers.data(name='values', shape=[dims], dtype='float32')

            normal_float = Normal(loc_float, scale_float)
            other_normal_float = Normal(other_loc_float, other_scale_float)

            normal_float_np_broadcast = Normal(loc_float, scale_np)
            other_normal_float_np_broadcast = Normal(other_loc_float,
                                                     other_scale_np)

            normal_np = Normal(loc_np, scale_np)
            other_normal_np = Normal(other_loc_np, other_scale_np)

            normal_variable = Normal(loc, scale)
            other_normal_variable = Normal(other_loc, other_scale)

            sample_float = normal_float.sample([batch_size, dims])
            sample_float_np_broadcast = normal_float_np_broadcast.sample(
                [batch_size, dims])
            sample_np = normal_np.sample([batch_size, dims])
            sample_variable = normal_variable.sample([batch_size, dims])

            entropy_float = normal_float.entropy()
            entropy_float_np_broadcast = normal_float_np_broadcast.entropy()
            entropy_np = normal_np.entropy()
            entropy_variable = normal_variable.entropy()

            lp_float_np_broadcast = normal_float_np_broadcast.log_prob(values)
            lp_np = normal_np.log_prob(values)
            lp_variable = normal_variable.log_prob(values)

            p_float_np_broadcast = normal_float_np_broadcast.probs(values)
            p_np = normal_np.probs(values)
            p_variable = normal_variable.probs(values)

            kl_float = normal_float.kl_divergence(other_normal_float)
            kl_float_np_broadcast = normal_float_np_broadcast.kl_divergence(
                other_normal_float_np_broadcast)
            kl_np = normal_np.kl_divergence(other_normal_np)
            kl_variable = normal_variable.kl_divergence(other_normal_variable)

        fetch_list = [
            sample_float, sample_float_np_broadcast, sample_np, sample_variable,
            entropy_float, entropy_float_np_broadcast, entropy_np,
            entropy_variable, lp_float_np_broadcast, lp_np, lp_variable,
            p_float_np_broadcast, p_np, p_variable, kl_float,
            kl_float_np_broadcast, kl_np, kl_variable
        ]
        feed_vars = {
            'loc': loc_np,
            'scale': scale_np,
            'other_loc': other_loc_np,
            'other_scale': other_scale_np,
            'values': values_np
        }
        return feed_vars, fetch_list

    def get_normal_random_input(self, batch_size, dims):
        loc_np = np.random.randn(batch_size, dims).astype('float32')
        other_loc_np = np.random.randn(batch_size, dims).astype('float32')

        loc_float = (np.random.ranf() - 0.5) * 4
        scale_float = (np.random.ranf() - 0.5) * 4
        while scale_float < 0:
            scale_float = (np.random.ranf() - 0.5) * 4

        other_loc_float = (np.random.ranf() - 0.5) * 4
        other_scale_float = (np.random.ranf() - 0.5) * 4
        while other_scale_float < 0:
            other_scale_float = (np.random.ranf() - 0.5) * 4

        scale_np = np.random.randn(batch_size, dims).astype('float32')
        other_scale_np = np.random.randn(batch_size, dims).astype('float32')
        values_np = np.random.randn(batch_size, dims).astype('float32')

        while not np.all(scale_np > 0):
            scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(other_scale_np > 0):
            other_scale_np = np.random.randn(batch_size, dims).astype('float32')
        return loc_np, other_loc_np, loc_float, scale_float, other_loc_float, \
               other_scale_float, scale_np, other_scale_np, values_np

    def test_normal_distribution(self, batch_size=2, dims=3, tolerance=1e-6):
        test_program = fluid.Program()
        loc_np, other_loc_np, loc_float, scale_float, other_loc_float, other_scale_float, scale_np, other_scale_np, values_np = self.get_normal_random_input(
            batch_size, dims)

        feed_vars, fetch_list = self.build_normal_program(
            test_program, batch_size, dims, loc_float, scale_float,
            other_loc_float, other_scale_float, scale_np, other_scale_np,
            loc_np, other_loc_np, values_np)
        self.executor.run(fluid.default_startup_program())

        np_normal_float = NormalNumpy(loc_float, scale_float)
        np_other_normal_float = NormalNumpy(other_loc_float, other_scale_float)
        np_normal_float_np_broadcast = NormalNumpy(loc_float, scale_np)
        np_other_normal_float_np_broadcast = NormalNumpy(other_loc_float,
                                                         other_scale_np)
        np_normal = NormalNumpy(loc_np, scale_np)
        np_other_normal = NormalNumpy(other_loc_np, other_scale_np)

        gt_sample_float = np_normal_float.sample([batch_size, dims])
        gt_sample_float_np_broadcast = np_normal_float_np_broadcast.sample(
            [batch_size, dims])
        gt_sample_np = np_normal.sample([batch_size, dims])
        gt_entropy_float = np_normal_float.entropy()
        gt_entropy_float_np_broadcast = np_normal_float_np_broadcast.entropy()
        gt_entropy = np_normal.entropy()
        gt_lp_float_np_broadcast = np_normal_float_np_broadcast.log_prob(
            values_np)
        gt_lp = np_normal.log_prob(values_np)
        gt_p_float_np_broadcast = np_normal_float_np_broadcast.probs(values_np)
        gt_p = np_normal.probs(values_np)
        gt_kl_float = np_normal_float.kl_divergence(np_other_normal_float)
        gt_kl_float_np_broadcast = np_normal_float_np_broadcast.kl_divergence(
            np_other_normal_float_np_broadcast)
        gt_kl = np_normal.kl_divergence(np_other_normal)

        [
            output_sample_float, output_sample_float_np_broadcast,
            output_sample_np, output_sample_variable, output_entropy_float,
            output_entropy_float_np_broadcast, output_entropy_np,
            output_entropy_variable, output_lp_float_np_broadcast, output_lp_np,
            output_lp_variable, output_p_float_np_broadcast, output_p_np,
            output_p_variable, output_kl_float, output_kl_float_np_broadcast,
            output_kl_np, output_kl_variable
        ] = self.executor.run(program=test_program,
                              feed=feed_vars,
                              fetch_list=fetch_list)

        np.testing.assert_allclose(
            output_sample_float.shape,
            gt_sample_float.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_float_np_broadcast.shape,
            gt_sample_float_np_broadcast.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_np.shape,
            gt_sample_np.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_variable.shape,
            gt_sample_np.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_float,
            gt_entropy_float,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_float_np_broadcast,
            gt_entropy_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_np, gt_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_variable, gt_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_lp_float_np_broadcast,
            gt_lp_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_lp_np, gt_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_lp_variable, gt_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_float_np_broadcast,
            gt_p_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_p_np, gt_p, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_variable, gt_p, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_kl_float, gt_kl_float, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_kl_float_np_broadcast,
            gt_kl_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_kl_np, gt_kl, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_kl_variable, gt_kl, rtol=tolerance, atol=tolerance)

    def build_uniform_program(self, test_program, batch_size, dims, low_float,
                              high_float, high_np, low_np, values_np):
        with fluid.program_guard(test_program):
            low = layers.data(name='low', shape=[dims], dtype='float32')
            high = layers.data(name='high', shape=[dims], dtype='float32')

            values = layers.data(name='values', shape=[dims], dtype='float32')

            uniform_float = Uniform(low_float, high_float)
            uniform_float_np_broadcast = Uniform(low_float, high_np)
            uniform_np = Uniform(low_np, high_np)
            uniform_variable = Uniform(low, high)

            sample_float = uniform_float.sample([batch_size, dims])
            sample_float_np_broadcast = uniform_float_np_broadcast.sample(
                [batch_size, dims])
            sample_np = uniform_np.sample([batch_size, dims])
            sample_variable = uniform_variable.sample([batch_size, dims])

            entropy_float = uniform_float.entropy()
            entropy_float_np_broadcast = uniform_float_np_broadcast.entropy()
            entropy_np = uniform_np.entropy()
            entropy_variable = uniform_variable.entropy()

            lp_float_np_broadcast = uniform_float_np_broadcast.log_prob(values)
            lp_np = uniform_np.log_prob(values)
            lp_variable = uniform_variable.log_prob(values)

            p_float_np_broadcast = uniform_float_np_broadcast.probs(values)
            p_np = uniform_np.probs(values)
            p_variable = uniform_variable.probs(values)

        fetch_list = [
            sample_float, sample_float_np_broadcast, sample_np, sample_variable,
            entropy_float, entropy_float_np_broadcast, entropy_np,
            entropy_variable, lp_float_np_broadcast, lp_np, lp_variable,
            p_float_np_broadcast, p_np, p_variable
        ]
        feed_vars = {'low': low_np, 'high': high_np, 'values': values_np}
        return feed_vars, fetch_list

    def test_uniform_distribution(self, batch_size=2, dims=3, tolerance=1e-6):
        test_program = fluid.Program()

        low_np = np.random.randn(batch_size, dims).astype('float32')
        low_float = np.random.uniform(-2, 1)
        high_float = np.random.uniform(1, 3)
        high_np = np.random.uniform(-5.0, 5.0,
                                    (batch_size, dims)).astype('float32')
        values_np = np.random.randn(batch_size, dims).astype('float32')

        feed_vars, fetch_list = self.build_uniform_program(
            test_program, batch_size, dims, low_float, high_float, high_np,
            low_np, values_np)

        self.executor.run(fluid.default_startup_program())

        np_uniform_float = UniformNumpy(low_float, high_float)
        np_uniform_float_np_broadcast = UniformNumpy(low_float, high_np)
        np_uniform = UniformNumpy(low_np, high_np)

        gt_sample_float = np_uniform_float.sample([batch_size, dims])
        gt_sample_float_np_broadcast = np_uniform_float_np_broadcast.sample(
            [batch_size, dims])
        gt_sample_np = np_uniform.sample([batch_size, dims])
        gt_entropy_float = np_uniform_float.entropy()
        gt_entropy_float_np_broadcast = np_uniform_float_np_broadcast.entropy()
        gt_entropy = np_uniform.entropy()
        gt_lp_float_np_broadcast = np_uniform_float_np_broadcast.log_prob(
            values_np)
        gt_lp = np_uniform.log_prob(values_np)
        gt_p_float_np_broadcast = np_uniform_float_np_broadcast.probs(values_np)
        gt_p = np_uniform.probs(values_np)

        # result calculated by paddle
        [
            output_sample_float, output_sample_float_np_broadcast,
            output_sample_np, output_sample_variable, output_entropy_float,
            output_entropy_float_np_broadcast, output_entropy_np,
            output_entropy_variable, output_lp_float_np_broadcast, output_lp_np,
            output_lp_variable, output_p_float_np_broadcast, output_p_np,
            output_p_variable
        ] = self.executor.run(program=test_program,
                              feed=feed_vars,
                              fetch_list=fetch_list)

        np.testing.assert_allclose(
            output_sample_float.shape,
            gt_sample_float.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_float_np_broadcast.shape,
            gt_sample_float_np_broadcast.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_np.shape,
            gt_sample_np.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_variable.shape,
            gt_sample_np.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_float,
            gt_entropy_float,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_float_np_broadcast,
            gt_entropy_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_np, gt_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_variable, gt_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_lp_float_np_broadcast,
            gt_lp_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_lp_np, gt_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_lp_variable, gt_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_float_np_broadcast,
            gt_p_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_p_np, gt_p, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_variable, gt_p, rtol=tolerance, atol=tolerance)


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
        loc = int(1)
        scale = int(1)

        # type of loc and scale must be float, list, numpy.ndarray, Variable
        self.assertRaises(TypeError, Normal, loc, 1.0)
        self.assertRaises(TypeError, Normal, 1.0, scale)

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
        low = int(1)
        high = int(1)

        # type of loc and scale must be float, list, numpy.ndarray, Variable
        self.assertRaises(TypeError, Uniform, low, 1.0)
        self.assertRaises(TypeError, Uniform, 1.0, high)

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


class DistributionTestName(unittest.TestCase):
    def test_normal_name(self):
        name = 'test_normal'
        normal1 = Normal(0.0, 1.0, name=name)
        self.assertEqual(normal1.name, name)

        normal2 = Normal(0.0, 1.0)
        self.assertEqual(normal2.name, 'Normal')

    def test_uniform_name(self):
        name = 'test_uniform'
        uniform1 = Uniform(0.0, 1.0, name=name)
        self.assertEqual(uniform1.name, name)

        uniform2 = Uniform(0.0, 1.0)
        self.assertEqual(uniform2.name, 'Uniform')


if __name__ == '__main__':
    unittest.main()
