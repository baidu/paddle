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

from __future__ import division

import unittest
import numpy as np
import paddle.fluid as fluid
from op_test import OpTest, skip_check_grad_ci

from paddle.fluid import core
from paddle.fluid.framework import program_guard, Program


def spectral_norm(weight, u, v, dim, power_iters, eps):
    shape = weight.shape
    weight_mat = weight.copy()
    h = shape[dim]
    w = np.prod(shape) // h
    if dim != 0:
        perm = [dim] + [d for d in range(len(shape)) if d != dim]
        weight_mat = weight_mat.transpose(perm)
    weight_mat = weight_mat.reshape((h, w))

    u = u.reshape((h, 1))
    v = v.reshape((w, 1))
    for i in range(power_iters):
        v = np.matmul(weight_mat.T, u)
        v_norm = np.sqrt((v * v).sum())
        v = v / (v_norm + eps)
        u = np.matmul(weight_mat, v)
        u_norm = np.sqrt((u * u).sum())
        u = u / (u_norm + eps)

    sigma = (u * np.matmul(weight_mat, v)).sum()
    return weight / sigma, u, v


@skip_check_grad_ci(
    reason="Spectral norm do not check grad when power_iters > 0 "
    "because grad is not calculated in power iterations, "
    "which cannot be checked by python grad unittests")
class TestSpectralNormOpNoGrad(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'spectral_norm'
        weight = np.random.random(self.weight_shape).astype('float64')
        u = np.random.normal(0., 1., self.u_shape).astype('float64')
        v = np.random.normal(0., 1., self.v_shape).astype('float64')

        self.attrs = {
            "dim": self.dim,
            "power_iters": self.power_iters,
            "eps": self.eps,
            "fix_state": self.fix_state,
        }

        self.inputs = {
            "Weight": weight,
            "U": u,
            "V": v,
        }

        output, new_u, new_v = spectral_norm(weight, u, v, self.dim, self.power_iters,
                                             self.eps)
        self.outputs = {"Out": output, "UOut": new_u, "VOut": new_v}

    def test_check_output(self):
        self.check_output()

    def initTestCase(self):
        self.weight_shape = (10, 12)
        self.u_shape = (10, )
        self.v_shape = (12, )
        self.dim = 0
        self.power_iters = 5
        self.eps = 1e-12
        self.fix_state = True


@skip_check_grad_ci(
    reason="Spectral norm do not check grad when power_iters > 0 "
    "because grad is not calculated in power iterations, "
    "which cannot be checked by python grad unittests")
class TestSpectralNormOpNoGrad2(TestSpectralNormOpNoGrad):
    def initTestCase(self):
        self.weight_shape = (2, 3, 3, 3)
        self.u_shape = (3, )
        self.v_shape = (18, )
        self.dim = 1
        self.power_iters = 10
        self.eps = 1e-12
        self.fix_state = True


class TestSpectralNormOp(TestSpectralNormOpNoGrad):
    def test_check_grad_ignore_uv(self):
        self.check_grad(
            ['Weight'],
            'Out',
            no_grad_set=set(["U", "V"]), )

    def initTestCase(self):
        self.weight_shape = (10, 12)
        self.u_shape = (10, )
        self.v_shape = (12, )
        self.dim = 0
        self.power_iters = 0
        self.eps = 1e-12
        self.fix_state = True


class TestSpectralNormOp2(TestSpectralNormOp):
    def initTestCase(self):
        self.weight_shape = (2, 6, 3, 3)
        self.u_shape = (6, )
        self.v_shape = (18, )
        self.dim = 1
        self.power_iters = 0
        self.eps = 1e-12
        self.fix_state = True


class TestSpectralNormOpFixState(TestSpectralNormOpNoGrad):
    def test_check_grad_ignore_uv(self):
        self.check_grad(
            ['Weight'],
            'Out',
            no_grad_set=set(["U", "V"]), )

    def initTestCase(self):
        self.weight_shape = (10, 12)
        self.u_shape = (10, )
        self.v_shape = (12, )
        self.dim = 0
        self.power_iters = 3
        self.eps = 1e-12
        self.fix_state = False


class TestSpectralNormOpUpdateState(TestSpectralNormOpNoGrad):
    def test_check_grad_ignore_uv(self):
        self.check_grad(
            ['Weight'],
            'Out',
            no_grad_set=set(["U", "V"]), )

    def initTestCase(self):
        self.weight_shape = (10, 12)
        self.u_shape = (10, )
        self.v_shape = (12, )
        self.dim = 0
        self.power_iters = 3
        self.eps = 1e-12
        self.fix_state = True


class TestSpectralNormOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                weight_1 = np.random.random((2, 4)).astype("float32")
                fluid.layers.spectral_norm(weight_1, dim=1, power_iters=2)

            # the weight type of spectral_norm must be Variable
            self.assertRaises(TypeError, test_Variable)

            def test_weight_dtype():
                weight_2 = np.random.random((2, 4)).astype("int32")
                fluid.layers.spectral_norm(weight_2, dim=1, power_iters=2)

            # the data type of type must be float32 or float64
            self.assertRaises(TypeError, test_weight_dtype)


class TestDygraphSpectralNormOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            shape = (2, 4, 3, 3)
            spectralNorm = fluid.dygraph.nn.SpectralNorm(
                shape, dim=1, power_iters=2)

            def test_Variable():
                weight_1 = np.random.random((2, 4)).astype("float32")
                spectralNorm(weight_1)

            # the weight type of SpectralNorm must be Variable
            self.assertRaises(TypeError, test_Variable)

            def test_weight_dtype():
                weight_2 = np.random.random((2, 4)).astype("int32")
                spectralNorm(weight_2)

            # the data type of type must be float32 or float64
            self.assertRaises(TypeError, test_weight_dtype)


if __name__ == "__main__":
    unittest.main()
