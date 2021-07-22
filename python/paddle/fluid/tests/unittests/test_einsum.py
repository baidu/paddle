#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import contextlib
import unittest
import paddle
from paddle.fluid import core


class TestErrors(unittest.TestCase):
    def setUp(self):
        pass

    def test_diagonalize_errors(self):
        a = np.arange(4 * 3 * 4 * 4).reshape(4, 3, 4, 4).astype('float')
        a = paddle.to_tensor(a)
        with self.assertRaisesRegex(AssertionError, (
                'Diagonal and trace not implemented yet.')):
            paddle.einsum('...ii->...i', a)
        with self.assertRaisesRegex(AssertionError, (
                'Diagonal and trace not implemented yet.')):
            paddle.einsum('i...i', a)
        with self.assertRaisesRegex(AssertionError, (
                'Diagonal and trace not implemented yet.')):
            paddle.einsum('i...i->i...', a)

    def test_param_errors(self):
        a = np.arange(4 * 3 * 4 * 4).reshape(4, 3, 4, 4).astype('float')
        a = paddle.to_tensor(a)
        with self.assertRaisesRegex(AssertionError,
                                    ('At least one operand is expected.')):
            paddle.einsum('ijk')
        with self.assertRaisesRegex(AssertionError, (
                'Invalid equation: multiple `->` were found.')):
            paddle.einsum('i -> j -> k', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: the number of operands is 2, "
                "but found 3 segments in the label equation.")):
            paddle.einsum('i,j,k', a, a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: the number of operands is 2, "
                "but found 1 segments in the label equation.")):
            paddle.einsum('ij -> k', a, a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: the number of operands is 1, "
                "but found 2 segments in the label equation.")):
            paddle.einsum('i, -> k', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: the label string '' misses dimensions.")):
            paddle.einsum('->', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: the label string 'i' misses dimensions.")):
            paddle.einsum('i', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: _ is not a valid label, "
                "which should be letters.")):
            paddle.einsum('i_', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: `.` is found outside of an ellipsis.")):
            paddle.einsum('i..j', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: `.` is found outside of an ellipsis.")):
            paddle.einsum('...k...', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: missing ellipsis in output labels.")):
            paddle.einsum('i...->i', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid equation: duplicate output labels are found.")):
            paddle.einsum('i...->i...i', a)
        with self.assertRaisesRegex(AssertionError, (
                "Invalid operands: label i "
                "corresponds to non-broadcastable dimensions.")):
            paddle.einsum('ij...,ji...', a, a)


class TestEinsum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

        cls.TEST_SAMPLES = {
            "x": np.random.rand(5),
            "y": np.random.rand(7),
            "A": np.random.rand(4, 5),
            "B": np.random.rand(2, 5),
            "C": np.random.rand(3, 7),
            "D": np.random.rand(3, 4, 5),
            "E": np.random.rand(3, 5, 2),
            "F": np.random.rand(2, 4, 5, 3),
            "G": np.random.rand(4, 2, 5),
            "H": np.random.rand(3, 2, 4),
            "I": np.random.rand(2, 2),
            "J": np.random.rand(1, 3, 5),
            "K": np.random.rand(1, 2, 3, 4),
        }

    def _get_place(self, force_to_use_cpu=False):
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    def check_output_equal(self, actual, expect, rtol=1.e-5, atol=1.e-8):
        error_msg = 'Output has diff at place:{}. \nExpect: {} \nBut Got: {} in class {}'
        self.assertTrue(
            np.allclose(
                actual, expect, rtol=rtol, atol=atol),
            error_msg.format(paddle.get_device(), expect, actual,
                             self.__class__.__name__))

    def setUp(self):
        self.sample = {"paradigm": "i->", "data": ["x"]}

    def test_forward(self):
        operands = [
            TestEinsum.TEST_SAMPLES[operand] for operand in self.sample["data"]
        ]
        expected_result = np.einsum(self.sample["paradigm"], *operands)
        equation = self.sample["paradigm"]

        with paddle.fluid.dygraph.guard(
                self._get_place(force_to_use_cpu=False)):
            pd_operands = [paddle.to_tensor(operand) for operand in operands]
            result = paddle.einsum(equation, *pd_operands)
            self.check_output_equal(result.numpy(), expected_result)

        with paddle.fluid.dygraph.guard(self._get_place(force_to_use_cpu=True)):
            pd_operands = [paddle.to_tensor(operand) for operand in operands]
            result = paddle.einsum(equation, *pd_operands)
            self.check_output_equal(result.numpy(), expected_result)


class TestEinsumVectorDot(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "i,i->", "data": ["x", "x"]}


class TestEinsumVectorMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "i,i->i", "data": ["x", "x"]}


class TestEinsumVectorOuter(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "i,j->ij", "data": ["x", "y"]}


class TestEinsumMatrixTranspose(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij->ji", "data": ["A"]}


class TestEinsumMatrixRowSum(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij->j", "data": ["A"]}


class TestEinsumMatrixColSum(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij->i", "data": ["A"]}


class TestEinsumMatrixEleMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,ij->ij", "data": ["A", "A"]}


class TestEinsumMatrixVecMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,j->i", "data": ["A", "x"]}


class TestEinsumMatrixMul(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,kj->ik", "data": ["A", "B"]}


class TestEinsumMatrixOuter(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,kl->ijkl", "data": ["A", "C"]}


class TestEinsumTensorBMM(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "bij,bjk->bik", "data": ["D", "E"]}


class TestEinsumTensorContract1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->i", "data": ["D", "A"]}


class TestEinsumTensorContract2(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,lk->ijl", "data": ["D", "B"]}


class TestEinsumTensorContract3(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "abcd,dfg->abcfg", "data": ["F", "D"]}


class TestEinsumTensorContract4(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->ik", "data": ["D", "A"]}


class TestEinsumTensorContract5(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk,jk->ij", "data": ["D", "A"]}


class TestEinsumTensorContract6(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ik, ijk->j", "data": ["A", "G"]}


class TestEinsumTensorContract7(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijk, ik->jk", "data": ["G", "A"]}


class TestEinsumEllipsis1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "i...->...", "data": ["G"]}


class TestEinsumEllipsis2(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ij,...i->j...", "data": ["A", "H"]}


class TestEinsumEllipsis3(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "k...,jk", "data": ["F", "I"]}


class TestEinsumTestEinsumBilinear(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "bn,anm,bm->ba", "data": ["B", "E", "I"]}


class TestEinsumTestEinsumOthers(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijkl, lmn->kmn", "data": ["F", "H"]}


class TestEinsumTestEinsumOthers(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "ijkl, lmn->ijn", "data": ["F", "H"]}


class TestEinsumBatch1(TestEinsum):
    def setUp(self):
        self.sample = {"paradigm": "blq,bhlk->bhlqk", "data": ["J", "K"]}


class TestNumpyTests(unittest.TestCase):
    def setUp(self):
        pass

    def _get_place(self, force_to_use_cpu=False):
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    def check_output_equal(self, actual, expect, rtol=1.e-5, atol=1.e-8):
        error_msg = 'Output has diff at place:{}. \nExpect: {} \nBut Got: {} in class {}'
        self.assertTrue(
            np.allclose(
                actual, expect, rtol=rtol, atol=atol),
            error_msg.format(paddle.get_device(), expect, actual,
                             self.__class__.__name__))

    def check_output(self, eqn, *ops):
        expect = np.einsum(eqn, *ops)
        with paddle.fluid.dygraph.guard(
                self._get_place(force_to_use_cpu=False)):
            pd_operands = [paddle.to_tensor(op) for op in ops]
            actual = paddle.einsum(eqn, *pd_operands)
            self.check_output_equal(actual.numpy(), expect)

    def test_sums(self):
        for n in range(1, 17):
            a = np.arange(n).astype('float')
            self.check_output("i->", a)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            self.check_output("...i->...", a)

        for n in range(1, 17):
            a = np.arange(2 * n).reshape(2, n).astype('float')
            self.check_output("i...->...", a)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            self.check_output("i...->...", a)

        for n in range(1, 17):
            a = np.arange(3 * n).reshape(3, n).astype('float')
            b = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            self.check_output("..., ...", a, b)

        for n in range(1, 17):
            a = np.arange(2 * 3 * n).reshape(2, 3, n).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("...i, ...i", a, b)

        for n in range(1, 11):
            a = np.arange(n * 3 * 2).reshape(n, 3, 2).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("i..., i...", a, b)

        for n in range(1, 17):
            a = (np.arange(3) + 1).astype('float')
            b = (np.arange(n) + 1).astype('float')
            self.check_output("i,j", a, b)

        for n in range(1, 17):
            a = np.arange(4 * n).reshape(4, n).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("ij, j", a, b)

        for n in range(1, 17):
            a = np.arange(4 * n).reshape(4, n).astype('float')
            b = np.arange(n).astype('float')
            self.check_output("ji,j", a.T, b.T)

        for n in range(1, 17):
            a = np.arange(4 * n).reshape(4, n).astype('float')
            b = np.arange(n * 6).reshape(n, 6).astype('float')
            self.check_output("ij,jk", a, b)

        a = np.arange(12).reshape(3, 4).astype('float')
        b = np.arange(20).reshape(4, 5).astype('float')
        c = np.arange(30).reshape(5, 6).astype('float')
        self.check_output("ij,jk,kl", a, b, c)

        a = np.arange(60).reshape(3, 4, 5).astype('float')
        b = np.arange(24).reshape(4, 3, 2).astype('float')
        self.check_output("ijk, jil -> kl", a, b)

        for n in range(1, 25):
            a = np.arange(n).astype('float')
            self.check_output("...,...", a, a)
            self.check_output("i,i", a, a)

        p = np.ones((10, 2)).astype('float')
        q = np.ones((1, 2)).astype('float')
        self.check_output('ij,ij->j', p, q)

        x = np.array([2., 3.]).astype('float')
        y = np.array([4.]).astype('float')
        self.check_output("i, i", x, y)

        p = np.ones((1, 5)) / 2
        q = np.ones((5, 5)) / 2
        self.check_output("...ij,...jk->...ik", p, p)
        self.check_output("...ij,...jk->...ik", p, q)

        x = np.eye(2).astype('float')
        y = np.ones(2).astype('float')
        self.check_output("ji,i->", x, y)
        self.check_output("i,ij->", y, x)
        self.check_output("ij,i->", x, y)

    def test_large_nops(self):
        a = np.arange(4 * 3 * 1 * 4).reshape(4, 3, 1, 4).astype('float')
        self.check_output('a...b,b...c,c...d', a, a, a)
        self.check_output('a...b,b...c,c...a', a, a, a)
        self.check_output('a...b,b...c,c...a', a, a, a)
        self.check_output('...ab,...ba,...ab,...ab', a, a, a, a)


if __name__ == "__main__":
    unittest.main()
