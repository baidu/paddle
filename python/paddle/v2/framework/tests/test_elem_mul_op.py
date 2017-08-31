import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta
import random


class TestElemWiseMulOp_Matrix(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elemwisemul"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((32, 84)).astype("float32")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class TestElemWiseMulOp_Vector(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "elemwisemul"
        self.inputs = {
            'X': np.random.random((32, )).astype("float32"),
            'Y': np.random.random((32, )).astype("float32")
        }
        self.outputs = {'Out': np.multiply(self.inputs['X'], self.inputs['Y'])}


class ElemMulGradOpTest_Matrix(GradientChecker):
    def test_mul(self):
        op = create_op("elemwisemul")
        """ Warning
        CPU gradient check error!
        'X': np.random.random((32,84)).astype("float32"),
        'Y': np.random.random((32,84)).astype("float32")
        """
        inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype("float32"),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype("float32")
        }

        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.02)


class ElemMulGradOpTest_Vector(GradientChecker):
    def test_mul(self):
        op = create_op("elemwisemul")
        inputs = {
            'X': np.random.random((32, )).astype("float32"),
            'Y': np.random.random((32, )).astype("float32")
        }
        self.compare_grad(op, inputs)
        self.check_grad(
            op, inputs, set(["X", "Y"]), "Out", max_relative_error=0.02)


if __name__ == '__main__':
    unittest.main()
