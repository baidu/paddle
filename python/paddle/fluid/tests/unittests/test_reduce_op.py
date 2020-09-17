#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp5D(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {
            'X': np.random.random((1, 2, 5, 6, 10)).astype("float64")
        }
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestSumOp6D(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {
            'X': np.random.random((1, 1, 2, 5, 6, 10)).astype("float64")
        }
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestMaxOp(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestMinOp(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


class TestProdOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_prod"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].prod(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestAllOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {'Out': self.inputs['X'].all()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output()


class TestAllOpWithDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': [1]}
        self.outputs = {'Out': self.inputs['X'].all(axis=1)}

    def test_check_output(self):
        self.check_output()


class TestAllOpWithKeepDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_all"
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].all(axis=1), axis=1)
        }

    def test_check_output(self):
        self.check_output()


class TestAllOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of reduce_all_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.reduce_all, input1)
            # The input dtype of reduce_all_op must be bool.
            input2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.reduce_all, input2)


class TestAnyOp(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.outputs = {'Out': self.inputs['X'].any()}
        self.attrs = {'reduce_all': True}

    def test_check_output(self):
        self.check_output()


class TestAnyOpWithDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': [1]}
        self.outputs = {'Out': self.inputs['X'].any(axis=1)}

    def test_check_output(self):
        self.check_output()


class TestAnyOpWithKeepDim(OpTest):
    def setUp(self):
        self.op_type = "reduce_any"
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].any(axis=1), axis=1)
        }

    def test_check_output(self):
        self.check_output()


class TestAnyOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of reduce_any_op must be Variable.
            input1 = 12
            self.assertRaises(TypeError, fluid.layers.reduce_any, input1)
            # The input dtype of reduce_any_op must be bool.
            input2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.reduce_any, input2)


class Test1DReduce(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random(120).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class Test2DReduce0(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [0]}
        self.inputs = {'X': np.random.random((20, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}


class Test2DReduce1(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [1]}
        self.inputs = {'X': np.random.random((20, 10)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


class Test3DReduce0(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [1]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


class Test3DReduce1(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


class Test3DReduce2(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [-2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


class Test3DReduce3(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.attrs = {'dim': [1, 2]}
        self.inputs = {'X': np.random.random((5, 6, 7)).astype("float64")}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']))
        }


class TestKeepDimReduce(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
                                        keepdims=self.attrs['keep_dim'])
        }


class TestReduceAll(Test1DReduce):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 2, 10)).astype("float64")}
        self.attrs = {'reduce_all': True}
        self.outputs = {'Out': self.inputs['X'].sum()}


@skip_check_grad_ci(
    reason="reduce_max is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMaxOpMultiAxises(OpTest):
    """Remove Max with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_max"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2, -1]}
        self.outputs = {
            'Out': self.inputs['X'].max(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(
    reason="reduce_min is discontinuous non-derivable function,"
    " its gradient check is not supported by unittest framework.")
class TestReduceMinOpMultiAxises(OpTest):
    """Remove Min with subgradient from gradient check to confirm the success of CI."""

    def setUp(self):
        self.op_type = "reduce_min"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [1, 2]}
        self.outputs = {
            'Out': self.inputs['X'].min(axis=tuple(self.attrs['dim']))
        }

    def test_check_output(self):
        self.check_output()


class TestKeepDimReduceSumMultiAxises(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((5, 6, 10)).astype("float64")}
        self.attrs = {'dim': [-2, -1], 'keep_dim': True}
        self.outputs = {
            'Out':
            self.inputs['X'].sum(axis=tuple(self.attrs['dim']), keepdims=True)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceSumWithDimOne(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float64")}
        self.attrs = {'dim': [1, 2], 'keep_dim': True}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
                                        keepdims=True)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceSumWithNumelOne(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((100, 1)).astype("float64")}
        self.attrs = {'dim': [1], 'keep_dim': False}
        self.outputs = {
            'Out': self.inputs['X'].sum(axis=tuple(self.attrs['dim']),
                                        keepdims=False)
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceAll(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((100, 1, 1)).astype("float64")}
        self.attrs = {'reduce_all': True, 'keep_dim': False}
        self.outputs = {'Out': self.inputs['X'].sum()}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class Test1DReduceWithAxes1(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random(100).astype("float64")}
        self.attrs = {'dim': [0], 'keep_dim': False}
        self.outputs = {'Out': self.inputs['X'].sum(axis=0)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceWithDtype(OpTest):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum().astype('float64')}
        self.attrs = {'reduce_all': True}
        self.attrs.update({
            'in_dtype': int(convert_np_dtype_to_dtype_(np.float32)),
            'out_dtype': int(convert_np_dtype_to_dtype_(np.float64))
        })

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestReduceWithDtype1(TestReduceWithDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=1)}
        self.attrs = {'dim': [1]}
        self.attrs.update({
            'in_dtype': int(convert_np_dtype_to_dtype_(np.float32)),
            'out_dtype': int(convert_np_dtype_to_dtype_(np.float64))
        })


class TestReduceWithDtype2(TestReduceWithDtype):
    def setUp(self):
        self.op_type = "reduce_sum"
        self.inputs = {'X': np.random.random((6, 2, 10)).astype("float64")}
        self.outputs = {'Out': self.inputs['X'].sum(axis=1, keepdims=True)}
        self.attrs = {'dim': [1], 'keep_dim': True}
        self.attrs.update({
            'in_dtype': int(convert_np_dtype_to_dtype_(np.float32)),
            'out_dtype': int(convert_np_dtype_to_dtype_(np.float64))
        })


class TestReduceSumOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of reduce_sum_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.reduce_sum, x1)
            # The input dtype of reduce_sum_op  must be float32 or float64 or int32 or int64.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, fluid.layers.reduce_sum, x2)


class API_TestSumOpError(unittest.TestCase):
    def test_errors(self):
        def test_dtype1():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float64")
                paddle.sum(data, dtype="float32")

        self.assertRaises(ValueError, test_dtype1)

        def test_dtype2():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="int64")
                paddle.sum(data, dtype="int32")

        self.assertRaises(ValueError, test_dtype2)

        def test_dtype3():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float64")
                paddle.sum(data, dtype="int32")

        self.assertRaises(ValueError, test_dtype3)

        def test_type():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="int32")
                paddle.sum(data, dtype="bool")

        self.assertRaises(TypeError, test_type)


class API_TestSumOp(unittest.TestCase):
    def run_static(self,
                   shape,
                   x_dtype,
                   attr_axis,
                   attr_dtype=None,
                   np_axis=None):
        if np_axis is None:
            np_axis = attr_axis

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data = fluid.data("data", shape=shape, dtype=x_dtype)
            result_sum = paddle.sum(x=data, axis=attr_axis, dtype=attr_dtype)

            exe = fluid.Executor(fluid.CPUPlace())
            input_data = np.random.rand(*shape).astype(x_dtype)
            res, = exe.run(feed={"data": input_data}, fetch_list=[result_sum])

        self.assertTrue(
            np.allclose(
                res, np.sum(input_data.astype(attr_dtype), axis=np_axis)))

    def test_static(self):
        shape = [10, 10]
        axis = 1

        self.run_static(shape, "int32", axis, attr_dtype=None)
        self.run_static(shape, "int32", axis, attr_dtype="int32")
        self.run_static(shape, "int32", axis, attr_dtype="int64")

        self.run_static(shape, "float32", axis, attr_dtype=None)
        self.run_static(shape, "float32", axis, attr_dtype="float32")
        self.run_static(shape, "float32", axis, attr_dtype="float64")

        shape = [5, 5, 5]
        self.run_static(shape, "int32", (0, 1), attr_dtype="int32")
        self.run_static(
            shape, "int32", (), attr_dtype="int32", np_axis=(0, 1, 2))

    def test_dygraph(self):
        np_x = np.random.random([2, 3, 4]).astype('int32')
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(np_x)
            out0 = paddle.sum(x).numpy()
            out1 = paddle.sum(x, axis=0).numpy()
            out2 = paddle.sum(x, axis=(0, 1)).numpy()
            out3 = paddle.sum(x, axis=(0, 1, 2)).numpy()

        self.assertTrue((out0 == np.sum(np_x, axis=(0, 1, 2))).all())
        self.assertTrue((out1 == np.sum(np_x, axis=0)).all())
        self.assertTrue((out2 == np.sum(np_x, axis=(0, 1))).all())
        self.assertTrue((out3 == np.sum(np_x, axis=(0, 1, 2))).all())


if __name__ == '__main__':
    """The framework of Paddle 2.0 is dynamic graph mode by default, but
     Unittest is implemented based on static graph mode.
     Here is a simple conversion from dygraph to static, and Unittest 
     needs to be modified later."""
    import paddle
    paddle.enable_static()
    unittest.main()
