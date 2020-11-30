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
import six

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid.framework import default_main_program, Program, convert_np_dtype_to_dtype_, in_dygraph_mode


class TestVarBase(unittest.TestCase):
    def setUp(self):
        self.shape = [512, 1234]
        self.dtype = np.float32
        self.array = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)

    def test_to_tensor(self):
        def _test_place(place):
            with fluid.dygraph.guard():
                paddle.set_default_dtype('float32')
                # set_default_dtype should not take effect on int
                x = paddle.to_tensor(1, place=place, stop_gradient=False)
                self.assertTrue(np.array_equal(x.numpy(), [1]))
                self.assertNotEqual(x.dtype, core.VarDesc.VarType.FP32)

                y = paddle.to_tensor(2, place=x.place)
                self.assertEqual(str(x.place), str(y.place))

                # set_default_dtype should not take effect on numpy
                x = paddle.to_tensor(
                    np.array([1.2]).astype('float16'),
                    place=place,
                    stop_gradient=False)
                self.assertTrue(
                    np.array_equal(x.numpy(), np.array([1.2], 'float16')))
                self.assertEqual(x.dtype, core.VarDesc.VarType.FP16)

                # set_default_dtype take effect on float
                x = paddle.to_tensor(1.2, place=place, stop_gradient=False)
                self.assertTrue(
                    np.array_equal(x.numpy(), np.array([1.2]).astype(
                        'float32')))
                self.assertEqual(x.dtype, core.VarDesc.VarType.FP32)
                clone_x = x.clone()
                self.assertTrue(
                    np.array_equal(clone_x.numpy(),
                                   np.array([1.2]).astype('float32')))
                self.assertEqual(clone_x.dtype, core.VarDesc.VarType.FP32)
                y = clone_x**2
                y.backward()
                self.assertTrue(
                    np.array_equal(x.grad, np.array([2.4]).astype('float32')))
                y = x.cpu()
                self.assertEqual(y.place.__repr__(), "CPUPlace")
                if core.is_compiled_with_cuda():
                    y = x.pin_memory()
                    self.assertEqual(y.place.__repr__(), "CUDAPinnedPlace")
                    y = x.cuda(blocking=False)
                    self.assertEqual(y.place.__repr__(), "CUDAPlace(0)")
                    y = x.cuda(blocking=True)
                    self.assertEqual(y.place.__repr__(), "CUDAPlace(0)")

                # set_default_dtype take effect on complex
                x = paddle.to_tensor(1 + 2j, place=place, stop_gradient=False)
                self.assertTrue(np.array_equal(x.numpy(), [1 + 2j]))
                self.assertEqual(x.dtype, core.VarDesc.VarType.COMPLEX64)

                paddle.set_default_dtype('float64')
                x = paddle.to_tensor(1.2, place=place, stop_gradient=False)
                self.assertTrue(np.array_equal(x.numpy(), [1.2]))
                self.assertEqual(x.dtype, core.VarDesc.VarType.FP64)

                x = paddle.to_tensor(1 + 2j, place=place, stop_gradient=False)
                self.assertTrue(np.array_equal(x.numpy(), [1 + 2j]))
                self.assertEqual(x.dtype, core.VarDesc.VarType.COMPLEX128)

                x = paddle.to_tensor(
                    1, dtype='float32', place=place, stop_gradient=False)
                self.assertTrue(np.array_equal(x.numpy(), [1.]))
                self.assertEqual(x.dtype, core.VarDesc.VarType.FP32)
                self.assertEqual(x.shape, [1])
                self.assertEqual(x.stop_gradient, False)
                self.assertEqual(x.type, core.VarDesc.VarType.LOD_TENSOR)

                x = paddle.to_tensor(
                    (1, 2), dtype='float32', place=place, stop_gradient=False)
                x = paddle.to_tensor(
                    [1, 2], dtype='float32', place=place, stop_gradient=False)
                self.assertTrue(np.array_equal(x.numpy(), [1., 2.]))
                self.assertEqual(x.dtype, core.VarDesc.VarType.FP32)
                self.assertEqual(x.grad, None)
                self.assertEqual(x.shape, [2])
                self.assertEqual(x.stop_gradient, False)
                self.assertEqual(x.type, core.VarDesc.VarType.LOD_TENSOR)

                x = paddle.to_tensor(
                    self.array,
                    dtype='float32',
                    place=place,
                    stop_gradient=False)
                self.assertTrue(np.array_equal(x.numpy(), self.array))
                self.assertEqual(x.dtype, core.VarDesc.VarType.FP32)
                self.assertEqual(x.shape, self.shape)
                self.assertEqual(x.stop_gradient, False)
                self.assertEqual(x.type, core.VarDesc.VarType.LOD_TENSOR)

                y = paddle.to_tensor(x)
                y = paddle.to_tensor(y, dtype='float64', place=place)
                self.assertTrue(np.array_equal(y.numpy(), self.array))
                self.assertEqual(y.dtype, core.VarDesc.VarType.FP64)
                self.assertEqual(y.shape, self.shape)
                self.assertEqual(y.stop_gradient, True)
                self.assertEqual(y.type, core.VarDesc.VarType.LOD_TENSOR)
                z = x + y
                self.assertTrue(np.array_equal(z.numpy(), 2 * self.array))

                x = paddle.to_tensor(
                    [1 + 2j, 1 - 2j], dtype='complex64', place=place)
                y = paddle.to_tensor(x)
                self.assertTrue(np.array_equal(x.numpy(), [1 + 2j, 1 - 2j]))
                self.assertEqual(y.type, core.VarDesc.VarType.LOD_TENSOR)
                self.assertEqual(y.shape, [2])

                with self.assertRaises(TypeError):
                    paddle.to_tensor('test')
                with self.assertRaises(TypeError):
                    paddle.to_tensor(1, dtype='test')
                with self.assertRaises(ValueError):
                    paddle.to_tensor([[1], [2, 3]])
                with self.assertRaises(ValueError):
                    paddle.to_tensor([[1], [2, 3]], place='test')
                with self.assertRaises(ValueError):
                    paddle.to_tensor([[1], [2, 3]], place=1)

        _test_place(core.CPUPlace())
        if core.is_compiled_with_cuda():
            _test_place(core.CUDAPinnedPlace())
            _test_place(core.CUDAPlace(0))

    def test_to_variable(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array, name="abc")
            self.assertTrue(np.array_equal(var.numpy(), self.array))
            self.assertEqual(var.name, 'abc')
            # default value
            self.assertEqual(var.persistable, False)
            self.assertEqual(var.stop_gradient, True)
            self.assertEqual(var.shape, self.shape)
            self.assertEqual(var.dtype, core.VarDesc.VarType.FP32)
            self.assertEqual(var.type, core.VarDesc.VarType.LOD_TENSOR)
            # The type of input must be 'ndarray' or 'Variable', it will raise TypeError
            with self.assertRaises(TypeError):
                var = fluid.dygraph.to_variable("test", name="abc")
            # test to_variable of LayerObjectHelper(LayerHelperBase)
            with self.assertRaises(TypeError):
                linear = fluid.dygraph.Linear(32, 64)
                var = linear._helper.to_variable("test", name="abc")

    def test_list_to_variable(self):
        with fluid.dygraph.guard():
            array = [[[1, 2], [1, 2], [1.0, 2]], [[1, 2], [1, 2], [1, 2]]]
            var = fluid.dygraph.to_variable(array, dtype='int32')
            self.assertTrue(np.array_equal(var.numpy(), array))
            self.assertEqual(var.shape, [2, 3, 2])
            self.assertEqual(var.dtype, core.VarDesc.VarType.INT32)
            self.assertEqual(var.type, core.VarDesc.VarType.LOD_TENSOR)

    def test_tuple_to_variable(self):
        with fluid.dygraph.guard():
            array = (((1, 2), (1, 2), (1, 2)), ((1, 2), (1, 2), (1, 2)))
            var = fluid.dygraph.to_variable(array, dtype='float32')
            self.assertTrue(np.array_equal(var.numpy(), array))
            self.assertEqual(var.shape, [2, 3, 2])
            self.assertEqual(var.dtype, core.VarDesc.VarType.FP32)
            self.assertEqual(var.type, core.VarDesc.VarType.LOD_TENSOR)

    def test_tensor_to_variable(self):
        with fluid.dygraph.guard():
            t = fluid.Tensor()
            t.set(np.random.random((1024, 1024)), fluid.CPUPlace())
            var = fluid.dygraph.to_variable(t)
            self.assertTrue(np.array_equal(t, var.numpy()))

    def test_detach(self):
        with fluid.dygraph.guard():
            x = paddle.to_tensor(1.0, dtype="float64", stop_gradient=False)
            detach_x = x.detach()
            self.assertTrue(detach_x.stop_gradient, True)

            detach_x[:] = 10.0
            self.assertTrue(np.array_equal(x.numpy(), [10.0]))

            y = x**2
            y.backward()
            self.assertTrue(np.array_equal(x.grad, [20.0]))
            self.assertEqual(detach_x.grad, None)

            detach_x.stop_gradient = False  # Set stop_gradient to be False, supported auto-grad
            z = 3 * detach_x**2
            z.backward()
            self.assertTrue(np.array_equal(x.grad, [20.0]))
            self.assertTrue(np.array_equal(detach_x.grad, [60.0]))
            # Due to sharing of data with origin Tensor, There are some unsafe operations:
            # with self.assertRaises(RuntimeError):
            #     y = 2 * x
            #     detach_x[:] = 5.0
            #     y.backward()

    def test_write_property(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)

            self.assertEqual(var.name, 'generated_tensor_0')
            var.name = 'test'
            self.assertEqual(var.name, 'test')

            self.assertEqual(var.persistable, False)
            var.persistable = True
            self.assertEqual(var.persistable, True)

            self.assertEqual(var.stop_gradient, True)
            var.stop_gradient = False
            self.assertEqual(var.stop_gradient, False)

    # test some patched methods
    def test_set_value(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            tmp1 = np.random.uniform(0.1, 1, [2, 2, 3]).astype(self.dtype)
            self.assertRaises(AssertionError, var.set_value, tmp1)

            tmp2 = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
            var.set_value(tmp2)
            self.assertTrue(np.array_equal(var.numpy(), tmp2))

    def test_to_string(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(isinstance(str(var), str))

    def test_backward(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            var.stop_gradient = False
            loss = fluid.layers.relu(var)
            loss.backward()
            grad_var = var._grad_ivar()
            self.assertEqual(grad_var.shape, self.shape)

    def test_gradient(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            var.stop_gradient = False
            loss = fluid.layers.relu(var)
            loss.backward()
            grad_var = var.gradient()
            self.assertEqual(grad_var.shape, self.array.shape)

    def test_block(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertEqual(var.block,
                             fluid.default_main_program().global_block())

    def _test_slice(self):
        w = fluid.dygraph.to_variable(
            np.random.random((784, 100, 100)).astype('float64'))

        for i in range(3):
            nw = w[i]
            self.assertEqual((100, 100), tuple(nw.shape))

        nw = w[:]
        self.assertEqual((784, 100, 100), tuple(nw.shape))

        nw = w[:, :]
        self.assertEqual((784, 100, 100), tuple(nw.shape))

        nw = w[:, :, -1]
        self.assertEqual((784, 100), tuple(nw.shape))

        nw = w[1, 1, 1]

        self.assertEqual(len(nw.shape), 1)
        self.assertEqual(nw.shape[0], 1)

        nw = w[:, :, :-1]
        self.assertEqual((784, 100, 99), tuple(nw.shape))

        tensor_array = np.array(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
             [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
             [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]).astype('float32')
        var = fluid.dygraph.to_variable(tensor_array)
        var1 = var[0, 1, 1]
        var2 = var[1:]
        var3 = var[0:1]
        var4 = var[::-1]
        var5 = var[1, 1:, 1:]
        var_reshape = fluid.layers.reshape(var, [3, -1, 3])
        var6 = var_reshape[:, :, -1]
        var7 = var[:, :, :-1]
        var8 = var[:1, :1, :1]
        var9 = var[:-1, :-1, :-1]
        var10 = var[::-1, :1, :-1]
        var11 = var[:-1, ::-1, -1:]
        var12 = var[1:2, 2:, ::-1]
        var13 = var[2:10, 2:, -2:-1]
        var14 = var[1:-1, 0:2, ::-1]
        var15 = var[::-1, ::-1, ::-1]

        vars = [
            var, var1, var2, var3, var4, var5, var6, var7, var8, var9, var10,
            var11, var12, var13, var14, var15
        ]
        local_out = [var.numpy() for var in vars]

        self.assertTrue(np.array_equal(local_out[1], tensor_array[0, 1, 1:2]))
        self.assertTrue(np.array_equal(local_out[2], tensor_array[1:]))
        self.assertTrue(np.array_equal(local_out[3], tensor_array[0:1]))
        self.assertTrue(np.array_equal(local_out[4], tensor_array[::-1]))
        self.assertTrue(np.array_equal(local_out[5], tensor_array[1, 1:, 1:]))
        self.assertTrue(
            np.array_equal(local_out[6],
                           tensor_array.reshape((3, -1, 3))[:, :, -1]))
        self.assertTrue(np.array_equal(local_out[7], tensor_array[:, :, :-1]))
        self.assertTrue(np.array_equal(local_out[8], tensor_array[:1, :1, :1]))
        self.assertTrue(
            np.array_equal(local_out[9], tensor_array[:-1, :-1, :-1]))
        self.assertTrue(
            np.array_equal(local_out[10], tensor_array[::-1, :1, :-1]))
        self.assertTrue(
            np.array_equal(local_out[11], tensor_array[:-1, ::-1, -1:]))
        self.assertTrue(
            np.array_equal(local_out[12], tensor_array[1:2, 2:, ::-1]))
        self.assertTrue(
            np.array_equal(local_out[13], tensor_array[2:10, 2:, -2:-1]))
        self.assertTrue(
            np.array_equal(local_out[14], tensor_array[1:-1, 0:2, ::-1]))
        self.assertTrue(
            np.array_equal(local_out[15], tensor_array[::-1, ::-1, ::-1]))

    def _test_for_var(self):
        np_value = np.random.random((30, 100, 100)).astype('float32')
        w = fluid.dygraph.to_variable(np_value)

        for i, e in enumerate(w):
            self.assertTrue(np.array_equal(e.numpy(), np_value[i]))

    def test_slice(self):
        with fluid.dygraph.guard():
            self._test_slice()
            self._test_for_var()

            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(np.array_equal(var[1, :].numpy(), self.array[1, :]))
            self.assertTrue(np.array_equal(var[::-1].numpy(), self.array[::-1]))

            with self.assertRaises(IndexError):
                y = var[self.shape[0]]

    def test_var_base_to_np(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(
                np.array_equal(var.numpy(),
                               fluid.framework._var_base_to_np(var)))

    def test_if(self):
        with fluid.dygraph.guard():
            var1 = fluid.dygraph.to_variable(np.array([[[0]]]))
            var2 = fluid.dygraph.to_variable(np.array([[[1]]]))

            var1_bool = False
            var2_bool = False

            if var1:
                var1_bool = True

            if var2:
                var2_bool = True

            assert var1_bool == False, "if var1 should be false"
            assert var2_bool == True, "if var2 should be true"
            assert bool(var1) == False, "bool(var1) is False"
            assert bool(var2) == True, "bool(var2) is True"

    def test_to_static_var(self):
        with fluid.dygraph.guard():
            # Convert VarBase into Variable or Parameter
            var_base = fluid.dygraph.to_variable(self.array, name="var_base_1")
            static_var = var_base._to_static_var()
            self._assert_to_static(var_base, static_var)

            var_base = fluid.dygraph.to_variable(self.array, name="var_base_2")
            static_param = var_base._to_static_var(to_parameter=True)
            self._assert_to_static(var_base, static_param, True)

            # Convert ParamBase into Parameter
            fc = fluid.dygraph.Linear(
                10,
                20,
                param_attr=fluid.ParamAttr(
                    learning_rate=0.001,
                    do_model_average=True,
                    regularizer=fluid.regularizer.L1Decay()))
            weight = fc.parameters()[0]
            static_param = weight._to_static_var()
            self._assert_to_static(weight, static_param, True)

    def _assert_to_static(self, var_base, static_var, is_param=False):
        if is_param:
            self.assertTrue(isinstance(static_var, fluid.framework.Parameter))
            self.assertTrue(static_var.persistable, True)
            if isinstance(var_base, fluid.framework.ParamBase):
                for attr in ['trainable', 'is_distributed', 'do_model_average']:
                    self.assertEqual(
                        getattr(var_base, attr), getattr(static_var, attr))

                self.assertEqual(static_var.optimize_attr['learning_rate'],
                                 0.001)
                self.assertTrue(
                    isinstance(static_var.regularizer,
                               fluid.regularizer.L1Decay))
        else:
            self.assertTrue(isinstance(static_var, fluid.framework.Variable))

        attr_keys = ['block', 'dtype', 'type', 'name']
        for attr in attr_keys:
            self.assertEqual(getattr(var_base, attr), getattr(static_var, attr))

        self.assertListEqual(list(var_base.shape), list(static_var.shape))

    def test_tensor_str(self):
        paddle.enable_static()
        paddle.disable_static(paddle.CPUPlace())
        paddle.seed(10)
        a = paddle.rand([10, 20])
        paddle.set_printoptions(4, 100, 3)
        a_str = str(a)

        expected = '''Tensor(shape=[10, 20], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[0.2727, 0.5489, 0.8655, ..., 0.2916, 0.8525, 0.9000],
        [0.3806, 0.8996, 0.0928, ..., 0.9535, 0.8378, 0.6409],
        [0.1484, 0.4038, 0.8294, ..., 0.0148, 0.6520, 0.4250],
        ...,
        [0.3426, 0.1909, 0.7240, ..., 0.4218, 0.2676, 0.5679],
        [0.5561, 0.2081, 0.0676, ..., 0.9778, 0.3302, 0.9559],
        [0.2665, 0.8483, 0.5389, ..., 0.4956, 0.6862, 0.9178]])'''

        self.assertEqual(a_str, expected)
        paddle.enable_static()

    def test_tensor_str2(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.to_tensor([[1.5111111, 1.0], [0, 0]])
        a_str = str(a)

        expected = '''Tensor(shape=[2, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[1.5111, 1.    ],
        [0.    , 0.    ]])'''

        self.assertEqual(a_str, expected)
        paddle.enable_static()

    def test_tensor_str3(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.to_tensor([[-1.5111111, 1.0], [0, -0.5]])
        a_str = str(a)

        expected = '''Tensor(shape=[2, 2], dtype=float32, place=CPUPlace, stop_gradient=True,
       [[-1.5111,  1.    ],
        [ 0.    , -0.5000]])'''

        self.assertEqual(a_str, expected)
        paddle.enable_static()

    def test_tensor_str_scaler(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.to_tensor(np.array(False))
        a_str = str(a)

        expected = '''Tensor(shape=[], dtype=bool, place=CPUPlace, stop_gradient=True,
       False)'''

        self.assertEqual(a_str, expected)
        paddle.enable_static()


class TestVarBaseSetitem(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.tensor_x = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
        self.np_value = np.random.random((2, 3)).astype(np.float32)
        self.tensor_value = paddle.to_tensor(self.np_value)

    def _test(self, value):
        paddle.disable_static()
        id_origin = id(self.tensor_x)

        self.tensor_x[0] = value

        if isinstance(value, (six.integer_types, float)):
            result = np.zeros((2, 3)).astype(np.float32) + value

        else:
            result = self.np_value

        self.assertTrue(np.array_equal(self.tensor_x[0].numpy(), result))
        self.assertEqual(id_origin, id(self.tensor_x))

        self.tensor_x[1:2] = value
        self.assertTrue(np.array_equal(self.tensor_x[1].numpy(), result))
        self.assertEqual(id_origin, id(self.tensor_x))

        self.tensor_x[...] = value
        self.assertTrue(np.array_equal(self.tensor_x[3].numpy(), result))
        self.assertEqual(id_origin, id(self.tensor_x))

    def test_value_tensor(self):
        paddle.disable_static()
        self._test(self.tensor_value)

    def test_value_numpy(self):
        paddle.disable_static()
        self._test(self.np_value)

    def test_value_int(self):
        paddle.disable_static()
        self._test(10)

    def test_value_float(self):
        paddle.disable_static()
        self._test(3.3)


if __name__ == '__main__':
    unittest.main()
