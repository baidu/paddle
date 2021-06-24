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
import copy

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


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
                    np.array_equal(x.grad.numpy(),
                                   np.array([2.4]).astype('float32')))
                y = x.cpu()
                self.assertEqual(y.place.__repr__(), "CPUPlace")
                if core.is_compiled_with_cuda():
                    y = x.pin_memory()
                    self.assertEqual(y.place.__repr__(), "CUDAPinnedPlace")
                    y = x.cuda(blocking=False)
                    self.assertEqual(y.place.__repr__(), "CUDAPlace(0)")
                    y = x.cuda(blocking=True)
                    self.assertEqual(y.place.__repr__(), "CUDAPlace(0)")

                # support 'dtype' is core.VarType
                x = paddle.rand((2, 2))
                y = paddle.to_tensor([2, 2], dtype=x.dtype)
                self.assertEqual(y.dtype, core.VarDesc.VarType.FP32)

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
                self.assertEqual(y.dtype, core.VarDesc.VarType.COMPLEX64)
                self.assertEqual(y.shape, [2])

                paddle.set_default_dtype('float32')
                x = paddle.randn([3, 4])
                x_array = np.array(x)
                self.assertEqual(x_array.shape, x.numpy().shape)
                self.assertEqual(x_array.dtype, x.numpy().dtype)
                self.assertTrue(np.array_equal(x_array, x.numpy()))

                x = paddle.to_tensor(1.0)
                self.assertEqual(x.item(), 1.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.randn([3, 2, 2])
                self.assertTrue(isinstance(x.item(5), float))
                self.assertTrue(isinstance(x.item(1, 0, 1), float))
                self.assertEqual(x.item(5), x.item(1, 0, 1))
                self.assertTrue(
                    np.array_equal(x.item(1, 0, 1), x.numpy().item(1, 0, 1)))

                x = paddle.to_tensor([[1.111111, 2.222222, 3.333333]])
                self.assertEqual(x.item(0, 2), x.item(2))
                self.assertAlmostEqual(x.item(2), 3.333333)
                self.assertTrue(isinstance(x.item(0, 2), float))

                x = paddle.to_tensor(1.0, dtype='float64')
                self.assertEqual(x.item(), 1.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.to_tensor(1.0, dtype='float16')
                self.assertEqual(x.item(), 1.0)
                self.assertTrue(isinstance(x.item(), float))

                x = paddle.to_tensor(1, dtype='uint8')
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype='int8')
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype='int16')
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype='int32')
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(1, dtype='int64')
                self.assertEqual(x.item(), 1)
                self.assertTrue(isinstance(x.item(), int))

                x = paddle.to_tensor(True)
                self.assertEqual(x.item(), True)
                self.assertTrue(isinstance(x.item(), bool))

                x = paddle.to_tensor(1 + 1j)
                self.assertEqual(x.item(), 1 + 1j)
                self.assertTrue(isinstance(x.item(), complex))

                numpy_array = np.random.randn(3, 4)
                # covert core.LoDTensor to paddle.Tensor
                lod_tensor = paddle.fluid.core.LoDTensor()
                place = paddle.fluid.framework._current_expected_place()
                lod_tensor.set(numpy_array, place)
                x = paddle.to_tensor(lod_tensor)
                self.assertTrue(np.array_equal(x.numpy(), numpy_array))
                self.assertEqual(x.type, core.VarDesc.VarType.LOD_TENSOR)
                self.assertEqual(str(x.place), str(place))

                # covert core.Tensor to paddle.Tensor
                x = paddle.to_tensor(numpy_array)
                dlpack = x.value().get_tensor()._to_dlpack()
                tensor_from_dlpack = paddle.fluid.core.from_dlpack(dlpack)
                x = paddle.to_tensor(tensor_from_dlpack)
                self.assertTrue(np.array_equal(x.numpy(), numpy_array))
                self.assertEqual(x.type, core.VarDesc.VarType.LOD_TENSOR)

                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item()
                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item(18)
                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item(1, 2)
                with self.assertRaises(ValueError):
                    paddle.randn([3, 2, 2]).item(2, 1, 2)
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
        _test_place("cpu")
        if core.is_compiled_with_cuda():
            _test_place(core.CUDAPinnedPlace())
            _test_place("gpu_pinned")
            _test_place(core.CUDAPlace(0))
            _test_place("gpu:0")

    def test_to_tensor_not_change_input_stop_gradient(self):
        with paddle.fluid.dygraph.guard(core.CPUPlace()):
            a = paddle.zeros([1024])
            a.stop_gradient = False
            b = paddle.to_tensor(a)
            self.assertEqual(a.stop_gradient, False)
            self.assertEqual(b.stop_gradient, True)

    def test_to_tensor_change_place(self):
        if core.is_compiled_with_cuda():
            a_np = np.random.rand(1024, 1024)
            with paddle.fluid.dygraph.guard(core.CPUPlace()):
                a = paddle.to_tensor(a_np, place=paddle.CUDAPinnedPlace())
                a = paddle.to_tensor(a)
                self.assertEqual(a.place.__repr__(), "CPUPlace")

            with paddle.fluid.dygraph.guard(core.CUDAPlace(0)):
                a = paddle.to_tensor(a_np, place=paddle.CUDAPinnedPlace())
                a = paddle.to_tensor(a)
                self.assertEqual(a.place.__repr__(), "CUDAPlace(0)")

            with paddle.fluid.dygraph.guard(core.CUDAPlace(0)):
                a = paddle.to_tensor(a_np, place=paddle.CPUPlace())
                a = paddle.to_tensor(a, place=paddle.CUDAPinnedPlace())
                self.assertEqual(a.place.__repr__(), "CUDAPinnedPlace")

    def test_to_tensor_with_lodtensor(self):
        if core.is_compiled_with_cuda():
            a_np = np.random.rand(1024, 1024)
            with paddle.fluid.dygraph.guard(core.CPUPlace()):
                lod_tensor = core.LoDTensor()
                lod_tensor.set(a_np, core.CPUPlace())
                a = paddle.to_tensor(lod_tensor)
                self.assertTrue(np.array_equal(a_np, a.numpy()))

            with paddle.fluid.dygraph.guard(core.CUDAPlace(0)):
                lod_tensor = core.LoDTensor()
                lod_tensor.set(a_np, core.CUDAPlace(0))
                a = paddle.to_tensor(lod_tensor, place=core.CPUPlace())
                self.assertTrue(np.array_equal(a_np, a.numpy()))
                self.assertTrue(a.place.__repr__(), "CPUPlace")

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

    def test_leaf_tensor(self):
        with fluid.dygraph.guard():
            x = paddle.to_tensor(np.random.uniform(-1, 1, size=[10, 10]))
            self.assertTrue(x.is_leaf)
            y = x + 1
            self.assertTrue(y.is_leaf)

            x = paddle.to_tensor(
                np.random.uniform(
                    -1, 1, size=[10, 10]), stop_gradient=False)
            self.assertTrue(x.is_leaf)
            y = x + 1
            self.assertFalse(y.is_leaf)

            linear = paddle.nn.Linear(10, 10)
            input = paddle.to_tensor(
                np.random.uniform(
                    -1, 1, size=[10, 10]).astype('float32'),
                stop_gradient=False)
            self.assertTrue(input.is_leaf)

            out = linear(input)
            self.assertTrue(linear.weight.is_leaf)
            self.assertTrue(linear.bias.is_leaf)
            self.assertFalse(out.is_leaf)

    def test_detach(self):
        with fluid.dygraph.guard():
            x = paddle.to_tensor(1.0, dtype="float64", stop_gradient=False)
            detach_x = x.detach()
            self.assertTrue(detach_x.stop_gradient, True)

            cmp_float = np.allclose if core.is_compiled_with_rocm(
            ) else np.array_equal
            detach_x[:] = 10.0
            self.assertTrue(cmp_float(x.numpy(), [10.0]))

            y = x**2
            y.backward()
            self.assertTrue(cmp_float(x.grad.numpy(), [20.0]))
            self.assertEqual(detach_x.grad, None)

            detach_x.stop_gradient = False  # Set stop_gradient to be False, supported auto-grad
            z = 3 * detach_x**2
            z.backward()
            self.assertTrue(cmp_float(x.grad.numpy(), [20.0]))
            self.assertTrue(cmp_float(detach_x.grad.numpy(), [60.0]))

            # Due to sharing of data with origin Tensor, There are some unsafe operations:
            with self.assertRaises(RuntimeError):
                y = 2**x
                detach_x[:] = 5.0
                y.backward()

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

    def test_deep_copy(self):
        with fluid.dygraph.guard():
            empty_var = core.VarBase()
            empty_var_copy = copy.deepcopy(empty_var)
            self.assertEqual(empty_var.stop_gradient,
                             empty_var_copy.stop_gradient)
            self.assertEqual(empty_var.persistable, empty_var_copy.persistable)
            self.assertEqual(empty_var.type, empty_var_copy.type)
            self.assertEqual(empty_var.dtype, empty_var_copy.dtype)

            x = paddle.to_tensor([2.], stop_gradient=False)
            y = paddle.to_tensor([3.], stop_gradient=False)
            z = x * y
            memo = {}
            x_copy = copy.deepcopy(x, memo)
            y_copy = copy.deepcopy(y, memo)

            self.assertEqual(x_copy.stop_gradient, y_copy.stop_gradient)
            self.assertEqual(x_copy.persistable, y_copy.persistable)
            self.assertEqual(x_copy.type, y_copy.type)
            self.assertEqual(x_copy.dtype, y_copy.dtype)
            self.assertTrue(np.array_equal(x.numpy(), x_copy.numpy()))
            self.assertTrue(np.array_equal(y.numpy(), y_copy.numpy()))

            self.assertNotEqual(id(x), id(x_copy))
            x_copy[:] = 5.
            self.assertTrue(np.array_equal(x_copy.numpy(), [5.]))
            self.assertTrue(np.array_equal(x.numpy(), [2.]))

            with self.assertRaises(RuntimeError):
                copy.deepcopy(z)

            x_copy2 = copy.deepcopy(x, memo)
            y_copy2 = copy.deepcopy(y, memo)
            self.assertEqual(id(x_copy), id(x_copy2))
            self.assertEqual(id(y_copy), id(y_copy2))

            # test copy selected rows
            x = core.VarBase(core.VarDesc.VarType.FP32, [3, 100],
                             "selected_rows",
                             core.VarDesc.VarType.SELECTED_ROWS, True)
            selected_rows = x.value().get_selected_rows()
            selected_rows.get_tensor().set(
                np.random.rand(3, 100), core.CPUPlace())
            selected_rows.set_height(10)
            selected_rows.set_rows([3, 5, 7])
            x_copy = copy.deepcopy(x)

            self.assertEqual(x_copy.stop_gradient, x.stop_gradient)
            self.assertEqual(x_copy.persistable, x.persistable)
            self.assertEqual(x_copy.type, x.type)
            self.assertEqual(x_copy.dtype, x.dtype)

            copy_selected_rows = x_copy.value().get_selected_rows()
            self.assertEqual(copy_selected_rows.height(),
                             selected_rows.height())
            self.assertEqual(copy_selected_rows.rows(), selected_rows.rows())
            self.assertTrue(
                np.array_equal(
                    np.array(copy_selected_rows.get_tensor()),
                    np.array(selected_rows.get_tensor())))

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
        var16 = var[-4:4]

        vars = [
            var, var1, var2, var3, var4, var5, var6, var7, var8, var9, var10,
            var11, var12, var13, var14, var15, var16
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
        self.assertTrue(np.array_equal(local_out[16], tensor_array[-4:4]))

    def _test_slice_for_tensor_attr(self):
        tensor_array = np.array(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
             [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
             [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]).astype('float32')

        var = paddle.to_tensor(tensor_array)

        one = paddle.ones(shape=[1], dtype="int32")
        two = paddle.full(shape=[1], fill_value=2, dtype="int32")
        negative_one = paddle.full(shape=[1], fill_value=-1, dtype="int32")
        four = paddle.full(shape=[1], fill_value=4, dtype="int32")

        var = fluid.dygraph.to_variable(tensor_array)
        var1 = var[0, one, one]
        var2 = var[one:]
        var3 = var[0:one]
        var4 = var[::negative_one]
        var5 = var[one, one:, one:]
        var_reshape = fluid.layers.reshape(var, [3, negative_one, 3])
        var6 = var_reshape[:, :, negative_one]
        var7 = var[:, :, :negative_one]
        var8 = var[:one, :one, :1]
        var9 = var[:-1, :negative_one, :negative_one]
        var10 = var[::negative_one, :one, :negative_one]
        var11 = var[:negative_one, ::-1, negative_one:]
        var12 = var[one:2, 2:, ::negative_one]
        var13 = var[two:10, 2:, -2:negative_one]
        var14 = var[1:negative_one, 0:2, ::negative_one]
        var15 = var[::negative_one, ::-1, ::negative_one]
        var16 = var[-4:4]

        vars = [
            var, var1, var2, var3, var4, var5, var6, var7, var8, var9, var10,
            var11, var12, var13, var14, var15, var16
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
        self.assertTrue(np.array_equal(local_out[16], tensor_array[-4:4]))

    def _test_for_var(self):
        np_value = np.random.random((30, 100, 100)).astype('float32')
        w = fluid.dygraph.to_variable(np_value)

        for i, e in enumerate(w):
            self.assertTrue(np.array_equal(e.numpy(), np_value[i]))

    def test_slice(self):
        with fluid.dygraph.guard():
            self._test_slice()
            self._test_slice_for_tensor_attr()
            self._test_for_var()

            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(np.array_equal(var[1, :].numpy(), self.array[1, :]))
            self.assertTrue(np.array_equal(var[::-1].numpy(), self.array[::-1]))

            with self.assertRaises(IndexError):
                y = var[self.shape[0]]

            with self.assertRaises(IndexError):
                y = var[0 - self.shape[0] - 1]

    def test_var_base_to_np(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(
                np.array_equal(var.numpy(),
                               fluid.framework._var_base_to_np(var)))

    def test_var_base_as_np(self):
        with fluid.dygraph.guard():
            var = fluid.dygraph.to_variable(self.array)
            self.assertTrue(np.array_equal(var.numpy(), np.array(var)))
            self.assertTrue(
                np.array_equal(
                    var.numpy(), np.array(
                        var, dtype=np.float32)))

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

    def test_tensor_str_shape_with_zero(self):
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.ones((10, 10))
        y = paddle.fluid.layers.where(x == 0)
        a_str = str(y)

        expected = '''Tensor(shape=[0, 2], dtype=int64, place=CPUPlace, stop_gradient=True,
       [])'''

        self.assertEqual(a_str, expected)
        paddle.enable_static()

    def test_print_tensor_dtype(self):
        paddle.disable_static(paddle.CPUPlace())
        a = paddle.rand([1])
        a_str = str(a.dtype)

        expected = 'paddle.float32'

        self.assertEqual(a_str, expected)
        paddle.enable_static()


class TestVarBaseSetitem(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.set_dtype()
        self.tensor_x = paddle.to_tensor(np.ones((4, 2, 3)).astype(self.dtype))
        self.np_value = np.random.random((2, 3)).astype(self.dtype)
        self.tensor_value = paddle.to_tensor(self.np_value)

    def set_dtype(self):
        self.dtype = "int32"

    def _test(self, value):
        paddle.disable_static()
        self.assertEqual(self.tensor_x.inplace_version, 0)

        id_origin = id(self.tensor_x)
        self.tensor_x[0] = value
        self.assertEqual(self.tensor_x.inplace_version, 1)

        if isinstance(value, (six.integer_types, float)):
            result = np.zeros((2, 3)).astype(self.dtype) + value

        else:
            result = self.np_value

        self.assertTrue(np.array_equal(self.tensor_x[0].numpy(), result))
        self.assertEqual(id_origin, id(self.tensor_x))

        self.tensor_x[1:2] = value
        self.assertEqual(self.tensor_x.inplace_version, 2)
        self.assertTrue(np.array_equal(self.tensor_x[1].numpy(), result))
        self.assertEqual(id_origin, id(self.tensor_x))

        self.tensor_x[...] = value
        self.assertEqual(self.tensor_x.inplace_version, 3)
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


class TestVarBaseSetitemInt64(TestVarBaseSetitem):
    def set_dtype(self):
        self.dtype = "int64"


class TestVarBaseSetitemFp32(TestVarBaseSetitem):
    def set_dtype(self):
        self.dtype = "float32"

    def test_value_float(self):
        paddle.disable_static()
        self._test(3.3)


class TestVarBaseSetitemFp64(TestVarBaseSetitem):
    def set_dtype(self):
        self.dtype = "float64"


class TestVarBaseInplaceVersion(unittest.TestCase):
    def test_setitem(self):
        paddle.disable_static()

        var = paddle.ones(shape=[4, 2, 3], dtype="float32")
        self.assertEqual(var.inplace_version, 0)

        var[1] = 1
        self.assertEqual(var.inplace_version, 1)

        var[1:2] = 1
        self.assertEqual(var.inplace_version, 2)

    def test_bump_inplace_version(self):
        paddle.disable_static()
        var = paddle.ones(shape=[4, 2, 3], dtype="float32")
        self.assertEqual(var.inplace_version, 0)

        var._bump_inplace_version()
        self.assertEqual(var.inplace_version, 1)

        var._bump_inplace_version()
        self.assertEqual(var.inplace_version, 2)


if __name__ == '__main__':
    unittest.main()
