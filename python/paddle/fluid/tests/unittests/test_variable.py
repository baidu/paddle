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
import paddle
from paddle.fluid.framework import default_main_program, Program, convert_np_dtype_to_dtype_, in_dygraph_mode
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import numpy as np

paddle.enable_static()


class TestVariable(unittest.TestCase):
    def test_np_dtype_convert(self):
        DT = core.VarDesc.VarType
        convert = convert_np_dtype_to_dtype_
        self.assertEqual(DT.FP32, convert(np.float32))
        self.assertEqual(DT.FP16, convert("float16"))
        self.assertEqual(DT.FP64, convert("float64"))
        self.assertEqual(DT.INT32, convert("int32"))
        self.assertEqual(DT.INT16, convert("int16"))
        self.assertEqual(DT.INT64, convert("int64"))
        self.assertEqual(DT.BOOL, convert("bool"))
        self.assertEqual(DT.INT8, convert("int8"))
        self.assertEqual(DT.UINT8, convert("uint8"))

    def test_var(self):
        b = default_main_program().current_block()
        w = b.create_var(
            dtype="float64", shape=[784, 100], lod_level=0, name="fc.w")
        self.assertNotEqual(str(w), "")
        self.assertEqual(core.VarDesc.VarType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual("fc.w@GRAD", w.grad_name)
        self.assertEqual(0, w.lod_level)

        w = b.create_var(name='fc.w')
        self.assertEqual(core.VarDesc.VarType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual("fc.w@GRAD", w.grad_name)
        self.assertEqual(0, w.lod_level)

        self.assertRaises(ValueError,
                          lambda: b.create_var(name="fc.w", shape=(24, 100)))

    def test_step_scopes(self):
        prog = Program()
        b = prog.current_block()
        var = b.create_var(
            name='step_scopes', type=core.VarDesc.VarType.STEP_SCOPES)
        self.assertEqual(core.VarDesc.VarType.STEP_SCOPES, var.type)

    def _test_slice(self, place):
        b = default_main_program().current_block()
        w = b.create_var(dtype="float64", shape=[784, 100, 100], lod_level=0)

        for i in range(3):
            nw = w[i]
            self.assertEqual((100, 100), nw.shape)

        nw = w[:]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[:, :]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[:, :, -1]
        self.assertEqual((784, 100), nw.shape)

        nw = w[1, 1, 1]

        self.assertEqual(len(nw.shape), 1)
        self.assertEqual(nw.shape[0], 1)

        nw = w[:, :, :-1]
        self.assertEqual((784, 100, 99), nw.shape)

        self.assertEqual(0, nw.lod_level)

        main = fluid.Program()
        with fluid.program_guard(main):
            exe = fluid.Executor(place)
            tensor_array = np.array(
                [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                 [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                 [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]).astype('float32')
            var = fluid.layers.assign(tensor_array)
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

            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.fc(input=x, size=1, act=None)
            y_1 = y[:, 0]
            feeder = fluid.DataFeeder(place=place, feed_list=[x])
            data = []
            data.append((np.random.randint(10, size=[13]).astype('float32')))
            exe.run(fluid.default_startup_program())

            local_out = exe.run(main,
                                feed=feeder.feed([data]),
                                fetch_list=[
                                    var, var1, var2, var3, var4, var5, var6,
                                    var7, var8, var9, var10, var11, var12,
                                    var13, var14, var15
                                ])

            self.assertTrue(
                np.array_equal(local_out[1], tensor_array[0, 1, 1:2]))
            self.assertTrue(np.array_equal(local_out[2], tensor_array[1:]))
            self.assertTrue(np.array_equal(local_out[3], tensor_array[0:1]))
            self.assertTrue(np.array_equal(local_out[4], tensor_array[::-1]))
            self.assertTrue(
                np.array_equal(local_out[5], tensor_array[1, 1:, 1:]))
            self.assertTrue(
                np.array_equal(local_out[6],
                               tensor_array.reshape((3, -1, 3))[:, :, -1]))
            self.assertTrue(
                np.array_equal(local_out[7], tensor_array[:, :, :-1]))
            self.assertTrue(
                np.array_equal(local_out[8], tensor_array[:1, :1, :1]))
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

    def _test_slice_index_tensor(self, place):
        data = np.random.rand(2, 3).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            idx0 = [1, 0]
            idx1 = [0, 1]
            idx2 = [0, 0]
            idx3 = [1, 1]

            out0 = x[paddle.assign(np.array(idx0))]
            out1 = x[paddle.assign(np.array(idx1))]
            out2 = x[paddle.assign(np.array(idx2))]
            out3 = x[paddle.assign(np.array(idx3))]

        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=[out0, out1, out2, out3])

        expected = [data[idx0], data[idx1], data[idx2], data[idx3]]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())

        with self.assertRaises(IndexError):
            one = paddle.ones(shape=[1])
            res = x[one, [0, 0]]

    def _test_slice_index_list(self, place):
        data = np.random.rand(2, 3).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            idx0 = [1, 0]
            idx1 = [0, 1]
            idx2 = [0, 0]
            idx3 = [1, 1]

            out0 = x[idx0]
            out1 = x[idx1]
            out2 = x[idx2]
            out3 = x[idx3]

        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=[out0, out1, out2, out3])

        expected = [data[idx0], data[idx1], data[idx2], data[idx3]]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())

    def _test_slice_index_ellipsis(self, place):
        data = np.random.rand(2, 3, 4).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            out1 = x[0:, ..., 1:]
            out2 = x[0:, ...]
            out3 = x[..., 1:]
            out4 = x[...]

        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=[out1, out2, out3, out4])

        expected = [data[0:, ..., 1:], data[0:, ...], data[..., 1:], data[...]]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())

        with self.assertRaises(IndexError):
            res = x[[1, 0], [0, 0]]

        with self.assertRaises(TypeError):
            res = x[[1.2, 0]]

    def _test_slice_index_list_bool(self, place):
        data = np.random.rand(2, 3).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            idx0 = [True, False]
            idx1 = [False, True]
            idx2 = [False, False]
            idx3 = [True, True]

            out0 = x[idx0]
            out1 = x[idx1]
            out2 = x[idx2]
            out3 = x[idx3]

        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=[out0, out1, out2, out3])

        expected = [data[idx0], data[idx1], data[idx2], data[idx3]]

        self.assertTrue((result[0] == expected[0]).all())
        self.assertTrue((result[1] == expected[1]).all())
        self.assertTrue((result[2] == expected[2]).all())
        self.assertTrue((result[3] == expected[3]).all())

        with self.assertRaises(TypeError):
            res = x[[True, 0]]

    def test_slice(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self._test_slice(place)
            self._test_slice_index_tensor(place)
            self._test_slice_index_list(place)
            self._test_slice_index_ellipsis(place)
            self._test_slice_index_list_bool(place)

    def _tostring(self):
        b = default_main_program().current_block()
        w = b.create_var(dtype="float64", lod_level=0)
        self.assertTrue(isinstance(str(w), str))

        if core.is_compiled_with_cuda():
            wc = b.create_var(dtype="int", lod_level=0)
            self.assertTrue(isinstance(str(wc), str))

    def test_tostring(self):
        with fluid.dygraph.guard():
            self._tostring()

        with fluid.program_guard(default_main_program()):
            self._tostring()

    def test_fake_interface_only_api(self):
        b = default_main_program().current_block()
        var = b.create_var(dtype="float64", lod_level=0)
        with fluid.dygraph.guard():
            self.assertRaises(AssertionError, var.numpy)
            self.assertRaises(AssertionError, var.backward)
            self.assertRaises(AssertionError, var.gradient)
            self.assertRaises(AssertionError, var.clear_gradient)

    def test_variable_in_dygraph_mode(self):
        b = default_main_program().current_block()
        var = b.create_var(dtype="float64", shape=[1, 1])
        with fluid.dygraph.guard():
            self.assertTrue(var.to_string(True).startswith('name:'))

            self.assertFalse(var.persistable)
            var.persistable = True
            self.assertTrue(var.persistable)

            self.assertFalse(var.stop_gradient)
            var.stop_gradient = True
            self.assertTrue(var.stop_gradient)

            self.assertTrue(var.name.startswith('_generated_var_'))
            self.assertEqual(var.shape, (1, 1))
            self.assertEqual(var.dtype, fluid.core.VarDesc.VarType.FP64)
            self.assertEqual(var.type, fluid.core.VarDesc.VarType.LOD_TENSOR)

    def test_create_selected_rows(self):
        b = default_main_program().current_block()

        var = b.create_var(
            name="var",
            shape=[1, 1],
            dtype="float32",
            type=fluid.core.VarDesc.VarType.SELECTED_ROWS,
            persistable=True)

        def _test():
            var.lod_level()

        self.assertRaises(Exception, _test)

    def test_size(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(np.random.rand(2, 3, 4).astype("float32"))
            exe = paddle.static.Executor(fluid.CPUPlace())
            exe.run(paddle.static.default_startup_program())

            output = exe.run(prog, fetch_list=[x.size()])
            self.assertEqual(output[0], [24])

    def test_detach(self):
        b = default_main_program().current_block()
        x = b.create_var(shape=[2, 3, 5], dtype="float64", lod_level=0)
        detach_x = x.detach()
        self.assertEqual(x.persistable, detach_x.persistable)
        self.assertEqual(x.shape, detach_x.shape)
        self.assertEqual(x.dtype, detach_x.dtype)
        self.assertEqual(x.type, detach_x.type)
        self.assertTrue(detach_x.stop_gradient)

        xx = b.create_var(name='xx', type=core.VarDesc.VarType.STEP_SCOPES)
        self.assertRaises(AssertionError, xx.detach)

        startup = paddle.static.Program()
        main = paddle.static.Program()
        scope = fluid.core.Scope()
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data(
                    name='x', shape=[3, 2, 1], dtype='float32')
                x.persistable = True
                feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)
                detach_x = x.detach()
                exe = paddle.static.Executor(paddle.CPUPlace())
                exe.run(startup)
                result = exe.run(main,
                                 feed={'x': feed_data},
                                 fetch_list=[x, detach_x])
                self.assertTrue((result[1] == feed_data).all())
                self.assertTrue((result[0] == result[1]).all())

                modified_value = np.zeros(shape=[3, 2, 1], dtype=np.float32)
                detach_x.set_value(modified_value, scope)
                result = exe.run(main, fetch_list=[x, detach_x])
                self.assertTrue((result[1] == modified_value).all())
                self.assertTrue((result[0] == result[1]).all())

                modified_value = np.random.uniform(
                    -1, 1, size=[3, 2, 1]).astype('float32')
                x.set_value(modified_value, scope)
                result = exe.run(main, fetch_list=[x, detach_x])
                self.assertTrue((result[1] == modified_value).all())
                self.assertTrue((result[0] == result[1]).all())


class TestVariableSlice(unittest.TestCase):
    def _test_item_none(self, place):
        data = np.random.rand(2, 3, 4).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            out0 = x[0:, None, 1:]
            out1 = x[0:, None]
            out2 = x[None, 1:]
            out3 = x[None]

        outs = [out0, out1, out2, out3]
        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=outs)

        expected = [
            data[0:, None, 1:], data[0:, None], data[None, 1:], data[None]
        ]
        for i in range(len(outs)):
            self.assertEqual(outs[i].shape, expected[i].shape)
            self.assertTrue((result[i] == expected[i]).all())

    def _test_item_none_and_decrease(self, place):
        data = np.random.rand(2, 3, 4).astype("float32")
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            x = paddle.assign(data)
            out0 = x[0, 1:, None]
            out1 = x[0, None]
            out2 = x[None, 1]
            out3 = x[None]
            out4 = x[0, 0, 0, None]
            out5 = x[None, 0, 0, 0, None]

        outs = [out0, out1, out2, out3, out4, out5]
        exe = paddle.static.Executor(place)
        result = exe.run(prog, fetch_list=outs)
        expected = [
            data[0, 1:, None], data[0, None], data[None, 1], data[None],
            data[0, 0, 0, None], data[None, 0, 0, 0, None]
        ]

        for i in range(len(outs)):
            self.assertEqual(outs[i].shape, expected[i].shape)
            self.assertTrue((result[i] == expected[i]).all())

    def test_slice(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self._test_item_none(place)
            self._test_item_none_and_decrease(place)


if __name__ == '__main__':
    unittest.main()
