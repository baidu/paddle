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

    def test_slice(self):
        place = fluid.CPUPlace()
        self._test_slice(place)

        if core.is_compiled_with_cuda():
            self._test_slice(core.CUDAPlace(0))

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
            self.assertRaises(AssertionError, var.detach)
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


if __name__ == '__main__':
    unittest.main()
