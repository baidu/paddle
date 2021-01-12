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

import op_test
import numpy as np
import unittest
import paddle
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.backward import append_backward


class TestPinnedMemcpyOp(op_test.OpTest):
    def setUp(self):
        paddle.enable_static()
        self.dtype = np.float16

        self.op_type = "pinned_memcpy"
        np_tensor = np.random.random(size=(5, 5)).astype('float16')

        self.inputs = {'X': np_tensor}
        self.attrs = {"to_pinned": True}
        self.outputs = {'Out': np_tensor}

    # pinned_memcpy only support paddle GPU version by now
    def test_copy_to_pinned(self):
        if core.is_compiled_with_cuda():
            self.attrs = {"to_pinned": True}
            self.check_output(check_dygraph=False)
        else:
            pass

    def test_copy_from_pinned(self):
        if core.is_compiled_with_cuda():
            self.attrs = {"to_pinned": False}
            self.check_output(check_dygraph=False)
        else:
            pass


class TestPinnedMemcpy_FillConstant(unittest.TestCase):
    def get_prog(self):
        paddle.enable_static()
        main_program = Program()
        startup_program = Program()

        with program_guard(main_program):
            pinned_var_name = "tensor@Pinned"
            gpu_var_name = "tensor@GPU"

            pinned_var = main_program.global_block().create_var(
                name=pinned_var_name,
                shape=[10, 10],
                dtype='float32',
                persistable=False,
                stop_gradient=True)

            gpu_var = main_program.global_block().create_var(
                name=gpu_var_name,
                shape=[10, 10],
                dtype='float32',
                persistable=False,
                stop_gradient=True)

            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": gpu_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": gpu_var.dtype,
                    "value": 1.0,
                    "place_type": 2,
                })

            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": pinned_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": gpu_var.dtype,
                    "value": 0.0,
                    "place_type": 3,
                })
        return main_program, gpu_var, pinned_var

    def test_gpu_cpoy_to_pinned(self):

        main_program, gpu_var, pinned_var = self.get_prog()
        main_program.global_block().append_op(
            type='pinned_memcpy',
            inputs={'X': gpu_var},
            outputs={'Out': pinned_var},
            attrs={'to_pinned': True, })

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        gpu_, pinned_ = exe.run(main_program,
                                feed={},
                                fetch_list=[gpu_var.name, pinned_var.name])

        self.assertTrue(np.allclose(gpu_, pinned_))
        self.assertTrue(np.allclose(pinned_, np.ones((10, 10))))

    def test_pinned_cpoy_gpu(self):

        main_program, gpu_var, pinned_var = self.get_prog()
        main_program.global_block().append_op(
            type='pinned_memcpy',
            inputs={'X': pinned_var},
            outputs={'Out': gpu_var},
            attrs={'to_pinned': False, })

        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        gpu_, pinned_ = exe.run(main_program,
                                feed={},
                                fetch_list=[gpu_var.name, pinned_var.name])

        self.assertTrue(np.allclose(gpu_, pinned_))
        self.assertTrue(np.allclose(gpu_, np.zeros((10, 10))))


class TestPinnedMemcpyOPError(unittest.TestCase):
    def test_SELECTED_ROWS(self):
        paddle.enable_static()
        main_program = Program()
        startup_program = Program()

        with program_guard(main_program):
            selected_row_var = main_program.global_block().create_var( \
                name="selected_row_0", dtype="float32", persistable=False, \
                type=fluid.core.VarDesc.VarType.SELECTED_ROWS, stop_gradient=True)

            pinned_var_name = "tensor@Pinned"
            pinned_var = main_program.global_block().create_var(
                name=pinned_var_name,
                shape=[10, 10],
                dtype='float32',
                persistable=False,
                stop_gradient=True)

            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": selected_row_var},
                attrs={
                    "shape": selected_row_var.shape,
                    "dtype": selected_row_var.dtype,
                    "value": 1.0,
                    "place_type": 2,
                })

            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": pinned_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": pinned_var.dtype,
                    "value": 0.0,
                    "place_type": 3,
                })

            main_program.global_block().append_op(
                type='pinned_memcpy',
                inputs={'X': selected_row_var},
                outputs={'Out': pinned_var},
                attrs={'to_pinned': True, })

        with self.assertRaises(NotImplementedError):
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            selected_row_var_, pinned_ = exe.run(
                main_program,
                feed={},
                fetch_list=[selected_row_var.name, pinned_var.name])

    def test_LoDTensorArray(self):
        paddle.enable_static()
        main_program = Program()
        startup_program = Program()

        with program_guard(main_program):

            tmp = fluid.layers.fill_constant(
                shape=[10, 10], dtype='float32', value=1.0)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            Lod_array_var = fluid.layers.array_write(tmp, i=i)

            pinned_var_name = "tensor@Pinned"
            pinned_var = main_program.global_block().create_var(
                name=pinned_var_name,
                shape=[10, 10],
                dtype='float32',
                persistable=False,
                stop_gradient=True)

            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": pinned_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": pinned_var.dtype,
                    "value": 0.0,
                    "place_type": 3,
                })

            main_program.global_block().append_op(
                type='pinned_memcpy',
                inputs={'X': Lod_array_var},
                outputs={'Out': pinned_var},
                attrs={'to_pinned': True, })

        with self.assertRaises(NotImplementedError):
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            Lod_array_var_, pinned_ = exe.run(
                main_program,
                feed={},
                fetch_list=[Lod_array_var.name, pinned_var.name])


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
