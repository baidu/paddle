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


class TestMemcpy_FillConstant(unittest.TestCase):
    def get_prog(self):
        paddle.enable_static()
        main_program = Program()
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
                    "place_type": 1
                })
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": pinned_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": gpu_var.dtype,
                    "value": 0.0,
                    "place_type": 2
                })
        return main_program, gpu_var, pinned_var

    def test_gpu_cpoy_to_pinned(self):
        main_program, gpu_var, pinned_var = self.get_prog()
        main_program.global_block().append_op(
            type='memcpy',
            inputs={'X': gpu_var},
            outputs={'Out': pinned_var},
            attrs={'dst_place_type': 2})
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
            type='memcpy',
            inputs={'X': pinned_var},
            outputs={'Out': gpu_var},
            attrs={'dst_place_type': 1})
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        gpu_, pinned_ = exe.run(main_program,
                                feed={},
                                fetch_list=[gpu_var.name, pinned_var.name])
        self.assertTrue(np.allclose(gpu_, pinned_))
        self.assertTrue(np.allclose(gpu_, np.zeros((10, 10))))


class TestMemcpyOPError(unittest.TestCase):
    def get_prog(self):
        paddle.enable_static()
        main_program = Program()
        with program_guard(main_program):
            pinned_var = main_program.global_block().create_var(
                name="tensor@Pinned_0",
                shape=[10, 10],
                dtype='float32',
                persistable=False,
                stop_gradient=True)
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": "tensor@Pinned_0"},
                attrs={
                    "shape": [10, 10],
                    "dtype": pinned_var.dtype,
                    "value": 0.0,
                    "place_type": 2
                })
        return main_program, pinned_var

    def test_SELECTED_ROWS(self):
        main_program, pinned_var = self.get_prog()
        selected_row_var = main_program.global_block().create_var( \
            name="selected_row_0", dtype="float32", persistable=False, \
            type=fluid.core.VarDesc.VarType.SELECTED_ROWS, stop_gradient=True)
        main_program.global_block().append_op(
            type="fill_constant",
            outputs={"Out": selected_row_var},
            attrs={
                "shape": selected_row_var.shape,
                "dtype": selected_row_var.dtype,
                "value": 1.0,
                "place_type": 1
            })
        main_program.global_block().append_op(
            type='memcpy',
            inputs={'X': selected_row_var},
            outputs={'Out': pinned_var},
            attrs={'dst_place_type': 2})
        with self.assertRaises(NotImplementedError):
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            selected_row_var_, pinned_ = exe.run(
                main_program,
                feed={},
                fetch_list=[selected_row_var.name, pinned_var.name])

    def test_OTHER_PLACE_NotImplementedError(self):
        main_program, pinned_var = self.get_prog()
        lod_tensor_var = main_program.global_block().create_var( \
            name="lod_tensor_0", dtype="float32", persistable=False, stop_gradient=True)
        main_program.global_block().append_op(
            type="fill_constant",
            outputs={"Out": lod_tensor_var},
            attrs={
                "shape": lod_tensor_var.shape,
                "dtype": lod_tensor_var.dtype,
                "value": 1.0,
                "place_type": 0
            })
        main_program.global_block().append_op(
            type='memcpy',
            inputs={'X': pinned_var},
            outputs={'Out': lod_tensor_var},
            attrs={'dst_place_type': 0, })
        with self.assertRaises(NotImplementedError):
            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            lod_tensor_var_, pinned_ = exe.run(
                main_program,
                feed={},
                fetch_list=[lod_tensor_var.name, pinned_var.name])


class TestMemcpyApi(unittest.TestCase):
    def test_api(self):
        a = paddle.ones([1024, 1024])
        b = paddle.tensor.creation._memcpy(a, paddle.CUDAPinnedPlace())
        self.assertEqual(b.place.__repr__(), "CUDAPinnedPlace")
        self.assertTrue(np.array_equal(a.numpy(), b.numpy()))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
