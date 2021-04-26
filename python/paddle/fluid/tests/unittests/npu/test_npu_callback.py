#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNpuCallback(unittest.TestCase):
    def test_callback(self):
        # NPU is not supported in ParallelExecutor
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(
                name="x", shape=[1024, 1024], dtype='float32')
            for _ in range(500):
                x = paddle.matmul(x, x)

            cpu_var_name = "tensor@Cpu"
            npu_var_name = "tensor@Npu"
            cpu_var = main_program.global_block().create_var(
                name=cpu_var_name,
                shape=[10, 10],
                dtype='float32',
                persistable=False,
                stop_gradient=True)
            npu_var = main_program.global_block().create_var(
                name=npu_var_name,
                shape=[10, 10],
                dtype='float32',
                persistable=False,
                stop_gradient=True)
            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": npu_var_name},
                attrs={
                    "shape": [10, 10],
                    "dtype": npu_var.dtype,
                    "value": 1.0,
                    "place_type": 1
                })
            with paddle.fluid.device_guard('cpu'):
                main_program.global_block().append_op(
                    type="fill_constant",
                    outputs={"Out": cpu_var_name},
                    attrs={
                        "shape": [10, 10],
                        "dtype": cpu_var.dtype,
                        "value": 2.0,
                        "place_type": 0
                    })

            main_program.global_block().append_op(
                type='memcpy',
                inputs={'X': cpu_var},
                outputs={'Out': npu_var},
                attrs={'dst_place_type': 4})

        x_np = np.random.random([1024, 1024]).astype('float32')
        place = paddle.NPUPlace(0)
        exe = paddle.static.Executor(place)
        copy_data = exe.run(main_program,
                            feed={"x": x_np},
                            fetch_list=[npu_var.name])
        print(copy_data)
        self.assertTrue(
            np.equal(copy_data[0], np.ones([10, 10]).astype('float32') * 2).all(
            ))


if __name__ == '__main__':
    unittest.main()
