# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestDeviceGuard(unittest.TestCase):
    def test_device_guard(self):
        data1 = fluid.layers.fill_constant(
            shape=[1, 3, 8, 8], value=0.5, dtype='float32')
        data2 = fluid.layers.fill_constant(
            shape=[1, 3, 5, 5], value=0.5, dtype='float32')
        shape = fluid.layers.shape(data2)
        with fluid.device_guard("cpu"):
            shape = fluid.layers.slice(shape, axes=[0], starts=[0], ends=[4])
            with fluid.device_guard("gpu"):
                out = fluid.layers.crop_tensor(data1, shape=shape)

        main_program = fluid.default_main_program()
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'slice':
                self.assertEqual(op.desc.attr(device_attr_name), "cpu")
            if op.type == 'crop_tensor':
                self.assertEqual(op.desc.attr(device_attr_name), "gpu")

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        exe.run(fetch_list=[out])

    def test_cpu_only_op(self):
        x = fluid.layers.fill_constant(
            shape=[2, 255, 13, 13], value=0.3, dtype='float32')
        gt_box = fluid.layers.fill_constant(
            shape=[2, 6, 4], value=0.5, dtype='float32')
        gt_label = fluid.layers.fill_constant(
            shape=[2, 6], value=1.0, dtype='int32')
        gt_score = fluid.layers.fill_constant(
            shape=[2, 6], value=0.5, dtype='float32')
        anchors = [
            10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
            373, 326
        ]
        anchor_mask = [0, 1, 2]
        with fluid.device_guard("gpu"):
            # yolov3_loss only has cpu kernel, so its cpu kernel will be executed
            loss = fluid.layers.yolov3_loss(
                x=x,
                gt_box=gt_box,
                gt_label=gt_label,
                gt_score=gt_score,
                anchors=anchors,
                anchor_mask=anchor_mask,
                class_num=80,
                ignore_thresh=0.7,
                downsample_ratio=32)
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        exe.run(fetch_list=[loss])

    def test_no_kernel_op(self):
        i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
        loop_len = fluid.layers.fill_constant(
            shape=[1], dtype='int64', value=10)
        cond = fluid.layers.less_than(x=i, y=loop_len)
        with fluid.device_guard("cpu"):
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                i = fluid.layers.increment(x=i, value=1, in_place=True)
                fluid.layers.less_than(x=i, y=loop_len, cond=cond)

        main_program = fluid.default_main_program()
        all_ops = main_program.global_block().ops
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        for op in all_ops:
            if op.type == 'while':
                self.assertEqual(op.desc.attr(device_attr_name), "")
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[i])

    def test_device_guard_error(self):
        def device_attr():
            with fluid.device_guard("cpu1"):
                out = fluid.layers.fill_constant(
                    shape=[1], value=0.2, dtype='float32')

        self.assertRaises(ValueError, device_attr)


if __name__ == '__main__':
    unittest.main()
