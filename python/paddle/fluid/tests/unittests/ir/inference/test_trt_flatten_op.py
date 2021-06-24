# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TRTFlattenTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            flatten_out = self.append_flatten(data)
            out = fluid.layers.batch_norm(flatten_out, is_test=True)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTFlattenTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_flatten(self, data):
        return fluid.layers.flatten(data, axis=1)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TRTFlattenDynamicTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            flatten_out = self.append_flatten(data)
            out = fluid.layers.batch_norm(flatten_out, is_test=True)
        self.feeds = {
            "data": np.random.random([2, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TRTFlattenDynamicTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = TRTFlattenDynamicTest.DynamicShapeParam({
            'data': [1, 6, 8, 8],
            'flatten_0.tmp_0': [1, 6 * 8 * 8]
        }, {'data': [3, 6, 128, 128],
            'flatten_0.tmp_0': [3, 6 * 128 * 128]}, {
                'data': [2, 6, 64, 64],
                'flatten_0.tmp_0': [2, 6 * 64 * 64]
            }, False)
        self.fetch_list = [out]

    def append_flatten(self, data):
        return fluid.layers.flatten(data, axis=1)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
