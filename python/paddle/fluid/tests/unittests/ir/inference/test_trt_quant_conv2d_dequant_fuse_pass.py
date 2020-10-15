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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import IrGraph
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.core import AnalysisConfig


class QuantDequantTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 32, 32], dtype="float32")
            param_attr = fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.0),
                trainable=False)
            quantized_op_out = self.append_quantized_op(data, param_attr)
            relu_out = fluid.layers.relu(quantized_op_out)
        self.set_quant_pattern()

        self.feeds = {
            "data": np.random.random([1, 3, 32, 32]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = QuantDequantTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Int8, False, False)
        self.fetch_list = [relu_out]

    def append_quantized_op(self, x, param_attr):
        return fluid.layers.conv2d(
            input=x,
            num_filters=3,
            filter_size=3,
            param_attr=param_attr,
            bias_attr=False,
            act=None)

    def set_quant_pattern(self):
        self.activation_quant_type = 'moving_average_abs_max'
        self.weight_quant_type = 'channel_wise_abs_max'
        self.quantized_op_type = 'conv2d'
        self.channels = 3

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, flatten=True, quant=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible(
                    'quant_conv2d_dequant_fuse_pass'))


class QuantFcDequantTest(QuantDequantTest):
    def append_quantized_op(self, x, param_attr):
        return fluid.layers.fc(x,
                               size=100,
                               num_flatten_dims=1,
                               param_attr=param_attr,
                               bias_attr=False,
                               act=None)

    def set_quant_pattern(self):
        self.activation_quant_type = 'moving_average_abs_max'
        self.weight_quant_type = 'abs_max'
        self.quantized_op_type = 'mul'
        self.channels = 1


if __name__ == "__main__":
    unittest.main()
