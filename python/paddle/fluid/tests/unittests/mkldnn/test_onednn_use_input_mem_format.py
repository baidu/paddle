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

import os
import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.ir.inference.inference_pass_test import (
    InferencePassTest)


class OneDnnUseInputMemFormatTestCheckConfig(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.abspath(os.path.curdir)
        self.config = fluid.core.AnalysisConfig(
            os.path.join(cur_dir, "model"), os.path.join(cur_dir, "params"))

    def test_is_use_input_mem_format_enabled_default(self):
        self.assertFalse(
            self.config.onednn_use_input_mem_format_enabled(),
            "The default value for \"onednn_use_input_mem_format\" should" +
            " be False.")

    def test_enable_use_input_mem_format(self):
        self.config.enable_onednn_use_input_mem_format()
        self.assertTrue(self.config.onednn_use_input_mem_format_enabled(),
                        "The value for \"onednn_use_input_mem_format\" should" +
                        " be True.")


class AnalysisConfigApiTest(InferencePassTest):
    def setupAnalysisConfig(self, config):
        return config

    def _get_analysis_config(self,
                             use_gpu=False,
                             use_trt=False,
                             use_mkldnn=False):
        config = super(AnalysisConfigApiTest, self)._get_analysis_config(
            use_gpu, use_trt, use_mkldnn)
        return self.setupAnalysisConfig(config)


class OneDnnUseInputMemFormatTestConv2D(AnalysisConfigApiTest):
    def setupAnalysisConfig(self, config):
        config.enable_onednn_use_input_mem_format()
        return config

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 100, 100], dtype="float32")
            conv2d = fluid.layers.conv2d(
                input=data,
                num_filters=3,
                filter_size=3,
                stride=[1, 1],
                use_cudnn=False,
                data_format="NCHW")
        self.feeds = {
            "data": np.random.random((1, 3, 100, 100)).astype("float32")
        }
        self.fetch_list = [conv2d]
        self.enable_mkldnn = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)


class OneDnnUseInputMemFormatTestFC(AnalysisConfigApiTest):
    def setupAnalysisConfig(self, config):
        config.enable_onednn_use_input_mem_format()
        return config

    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 128, 768], dtype="float32")
            data_y = fluid.data(name="y", shape=[-1, 128, 768], dtype="float32")
            fc_out1 = fluid.layers.fc(input=data,
                                      size=3072,
                                      num_flatten_dims=2,
                                      act="relu")

        self.feeds = {"data": np.random.random((4, 128, 768)).astype("float32")}
        self.fetch_list = [fc_out1]
        self.enable_mkldnn = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
