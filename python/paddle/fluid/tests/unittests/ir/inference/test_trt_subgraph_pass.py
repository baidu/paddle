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
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTSubgraphPassConvTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d(
                input=data,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                act=None)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConvTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = [1, 1]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassConvValidPaddingTest(TensorRTSubgraphPassConvTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'VALID'


'''
# conv2d padded in 'SAME' mode is not yet supported in TRT, reopen this when support is complete.
class TensorRTSubgraphPassConvSamePaddingTest(InferencePassTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 3
        self.conv_padding = 'SAME'
'''


class TensorRTSubgraphPassDepthwiseConvTest(TensorRTSubgraphPassConvTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 6
        self.conv_padding = [1, 1]


class TensorRTSubgraphPassConvTransposeTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            conv_out = fluid.layers.conv2d_transpose(
                input=data,
                num_filters=self.conv_num_filters,
                filter_size=self.conv_filter_size,
                groups=self.conv_groups,
                padding=self.conv_padding,
                bias_attr=False,
                act=None)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConvTransposeTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [conv_out]

    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = [1, 1]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassConvTransposeValidPaddingTest(
        TensorRTSubgraphPassConvTransposeTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'VALID'


'''
# conv2d_transpose padded in 'SAME' mode is not yet supported in TRT, reopen this when support is complete.
class TensorRTSubgraphPassConvTransposeSamePaddingTest(TensorRTSubgraphPassConvTransposeTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = 'SAME'
'''


class TensorRTSubgraphPassDepthwiseConvTransposeTest(
        TensorRTSubgraphPassConvTransposeTest):
    def set_params(self):
        self.conv_num_filters = 6
        self.conv_filter_size = 6
        self.conv_groups = 1
        self.conv_padding = [1, 1]


class TensorRTSubgraphPassFcTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            fc_out = fluid.layers.fc(input=[data], act=None, size=1000)
            reshape_out = fluid.layers.reshape(x=fc_out, shape=[1, 1000])
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassFcTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [reshape_out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            # TRT output shape of fc is (1, 1000, 1, 1). To compare the output value only, flatten the results.
            self.check_output_with_option(use_gpu, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassPoolTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            pool_out = fluid.layers.pool2d(
                input=data,
                pool_size=self.pool_size,
                pool_type=self.pool_type,
                pool_stride=self.pool_stride,
                pool_padding=self.pool_padding,
                global_pooling=self.global_pooling,
                ceil_mode=self.ceil_mode,
                exclusive=self.exclusive)
            out = fluid.layers.batch_norm(pool_out, is_test=True)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassPoolTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassAvgPoolTest(TensorRTSubgraphPassPoolTest):
    def set_params(self):
        self.pool_size = 2
        self.pool_type = 'avg'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTSubgraphPassGlobalPoolTest(TensorRTSubgraphPassPoolTest):
    def set_params(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = True
        self.ceil_mode = False
        self.exclusive = False


class TensorRTSubgraphPassCeilPoolTest(TensorRTSubgraphPassPoolTest):
    def set_params(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = True
        self.exclusive = False


class TensorRTSubgraphPassExclusivePoolTest(TensorRTSubgraphPassPoolTest):
    def set_params(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 0
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = True


class TensorRTSubgraphPassSamePaddingPoolTest(InferencePassTest):
    def set_params(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 'SAME'
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTSubgraphPassValidPaddingPoolTest(InferencePassTest):
    def set_params(self):
        self.pool_size = 2
        self.pool_type = 'max'
        self.pool_stride = 1
        self.pool_padding = 'VALID'
        self.global_pooling = False
        self.ceil_mode = False
        self.exclusive = False


class TensorRTSubgraphPassActivationTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            act_out = self.append_act(data)
            out = fluid.layers.batch_norm(act_out, is_test=True)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassActivationTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_act(self, x):
        return fluid.layers.relu(x)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassLeakyReluTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.leaky_relu(x)


class TensorRTSubgraphPassRelu6Test(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.relu6(x)


class TensorRTSubgraphPassSoftMaxTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.softmax(x)


class TensorRTSubgraphPassSigmoidTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.sigmoid(x)


class TensorRTSubgraphPassHardSwishTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.hard_swish(x)


class TensorRTSubgraphPassHardSigmoidTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.hard_sigmoid(x)


class TensorRTSubgraphPassTanhTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.tanh(x)


class TensorRTSubgraphPassSwishTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.swish(x)


class TensorRTSubgraphPassPreluAllTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.prelu(x, mode='all')


class TensorRTSubgraphPassPreluChannelTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.prelu(x, mode='channel')


class TensorRTSubgraphPassPreluElementTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.prelu(x, mode='element')


class TensorRTSubgraphPassGeluTest(TensorRTSubgraphPassActivationTest):
    def append_act(self, x):
        return fluid.layers.gelu(x)


class TensorRTSubgraphPassConcatTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(
                name="data1", shape=[-1, 3, 64, 64], dtype="float32")
            data2 = fluid.data(
                name="data2", shape=[-1, 3, 64, 64], dtype="float32")
            concat_out = fluid.layers.concat([data1, data2], axis=2)
            out = fluid.layers.batch_norm(concat_out, is_test=True)
        self.feeds = {
            "data1": np.random.random([1, 3, 64, 64]).astype("float32"),
            "data2": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassConcatTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassSplitTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            split_out = fluid.layers.split(data, dim=-1, num_or_sections=2)
            out = fluid.layers.batch_norm(split_out[0], is_test=True)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassSplitTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassInstanceNormTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            fc_out = fluid.layers.fc(input=data, size=200)
            param_attr = fluid.ParamAttr(
                name='instance_norm_w',
                initializer=fluid.initializer.Constant(value=1.0))
            bias_attr = fluid.ParamAttr(
                name='instance_norm_b',
                initializer=fluid.initializer.Constant(value=0.0))
            out = fluid.layers.instance_norm(
                input=fc_out, param_attr=param_attr, bias_attr=bias_attr)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassInstanceNormTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu, atol=1e-4, flatten=True)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassLayerNormTest(InferencePassTest):
    def setUp(self):
        self.set_params()
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 3, 64, 64], dtype="float32")
            out = fluid.layers.layer_norm(
                data, begin_norm_axis=self.begin_norm_axis)
        self.feeds = {
            "data": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassLayerNormTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def set_params(self):
        self.begin_norm_axis = 1

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassLayerNormBeginNormAxis2Test(
        TensorRTSubgraphPassLayerNormTest):
    def set_params(self):
        self.begin_norm_axis = 2


class TensorRTSubgraphPassLayerNormBeginNormAxis3Test(
        TensorRTSubgraphPassLayerNormTest):
    def set_params(self):
        self.begin_norm_axis = 3


class TensorRTSubgraphPassElementwiseTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(
                name="data1", shape=[-1, 3, 64, 64], dtype="float32")
            data2 = fluid.data(
                name="data2", shape=[-1, 3, 64, 64], dtype="float32")
            eltwise_out = self.append_eltwise(data1, data2)
            out = fluid.layers.batch_norm(eltwise_out, is_test=True)
        self.feeds = {
            "data1": np.random.random([1, 3, 64, 64]).astype("float32"),
            "data2": np.random.random([1, 3, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassElementwiseTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def append_eltwise(self, data1, data2):
        return fluid.layers.elementwise_add(x=data1, y=data2)

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


class TensorRTSubgraphPassElementwiseMulTest(
        TensorRTSubgraphPassElementwiseTest):
    def append_eltwise(self, data1, data2):
        return fluid.layers.elementwise_mul(x=data1, y=data2)


class TensorRTSubgraphPassShuffleChannelTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name="data", shape=[-1, 6, 64, 64], dtype="float32")
            sc_out = fluid.layers.shuffle_channel(data, group=3)
            out = fluid.layers.batch_norm(sc_out, is_test=True)
        self.feeds = {
            "data": np.random.random([1, 6, 64, 64]).astype("float32"),
        }
        self.enable_trt = True
        self.trt_parameters = TensorRTSubgraphPassShuffleChannelTest.TensorRTParam(
            1 << 30, 32, 0, AnalysisConfig.Precision.Float32, False, False)
        self.fetch_list = [out]

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))


if __name__ == "__main__":
    unittest.main()
