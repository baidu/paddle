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

import os
import six
import random
import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass


class InferencePassTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        paddle.enable_static()
        super(InferencePassTest, self).__init__(methodName)
        paddle.enable_static()
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.feeds = None
        self.fetch_list = None

        self.enable_mkldnn = False
        self.enable_mkldnn_bfloat16 = False
        self.enable_trt = False
        self.enable_tensorrt_oss = True
        self.trt_parameters = None
        self.dynamic_shape_params = None
        self.enable_lite = False
        self.lite_parameters = None
        self.path = "./inference_pass/" + self.__class__.__name__ + "/"
        np.random.seed(1)
        random.seed(1)

    def _get_place(self):
        return set([False, core.is_compiled_with_cuda()])

    def _save_models(self, executor, program, scope):
        with fluid.scope_guard(scope):
            outs = executor.run(program=program,
                                feed=self.feeds,
                                fetch_list=self.fetch_list,
                                return_numpy=False)
            # save models as combined to ensure that 
            # there won't be too many useless files 
            # after finishing a couple of tests.
            fluid.io.save_inference_model(
                dirname=self.path,
                feeded_var_names=list(self.feeds.keys()),
                target_vars=self.fetch_list,
                executor=executor,
                main_program=program,
                model_filename="model",
                params_filename="params")

        return outs

    def _get_analysis_outputs(self, config):
        '''
        Return AnalysisPredictor outputs. 
        '''
        predictor = create_paddle_predictor(config)
        tensor_shapes = predictor.get_input_tensor_shape()
        names = predictor.get_input_names()
        for i, name in enumerate(names):
            shape = tensor_shapes[name]
            shape[0] = 1
            tensor = predictor.get_input_tensor(name)
            feed_data = list(self.feeds.values())[i]
            tensor.copy_from_cpu(np.array(feed_data))
            if type(feed_data) == fluid.LoDTensor:
                tensor.set_lod(feed_data.lod())

        predictor.zero_copy_run()

        output_names = predictor.get_output_names()
        outs = [
            predictor.get_output_tensor(out_name).copy_to_cpu()
            for out_name in output_names
        ]

        return outs

    def _get_analysis_config(self,
                             use_gpu=False,
                             use_trt=False,
                             use_mkldnn=False):
        '''
        Return a new object of AnalysisConfig. 
        '''
        config = AnalysisConfig(
            os.path.join(self.path, "model"), os.path.join(self.path, "params"))
        config.disable_gpu()
        config.switch_specify_input_names(True)
        config.switch_ir_optim(True)
        config.switch_use_feed_fetch_ops(False)
        if use_gpu:
            config.enable_use_gpu(100, 0)
            if use_trt:
                config.enable_tensorrt_engine(
                    self.trt_parameters.workspace_size,
                    self.trt_parameters.max_batch_size,
                    self.trt_parameters.min_subgraph_size,
                    self.trt_parameters.precision,
                    self.trt_parameters.use_static,
                    self.trt_parameters.use_calib_mode)

                if self.dynamic_shape_params:
                    config.set_trt_dynamic_shape_info(
                        self.dynamic_shape_params.min_input_shape,
                        self.dynamic_shape_params.max_input_shape,
                        self.dynamic_shape_params.optim_input_shape,
                        self.dynamic_shape_params.disable_trt_plugin_fp16)
                if self.enable_tensorrt_oss:
                    config.enable_tensorrt_oss()

        elif use_mkldnn:
            config.enable_mkldnn()
            if self.enable_mkldnn_bfloat16:
                config.enable_mkldnn_bfloat16()

        return config

    def check_output(self, atol=1e-5):
        '''
        Check whether calculating on CPU and GPU, enable TensorRT 
        or disable TensorRT, enable MKLDNN or disable MKLDNN 
        are all the same. 
        '''
        self.assertFalse(self.feeds is None,
                         "The inputs of the model is None. ")
        use_gpu = self._get_place()
        for place_ in use_gpu:
            self.check_output_with_option(place_, atol)

    def check_output_with_option(self,
                                 use_gpu,
                                 atol=1e-5,
                                 flatten=False,
                                 quant=False,
                                 rtol=1e-5):
        '''
        Check whether calculating on CPU and GPU, enable TensorRT 
        or disable TensorRT, enable MKLDNN or disable MKLDNN 
        are all the same. 
        '''
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(place)
        scope = fluid.Scope()
        device = "GPU" if use_gpu else "CPU"
        with fluid.scope_guard(scope):
            executor.run(self.startup_program)

        if quant:
            main_graph = IrGraph(
                core.Graph(self.main_program.desc), for_test=True)

            transform_pass = QuantizationTransformPass(
                scope=scope,
                place=place,
                activation_quantize_type=self.activation_quant_type,
                weight_quantize_type=self.weight_quant_type,
                quantizable_op_type=[
                    'conv2d', 'mul', 'depthwise_conv2d', 'conv2d_transpose'
                ])
            transform_pass.apply(main_graph)
            weight_scale_map = {
                "conv2d": "conv2d_0.w_0.scale",
                "mul": "fc_0.w_0.scale"
            }

            weight_scale_tensor = scope.var(weight_scale_map[
                self.quantized_op_type]).get_tensor()
            weight_scale = np.ones(self.channels).astype("float32")
            weight_scale_tensor.set(weight_scale, place)

            op_nodes = main_graph.all_op_nodes()
            for op_node in op_nodes:
                if op_node.name() in [self.quantized_op_type, "relu"]:
                    op_node.op()._set_attr("out_threshold", 0.5)

            with fluid.scope_guard(scope):
                executor.run(program=self.main_program,
                             feed=self.feeds,
                             fetch_list=self.fetch_list)

            freeze_pass = QuantizationFreezePass(
                scope=scope,
                place=place,
                weight_quantize_type=self.weight_quant_type)
            freeze_pass.apply(main_graph)
            self.main_program = main_graph.to_program()

        outs = self._save_models(executor, self.main_program, scope)

        analysis_outputs = self._get_analysis_outputs(
            self._get_analysis_config(use_gpu=use_gpu))

        # Check whether the results calculated on CPU and on GPU are the same. 
        self.assertTrue(
            len(outs) == len(analysis_outputs),
            "The number of outputs is different between inference and training forward at {}".
            format(device))

        for out, analysis_output in zip(outs, analysis_outputs):
            out = np.array(out)
            if flatten:
                out = out.flatten()
                analysis_output = analysis_output.flatten()

            self.assertTrue(
                np.allclose(
                    out, analysis_output, atol=atol),
                "Output has diff between inference and training forward at {} ".
                format(device))

        # Check whether the trt results and the GPU results are the same. 
        if use_gpu and self.enable_trt:
            tensorrt_outputs = self._get_analysis_outputs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_trt=self.enable_trt))

            if self.trt_parameters.use_static:
                #deserialize
                tensorrt_outputs = self._get_analysis_outputs(
                    self._get_analysis_config(
                        use_gpu=use_gpu, use_trt=self.enable_trt))

            self.assertTrue(
                len(tensorrt_outputs) == len(outs),
                "The number of outputs is different between GPU and TensorRT. ")

            for out, tensorrt_output in zip(outs, tensorrt_outputs):
                out = np.array(out)
                if flatten:
                    out = out.flatten()
                    tensorrt_output = tensorrt_output.flatten()

                self.assertTrue(
                    np.allclose(
                        out, tensorrt_output, rtol=rtol, atol=atol),
                    "Output has diff between GPU and TensorRT. ")

        # Check whether the mkldnn results and the CPU results are the same. 
        if (not use_gpu) and self.enable_mkldnn:
            mkldnn_outputs = self._get_analysis_outputs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_mkldnn=self.enable_mkldnn))

            self.assertTrue(
                len(outs) == len(mkldnn_outputs),
                "The number of outputs is different between CPU and MKLDNN. ")

            if self.enable_mkldnn_bfloat16:
                atol = 0.01
            for out, mkldnn_output in zip(outs, mkldnn_outputs):
                self.assertTrue(
                    np.allclose(
                        np.array(out), mkldnn_output, atol=atol),
                    "Output has diff between CPU and MKLDNN. ")

    class TensorRTParam:
        '''
        Prepare TensorRT subgraph engine parameters. 
        '''

        def __init__(self, workspace_size, max_batch_size, min_subgraph_size,
                     precision, use_static, use_calib_mode):
            self.workspace_size = workspace_size
            self.max_batch_size = max_batch_size
            self.min_subgraph_size = min_subgraph_size
            self.precision = precision
            self.use_static = use_static
            self.use_calib_mode = use_calib_mode

    class DynamicShapeParam:
        '''
        Prepare TensorRT subgraph engine dynamic shape parameters. 
        '''

        def __init__(self, min_input_shape, max_input_shape, optim_input_shape,
                     disable_trt_plugin_fp16):
            self.min_input_shape = min_input_shape
            self.max_input_shape = max_input_shape
            self.optim_input_shape = optim_input_shape
            self.disable_trt_plugin_fp16 = disable_trt_plugin_fp16

    class LiteParam:
        '''
        Prepare Lite subgraph engine parameters. 
        '''

        def __init__(self, precision, passes_filter, ops_filter):
            self.precision = precision
            self.passes_filter = passes_filter
            self.ops_filter = ops_filter
