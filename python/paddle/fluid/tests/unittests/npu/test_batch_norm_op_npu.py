#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator
from op_test import OpTest, _set_use_system_allocator
from paddle.fluid import Program, program_guard

from test_batch_norm_op import _reference_testing, _cal_mean_variance, _reference_training, _reference_grad

_set_use_system_allocator(False)
paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestBatchNormOpInference(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.init_kernel_type()
        self.data_formats = ["NCHW", "NHWC"]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def check_with_place(self, place, data_layout, dtype, shape):
        epsilon = epsilon = 0.00001
        if len(shape) == 2:
            x_shape = shape
            c = x_shape[1]
        else:
            n, h, w, c = shape[0], shape[1], shape[2], shape[3]
            if data_layout == "NHWC":
                x_shape = [n, h, w, c]
            elif data_layout == "NCHW":
                x_shape = [n, c, h, w]
            else:
                raise ValueError("Unknown data layout.")
        scale_shape = [c]

        x = np.random.random_sample(x_shape).astype(dtype)
        x = x - 0.5
        scale = np.random.random_sample(scale_shape).astype(np.float32)
        bias = np.random.random_sample(scale_shape).astype(np.float32)
        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.ones(scale_shape).astype(np.float32)
        y = _reference_testing(x, scale, bias, mean, variance, epsilon,
                               data_layout).astype(dtype)
        var_dict = locals()
        var_names = ["x", "scale", "bias", "mean", "variance", "y"]
        ground_truth = {name: var_dict[name] for name in var_names}
        ground_truth["saved_mean"] = mean
        ground_truth["saved_variance"] = variance

        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in ground_truth:
                block.create_var(
                    name=name, dtype="float32", shape=ground_truth[name].shape)
            inputs = {
                "X": block.var("x"),
                "Scale": block.var("scale"),
                "Bias": block.var("bias"),
                "Mean": block.var("mean"),
                "Variance": block.var("variance")
            }
            attrs = {
                "epsilon": epsilon,
                "is_test": True,
                "data_layout": data_layout,
                "use_mkldnn": False,
                "fuse_with_relu": False,
            }
            outputs = {
                "Y": block.var("y"),
                "MeanOut": block.var("mean"),  # share memory
                "VarianceOut": block.var("variance"),  # share memory
                "SavedMean": block.var("saved_mean"),
                "SavedVariance": block.var("saved_variance")
            }
            block.create_var(name="reserve_space", dtype='float32')
            outputs["ReserveSpace"] = block.var('reserve_space')
            bn_op = block.append_op(
                type="batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)

            program._sync_with_cpp()

            exe = fluid.Executor(place)
            out = exe.run(
                program,
                feed={
                    name: ground_truth[name]
                    for name in ["x", "scale", "bias", "mean", "variance"]
                },
                fetch_list=["y"])
            self.__assert_close(var_dict["y"], out[0], "y", atol=1e-3)

    def test_check_output(self):
        place = core.NPUPlace(0)
        for data_format in self.data_formats:
            self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])

    def init_kernel_type(self):
        pass


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestFP16BatchNormOpInference(TestBatchNormOpInference):
    def setUp(self):
        self.dtype = np.float16
        self.init_kernel_type()
        self.data_formats = ["NCHW", "NHWC"]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestBatchNormOpTraining(unittest.TestCase):
    def set_npu(self):
        self.__class__.use_npu = True

    def setUp(self):
        self.set_npu()
        self.use_mkldnn = False
        self.fuse_with_relu = False
        self.data_formats = ["NCHW", "NHWC"]
        self.momentum = 0.9
        self.use_momentum_variable = False
        self.epsilon = 0.00001
        self.init_kernel_type()
        self.init_test_case()

    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = set()
        self.fetch_list = [
            "y", 'mean', 'variance', 'saved_mean', 'saved_variance', 'x@GRAD',
            'scale@GRAD', 'bias@GRAD'
        ]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.allclose(np.array(tensor), np_array, atol=atol)

    def ref_forward_backward(self, x, y_grad, scale, bias, mean, variance,
                             epsilon, momentum, shape, data_layout):
        # run forward
        y, saved_mean, var_ref = _reference_training(x, scale, bias, epsilon,
                                                     data_layout)
        mean_out = saved_mean * (1. - momentum) + momentum * mean
        variance_out = var_ref * (1. - momentum) + momentum * variance
        saved_variance = 1. / np.sqrt(var_ref + epsilon)
        # run backward
        x_grad, scale_grad, bias_grad = _reference_grad(
            x, y_grad, scale, saved_mean, var_ref, epsilon, data_layout)

        return y, mean_out, variance_out, saved_mean, saved_variance, x_grad, scale_grad, bias_grad

    def set_mean_variance(self, scale_shape, x, data_layout):
        mean, variance = _cal_mean_variance(x, self.epsilon, data_layout)
        mean_pre = np.zeros(scale_shape).astype(np.float32)
        variance_pre = np.ones(scale_shape).astype(np.float32)
        # computing global mean/variance for one step
        if self.use_global_stats:
            mom = self.momentum
            mean = mean * (1. - mom) + mom * mean_pre
            variance = variance * (1. - mom) + mom * variance_pre
        return mean, variance

    def test_forward_backward(self):
        def test_with_place(place, data_layout, shape):
            # attr
            epsilon = self.epsilon
            momentum = self.momentum
            if data_layout == "NCHW":
                n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            else:
                n, h, w, c = shape[0], shape[1], shape[2], shape[3]
            scale_shape = [c]

            np.random.seed(123)
            x = np.random.random_sample(shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            mean, variance = self.set_mean_variance(scale_shape, x, data_layout)
            y_grad = np.random.random_sample(shape).astype(np.float32)
            momentum_var = np.array([momentum]).astype(np.float32)

            y, mean_out, variance_out, saved_mean, saved_variance, x_grad, scale_grad, bias_grad = self.ref_forward_backward(
                x, y_grad, scale, bias, mean, variance, epsilon, momentum,
                shape, data_layout)

            var_dict = locals()
            var_dict['y@GRAD'] = y_grad
            var_dict['x@GRAD'] = x_grad
            var_dict['scale@GRAD'] = scale_grad
            var_dict['bias@GRAD'] = bias_grad

            var_names = [
                'x', 'scale', 'bias', 'mean', 'variance', "y", 'saved_mean',
                'saved_variance', 'momentum_var'
            ]
            ground_truth = {name: var_dict[name] for name in var_names}

            program = fluid.Program()
            with fluid.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(
                        name=name,
                        dtype='float32',
                        shape=ground_truth[name].shape)
                inputs = {
                    "X": block.var('x'),
                    "Scale": block.var('scale'),
                    "Bias": block.var('bias'),
                    "Mean": block.var('mean'),
                    "Variance": block.var('variance')
                }
                attrs = {
                    "epsilon": epsilon,
                    "is_test": False,
                    "data_layout": data_layout,
                    "use_mkldnn": self.use_mkldnn,
                    "fuse_with_relu": self.fuse_with_relu,
                    "use_global_stats": self.use_global_stats
                }
                if self.use_momentum_variable:
                    inputs['MomentumTensor'] = block.var('momentum_var')
                else:
                    attrs['momentum'] = momentum

                outputs = {
                    "Y": block.var("y"),
                    "MeanOut": block.var('mean'),  # share memory
                    "VarianceOut": block.var('variance'),  # share memory
                    "SavedMean": block.var('saved_mean'),
                    "SavedVariance": block.var('saved_variance')
                }
                block.create_var(name="reserve_space", dtype='float32')
                outputs["ReserveSpace"] = block.var('reserve_space')
                bn_op = block.append_op(
                    type="batch_norm",
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs)
                block.create_var(name='y@GRAD', dtype='float32', shape=y.shape)

                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    bn_op.desc, self.no_grad_set, [])
                grad_op_desc = grad_op_desc_list[0]
                new_op_desc = block.desc.append_op()
                new_op_desc.copy_from(grad_op_desc)
                for var_name in grad_op_desc.output_arg_names():
                    block.desc.var(var_name.encode("ascii"))
                grad_op_desc.infer_var_type(block.desc)
                grad_op_desc.infer_shape(block.desc)
                for arg in grad_op_desc.output_arg_names():
                    grad_var = block.desc.find_var(arg.encode("ascii"))
                    grad_var.set_dtype(core.VarDesc.VarType.FP32)

                program._sync_with_cpp()

                exe = fluid.Executor(place)
                out = exe.run(program,
                              feed={
                                  name: var_dict[name]
                                  for name in [
                                      'x', 'scale', 'bias', 'mean', 'variance',
                                      'y@GRAD', 'momentum_var'
                                  ]
                              },
                              fetch_list=self.fetch_list)

            for id, name in enumerate(self.fetch_list):
                if name == 'variance':
                    self.__assert_close(
                        var_dict[name], out[id], name, atol=1e-3)
                    continue
                self.__assert_close(var_dict[name], out[id], name)
            print("op test forward passed: ", str(place), data_layout)

        for data_format in self.data_formats:
            test_with_place(core.NPUPlace(0), data_format, [2, 3, 4, 5])

    def init_kernel_type(self):
        pass


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestBatchNormOpTrainingCase1(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = set(['scale@GRAD', 'bias@GRAD'])
        self.fetch_list = ['y', 'mean', 'variance', 'x@GRAD']


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestBatchNormOpTrainingMomentumVariable(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_momentum_variable = True
        self.use_global_stats = False
        self.no_grad_set = set()
        self.fetch_list = [
            'y', 'mean', 'variance', 'saved_mean', 'saved_variance', 'x@GRAD',
            'scale@GRAD', 'bias@GRAD'
        ]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestBatchNormOpFreezeStatsTraining(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = True
        self.no_grad_set = set()
        self.fetch_list = [
            'y', 'mean', 'variance', 'x@GRAD', 'scale@GRAD', 'bias@GRAD'
        ]

    def reference_grad(self, x, y_grad, scale, mean, var, epsilon, data_format):
        if data_format == "NCHW":
            x = np.transpose(x, (0, 2, 3, 1))
            y_grad = np.transpose(y_grad, (0, 2, 3, 1))

        x_grad = scale * y_grad / np.sqrt(var + epsilon)
        grad_scale = np.sum(y_grad * (x - mean) / np.sqrt(var + epsilon),
                            axis=(0, 1, 2))
        grad_offset = np.sum(y_grad, axis=(0, 1, 2))

        # transfer back to N, C, H, W
        if data_format == "NCHW":
            x_grad = np.transpose(x_grad, (0, 3, 1, 2))
            x = np.transpose(x, (0, 3, 1, 2))
            y_grad = np.transpose(y_grad, (0, 3, 1, 2))

        return x_grad, grad_scale, grad_offset

    def ref_forward_backward(self, x, y_grad, scale, bias, mean, variance,
                             epsilon, momentum, shape, data_layout):
        if data_layout != "NCHW" and data_layout != "NHWC":
            raise ValueError("Unknown data order.")

        if data_layout == "NCHW":
            x = np.transpose(x, (0, 2, 3, 1))

        # run normalizaton
        normalized = (x - mean) / np.sqrt(variance + epsilon)
        y = normalized * scale + bias

        # transfer back to N, C, H, W
        if data_layout == "NCHW":
            x = np.transpose(x, (0, 3, 1, 2))
            y = np.transpose(y, (0, 3, 1, 2))

        mean_out = mean
        variance_out = variance
        saved_variance = 1. / np.sqrt(variance + epsilon)
        # run backward
        x_grad, scale_grad, bias_grad = self.reference_grad(
            x, y_grad, scale, mean, variance, epsilon, data_layout)

        return y, mean_out, variance_out, mean, saved_variance, x_grad, scale_grad, bias_grad


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestBatchNormOpFreezeStatsAndScaleBiasTraining(
        TestBatchNormOpFreezeStatsTraining):
    def init_test_case(self):
        self.use_global_stats = True
        self.no_grad_set = set(['scale@GRAD', 'bias@GRAD'])
        self.fetch_list = ['y', 'mean', 'variance', 'x@GRAD']


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestDygraphBatchNormTrainableStats(unittest.TestCase):
    def test_dygraph(self):
        places = [fluid.NPUPlace(0)]
        for p in places:
            shape = [4, 10, 4, 4]

            def compute(x, is_test, trainable_statistics):
                with fluid.dygraph.guard(p):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics)
                    y = bn(fluid.dygraph.to_variable(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute(x, False, False)
            y2 = compute(x, True, True)
            self.assertTrue(np.allclose(y1, y2))

    def test_static(self):
        places = [fluid.NPUPlace(0)]
        for p in places:
            exe = fluid.Executor(p)
            shape = [4, 10, 16, 16]

            def compute(x_np, is_test, trainable_statistics):
                with program_guard(Program(), Program()):
                    bn = fluid.dygraph.BatchNorm(
                        shape[1],
                        is_test=is_test,
                        trainable_statistics=trainable_statistics)
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = bn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute(x, False, False)
            y2 = compute(x, True, True)
            self.assertTrue(np.allclose(y1, y2))


if __name__ == "__main__":
    unittest.main()
