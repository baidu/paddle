#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
from scipy.special import expit, erf
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid import compiler, Program, program_guard


def ref_batch_norm_infer(x, scale, bias, mean, variance, momentum, epsilon,
                         data_layout):
    if data_layout == "NCHW":
        n, c, h, w = x.shape
        mean_tile = np.reshape(mean, (1, c, 1, 1))
        mean_tile = np.tile(mean_tile, (n, 1, h, w))
        variance_tile = np.reshape(variance, (1, c, 1, 1))
        variance_tile = np.tile(variance_tile, (n, 1, h, w))
        normalized_x = (x - mean_tile) / np.sqrt(variance_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        bias_tile = np.reshape(bias, (1, c, 1, 1))
        bias_tile = np.reshape(bias_tile, (1, c, 1, 1))
        y = normalized_x * scale_tile + bias_tile
    elif data_layout == "NHWC":
        normalized_x = (x - mean) / np.sqrt(variance + epsilon)
        y = normalized_x * scale + bias
    else:
        raise ValueError(
            "Unsupported data layout! Only NCHW and NHWC is supported, but received "
            + data_layout)
    return y


def ref_batch_norm_train(x, y_grad, scale, bias, mean, variance, momentum,
                         epsilon, data_layout):
    # Forward
    if data_layout == "NCHW":
        n, c, h, w = x.shape
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 2, 3))
        x_sum = np.sum(x, axis=(0, 2, 3))
        element_count = np.size(x) / int(np.shape(x)[1])
        saved_mean = x_sum / element_count
        saved_variance = x_square_sum / element_count - saved_mean * saved_mean
        saved_mean_tile = np.reshape(saved_mean, (1, c, 1, 1))
        saved_mean_tile = np.tile(saved_mean_tile, (n, 1, h, w))
        saved_variance_tile = np.reshape(saved_variance, (1, c, 1, 1))
        saved_variance_tile = np.tile(saved_variance_tile, (n, 1, h, w))
        normalized_x = (
            x - saved_mean_tile) / np.sqrt(saved_variance_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        bias_tile = np.reshape(bias, (1, c, 1, 1))
        bias_tile = np.reshape(bias_tile, (1, c, 1, 1))
        y = normalized_x * scale_tile + bias_tile
    elif data_layout == "NHWC":
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 1, 2))
        x_sum = np.sum(x, axis=(0, 1, 2))
        element_count = np.size(x) / int(np.shape(x)[-1])
        saved_mean = x_sum / element_count
        saved_variance = x_square_sum / element_count - saved_mean * saved_mean
        normalized_x = (x - saved_mean) / np.sqrt(saved_variance + epsilon)
        y = normalized_x * scale + bias
    else:
        raise ValueError(
            "Unsupported data layout! Only NCHW and NHWC is supported, but received "
            + data_layout)
    mean_out = saved_mean * (1. - momentum) + momentum * mean
    variance_out = saved_variance * (1. - momentum) + momentum * variance
    saved_inv_std = 1. / np.sqrt(saved_variance + epsilon)
    # Backward
    # Use the following formulas to calculate gradients:
    # grad_scale =
    #   sum(grad_y * (x - mean)) * rsqrt(variance + epsilon)
    #
    # grad_bias = sum(y)
    #
    # x_grad =
    #   1/N * scale * rsqrt(variance + epsilon) * (N * grad_y - sum(grad_y) -
    #   (x - mean) * sum(grad_y * (x - mean)) / (variance + epsilon))
    # Transfer from (N, C, H, W) to (N, H, W, C) to simplify computation
    if data_layout == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        y_grad = np.transpose(y_grad, (0, 2, 3, 1))
    x_grad = scale * (
        y_grad - np.mean(
            y_grad, axis=(0, 1, 2)) - (x - saved_mean) * np.mean(
                y_grad * (x - saved_mean), axis=(0, 1, 2)) /
        (saved_variance + epsilon)) / np.sqrt(saved_variance + epsilon)
    scale_grad = np.sum(y_grad * (x - saved_mean) /
                        np.sqrt(saved_variance + epsilon),
                        axis=(0, 1, 2))
    bias_grad = np.sum(y_grad, axis=(0, 1, 2))
    # Transfer back to N, C, H, W
    if data_layout == "NCHW":
        x_grad = np.transpose(x_grad, (0, 3, 1, 2))
        x = np.transpose(x, (0, 3, 1, 2))
        y_grad = np.transpose(y_grad, (0, 3, 1, 2))
    return y, mean_out, variance_out, saved_mean, saved_inv_std, x_grad, scale_grad, bias_grad


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUBatchNormOp(unittest.TestCase):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.op_type = "batch_norm"
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.data_layout = "NCHW"
        self.epsilon = 1e-05
        self.momentum = 0.9
        self.set_attrs()

        if self.data_layout == "NHWC":
            channel_size = self.shape[3]
        elif self.data_layout == "NCHW":
            channel_size = self.shape[1]
        else:
            raise ValueError(
                "Unsupported data layout! Only NCHW and NHWC is supported, but received "
                + self.data_layout)
        np.random.seed(1024)
        self.x_np = np.random.random_sample(self.shape).astype(self.dtype)
        self.scale_np = np.random.random_sample(
            [channel_size]).astype(self.dtype)
        self.bias_np = np.random.random_sample(
            [channel_size]).astype(self.dtype)
        self.mean_np = np.zeros([channel_size]).astype(self.dtype)
        self.variance_np = np.ones([channel_size]).astype(self.dtype)
        self.saved_mean_np = np.zeros([channel_size]).astype(self.dtype)
        self.saved_variance_np = np.ones([channel_size]).astype(self.dtype)

    def set_attrs(self):
        pass

    def test_infer(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            scale = paddle.fluid.data('Scale', self.scale_np.shape,
                                      self.scale_np.dtype)
            bias = paddle.fluid.data('Bias', self.bias_np.shape,
                                     self.bias_np.dtype)
            mean = paddle.fluid.data('Mean', self.mean_np.shape,
                                     self.mean_np.dtype)
            variance = paddle.fluid.data('Variance', self.variance_np.shape,
                                         self.variance_np.dtype)
            y = F.batch_norm(x, mean, variance, scale, bias, False,
                             self.momentum, self.epsilon, self.data_layout)
            exe = paddle.static.Executor(self.place)
            [y_np] = exe.run(feed={
                'X': self.x_np,
                'Scale': self.scale_np,
                'Bias': self.bias_np,
                'Mean': self.mean_np,
                'Variance': self.variance_np
            },
                             fetch_list=[y])
        y_np_ref = ref_batch_norm_infer(
            self.x_np, self.scale_np, self.bias_np, self.mean_np,
            self.variance_np, self.momentum, self.epsilon, self.data_layout)
        self.assertEqual(np.allclose(y_np_ref, y_np), True)

    def test_train(self):
        y_grad_np = np.random.random_sample(self.shape).astype(self.dtype)
        y_np, mean_out_np, variance_out_np, saved_mean_np, saved_variance_np, x_grad_np, scale_grad_np, bias_grad_np = ref_batch_norm_train(
            self.x_np, y_grad_np, self.scale_np, self.bias_np, self.mean_np,
            self.variance_np, self.momentum, self.epsilon, self.data_layout)
        inputs = {
            'X': self.x_np,
            'Scale': self.scale_np,
            'Bias': self.bias_np,
            'Mean': self.mean_np,
            'Variance': self.variance_np,
            'Y@GRAD': y_grad_np
        }
        outputs = {
            'Y': y_np,
            'Mean': mean_out_np,
            'Variance': variance_out_np,
            'SavedMean': saved_mean_np,
            'SavedVariance': saved_variance_np,
            'X@GRAD': x_grad_np,
            'Scale@GRAD': scale_grad_np,
            'Bias@GRAD': bias_grad_np
        }
        attrs = {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'is_test': False,
            'data_layout': self.data_layout,
            'use_mkldnn': False,
            'fuse_with_relu': False,
            'use_global_stats': False,
        }
        paddle.enable_static()
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            block = program.global_block()
            # Set inputs, outputs and attributes to the forward op of batch_norm 
            input_vars = {}
            for var_name in inputs:
                arg_name = var_name
                np_value = inputs[var_name]
                if not block.has_var(var_name):
                    block.create_var(
                        name=var_name,
                        shape=np_value.shape,
                        dtype=np_value.dtype)
                input_vars[arg_name] = block.var(var_name)
            fetch_list = []
            output_vars = {}
            for var_name in outputs:
                arg_name = var_name
                np_value = outputs[var_name]
                if not block.has_var(var_name):
                    block.create_var(
                        name=var_name,
                        shape=np_value.shape,
                        dtype=np_value.dtype)
                if var_name == 'Mean':
                    arg_name = 'MeanOut'  # Share memory
                if var_name == 'Variance':
                    arg_name = 'VarianceOut'  # Share memory
                output_vars[arg_name] = block.var(var_name)
                fetch_list.append(var_name)
            batch_norm_op = block.append_op(
                type="batch_norm",
                inputs=input_vars,
                outputs=output_vars,
                attrs=attrs)
            # Generate the backward op_desc of batch_norm
            grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                batch_norm_op.desc, set(), [])
            grad_op_desc = grad_op_desc_list[0]
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(grad_op_desc)
            program._sync_with_cpp()
            exe = paddle.static.Executor(self.place)
            outs = exe.run(program, feed=inputs, fetch_list=fetch_list)
            for id, name in enumerate(fetch_list):
                self.assertEqual(
                    np.allclose(
                        outputs[name], outs[id], atol=1e-4), True)


if __name__ == "__main__":
    unittest.main()
