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

import math
import numpy as np

import paddle
from paddle.fluid.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

from . import quant_nn

layer_name_map = {
    'Conv2D': paddle.nn.Conv2D,
    'Linear': paddle.nn.Linear,
    'AdaptiveAvgPool2D': paddle.nn.AdaptiveAvgPool2D,
    'AdaptiveMaxPool2D': paddle.nn.AdaptiveMaxPool2D,
    'AvgPool2D': paddle.nn.AvgPool2D,
    'MaxPool2D': paddle.nn.MaxPool2D,
    'Hardswish': paddle.nn.Hardswish,
    'LeakyReLU': paddle.nn.LeakyReLU,
    'PReLU': paddle.nn.PReLU,
    'ReLU': paddle.nn.ReLU,
    'ReLU6': paddle.nn.ReLU6,
    'Sigmoid': paddle.nn.Sigmoid,
    'Softmax': paddle.nn.Softmax,
    'Swish': paddle.nn.Swish,
    'Tanh': paddle.nn.Tanh,
    'Hardswish': paddle.nn.Hardswish,
    'BatchNorm': paddle.nn.BatchNorm,
    'GroupNorm': paddle.nn.GroupNorm,
    'LayerNorm': paddle.nn.LayerNorm,
}

# Apply fake quant for the inputs of these layers
# TODO (jc): support paddle.nn.Conv2DTranspose
fake_quant_input_layers = [paddle.nn.Conv2D, paddle.nn.Linear]

# Apply fake quant for the output of these layers
# TODO(jc): fix the problem of adding duplicate fake_quant ops
# paddle.nn.AdaptiveAvgPool2D, paddle.nn.AvgPool2D, paddle.nn.ReLU,paddle.nn.LeakyReLU
fake_quant_output_layers = [
    paddle.nn.quant.add, paddle.nn.quant.subtract, paddle.nn.quant.multiply,
    paddle.nn.quant.divide
]

fake_quant_leaf_layers = [
    quant_nn.FakeQuantAbsMax,
    quant_nn.FakeQuantChannelWiseAbsMax,
    quant_nn.FakeQuantMovingAverageAbsMax,
    quant_nn.MovingAverageAbsMaxScale,
]

fake_quant_wrap_layers = [quant_nn.QuantizedConv2D, quant_nn.QuantizedLinear]

# The weight format of these layers is Cin * Cout * H * W 
spec_channel_axis_layers = [paddle.nn.Conv2D, paddle.nn.Conv2DTranspose]

weight_op_types = [
    "conv2d", "depthwise_conv2d", "matmul", "conv2d_transpose",
    "depthwise_conv2d_transpose"
]

fake_quantize_dequantize_op_types = [
    "fake_quantize_dequantize_abs_max",
    "fake_channel_wise_quantize_dequantize_abs_max",
    "fake_quantize_dequantize_moving_average_abs_max"
]


def load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Can not find " + var_name + " in the scope."
    return np.array(var_node.get_tensor())


def find_previous_op(block, var_name):
    """
    Find the previous op for the input variable.
    """
    for op in block.ops:
        if var_name in op.output_arg_names:
            return op


def find_next_ops(block, var_name):
    """
    Find all followed ops for the input variable.
    """
    res_ops = []
    for op in block.ops:
        if var_name in op.input_arg_names:
            res_ops.append(op)
    return res_ops


def find_parent_layer_and_sub_name(model, name):
    """
    Given the model and the name of a layer, find the parent layer and
    the sub_name of the layer.
    For example, if name is 'block_1/convbn_1/conv_1', the parent layer is
    'block_1/convbn_1' and the sub_name is `conv_1`.
    """
    assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."
    assert len(name) > 0, "The input (name) should not be empty."

    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == '.':
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name


def is_leaf_layer(layer):
    """
    Whether the layer is leaf layer.
    """
    return isinstance(layer, paddle.nn.Layer) \
        and len(layer.sublayers()) == 0


def expand_quantized_bins(quantized_bins, reference_bins):
    """
    """
    expanded_quantized_bins = [0] * len(reference_bins)
    num_merged_bins = int(len(reference_bins) / len(quantized_bins))
    j_start = 0
    j_end = num_merged_bins
    for idx in range(len(quantized_bins)):
        zero_count = reference_bins[j_start:j_end].count(0)
        num_merged_bins = j_end - j_start
        if zero_count == num_merged_bins:
            avg_bin_ele = 0
        else:
            avg_bin_ele = quantized_bins[idx] / (
                num_merged_bins - zero_count + 0.0)
        for idx1 in range(j_start, j_end):
            expanded_quantized_bins[idx1] = (0 if reference_bins[idx1] == 0 else
                                             avg_bin_ele)
        j_start += num_merged_bins
        j_end += num_merged_bins
        if (idx + 1) == len(quantized_bins) - 1:
            j_end = len(reference_bins)
    return expanded_quantized_bins


def safe_entropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
    '''
    Calculate the entropy.
    '''
    assert len(reference_distr_P) == len(candidate_distr_Q)
    tmp_sum1 = 0
    tmp_sum2 = 0
    for idx in range(len(reference_distr_P)):
        p_idx = reference_distr_P[idx]
        q_idx = candidate_distr_Q[idx]
        if p_idx == 0:
            tmp_sum1 += 0
            tmp_sum2 += 0
        else:
            if q_idx == 0:
                _logger.error("Fatal error!, idx = " + str(idx) +
                              " qindex = 0! p_idx = " + str(p_idx))
            tmp_sum1 += p_idx * (math.log(Q_sum * p_idx))
            tmp_sum2 += p_idx * (math.log(P_sum * q_idx))
    return (tmp_sum1 - tmp_sum2) / P_sum


def cal_kl_scaling_factor(hist, abs_max, bits):
    '''
    Using the KL-divergenc method to get the more precise scaling factor.
    '''
    assert hist.ndim == 1
    hist_bins = hist.shape[0]
    starting_iter = int((hist_bins - 1) * 0.5)
    bin_width = abs_max / hist_bins
    quant_range = 2**(bits - 1) - 1

    P_sum = np.sum(np.array(hist).ravel())
    min_kl_divergence = 0
    min_kl_index = 0
    kl_inited = False

    for i in range(starting_iter, hist_bins):
        reference_distr_P = hist[0:i].tolist()
        outliers_count = sum(hist[i:])
        if reference_distr_P[i - 1] == 0:
            continue
        reference_distr_P[i - 1] += outliers_count
        reference_distr_bins = reference_distr_P[:]
        candidate_distr_Q = hist[0:i].tolist()
        num_merged_bins = int(i / quant_range)
        candidate_distr_Q_quantized = [0] * quant_range
        j_start = 0
        j_end = num_merged_bins
        for idx in range(quant_range):
            candidate_distr_Q_quantized[idx] = sum(candidate_distr_Q[j_start:
                                                                     j_end])
            j_start += num_merged_bins
            j_end += num_merged_bins
            if (idx + 1) == quant_range - 1:
                j_end = i
        candidate_distr_Q = expand_quantized_bins(candidate_distr_Q_quantized,
                                                  reference_distr_bins)
        Q_sum = sum(candidate_distr_Q)
        kl_divergence = safe_entropy(reference_distr_P, P_sum,
                                     candidate_distr_Q, Q_sum)
        if not kl_inited:
            min_kl_divergence = kl_divergence
            min_kl_index = i
            kl_inited = True
        elif kl_divergence < min_kl_divergence:
            min_kl_divergence = kl_divergence
            min_kl_index = i
        else:
            pass
    if min_kl_index == 0:
        while starting_iter > 0:
            if hist[starting_iter] == 0:
                starting_iter -= 1
                continue
            else:
                break
        min_kl_index = starting_iter
    return (min_kl_index + 0.5) * bin_width
