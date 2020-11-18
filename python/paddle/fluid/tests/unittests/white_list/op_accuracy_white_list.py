# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# For op in NO_FP64_CHECK_GRAD_OP_LIST, the op test requires check_grad with fp64 precision
NO_FP64_CHECK_GRAD_OP_LIST = [
    'affine_grid', \
    'clip', \
    'conv2d', \
    'conv2d_transpose', \
    'conv3d', \
    'conv3d_transpose', \
    'conv_shift', \
    'cos_sim', \
    'cudnn_lstm', \
    'cvm', \
    'data_norm', \
    'deformable_conv', \
    'deformable_conv_v1', \
    'deformable_psroi_pooling', \
    'depthwise_conv2d', \
    'depthwise_conv2d_transpose', \
    'dropout', \
    'fused_elemwise_activation', \
    'hinge_loss', \
    'huber_loss', \
    'im2sequence', \
    'increment', \
    'l1_norm', \
    'log_loss', \
    'lrn', \
    'margin_rank_loss', \
    'match_matrix_tensor', \
    'matmul', \
    'max_pool2d_with_index', \
    'max_pool3d_with_index', \
    'minus', \
    'modified_huber_loss', \
    'nce', \
    'pool2d', \
    'pool3d', \
    'prroi_pool', \
    'rank_loss', \
    'reduce_max', \
    'reduce_min', \
    'reshape2', \
    'roi_perspective_transform', \
    'row_conv', \
    'rnn', \
    'scatter', \
    'sequence_conv', \
    'sequence_pool', \
    'sequence_reverse', \
    'sequence_slice', \
    'sequence_topk_avg_pooling', \
    'shuffle_channel', \
    'sigmoid', \
    'smooth_l1_loss', \
    'softmax', \
    'spectral_norm', \
    'squared_l2_distance', \
    'squared_l2_norm', \
    'tanh', \
    'mish', \
    'transpose2', \
    'trilinear_interp', \
    'trilinear_interp_v2', \
    'var_conv_2d', \
    'warpctc', \
    'bilateral_slice'
]

NO_FP16_CHECK_GRAD_OP_LIST = [
    'fused_elemwise_activation', \
    'pool2d', \
    'pool3d', \
    'softmax',\
    'conv2d_transpose'
]
