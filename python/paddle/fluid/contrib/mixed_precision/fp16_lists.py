#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import copy

__all__ = ["AutoMixedPrecisionLists"]


class AutoMixedPrecisionLists(object):
    """
    AutoMixedPrecisionLists is a class for black/white list. It can update
    pre-defined black list and white list according to users' custom black
    white lists. The lists are used for an algorithm which determines op's
    execution mode (fp32 or fp16).

    Args:
        custom_white_list (set): Users' custom white list.
        custom_black_list (set): Users' custom black list.
    """

    def __init__(self,
                 custom_white_list=None,
                 custom_black_list=None,
                 custom_black_varnames=None):
        self._custom_white_list = custom_white_list
        self._custom_black_list = custom_black_list
        self.white_list = copy.copy(white_list)
        self.black_list = copy.copy(black_list)
        self.gray_list = copy.copy(gray_list)
        self.black_varnames = copy.copy(custom_black_varnames)
        self._update_list()

    def _update_list(self):
        """
        Update black and white list according to users' custom list.
        """
        if self._custom_white_list and self._custom_black_list:
            for op_name in self._custom_white_list:
                if op_name in self._custom_black_list:
                    raise ValueError("Custom white list overlap "
                                     "custom black list")
        if self._custom_white_list:
            for op_name in self._custom_white_list:
                if op_name in self.black_list:
                    self.black_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.white_list.add(op_name)
        if self._custom_black_list:
            for op_name in self._custom_black_list:
                if op_name in self.white_list:
                    self.white_list.remove(op_name)
                elif op_name in self.gray_list:
                    self.gray_list.remove(op_name)
                self.black_list.add(op_name)


# The three sets listed below are changed dynamiclly. They don't contain all  
# paddle ops currently.

# The set of ops that support fp16 calculation and are considered numerically-
# safe and performance-critical. These ops are always converted to fp16.
white_list = {
    'conv2d',
    'matmul',
    'matmul_v2',
    'mul',
}

# The set of ops that support fp16 calculation and are considered numerically-
# dangerous and whose effects may also be observed in downstream ops.
black_list = {
    'exp',
    'square',
    'log',
    'mean',
    'sum',
    'reduce_sum'
    'cos_sim',
    'softmax',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'cross_entropy',
    'cross_entropy2',
}

# This set contains two types of ops. All ops supported fp16 calculation. One 
# of two types is considered numerically-safe, but may be made unsafe by an
# upstream blacklist op. Another type do not have numerically-significant
# effects, like stack, flatten2.
gray_list = {
    'elementwise_add',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_div',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'batch_norm',
    'layer_norm',
    'tanh',
    'sigmoid',
    'lookup_table',
    'lookup_table_v2',
    'top_k',
    'pool2d',
    'pool3d',
    'dropout',
    'relu',
    'relu6',
    'leaky_relu',
    'soft_relu',
    'flatten2',
    'stack',
    'unstack',
    'uniform_random',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'gaussian_random_batch_size_like',
    'slice',
    'rank',
    'scale',
    'transpose2',
    'reshape2',
    'gather',
    'fill_constant',
    'get_tensor_from_selected_rows',
    'sign',
    'cast',
    'fused_bn_add_activation',
}
'''
# The set of ops that don't support fp16 calculation
unsupported_fp16_list = {
	# from python/paddle/fluid/layers/io.py
    'send',
    'send_barrier',
    'recv',
    'fetch_barrier',
    'create_py_reader',
    'create_double_buffer_reader',
    'read',
    'load',
    
   	# from python/paddle/fluid/control_flow.py
    'increment',
    'less_than',
    'less_equal',
    'greater_than',
    'greater_equal',
    'equal',
    'not_equal',
    'read_from_array',
    'shrink_rnn_memory',
    'lod_array_length',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
    'print',
    'conditional_block',
    'while',
    'ifelse',
    'is_empty',

    'lstm',
    'cudnn_lstm',
    'lstmp',
    'gru',
    'gru_unit',
    'linear_chain_crf',
    'crf_decoding',
    'bpr_loss',
    'chunk_eval',
    'sequence_conv',
    'sequence_softmax',
    # Depthwise conv2d isn't fast and safe currently.
    # ref: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h#L79
    'depthwise_conv2d',
    # Tensor Core kernels are not available for 3D convolutions currently.
    'conv3d',
    'sequence_pool',
    'sequence_concat',
    'sequence_slice',
    'data_norm',
    'group_norm',
    'spectral_norm',
    'depthwise_conv2d_transpose',
    'sequence_expand',
    'conv_transposed2d',
    'conv_transposed3d',
    'sequence_expand_as',
    'sequence_pad',
    'sequence_unpad',
    'sequence_erase',
    'beam_search',
    'beam_search_decode',
    'lstm_unit',
    'reduce_sum',
    'reduce_mean',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'reduce_all',
    'reduce_any',
    'split',
    'edit_distance',
    'ctc_align',
    'warpctc',
    'sequence_reshape',
    'nce',
    'hierarchical_sigmoid',
    'im2sequence',
    'row_conv',
    'multiplex',
    'sample_logits',
    'one_hot',
    'smooth_l1_loss',
    'squeeze2',
    'unsqueeze2',
    'lod_reset',
    'lrn',
    'pad',
    'pad_constant_like',
    'label_smooth',
    'scatter',
    'sequence_scatter',
    'random_crop',
    'mean_iou',
    'selu',
    'crop',
    'affine_grid',
    'rank_loss',
    'margin_rank_loss',
    'pad2d',
    'elu',
    'pow',
    'stanh',
    'hard_sigmoid',
    'swish',
    'prelu',
    'brelu',
    'sequence_enumerate',
    'sequence_mask',
    'expand',
    'sampling_id',
    'maxout',
    'space_to_depth',
    'sequence_reverse',
    'similarity_focus',
    'hash',
    'grid_sampler',
    'log_loss',
    'teacher_student_sigmoid_loss',
    'add_position_encoding',
    'bilinear_tensor_product',
    'shuffle_channel',
    'temporal_shift',
    'psroi_pool',
    'huber_loss',
    'kldiv_loss',
    'tree_conv',
    'pixel_shuffle',
    'fsp',
    'cvm',

    'affine_channel',
    'roi_pool',
    'roi_align',
    'anchor_generator',
    'generate_proposals',
    'generate_proposal_labels',
    'generate_mask_labels',
		
}
'''
