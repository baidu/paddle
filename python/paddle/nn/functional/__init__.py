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

# TODO: import all neural network related api under this directory,
# including layers, linear, conv, rnn etc.
__all__ = []

# TODO: define alias in functional directory
from . import conv
__all__ += conv.__all__
from . import activation
__all__ += activation.__all__
from . import extension
__all__ += extension.__all__
from . import common
__all__ += common.__all__
from . import pooling
__all__ += pooling.__all__
from . import loss
__all__ += loss.__all__
from .activation import elu  #DEFINE_ALIAS
# from .activation import erf  #DEFINE_ALIAS
from .activation import gelu  #DEFINE_ALIAS
from .activation import hardshrink  #DEFINE_ALIAS
from .activation import hardtanh  #DEFINE_ALIAS
from .activation import hardsigmoid  #DEFINE_ALIAS
from .activation import hardswish  #DEFINE_ALIAS
from .activation import leaky_relu  #DEFINE_ALIAS
from .activation import log_sigmoid  #DEFINE_ALIAS
from .activation import maxout  #DEFINE_ALIAS
from .activation import prelu  #DEFINE_ALIAS
from .activation import relu  #DEFINE_ALIAS
from .activation import relu6  #DEFINE_ALIAS
from .activation import selu  #DEFINE_ALIAS
from .activation import sigmoid  #DEFINE_ALIAS
# from .activation import soft_relu  #DEFINE_ALIAS
from .activation import softmax  #DEFINE_ALIAS
from .activation import softplus  #DEFINE_ALIAS
from .activation import softshrink  #DEFINE_ALIAS
from .activation import softsign  #DEFINE_ALIAS
from .activation import swish  #DEFINE_ALIAS
from .activation import tanh  #DEFINE_ALIAS
from .activation import tanhshrink  #DEFINE_ALIAS
from .activation import thresholded_relu  #DEFINE_ALIAS
from .activation import log_softmax  #DEFINE_ALIAS
from .common import dropout  #DEFINE_ALIAS
from .common import dropout2d  #DEFINE_ALIAS
from .common import dropout3d  #DEFINE_ALIAS
from .common import alpha_dropout  #DEFINE_ALIAS
# from .common import embedding        #DEFINE_ALIAS
# from .common import fc  #DEFINE_ALIAS
from .common import label_smooth
# from .common import one_hot  #DEFINE_ALIAS
from .common import pad  #DEFINE_ALIAS
# from .common import pad_constant_like  #DEFINE_ALIAS
# from .common import pad2d  #DEFINE_ALIAS
from .common import cosine_similarity  #DEFINE_ALIAS
from .common import unfold  #DEFINE_ALIAS
# from .common import bilinear_tensor_product        #DEFINE_ALIAS
from .common import assign  #DEFINE_ALIAS
from .common import interpolate  #DEFINE_ALIAS
from .common import upsample  #DEFINE_ALIAS
from .common import bilinear  #DEFINE_ALIAS
from .conv import conv1d  #DEFINE_ALIAS
from .conv import conv_transpose1d  #DEFINE_ALIAS
from .common import linear  #DEFINE_ALIAS
from .conv import conv2d  #DEFINE_ALIAS
from .conv import conv_transpose2d  #DEFINE_ALIAS
from .conv import conv3d  #DEFINE_ALIAS
from .conv import conv_transpose3d  #DEFINE_ALIAS
# from .extension import add_position_encoding  #DEFINE_ALIAS
# from .extension import autoincreased_step_counter        #DEFINE_ALIAS
# from .extension import continuous_value_model  #DEFINE_ALIAS
# from .extension import filter_by_instag  #DEFINE_ALIAS
# from .extension import linear_chain_crf        #DEFINE_ALIAS
# from .extension import merge_selected_rows        #DEFINE_ALIAS
# from .extension import multiclass_nms  #DEFINE_ALIAS
# from .extension import polygon_box_transform  #DEFINE_ALIAS
# from .extension import random_crop  #DEFINE_ALIAS
# from .extension import row_conv  #DEFINE_ALIAS
# from .extension import rpn_target_assign  #DEFINE_ALIAS
# from .extension import similarity_focus  #DEFINE_ALIAS
# from .extension import target_assign  #DEFINE_ALIAS
# from .extension import temporal_shift  #DEFINE_ALIAS
# from .extension import warpctc  #DEFINE_ALIAS
from .extension import diag_embed  #DEFINE_ALIAS
# from .lod import sequence_concat        #DEFINE_ALIAS
# from .lod import sequence_conv        #DEFINE_ALIAS
# from .lod import sequence_enumerate        #DEFINE_ALIAS
# from .lod import sequence_expand_as        #DEFINE_ALIAS
# from .lod import sequence_expand        #DEFINE_ALIAS
# from .lod import sequence_first_step        #DEFINE_ALIAS
# from .lod import sequence_last_step        #DEFINE_ALIAS
# from .lod import sequence_mask        #DEFINE_ALIAS
# from .lod import sequence_pad        #DEFINE_ALIAS
# from .lod import sequence_pool        #DEFINE_ALIAS
# from .lod import sequence_reshape        #DEFINE_ALIAS
# from .lod import sequence_reverse        #DEFINE_ALIAS
# from .lod import sequence_scatter        #DEFINE_ALIAS
# from .lod import sequence_slice        #DEFINE_ALIAS
# from .lod import sequence_softmax        #DEFINE_ALIAS
# from .lod import sequence_unpad        #DEFINE_ALIAS
# from .lod import array_length        #DEFINE_ALIAS
# from .lod import array_read        #DEFINE_ALIAS
# from .lod import array_write        #DEFINE_ALIAS
# from .lod import create_array        #DEFINE_ALIAS
# from .lod import hash  #DEFINE_ALIAS
# from .lod import im2sequence        #DEFINE_ALIAS
# from .lod import lod_append        #DEFINE_ALIAS
# from .lod import lod_reset        #DEFINE_ALIAS
# from .lod import reorder_lod_tensor_by_rank        #DEFINE_ALIAS
# from .lod import tensor_array_to_tensor        #DEFINE_ALIAS
# from .lod import dynamic_gru        #DEFINE_ALIAS
# from .lod import dynamic_lstm        #DEFINE_ALIAS
# from .lod import dynamic_lstmp        #DEFINE_ALIAS
from .loss import binary_cross_entropy  #DEFINE_ALIAS
from .loss import binary_cross_entropy_with_logits  #DEFINE_ALIAS
# from .loss import bpr_loss  #DEFINE_ALIAS
# from .loss import center_loss  #DEFINE_ALIAS
from .loss import cross_entropy  #DEFINE_ALIAS
from .loss import dice_loss  #DEFINE_ALIAS
from .loss import hsigmoid_loss  #DEFINE_ALIAS
from .loss import iou_similarity  #DEFINE_ALIAS
from .loss import kl_div  #DEFINE_ALIAS
from .loss import l1_loss  #DEFINE_ALIAS
from .loss import log_loss  #DEFINE_ALIAS
from .loss import margin_ranking_loss  #DEFINE_ALIAS
from .loss import mse_loss  #DEFINE_ALIAS
from .loss import nll_loss  #DEFINE_ALIAS
# from .loss import nce        #DEFINE_ALIAS
from .loss import npair_loss  #DEFINE_ALIAS
from .loss import sigmoid_focal_loss  #DEFINE_ALIAS
# from .loss import smooth_l1  #DEFINE_ALIAS
from .loss import smooth_l1_loss  #DEFINE_ALIAS
from .loss import softmax_with_cross_entropy  #DEFINE_ALIAS
from .loss import square_error_cost  #DEFINE_ALIAS
from .loss import ssd_loss  #DEFINE_ALIAS
# from .loss import teacher_student_sigmoid_loss  #DEFINE_ALIAS
from .loss import ctc_loss  #DEFINE_ALIAS
# from .norm import data_norm        #DEFINE_ALIAS
# from .norm import group_norm        #DEFINE_ALIAS
from .norm import batch_norm  #DEFINE_ALIAS
from .norm import instance_norm  #DEFINE_ALIAS
from .norm import layer_norm  #DEFINE_ALIAS
from .norm import local_response_norm  #DEFINE_ALIAS
from .norm import normalize  #DEFINE_ALIAS
# from .norm import spectral_norm        #DEFINE_ALIAS
# from .pooling import pool2d  #DEFINE_ALIAS
# from .pooling import pool3d  #DEFINE_ALIAS
from .pooling import avg_pool1d  #DEFINE_ALIAS
from .pooling import avg_pool2d  #DEFINE_ALIAS
from .pooling import avg_pool3d  #DEFINE_ALIAS
from .pooling import max_pool1d  #DEFINE_ALIAS
from .pooling import max_pool2d  #DEFINE_ALIAS
from .pooling import max_pool3d  #DEFINE_ALIAS

from .pooling import adaptive_max_pool1d  #DEFINE_ALIAS
from .pooling import adaptive_max_pool2d  #DEFINE_ALIAS
from .pooling import adaptive_max_pool3d  #DEFINE_ALIAS
from .pooling import adaptive_avg_pool1d  #DEFINE_ALIAS
from .pooling import adaptive_avg_pool2d  #DEFINE_ALIAS
from .pooling import adaptive_avg_pool3d  #DEFINE_ALIAS

# from .rnn import rnn  #DEFINE_ALIAS
# from .rnn import birnn  #DEFINE_ALIAS
# from .rnn import gru_unit        #DEFINE_ALIAS
# from .rnn import lstm        #DEFINE_ALIAS
# from .rnn import lstm_unit        #DEFINE_ALIAS
# from .vision import affine_channel  #DEFINE_ALIAS
from .vision import affine_grid  #DEFINE_ALIAS
# from .vision import anchor_generator  #DEFINE_ALIAS
# from .vision import bipartite_match  #DEFINE_ALIAS
# from .vision import box_clip  #DEFINE_ALIAS
# from .vision import box_coder  #DEFINE_ALIAS
# from .vision import box_decoder_and_assign  #DEFINE_ALIAS
# from .vision import collect_fpn_proposals  #DEFINE_ALIAS
# from .vision import deformable_conv  #DEFINE_ALIAS
# from .vision import deformable_roi_pooling  #DEFINE_ALIAS
# from .vision import density_prior_box  #DEFINE_ALIAS
# from .vision import detection_output  #DEFINE_ALIAS
# from .vision import distribute_fpn_proposals  #DEFINE_ALIAS
# from .vision import fsp_matrix  #DEFINE_ALIAS
# from .vision import generate_mask_labels  #DEFINE_ALIAS
# from .vision import generate_proposal_labels  #DEFINE_ALIAS
# from .vision import generate_proposals  #DEFINE_ALIAS
from .vision import grid_sample  #DEFINE_ALIAS
# from .vision import image_resize  #DEFINE_ALIAS
# from .vision import image_resize_short  #DEFINE_ALIAS
# from .vision import multi_box_head  #DEFINE_ALIAS
from .vision import pixel_shuffle  #DEFINE_ALIAS
# from .vision import prior_box  #DEFINE_ALIAS
# from .vision import prroi_pool  #DEFINE_ALIAS
# from .vision import psroi_pool  #DEFINE_ALIAS
# from .vision import resize_bilinear  #DEFINE_ALIAS
# from .vision import resize_nearest  #DEFINE_ALIAS
# from .vision import resize_trilinear  #DEFINE_ALIAS
# from .vision import retinanet_detection_output  #DEFINE_ALIAS
# from .vision import retinanet_target_assign  #DEFINE_ALIAS
# from .vision import roi_align  #DEFINE_ALIAS
# from .vision import roi_perspective_transform  #DEFINE_ALIAS
# from .vision import roi_pool  #DEFINE_ALIAS
# from .vision import shuffle_channel  #DEFINE_ALIAS
# from .vision import space_to_depth  #DEFINE_ALIAS
# from .vision import yolo_box  #DEFINE_ALIAS
# from .vision import yolov3_loss  #DEFINE_ALIAS
from .input import one_hot  #DEFINE_ALIAS
from .input import embedding  #DEFINE_ALIAS
