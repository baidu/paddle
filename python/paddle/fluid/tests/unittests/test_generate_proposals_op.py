#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import math
import paddle
import paddle.fluid as fluid
from op_test import OpTest
from test_multiclass_nms_op import nms
from test_anchor_generator_op import anchor_generator_in_python
import copy


def generate_proposals_in_python(scores, bbox_deltas, im_info, anchors,
                                 variances, pre_nms_topN, post_nms_topN,
                                 nms_thresh, min_size, eta):
    all_anchors = anchors.reshape(-1, 4)
    rois = np.empty((0, 5), dtype=np.float32)
    roi_probs = np.empty((0, 1), dtype=np.float32)

    rpn_rois = []
    rpn_roi_probs = []
    rois_num = []
    num_images = scores.shape[0]
    for img_idx in range(num_images):
        img_i_boxes, img_i_probs = proposal_for_one_image(
            im_info[img_idx, :], all_anchors, variances,
            bbox_deltas[img_idx, :, :, :], scores[img_idx, :, :, :],
            pre_nms_topN, post_nms_topN, nms_thresh, min_size, eta)
        rois_num.append(img_i_probs.shape[0])
        rpn_rois.append(img_i_boxes)
        rpn_roi_probs.append(img_i_probs)

    return rpn_rois, rpn_roi_probs, rois_num


def proposal_for_one_image(im_info, all_anchors, variances, bbox_deltas, scores,
                           pre_nms_topN, post_nms_topN, nms_thresh, min_size,
                           eta):
    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #   - bbox deltas will be (4 * A, H, W) format from conv output
    #   - transpose to (H, W, 4 * A)
    #   - reshape to (H * W * A, 4) where rows are ordered by (H, W, A)
    #     in slowest to fastest order to match the enumerated anchors
    bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape(-1, 4)
    all_anchors = all_anchors.reshape(-1, 4)
    variances = variances.reshape(-1, 4)
    # Same story for the scores:
    #   - scores are (A, H, W) format from conv output
    #   - transpose to (H, W, A)
    #   - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
    #     to match the order of anchors and bbox_deltas
    scores = scores.transpose((1, 2, 0)).reshape(-1, 1)

    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN (e.g. 6000)
    if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
        order = np.argsort(-scores.squeeze())
    else:
        # Avoid sorting possibly large arrays;
        # First partition to get top K unsorted
        # and then sort just those
        inds = np.argpartition(-scores.squeeze(), pre_nms_topN)[:pre_nms_topN]
        order = np.argsort(-scores[inds].squeeze())
        order = inds[order]
    scores = scores[order, :]
    bbox_deltas = bbox_deltas[order, :]
    all_anchors = all_anchors[order, :]
    proposals = box_coder(all_anchors, bbox_deltas, variances)
    # clip proposals to image (may result in proposals with zero area
    # that will be removed in the next step)
    proposals = clip_tiled_boxes(proposals, im_info[:2])
    # remove predicted boxes with height or width < min_size
    keep = filter_boxes(proposals, min_size, im_info)
    if len(keep) == 0:
        proposals = np.zeros((1, 4)).astype('float32')
        scores = np.zeros((1, 1)).astype('float32')
        return proposals, scores
    proposals = proposals[keep, :]
    scores = scores[keep, :]

    # apply loose nms (e.g. threshold = 0.7)
    # take post_nms_topN (e.g. 1000)
    # return the top proposals
    if nms_thresh > 0:
        keep = nms(boxes=proposals,
                   scores=scores,
                   nms_threshold=nms_thresh,
                   eta=eta)
        if post_nms_topN > 0 and post_nms_topN < len(keep):
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep, :]

    return proposals, scores


def box_coder(all_anchors, bbox_deltas, variances):
    """
    Decode proposals by anchors and bbox_deltas from RPN 
    """
    #proposals: xmin, ymin, xmax, ymax
    proposals = np.zeros_like(bbox_deltas, dtype=np.float32)

    #anchor_loc: width, height, center_x, center_y
    anchor_loc = np.zeros_like(bbox_deltas, dtype=np.float32)

    anchor_loc[:, 0] = all_anchors[:, 2] - all_anchors[:, 0] + 1
    anchor_loc[:, 1] = all_anchors[:, 3] - all_anchors[:, 1] + 1
    anchor_loc[:, 2] = all_anchors[:, 0] + 0.5 * anchor_loc[:, 0]
    anchor_loc[:, 3] = all_anchors[:, 1] + 0.5 * anchor_loc[:, 1]

    #predicted bbox: bbox_center_x, bbox_center_y, bbox_width, bbox_height 
    pred_bbox = np.zeros_like(bbox_deltas, dtype=np.float32)
    if variances is not None:
        for i in range(bbox_deltas.shape[0]):
            pred_bbox[i, 0] = variances[i, 0] * bbox_deltas[i, 0] * anchor_loc[
                i, 0] + anchor_loc[i, 2]
            pred_bbox[i, 1] = variances[i, 1] * bbox_deltas[i, 1] * anchor_loc[
                i, 1] + anchor_loc[i, 3]
            pred_bbox[i, 2] = math.exp(
                min(variances[i, 2] * bbox_deltas[i, 2], math.log(
                    1000 / 16.0))) * anchor_loc[i, 0]
            pred_bbox[i, 3] = math.exp(
                min(variances[i, 3] * bbox_deltas[i, 3], math.log(
                    1000 / 16.0))) * anchor_loc[i, 1]
    else:
        for i in range(bbox_deltas.shape[0]):
            pred_bbox[i, 0] = bbox_deltas[i, 0] * anchor_loc[i, 0] + anchor_loc[
                i, 2]
            pred_bbox[i, 1] = bbox_deltas[i, 1] * anchor_loc[i, 1] + anchor_loc[
                i, 3]
            pred_bbox[i, 2] = math.exp(
                min(bbox_deltas[i, 2], math.log(1000 / 16.0))) * anchor_loc[i,
                                                                            0]
            pred_bbox[i, 3] = math.exp(
                min(bbox_deltas[i, 3], math.log(1000 / 16.0))) * anchor_loc[i,
                                                                            1]

    proposals[:, 0] = pred_bbox[:, 0] - pred_bbox[:, 2] / 2
    proposals[:, 1] = pred_bbox[:, 1] - pred_bbox[:, 3] / 2
    proposals[:, 2] = pred_bbox[:, 0] + pred_bbox[:, 2] / 2 - 1
    proposals[:, 3] = pred_bbox[:, 1] + pred_bbox[:, 3] / 2 - 1

    return proposals


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def filter_boxes(boxes, min_size, im_info):
    """Only keep boxes with both sides >= min_size and center within the image.
    """
    # Scale min_size to match image scale
    im_scale = im_info[2]
    min_size = max(min_size, 1.0)
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    ws_orig_scale = (boxes[:, 2] - boxes[:, 0]) / im_scale + 1
    hs_orig_scale = (boxes[:, 3] - boxes[:, 1]) / im_scale + 1
    x_ctr = boxes[:, 0] + ws / 2.
    y_ctr = boxes[:, 1] + hs / 2.
    keep = np.where((ws_orig_scale >= min_size) & (hs_orig_scale >= min_size) &
                    (x_ctr < im_info[1]) & (y_ctr < im_info[0]))[0]
    return keep


def iou(box_a, box_b):
    """
	Apply intersection-over-union overlap between box_a and box_b
    """
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])

    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])

    area_a = (ymax_a - ymin_a + 1) * (xmax_a - xmin_a + 1)
    area_b = (ymax_b - ymin_b + 1) * (xmax_b - xmin_b + 1)
    if area_a <= 0 and area_b <= 0:
        return 0.0

    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)

    inter_area = max(xb - xa + 1, 0.0) * max(yb - ya + 1, 0.0)

    iou_ratio = inter_area / (area_a + area_b - inter_area)

    return iou_ratio


def nms(boxes, scores, nms_threshold, eta=1.0):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        nms_threshold: (float) The overlap thresh for suppressing unnecessary
            boxes.
        eta: (float) The parameter for adaptive NMS.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    all_scores = copy.deepcopy(scores)
    all_scores = all_scores.flatten()

    sorted_indices = np.argsort(-all_scores, axis=0, kind='mergesort')
    sorted_scores = all_scores[sorted_indices]
    selected_indices = []
    adaptive_threshold = nms_threshold
    for i in range(sorted_scores.shape[0]):
        idx = sorted_indices[i]
        keep = True
        for k in range(len(selected_indices)):
            if keep:
                kept_idx = selected_indices[k]
                overlap = iou(boxes[idx], boxes[kept_idx])
                keep = True if overlap <= adaptive_threshold else False
            else:
                break
        if keep:
            selected_indices.append(idx)
        if keep and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold *= eta
    return selected_indices


class TestGenerateProposalsOp(OpTest):
    def set_data(self):
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {
            'Scores': self.scores,
            'BboxDeltas': self.bbox_deltas,
            'ImInfo': self.im_info.astype(np.float32),
            'Anchors': self.anchors,
            'Variances': self.variances
        }

        self.attrs = {
            'pre_nms_topN': self.pre_nms_topN,
            'post_nms_topN': self.post_nms_topN,
            'nms_thresh': self.nms_thresh,
            'min_size': self.min_size,
            'eta': self.eta
        }

        self.outputs = {
            'RpnRois': (self.rpn_rois[0], [self.rois_num]),
            'RpnRoiProbs': (self.rpn_roi_probs[0], [self.rois_num]),
        }

    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "generate_proposals"
        self.set_data()

    def init_test_params(self):
        self.pre_nms_topN = 12000  # train 12000, test 2000
        self.post_nms_topN = 5000  # train 6000, test 1000
        self.nms_thresh = 0.7
        self.min_size = 3.0
        self.eta = 1.

    def init_test_input(self):
        batch_size = 1
        input_channels = 20
        layer_h = 16
        layer_w = 16
        input_feat = np.random.random(
            (batch_size, input_channels, layer_h, layer_w)).astype('float32')
        self.anchors, self.variances = anchor_generator_in_python(
            input_feat=input_feat,
            anchor_sizes=[16., 32.],
            aspect_ratios=[0.5, 1.0],
            variances=[1.0, 1.0, 1.0, 1.0],
            stride=[16.0, 16.0],
            offset=0.5)
        self.im_info = np.array([[64., 64., 8.]])  #im_height, im_width, scale
        num_anchors = self.anchors.shape[2]
        self.scores = np.random.random(
            (batch_size, num_anchors, layer_h, layer_w)).astype('float32')
        self.bbox_deltas = np.random.random(
            (batch_size, num_anchors * 4, layer_h, layer_w)).astype('float32')

    def init_test_output(self):
        self.rpn_rois, self.rpn_roi_probs, self.rois_num = generate_proposals_in_python(
            self.scores, self.bbox_deltas, self.im_info, self.anchors,
            self.variances, self.pre_nms_topN, self.post_nms_topN,
            self.nms_thresh, self.min_size, self.eta)


class TestGenerateProposalsOutLodOp(TestGenerateProposalsOp):
    def set_data(self):
        self.init_test_params()
        self.init_test_input()
        self.init_test_output()
        self.inputs = {
            'Scores': self.scores,
            'BboxDeltas': self.bbox_deltas,
            'ImInfo': self.im_info.astype(np.float32),
            'Anchors': self.anchors,
            'Variances': self.variances
        }

        self.attrs = {
            'pre_nms_topN': self.pre_nms_topN,
            'post_nms_topN': self.post_nms_topN,
            'nms_thresh': self.nms_thresh,
            'min_size': self.min_size,
            'eta': self.eta,
            'return_rois_num': True
        }

        self.outputs = {
            'RpnRois': (self.rpn_rois[0], [self.rois_num]),
            'RpnRoiProbs': (self.rpn_roi_probs[0], [self.rois_num]),
            'RpnRoisNum': (np.asarray(
                self.rois_num, dtype=np.int32))
        }


class TestGenerateProposalsOpNoBoxLeft(TestGenerateProposalsOp):
    def init_test_params(self):
        self.pre_nms_topN = 12000  # train 12000, test 2000
        self.post_nms_topN = 5000  # train 6000, test 1000
        self.nms_thresh = 0.7
        self.min_size = 1000.0
        self.eta = 1.


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
