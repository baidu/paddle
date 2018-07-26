#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
All layers just related to the detection neural network.
"""

from .layer_function_generator import generate_layer_fn
from .layer_function_generator import autodoc, templatedoc
from ..layer_helper import LayerHelper
from . import tensor
from . import nn
import math
from functools import reduce

__all__ = [
    'prior_box',
    'multi_box_head',
    'bipartite_match',
    'target_assign',
    'detection_output',
    'ssd_loss',
    'detection_map',
    'rpn_target_assign',
    'anchor_generator',
]

__auto__ = [
    'iou_similarity',
    'box_coder',
]

__all__ += __auto__

for _OP in set(__auto__):
    globals()[_OP] = generate_layer_fn(_OP)


def rpn_target_assign(loc,
                      scores,
                      anchor_box,
                      gt_box,
                      rpn_batch_size_per_im=256,
                      fg_fraction=0.25,
                      rpn_positive_overlap=0.7,
                      rpn_negative_overlap=0.3):
    """
    ** Target Assign Layer for region proposal network (RPN) in Faster-RCNN detection. **

    This layer can be, for given the  Intersection-over-Union (IoU) overlap
    between anchors and ground truth boxes, to assign classification and
    regression targets to each each anchor, these target labels are used for
    train RPN. The classification targets is a binary class label (of being
    an object or not). Following the paper of Faster-RCNN, the positive labels
    are two kinds of anchors: (i) the anchor/anchors with the highest IoU
    overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap
    higher than rpn_positive_overlap(0.7) with any ground-truth box. Note
    that a single ground-truth box may assign positive labels to multiple
    anchors. A non-positive anchor is when its IoU ratio is lower than
    rpn_negative_overlap (0.3) for all ground-truth boxes. Anchors that are
    neither positive nor negative do not contribute to the training objective.
    The regression targets are the encoded ground-truth boxes associated with
    the positive anchors.

    Args:
        loc(Variable): A 3-D Tensor with shape [N, M, 4] represents the
            predicted locations of M bounding bboxes. N is the batch size,
            and each bounding box has four coordinate values and the layout
            is [xmin, ymin, xmax, ymax].
        scores(Variable): A 3-D Tensor with shape [N, M, C] represents the
            predicted confidence predictions. N is the batch size, C is the
            class number, M is number of bounding boxes. For each category
            there are total M scores which corresponding M bounding boxes.
        anchor_box(Variable): A 2-D Tensor with shape [M, 4] holds M boxes,
            each box is represented as [xmin, ymin, xmax, ymax],
            [xmin, ymin] is the left top coordinate of the anchor box,
            if the input is image feature map, they are close to the origin
            of the coordinate system. [xmax, ymax] is the right bottom
            coordinate of the anchor box.
        gt_box (Variable): The ground-truth boudding boxes (bboxes) are a 2D
            LoDTensor with shape [Ng, 4], Ng is the total number of ground-truth
            bboxes of mini-batch input.
        rpn_batch_size_per_im(int): Total number of RPN examples per image.
        fg_fraction(float): Target fraction of RoI minibatch that is labeled
            foreground (i.e. class > 0), 0-th class is background.
        rpn_positive_overlap(float): Minimum overlap required between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a positive
            example.
        rpn_negative_overlap(float): Maximum overlap allowed between an anchor
            and ground-truth box for the (anchor, gt box) pair to be a negative
            examples.

    Returns:
        tuple: 
               A tuple(predicted_scores, predicted_location, target_label,
               target_bbox) is returned. The predicted_scores and
               predicted_location is the predicted result of the RPN.
               The target_label and target_bbox is the ground truth,
               respectively. The predicted_location is a 2D Tensor with shape
               [F, 4], and the shape of target_bbox is same as the shape of
               the predicted_location, F is the number of the foreground
               anchors. The predicted_scores is a 2D Tensor with shape
               [F + B, 1], and the shape of target_label is same as the shape
               of the predicted_scores, B is the number of the background
               anchors, the F and B is depends on the input of this operator. 

    Examples:
        .. code-block:: python

        loc = layers.data(name='location', shape=[2, 80],
                          append_batch_size=False, dtype='float32')
        scores = layers.data(name='scores', shape=[2, 40],
                          append_batch_size=False, dtype='float32')
        anchor_box = layers.data(name='anchor_box', shape=[20, 4],
                          append_batch_size=False, dtype='float32')
        gt_box = layers.data(name='gt_box', shape=[10, 4],
                         append_batch_size=False, dtype='float32')
        loc_pred, score_pred, loc_target, score_target =
            fluid.layers.detection_output(loc=location,
                                          scores=scores,
                                          anchor_box=anchor_box,
                                          gt_box=gt_box)
    """

    helper = LayerHelper('rpn_target_assign', **locals())
    # 1. Compute the regression target bboxes
    target_bbox = box_coder(
        prior_box=anchor_box,
        target_box=gt_box,
        code_type='encode_center_size',
        box_normalized=False)

    # 2. Compute overlaps between the prior boxes and the gt boxes overlaps
    iou = iou_similarity(x=gt_box, y=anchor_box)

    # 3. Assign target label to anchors
    loc_index = helper.create_tmp_variable(dtype=anchor_box.dtype)
    score_index = helper.create_tmp_variable(dtype=anchor_box.dtype)
    target_label = helper.create_tmp_variable(dtype=anchor_box.dtype)
    helper.append_op(
        type="rpn_target_assign",
        inputs={'Overlap': iou, },
        outputs={
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': target_label,
        },
        attrs={
            'rpn_batch_size_per_im': rpn_batch_size_per_im,
            'rpn_positive_overlap': rpn_positive_overlap,
            'rpn_negative_overlap': rpn_negative_overlap,
            'fg_fraction': fg_fraction,
        })

    # 4. Reshape and gather the target entry
    scores = nn.reshape(x=scores, shape=(-1, 1))
    loc = nn.reshape(x=loc, shape=(-1, 4))
    target_label = nn.reshape(x=target_label, shape=(-1, 1))
    target_bbox = nn.reshape(x=target_bbox, shape=(-1, 4))

    predicted_scores = nn.gather(scores, score_index)
    predicted_location = nn.gather(loc, loc_index)
    target_label = nn.gather(target_label, score_index)
    target_bbox = nn.gather(target_bbox, loc_index)
    return predicted_scores, predicted_loc, target_label, target_bbox


def detection_output(loc,
                     scores,
                     prior_box,
                     prior_box_var,
                     background_label=0,
                     nms_threshold=0.3,
                     nms_top_k=400,
                     keep_top_k=200,
                     score_threshold=0.01,
                     nms_eta=1.0):
    """
    **Detection Output Layer for Single Shot Multibox Detector (SSD).**

    This operation is to get the detection results by performing following
    two steps:

    1. Decode input bounding box predictions according to the prior boxes.
    2. Get the final detection results by applying multi-class non maximum
       suppression (NMS).

    Please note, this operation doesn't clip the final output bounding boxes
    to the image window.

    Args:
        loc(Variable): A 3-D Tensor with shape [N, M, 4] represents the
            predicted locations of M bounding bboxes. N is the batch size,
            and each bounding box has four coordinate values and the layout
            is [xmin, ymin, xmax, ymax].
        scores(Variable): A 3-D Tensor with shape [N, M, C] represents the
            predicted confidence predictions. N is the batch size, C is the
            class number, M is number of bounding boxes. For each category
            there are total M scores which corresponding M bounding boxes.
        prior_box(Variable): A 2-D Tensor with shape [M, 4] holds M boxes,
            each box is represented as [xmin, ymin, xmax, ymax],
            [xmin, ymin] is the left top coordinate of the anchor box,
            if the input is image feature map, they are close to the origin
            of the coordinate system. [xmax, ymax] is the right bottom
            coordinate of the anchor box.
        prior_box_var(Variable): A 2-D Tensor with shape [M, 4] holds M group
            of variance.
        background_label(float): The index of background label,
            the background label will be ignored. If set to -1, then all
            categories will be considered.
        nms_threshold(float): The threshold to be used in NMS.
        nms_top_k(int): Maximum number of detections to be kept according
            to the confidences aftern the filtering detections based on
            score_threshold.
        keep_top_k(int): Number of total bboxes to be kept per image after
            NMS step. -1 means keeping all bboxes after NMS step.
        score_threshold(float): Threshold to filter out bounding boxes with
            low confidence score. If not provided, consider all boxes.
        nms_eta(float): The parameter for adaptive NMS.

    Returns:
        Variable: 
        
            The detection outputs is a LoDTensor with shape [No, 6].
            Each row has six values: [label, confidence, xmin, ymin, xmax, ymax].
            `No` is the total number of detections in this mini-batch. For each
            instance, the offsets in first dimension are called LoD, the offset
            number is N + 1, N is the batch size. The i-th image has
            `LoD[i + 1] - LoD[i]` detected results, if it is 0, the i-th image
            has no detected results. If all images have not detected results,
            all the elements in LoD are 0, and output tensor only contains one
            value, which is -1.

    Examples:
        .. code-block:: python

            pb = layers.data(name='prior_box', shape=[10, 4],
                         append_batch_size=False, dtype='float32')
            pbv = layers.data(name='prior_box_var', shape=[10, 4],
                          append_batch_size=False, dtype='float32')
            loc = layers.data(name='target_box', shape=[2, 21, 4],
                          append_batch_size=False, dtype='float32')
            scores = layers.data(name='scores', shape=[2, 21, 10],
                          append_batch_size=False, dtype='float32')
            nmsed_outs = fluid.layers.detection_output(scores=scores,
                                       loc=loc,
                                       prior_box=pb,
                                       prior_box_var=pbv)
    """
    helper = LayerHelper("detection_output", **locals())
    decoded_box = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=loc,
        code_type='decode_center_size')
    old_shape = scores.shape
    scores = nn.reshape(x=scores, shape=(-1, old_shape[-1]))
    scores = nn.softmax(input=scores)
    scores = nn.reshape(x=scores, shape=old_shape)
    scores = nn.transpose(scores, perm=[0, 2, 1])
    scores.stop_gradient = True
    nmsed_outs = helper.create_tmp_variable(dtype=decoded_box.dtype)
    helper.append_op(
        type="multiclass_nms",
        inputs={'Scores': scores,
                'BBoxes': decoded_box},
        outputs={'Out': nmsed_outs},
        attrs={
            'background_label': 0,
            'nms_threshold': nms_threshold,
            'nms_top_k': nms_top_k,
            'keep_top_k': keep_top_k,
            'score_threshold': score_threshold,
            'nms_eta': 1.0
        })
    nmsed_outs.stop_gradient = True
    return nmsed_outs


@templatedoc()
def detection_map(detect_res,
                  label,
                  class_num,
                  background_label=0,
                  overlap_threshold=0.3,
                  evaluate_difficult=True,
                  has_state=None,
                  input_states=None,
                  out_states=None,
                  ap_version='integral'):
    """
    ${comment}

    Args:
        detect_res: ${detect_res_comment}
        label:  ${label_comment}
        class_num: ${class_num_comment}
        background_label: ${background_label_comment}
        overlap_threshold: ${overlap_threshold_comment}
        evaluate_difficult: ${evaluate_difficult_comment}
        has_state: ${has_state_comment}
        input_states: If not None, It contains 3 elements:
            1. pos_count ${pos_count_comment}.
            2. true_pos ${true_pos_comment}.
            3. false_pos ${false_pos_comment}.
        out_states: If not None, it contains 3 elements.
            1. accum_pos_count ${accum_pos_count_comment}.
            2. accum_true_pos ${accum_true_pos_comment}.
            3. accum_false_pos ${accum_false_pos_comment}.
        ap_version: ${ap_type_comment}

    Returns:
        ${map_comment}


    Examples:
          .. code-block:: python

            detect_res = fluid.layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = fluid.layers.data(
                name='label',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')

            map_out = fluid.layers.detection_map(detect_res, label, 21)
    """
    helper = LayerHelper("detection_map", **locals())

    def __create_var(type):
        return helper.create_tmp_variable(dtype=type)

    map_out = __create_var('float32')
    accum_pos_count_out = out_states[0] if out_states else __create_var('int32')
    accum_true_pos_out = out_states[1] if out_states else __create_var(
        'float32')
    accum_false_pos_out = out_states[2] if out_states else __create_var(
        'float32')

    pos_count = input_states[0] if input_states else None
    true_pos = input_states[1] if input_states else None
    false_pos = input_states[2] if input_states else None

    helper.append_op(
        type="detection_map",
        inputs={
            'Label': label,
            'DetectRes': detect_res,
            'HasState': has_state,
            'PosCount': pos_count,
            'TruePos': true_pos,
            'FalsePos': false_pos
        },
        outputs={
            'MAP': map_out,
            'AccumPosCount': accum_pos_count_out,
            'AccumTruePos': accum_true_pos_out,
            'AccumFalsePos': accum_false_pos_out
        },
        attrs={
            'overlap_threshold': overlap_threshold,
            'evaluate_difficult': evaluate_difficult,
            'ap_type': ap_version,
            'class_num': class_num,
        })
    return map_out


def bipartite_match(dist_matrix,
                    match_type=None,
                    dist_threshold=None,
                    name=None):
    """
    This operator implements a greedy bipartite matching algorithm, which is
    used to obtain the matching with the maximum distance based on the input
    distance matrix. For input 2D matrix, the bipartite matching algorithm can
    find the matched column for each row (matched means the largest distance),
    also can find the matched row for each column. And this operator only
    calculate matched indices from column to row. For each instance,
    the number of matched indices is the column number of the input distance
    matrix.

    There are two outputs, matched indices and distance.
    A simple description, this algorithm matched the best (maximum distance)
    row entity to the column entity and the matched indices are not duplicated
    in each row of ColToRowMatchIndices. If the column entity is not matched
    any row entity, set -1 in ColToRowMatchIndices.

    NOTE: the input DistMat can be LoDTensor (with LoD) or Tensor.
    If LoDTensor with LoD, the height of ColToRowMatchIndices is batch size.
    If Tensor, the height of ColToRowMatchIndices is 1.

    NOTE: This API is a very low level API. It is used by :code:`ssd_loss`
    layer. Please consider to use :code:`ssd_loss` instead.

    Args:
        dist_matrix(Variable): This input is a 2-D LoDTensor with shape
            [K, M]. It is pair-wise distance matrix between the entities
            represented by each row and each column. For example, assumed one
            entity is A with shape [K], another entity is B with shape [M]. The
            dist_matrix[i][j] is the distance between A[i] and B[j]. The bigger
            the distance is, the better matching the pairs are.

            NOTE: This tensor can contain LoD information to represent a batch
            of inputs. One instance of this batch can contain different numbers
            of entities.
        match_type(string|None): The type of matching method, should be
           'bipartite' or 'per_prediction'. [default 'bipartite'].
        dist_threshold(float|None): If `match_type` is 'per_prediction',
            this threshold is to determine the extra matching bboxes based
            on the maximum distance, 0.5 by default.
    Returns:
        tuple: a tuple with two elements is returned. The first is
        matched_indices, the second is matched_distance.

        The matched_indices is a 2-D Tensor with shape [N, M] in int type.
        N is the batch size. If match_indices[i][j] is -1, it
        means B[j] does not match any entity in i-th instance.
        Otherwise, it means B[j] is matched to row
        match_indices[i][j] in i-th instance. The row number of
        i-th instance is saved in match_indices[i][j].

        The matched_distance is a 2-D Tensor with shape [N, M] in float type
        . N is batch size. If match_indices[i][j] is -1,
        match_distance[i][j] is also -1.0. Otherwise, assumed
        match_distance[i][j] = d, and the row offsets of each instance
        are called LoD. Then match_distance[i][j] =
        dist_matrix[d+LoD[i]][j].

    Examples:

        >>> x = fluid.layers.data(name='x', shape=[4], dtype='float32')
        >>> y = fluid.layers.data(name='y', shape=[4], dtype='float32')
        >>> iou = fluid.layers.iou_similarity(x=x, y=y)
        >>> matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
    """
    helper = LayerHelper('bipartite_match', **locals())
    match_indices = helper.create_tmp_variable(dtype='int32')
    match_distance = helper.create_tmp_variable(dtype=dist_matrix.dtype)
    helper.append_op(
        type='bipartite_match',
        inputs={'DistMat': dist_matrix},
        attrs={
            'match_type': match_type,
            'dist_threshold': dist_threshold,
        },
        outputs={
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_distance
        })
    return match_indices, match_distance


def target_assign(input,
                  matched_indices,
                  negative_indices=None,
                  mismatch_value=None,
                  name=None):
    """
    This operator can be, for given the target bounding boxes or labels,
    to assign classification and regression targets to each prediction as well as
    weights to prediction. The weights is used to specify which prediction would
    not contribute to training loss.

    For each instance, the output `out` and`out_weight` are assigned based on
    `match_indices` and `negative_indices`.
    Assumed that the row offset for each instance in `input` is called lod,
    this operator assigns classification/regression targets by performing the
    following steps:

    1. Assigning all outpts based on `match_indices`:

    .. code-block:: text

        If id = match_indices[i][j] > 0,

            out[i][j][0 : K] = X[lod[i] + id][j % P][0 : K]
            out_weight[i][j] = 1.

        Otherwise,

            out[j][j][0 : K] = {mismatch_value, mismatch_value, ...}
            out_weight[i][j] = 0.

    2. Assigning out_weight based on `neg_indices` if `neg_indices` is provided:

    Assumed that the row offset for each instance in `neg_indices` is called neg_lod,
    for i-th instance and each `id` of neg_indices in this instance:
    
    .. code-block:: text

        out[i][id][0 : K] = {mismatch_value, mismatch_value, ...}
        out_weight[i][id] = 1.0

    Args:
       inputs (Variable): This input is a 3D LoDTensor with shape [M, P, K].
       matched_indices (Variable): Tensor<int>), The input matched indices
           is 2D Tenosr<int32> with shape [N, P], If MatchIndices[i][j] is -1,
           the j-th entity of column is not matched to any entity of row in
           i-th instance.
       negative_indices (Variable): The input negative example indices are
           an optional input with shape [Neg, 1] and int32 type, where Neg is
           the total number of negative example indices.
       mismatch_value (float32): Fill this value to the mismatched location.

    Returns:
        tuple: 
               A tuple(out, out_weight) is returned. out is a 3D Tensor with 
               shape [N, P, K], N and P is the same as they are in 
               `neg_indices`, K is the same as it in input of X. If 
               `match_indices[i][j]`. out_weight is the weight for output with 
               the shape of [N, P, 1].

    Examples:

        .. code-block:: python

            matched_indices, matched_dist = fluid.layers.bipartite_match(iou)
            gt = layers.data(
                        name='gt', shape=[1, 1], dtype='int32', lod_level=1)
            trg, trg_weight = layers.target_assign(
                            gt, matched_indices, mismatch_value=0)
    """
    helper = LayerHelper('target_assign', **locals())
    out = helper.create_tmp_variable(dtype=input.dtype)
    out_weight = helper.create_tmp_variable(dtype='float32')
    helper.append_op(
        type='target_assign',
        inputs={
            'X': input,
            'MatchIndices': matched_indices,
            'NegIndices': negative_indices
        },
        outputs={'Out': out,
                 'OutWeight': out_weight},
        attrs={'mismatch_value': mismatch_value})
    return out, out_weight


def ssd_loss(location,
             confidence,
             gt_box,
             gt_label,
             prior_box,
             prior_box_var=None,
             background_label=0,
             overlap_threshold=0.5,
             neg_pos_ratio=3.0,
             neg_overlap=0.5,
             loc_loss_weight=1.0,
             conf_loss_weight=1.0,
             match_type='per_prediction',
             mining_type='max_negative',
             normalize=True,
             sample_size=None):
    """
    **Multi-box loss layer for object detection algorithm of SSD**

    This layer is to compute dection loss for SSD given the location offset
    predictions, confidence predictions, prior boxes and ground-truth boudding
    boxes and labels, and the type of hard example mining. The returned loss
    is a weighted sum of the localization loss (or regression loss) and
    confidence loss (or classification loss) by performing the following steps:

    1. Find matched bounding box by bipartite matching algorithm.

      1.1 Compute IOU similarity between ground-truth boxes and prior boxes.

      1.2 Compute matched boundding box by bipartite matching algorithm.

    2. Compute confidence for mining hard examples

      2.1. Get the target label based on matched indices.

      2.2. Compute confidence loss.

    3. Apply hard example mining to get the negative example indices and update
       the matched indices.

    4. Assign classification and regression targets

      4.1. Encoded bbox according to the prior boxes.

      4.2. Assign regression targets.

      4.3. Assign classification targets.

    5. Compute the overall objective loss.

      5.1 Compute confidence loss.

      5.1 Compute localization loss.

      5.3 Compute the overall weighted loss.

    Args:
        location (Variable): The location predictions are a 3D Tensor with
            shape [N, Np, 4], N is the batch size, Np is total number of
            predictions for each instance. 4 is the number of coordinate values,
            the layout is [xmin, ymin, xmax, ymax].
        confidence (Variable): The confidence predictions are a 3D Tensor
            with shape [N, Np, C], N and Np are the same as they are in
            `location`, C is the class number.
        gt_box (Variable): The ground-truth boudding boxes (bboxes) are a 2D
            LoDTensor with shape [Ng, 4], Ng is the total number of ground-truth
            bboxes of mini-batch input.
        gt_label (Variable): The ground-truth labels are a 2D LoDTensor
            with shape [Ng, 1].
        prior_box (Variable): The prior boxes are a 2D Tensor with shape [Np, 4].
        prior_box_var (Variable): The variance of prior boxes are a 2D Tensor
            with shape [Np, 4].
        background_label (int): The index of background label, 0 by default.
        overlap_threshold (float): If match_type is 'per_prediction', use
            `overlap_threshold` to determine the extra matching bboxes when
             finding matched boxes. 0.5 by default.
        neg_pos_ratio (float): The ratio of the negative boxes to the positive
            boxes, used only when mining_type is 'max_negative', 3.0 by defalut.
        neg_overlap (float): The negative overlap upper bound for the unmatched
            predictions. Use only when mining_type is 'max_negative',
            0.5 by default.
        loc_loss_weight (float): Weight for localization loss, 1.0 by default.
        conf_loss_weight (float): Weight for confidence loss, 1.0 by default.
        match_type (str): The type of matching method during training, should
            be 'bipartite' or 'per_prediction', 'per_prediction' by defalut.
        mining_type (str): The hard example mining type, should be 'hard_example'
            or 'max_negative', now only support `max_negative`.
        normalize (bool): Whether to normalize the SSD loss by the total number
            of output locations, True by default.
        sample_size (int): The max sample size of negative box, used only when
            mining_type is 'hard_example'.

    Returns:
        The weighted sum of the localization loss and confidence loss, with \
        shape [N * Np, 1], N and Np are the same as they are in `location`.

    Raises:
        ValueError: If mining_type is 'hard_example', now only support mining \
        type of `max_negative`.

    Examples:
        >>> pb = fluid.layers.data(
        >>>                   name='prior_box',
        >>>                   shape=[10, 4],
        >>>                   append_batch_size=False,
        >>>                   dtype='float32')
        >>> pbv = fluid.layers.data(
        >>>                   name='prior_box_var',
        >>>                   shape=[10, 4],
        >>>                   append_batch_size=False,
        >>>                   dtype='float32')
        >>> loc = fluid.layers.data(name='target_box', shape=[10, 4], dtype='float32')
        >>> scores = fluid.layers.data(name='scores', shape=[10, 21], dtype='float32')
        >>> gt_box = fluid.layers.data(
        >>>         name='gt_box', shape=[4], lod_level=1, dtype='float32')
        >>> gt_label = fluid.layers.data(
        >>>         name='gt_label', shape=[1], lod_level=1, dtype='float32')
        >>> loss = fluid.layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
    """

    helper = LayerHelper('ssd_loss', **locals())
    if mining_type != 'max_negative':
        raise ValueError("Only support mining_type == max_negative now.")

    num, num_prior, num_class = confidence.shape

    def __reshape_to_2d(var):
        return nn.reshape(x=var, shape=[-1, var.shape[-1]])

    # 1. Find matched boundding box by prior box.
    #   1.1 Compute IOU similarity between ground-truth boxes and prior boxes.
    iou = iou_similarity(x=gt_box, y=prior_box)
    #   1.2 Compute matched boundding box by bipartite matching algorithm.
    matched_indices, matched_dist = bipartite_match(iou, match_type,
                                                    overlap_threshold)

    # 2. Compute confidence for mining hard examples
    # 2.1. Get the target label based on matched indices
    gt_label = nn.reshape(x=gt_label, shape=gt_label.shape + (1, ))
    gt_label.stop_gradient = True
    target_label, _ = target_assign(
        gt_label, matched_indices, mismatch_value=background_label)
    # 2.2. Compute confidence loss.
    # Reshape confidence to 2D tensor.
    confidence = __reshape_to_2d(confidence)
    target_label = tensor.cast(x=target_label, dtype='int64')
    target_label = __reshape_to_2d(target_label)
    target_label.stop_gradient = True
    conf_loss = nn.softmax_with_cross_entropy(confidence, target_label)

    # 3. Mining hard examples
    conf_loss = nn.reshape(x=conf_loss, shape=(num, num_prior))
    conf_loss.stop_gradient = True
    neg_indices = helper.create_tmp_variable(dtype='int32')
    dtype = matched_indices.dtype
    updated_matched_indices = helper.create_tmp_variable(dtype=dtype)
    helper.append_op(
        type='mine_hard_examples',
        inputs={
            'ClsLoss': conf_loss,
            'LocLoss': None,
            'MatchIndices': matched_indices,
            'MatchDist': matched_dist,
        },
        outputs={
            'NegIndices': neg_indices,
            'UpdatedMatchIndices': updated_matched_indices
        },
        attrs={
            'neg_pos_ratio': neg_pos_ratio,
            'neg_dist_threshold': neg_pos_ratio,
            'mining_type': mining_type,
            'sample_size': sample_size,
        })

    # 4. Assign classification and regression targets
    # 4.1. Encoded bbox according to the prior boxes.
    encoded_bbox = box_coder(
        prior_box=prior_box,
        prior_box_var=prior_box_var,
        target_box=gt_box,
        code_type='encode_center_size')
    # 4.2. Assign regression targets
    target_bbox, target_loc_weight = target_assign(
        encoded_bbox, updated_matched_indices, mismatch_value=background_label)
    # 4.3. Assign classification targets
    target_label, target_conf_weight = target_assign(
        gt_label,
        updated_matched_indices,
        negative_indices=neg_indices,
        mismatch_value=background_label)

    # 5. Compute loss.
    # 5.1 Compute confidence loss.
    target_label = __reshape_to_2d(target_label)
    target_label = tensor.cast(x=target_label, dtype='int64')

    conf_loss = nn.softmax_with_cross_entropy(confidence, target_label)
    target_conf_weight = __reshape_to_2d(target_conf_weight)
    conf_loss = conf_loss * target_conf_weight

    # the target_label and target_conf_weight do not have gradient.
    target_label.stop_gradient = True
    target_conf_weight.stop_gradient = True

    # 5.2 Compute regression loss.
    location = __reshape_to_2d(location)
    target_bbox = __reshape_to_2d(target_bbox)

    loc_loss = nn.smooth_l1(location, target_bbox)
    target_loc_weight = __reshape_to_2d(target_loc_weight)
    loc_loss = loc_loss * target_loc_weight

    # the target_bbox and target_loc_weight do not have gradient.
    target_bbox.stop_gradient = True
    target_loc_weight.stop_gradient = True

    # 5.3 Compute overall weighted loss.
    loss = conf_loss_weight * conf_loss + loc_loss_weight * loc_loss
    # reshape to [N, Np], N is the batch size and Np is the prior box number.
    loss = nn.reshape(x=loss, shape=[-1, num_prior])
    loss = nn.reduce_sum(loss, dim=1, keep_dim=True)
    if normalize:
        normalizer = nn.reduce_sum(target_loc_weight)
        loss = loss / normalizer

    return loss


def prior_box(input,
              image,
              min_sizes,
              max_sizes=None,
              aspect_ratios=[1.],
              variance=[0.1, 0.1, 0.2, 0.2],
              flip=False,
              clip=False,
              steps=[0.0, 0.0],
              offset=0.5,
              name=None,
              min_max_aspect_ratios_order=False):
    """
    **Prior Box Operator**

    Generate prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
    Each position of the input produce N prior boxes, N is determined by
    the count of min_sizes, max_sizes and aspect_ratios, The size of the
    box is in range(min_size, max_size) interval, which is generated in
    sequence according to the aspect_ratios.

    Args:
       input(Variable): The Input Variables, the format is NCHW.
       image(Variable): The input image data of PriorBoxOp,
            the layout is NCHW.
       min_sizes(list|tuple|float value): min sizes of generated prior boxes.
       max_sizes(list|tuple|None): max sizes of generated prior boxes.
            Default: None.
       aspect_ratios(list|tuple|float value): the aspect ratios of generated
            prior boxes. Default: [1.].
       variance(list|tuple): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip(bool): Whether to flip aspect ratios. Default:False.
       clip(bool): Whether to clip out-of-boundary boxes. Default: False.
       step(list|turple): Prior boxes step across width and height, If
            step[0] == 0.0/step[1] == 0.0, the prior boxes step across
            height/weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset(float): Prior boxes center offset. Default: 0.5
       name(str): Name of the prior box op. Default: None.
       min_max_aspect_ratios_order(bool): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with 
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.

    Returns:
        tuple: A tuple with two Variable (boxes, variances)

        boxes: the output prior boxes of PriorBox.
        The layout is [H, W, num_priors, 4].
        H is the height of input, W is the width of input,
        num_priors is the total
        box count of each position of input.

        variances: the expanded variances of PriorBox.
        The layout is [H, W, num_priors, 4].
        H is the height of input, W is the width of input
        num_priors is the total
        box count of each position of input


    Examples:
        .. code-block:: python

            box, var = fluid.layers.prior_box(
                input=conv1,
                image=images,
                min_sizes=[100.],
                flip=True,
                clip=True)
    """
    helper = LayerHelper("prior_box", **locals())
    dtype = helper.input_dtype()

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(min_sizes):
        min_sizes = [min_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(steps) and len(steps) == 2):
        raise ValueError('steps should be a list or tuple ',
                         'with length 2, (step_width, step_height).')

    min_sizes = list(map(float, min_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    steps = list(map(float, steps))

    attrs = {
        'min_sizes': min_sizes,
        'aspect_ratios': aspect_ratios,
        'variances': variance,
        'flip': flip,
        'clip': clip,
        'step_w': steps[0],
        'step_h': steps[1],
        'offset': offset,
        'min_max_aspect_ratios_order': min_max_aspect_ratios_order
    }
    if max_sizes is not None and len(max_sizes) > 0 and max_sizes[0] > 0:
        if not _is_list_or_tuple_(max_sizes):
            max_sizes = [max_sizes]
        attrs['max_sizes'] = max_sizes

    box = helper.create_tmp_variable(dtype)
    var = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="prior_box",
        inputs={"Input": input,
                "Image": image},
        outputs={"Boxes": box,
                 "Variances": var},
        attrs=attrs, )
    box.stop_gradient = True
    var.stop_gradient = True
    return box, var


def multi_box_head(inputs,
                   image,
                   base_size,
                   num_classes,
                   aspect_ratios,
                   min_ratio=None,
                   max_ratio=None,
                   min_sizes=None,
                   max_sizes=None,
                   steps=None,
                   step_w=None,
                   step_h=None,
                   offset=0.5,
                   variance=[0.1, 0.1, 0.2, 0.2],
                   flip=True,
                   clip=False,
                   kernel_size=1,
                   pad=0,
                   stride=1,
                   name=None,
                   min_max_aspect_ratios_order=False):
    """
    Generate prior boxes for SSD(Single Shot MultiBox Detector)
    algorithm. The details of this algorithm, please refer the
    section 2.2 of SSD paper `SSD: Single Shot MultiBox Detector
    <https://arxiv.org/abs/1512.02325>`_ .

    Args:
       inputs(list|tuple): The list of input Variables, the format
            of all Variables is NCHW.
       image(Variable): The input image data of PriorBoxOp,
            the layout is NCHW.
       base_size(int): the base_size is used to get min_size
            and max_size according to min_ratio and max_ratio.
       num_classes(int): The number of classes.
       aspect_ratios(list|tuple): the aspect ratios of generated prior
            boxes. The length of input and aspect_ratios must be equal.
       min_ratio(int): the min ratio of generated prior boxes.
       max_ratio(int): the max ratio of generated prior boxes.
       min_sizes(list|tuple|None): If `len(inputs) <=2`,
            min_sizes must be set up, and the length of min_sizes
            should equal to the length of inputs. Default: None.
       max_sizes(list|tuple|None): If `len(inputs) <=2`,
            max_sizes must be set up, and the length of min_sizes
            should equal to the length of inputs. Default: None.
       steps(list|tuple): If step_w and step_h are the same,
            step_w and step_h can be replaced by steps.
       step_w(list|tuple): Prior boxes step
            across width. If step_w[i] == 0.0, the prior boxes step
            across width of the inputs[i] will be automatically
            calculated. Default: None.
       step_h(list|tuple): Prior boxes step across height, If
            step_h[i] == 0.0, the prior boxes step across height of
            the inputs[i] will be automatically calculated. Default: None.
       offset(float): Prior boxes center offset. Default: 0.5
       variance(list|tuple): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip(bool): Whether to flip aspect ratios. Default:False.
       clip(bool): Whether to clip out-of-boundary boxes. Default: False.
       kernel_size(int): The kernel size of conv2d. Default: 1.
       pad(int|list|tuple): The padding of conv2d. Default:0.
       stride(int|list|tuple): The stride of conv2d. Default:1,
       name(str): Name of the prior box layer. Default: None.
       min_max_aspect_ratios_order(bool): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with 
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the fininal
            detection results. Default: False.

    Returns:
        tuple: A tuple with four Variables. (mbox_loc, mbox_conf, boxes, variances)

        mbox_loc: The predicted boxes' location of the inputs. The layout
        is [N, H*W*Priors, 4]. where Priors is the number of predicted
        boxes each position of each input.

        mbox_conf: The predicted boxes' confidence of the inputs. The layout
        is [N, H*W*Priors, C]. where Priors is the number of predicted boxes
        each position of each input and C is the number of Classes.

        boxes: the output prior boxes of PriorBox. The layout is [num_priors, 4].
        num_priors is the total box count of each position of inputs.

        variances: the expanded variances of PriorBox. The layout is
        [num_priors, 4]. num_priors is the total box count of each position of inputs


    Examples:
        .. code-block:: python

          mbox_locs, mbox_confs, box, var = fluid.layers.multi_box_head(
            inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
            image=images,
            num_classes=21,
            min_ratio=20,
            max_ratio=90,
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True)
    """

    def _reshape_with_axis_(input, axis=1):
        if not (axis > 0 and axis < len(input.shape)):
            raise ValueError("The axis should be smaller than "
                             "the arity of input and bigger than 0.")
        new_shape = [
            -1, reduce(lambda x, y: x * y, input.shape[axis:len(input.shape)])
        ]
        out = nn.reshape(x=input, shape=new_shape)
        return out

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    def _is_list_or_tuple_and_equal(data, length, err_info):
        if not (_is_list_or_tuple_(data) and len(data) == length):
            raise ValueError(err_info)

    if not _is_list_or_tuple_(inputs):
        raise ValueError('inputs should be a list or tuple.')

    num_layer = len(inputs)

    if num_layer <= 2:
        assert min_sizes is not None and max_sizes is not None
        assert len(min_sizes) == num_layer and len(max_sizes) == num_layer
    elif min_sizes is None and max_sizes is None:
        min_sizes = []
        max_sizes = []
        step = int(math.floor(((max_ratio - min_ratio)) / (num_layer - 2)))
        for ratio in range(min_ratio, max_ratio + 1, step):
            min_sizes.append(base_size * ratio / 100.)
            max_sizes.append(base_size * (ratio + step) / 100.)
        min_sizes = [base_size * .10] + min_sizes
        max_sizes = [base_size * .20] + max_sizes

    if aspect_ratios:
        _is_list_or_tuple_and_equal(
            aspect_ratios, num_layer,
            'aspect_ratios should be list or tuple, and the length of inputs '
            'and aspect_ratios should be the same.')
    if step_h:
        _is_list_or_tuple_and_equal(
            step_h, num_layer,
            'step_h should be list or tuple, and the length of inputs and '
            'step_h should be the same.')
    if step_w:
        _is_list_or_tuple_and_equal(
            step_w, num_layer,
            'step_w should be list or tuple, and the length of inputs and '
            'step_w should be the same.')
    if steps:
        _is_list_or_tuple_and_equal(
            steps, num_layer,
            'steps should be list or tuple, and the length of inputs and '
            'step_w should be the same.')
        step_w = steps
        step_h = steps

    mbox_locs = []
    mbox_confs = []
    box_results = []
    var_results = []
    for i, input in enumerate(inputs):
        min_size = min_sizes[i]
        max_size = max_sizes[i]

        if not _is_list_or_tuple_(min_size):
            min_size = [min_size]
        if not _is_list_or_tuple_(max_size):
            max_size = [max_size]

        aspect_ratio = []
        if aspect_ratios is not None:
            aspect_ratio = aspect_ratios[i]
            if not _is_list_or_tuple_(aspect_ratio):
                aspect_ratio = [aspect_ratio]
        step = [step_w[i] if step_w else 0.0, step_h[i] if step_w else 0.0]

        box, var = prior_box(input, image, min_size, max_size, aspect_ratio,
                             variance, flip, clip, step, offset, None,
                             min_max_aspect_ratios_order)

        box_results.append(box)
        var_results.append(var)

        num_boxes = box.shape[2]

        # get loc
        num_loc_output = num_boxes * 4
        mbox_loc = nn.conv2d(
            input=input,
            num_filters=num_loc_output,
            filter_size=kernel_size,
            padding=pad,
            stride=stride)

        mbox_loc = nn.transpose(mbox_loc, perm=[0, 2, 3, 1])
        new_shape = [
            mbox_loc.shape[0],
            mbox_loc.shape[1] * mbox_loc.shape[2] * mbox_loc.shape[3] / 4, 4
        ]
        mbox_loc_flatten = nn.reshape(mbox_loc, shape=new_shape)
        mbox_locs.append(mbox_loc_flatten)

        # get conf
        num_conf_output = num_boxes * num_classes
        conf_loc = nn.conv2d(
            input=input,
            num_filters=num_conf_output,
            filter_size=kernel_size,
            padding=pad,
            stride=stride)
        conf_loc = nn.transpose(conf_loc, perm=[0, 2, 3, 1])
        new_shape = [
            conf_loc.shape[0], conf_loc.shape[1] * conf_loc.shape[2] *
            conf_loc.shape[3] / num_classes, num_classes
        ]
        conf_loc_flatten = nn.reshape(conf_loc, shape=new_shape)
        mbox_confs.append(conf_loc_flatten)

    if len(box_results) == 1:
        box = box_results[0]
        var = var_results[0]
        mbox_locs_concat = mbox_locs[0]
        mbox_confs_concat = mbox_confs[0]
    else:
        reshaped_boxes = []
        reshaped_vars = []
        for i in range(len(box_results)):
            reshaped_boxes.append(_reshape_with_axis_(box_results[i], axis=3))
            reshaped_vars.append(_reshape_with_axis_(var_results[i], axis=3))

        box = tensor.concat(reshaped_boxes)
        var = tensor.concat(reshaped_vars)
        mbox_locs_concat = tensor.concat(mbox_locs, axis=1)
        mbox_confs_concat = tensor.concat(mbox_confs, axis=1)

    box.stop_gradient = True
    var.stop_gradient = True
    return mbox_locs_concat, mbox_confs_concat, box, var


def anchor_generator(input,
                     anchor_sizes=None,
                     aspect_ratios=None,
                     variance=[0.1, 0.1, 0.2, 0.2],
                     stride=None,
                     offset=0.5,
                     name=None):
    """
    **Anchor generator operator**

    Generate anchors for Faster RCNN algorithm.
    Each position of the input produce N anchors, N =
    size(anchor_sizes) * size(aspect_ratios). The order of generated anchors
    is firstly aspect_ratios loop then anchor_sizes loop.

    Args:
       input(Variable): The input feature map, the format is NCHW.
       anchor_sizes(list|tuple|float): The anchor sizes of generated anchors,
       given in absolute pixels e.g. [64., 128., 256., 512.].
       For instance, the anchor size of 64 means the area of this anchor equals to 64**2.
       aspect_ratios(list|tuple|float): The height / width ratios of generated
            anchors, e.g. [0.5, 1.0, 2.0].
       variance(list|tuple): The variances to be used in box regression deltas.
            Default:[0.1, 0.1, 0.2, 0.2].
       stride(list|turple): The anchors stride across width and height,
            e.g. [16.0, 16.0]
       offset(float): Prior boxes center offset. Default: 0.5
       name(str): Name of the prior box op. Default: None.

    Returns:
        Anchors(Variable):  The output anchors with a layout of [H, W, num_anchors, 4].
              H is the height of input, W is the width of input,
              num_anchors is the box count of each position.
              Each anchor is in (xmin, ymin, xmax, ymax) format an unnormalized.
        Variances(Variable): The expanded variances of anchors
              with a layout of [H, W, num_priors, 4].
              H is the height of input, W is the width of input
              num_anchors is the box count of each position.
              Each variance is in (xcenter, ycenter, w, h) format.


    Examples:

        .. code-block:: python

            anchor, var = anchor_generator(
                input=conv1,
                anchor_sizes=[64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                variance=[0.1, 0.1, 0.2, 0.2],
                stride=[16.0, 16.0],
                offset=0.5)
    """
    helper = LayerHelper("anchor_generator", **locals())
    dtype = helper.input_dtype()

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(anchor_sizes):
        anchor_sizes = [anchor_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(stride) and len(stride) == 2):
        raise ValueError('stride should be a list or tuple ',
                         'with length 2, (stride_width, stride_height).')

    anchor_sizes = list(map(float, anchor_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    stride = list(map(float, stride))

    attrs = {
        'anchor_sizes': anchor_sizes,
        'aspect_ratios': aspect_ratios,
        'variances': variance,
        'stride': stride,
        'offset': offset
    }

    anchor = helper.create_tmp_variable(dtype)
    var = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="anchor_generator",
        inputs={"Input": input},
        outputs={"Anchors": anchor,
                 "Variances": var},
        attrs=attrs, )
    anchor.stop_gradient = True
    var.stop_gradient = True
    return anchor, var
