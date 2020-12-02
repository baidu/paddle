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

import numpy as np
from ..fluid.layer_helper import LayerHelper
from ..fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from ..fluid import core, layers

from paddle.common_ops_import import *

__all__ = ['yolo_loss', 'yolo_box']


def yolo_loss(x,
              gt_box,
              gt_label,
              anchors,
              anchor_mask,
              class_num,
              ignore_thresh,
              downsample_ratio,
              gt_score=None,
              use_label_smooth=True,
              name=None,
              scale_x_y=1.):
    r"""

    This operator generates YOLOv3 loss based on given predict result and ground
    truth boxes.
    
    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict 
    given number bounding boxes, this given number, which following will be represented as S,
    is specified by the number of anchor clusters in each scale. In the second dimension(the channel
    dimension), C should be equal to S * (class_num + 5), class_num is the object 
    category number of source dataset(such as 80 in coco dataset), so in the 
    second(channel) dimension, apart from 4 box location coordinates x, y, w, h, 
    also includes confidence score of the box and class one-hot key of each anchor box.

    Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box predictions
    should be as follows:

    $$
    b_x = \\sigma(t_x) + c_x
    $$
    $$
    b_y = \\sigma(t_y) + c_y
    $$
    $$
    b_w = p_w e^{t_w}
    $$
    $$
    b_h = p_h e^{t_h}
    $$

    In the equation above, :math:`c_x, c_y` is the left top corner of current grid
    and :math:`p_w, p_h` is specified by anchors.

    As for confidence score, it is the logistic regression value of IoU between
    anchor boxes and ground truth boxes, the score of the anchor box which has 
    the max IoU should be 1, and if the anchor box has IoU bigger than ignore 
    thresh, the confidence score loss of this anchor box will be ignored.

    Therefore, the YOLOv3 loss consists of three major parts: box location loss,
    objectness loss and classification loss. The L1 loss is used for 
    box coordinates (w, h), sigmoid cross entropy loss is used for box 
    coordinates (x, y), objectness loss and classification loss.

    Each groud truth box finds a best matching anchor box in all anchors. 
    Prediction of this anchor box will incur all three parts of losses, and
    prediction of anchor boxes with no GT box matched will only incur objectness
    loss.

    In order to trade off box coordinate losses between big boxes and small 
    boxes, box coordinate losses will be mutiplied by scale weight, which is
    calculated as follows.

    $$
    weight_{box} = 2.0 - t_w * t_h
    $$

    Final loss will be represented as follows.

    $$
    loss = (loss_{xy} + loss_{wh}) * weight_{box}
         + loss_{conf} + loss_{class}
    $$

    While :attr:`use_label_smooth` is set to be :attr:`True`, the classification
    target will be smoothed when calculating classification loss, target of 
    positive samples will be smoothed to :math:`1.0 - 1.0 / class\_num` and target of
    negetive samples will be smoothed to :math:`1.0 / class\_num`.

    While :attr:`gt_score` is given, which means the mixup score of ground truth 
    boxes, all losses incured by a ground truth box will be multiplied by its 
    mixup score.

    Args:
        x (Tensor): The input tensor of YOLOv3 loss operator, This is a 4-D
                      tensor with shape of [N, C, H, W]. H and W should be same,
                      and the second dimension(C) stores box locations, confidence
                      score and classification one-hot keys of each anchor box.
                      The data type is float32 or float64. 
        gt_box (Tensor): groud truth boxes, should be in shape of [N, B, 4],
                          in the third dimension, x, y, w, h should be stored. 
                          x,y is the center coordinate of boxes, w, h are the
                          width and height, x, y, w, h should be divided by 
                          input image height to scale to [0, 1].
                          N is the batch number and B is the max box number in 
                          an image.The data type is float32 or float64. 
        gt_label (Tensor): class id of ground truth boxes, should be in shape
                            of [N, B].The data type is int32. 
        anchors (list|tuple): The anchor width and height, it will be parsed
                              pair by pair.
        anchor_mask (list|tuple): The mask index of anchors used in current
                                  YOLOv3 loss calculation.
        class_num (int): The number of classes.
        ignore_thresh (float): The ignore threshold to ignore confidence loss.
        downsample_ratio (int): The downsample ratio from network input to YOLOv3
                                loss input, so 32, 16, 8 should be set for the
                                first, second, and thrid YOLOv3 loss operators. 
        name (string): The default value is None.  Normally there is no need 
                       for user to set this property.  For more information, 
                       please refer to :ref:`api_guide_Name`
        gt_score (Tensor): mixup score of ground truth boxes, should be in shape
                            of [N, B]. Default None.
        use_label_smooth (bool): Whether to use label smooth. Default True. 
        scale_x_y (float): Scale the center point of decoded bounding box.
                           Default 1.0

    Returns:
        Tensor: A 1-D tensor with shape [N], the value of yolov3 loss

    Raises:
        TypeError: Input x of yolov3_loss must be Tensor
        TypeError: Input gtbox of yolov3_loss must be Tensor 
        TypeError: Input gtlabel of yolov3_loss must be Tensor 
        TypeError: Input gtscore of yolov3_loss must be None or Tensor 
        TypeError: Attr anchors of yolov3_loss must be list or tuple
        TypeError: Attr class_num of yolov3_loss must be an integer
        TypeError: Attr ignore_thresh of yolov3_loss must be a float number
        TypeError: Attr use_label_smooth of yolov3_loss must be a bool value

    Examples:
      .. code-block:: python

          import paddle
          import numpy as np

          x = np.random.random([2, 14, 8, 8]).astype('float32')
          gt_box = np.random.random([2, 10, 4]).astype('float32')
          gt_label = np.random.random([2, 10]).astype('int32')

          x = paddle.to_tensor(x)
          gt_box = paddle.to_tensor(gt_box)
          gt_label = paddle.to_tensor(gt_label)

          loss = paddle.vision.ops.yolo_loss(x,
                                             gt_box=gt_box,
                                             gt_label=gt_label,
                                             anchors=[10, 13, 16, 30],
                                             anchor_mask=[0, 1],
                                             class_num=2,
                                             ignore_thresh=0.7,
                                             downsample_ratio=8,
                                             use_label_smooth=True,
                                             scale_x_y=1.)
    """

    if in_dygraph_mode() and gt_score is None:
        loss = core.ops.yolov3_loss(
            x, gt_box, gt_label, 'anchors', anchors, 'anchor_mask', anchor_mask,
            'class_num', class_num, 'ignore_thresh', ignore_thresh,
            'downsample_ratio', downsample_ratio, 'use_label_smooth',
            use_label_smooth, 'scale_x_y', scale_x_y)
        return loss

    helper = LayerHelper('yolov3_loss', **locals())

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'yolo_loss')
    check_variable_and_dtype(gt_box, 'gt_box', ['float32', 'float64'],
                             'yolo_loss')
    check_variable_and_dtype(gt_label, 'gt_label', 'int32', 'yolo_loss')
    check_type(anchors, 'anchors', (list, tuple), 'yolo_loss')
    check_type(anchor_mask, 'anchor_mask', (list, tuple), 'yolo_loss')
    check_type(class_num, 'class_num', int, 'yolo_loss')
    check_type(ignore_thresh, 'ignore_thresh', float, 'yolo_loss')
    check_type(use_label_smooth, 'use_label_smooth', bool, 'yolo_loss')

    loss = helper.create_variable_for_type_inference(dtype=x.dtype)

    objectness_mask = helper.create_variable_for_type_inference(dtype='int32')
    gt_match_mask = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {
        "X": x,
        "GTBox": gt_box,
        "GTLabel": gt_label,
    }
    if gt_score is not None:
        inputs["GTScore"] = gt_score

    attrs = {
        "anchors": anchors,
        "anchor_mask": anchor_mask,
        "class_num": class_num,
        "ignore_thresh": ignore_thresh,
        "downsample_ratio": downsample_ratio,
        "use_label_smooth": use_label_smooth,
        "scale_x_y": scale_x_y,
    }

    helper.append_op(
        type='yolov3_loss',
        inputs=inputs,
        outputs={
            'Loss': loss,
            'ObjectnessMask': objectness_mask,
            'GTMatchMask': gt_match_mask
        },
        attrs=attrs)
    return loss


def yolo_box(x,
             img_size,
             anchors,
             class_num,
             conf_thresh,
             downsample_ratio,
             clip_bbox=True,
             name=None,
             scale_x_y=1.):
    r"""

    This operator generates YOLO detection boxes from output of YOLOv3 network.
    
    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict 
    given number boxes, this given number, which following will be represented as S,
    is specified by the number of anchors. In the second dimension(the channel
    dimension), C should be equal to S * (5 + class_num), class_num is the object 
    category number of source dataset(such as 80 in coco dataset), so the 
    second(channel) dimension, apart from 4 box location coordinates x, y, w, h, 
    also includes confidence score of the box and class one-hot key of each anchor 
    box.

    Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box 
    predictions should be as follows:

    $$
    b_x = \\sigma(t_x) + c_x
    $$
    $$
    b_y = \\sigma(t_y) + c_y
    $$
    $$
    b_w = p_w e^{t_w}
    $$
    $$
    b_h = p_h e^{t_h}
    $$

    in the equation above, :math:`c_x, c_y` is the left top corner of current grid
    and :math:`p_w, p_h` is specified by anchors.

    The logistic regression value of the 5th channel of each anchor prediction boxes
    represents the confidence score of each prediction box, and the logistic
    regression value of the last :attr:`class_num` channels of each anchor prediction 
    boxes represents the classifcation scores. Boxes with confidence scores less than
    :attr:`conf_thresh` should be ignored, and box final scores is the product of 
    confidence scores and classification scores.

    $$
    score_{pred} = score_{conf} * score_{class}
    $$

    Args:
        x (Tensor): The input tensor of YoloBox operator is a 4-D tensor with
                      shape of [N, C, H, W]. The second dimension(C) stores box
                      locations, confidence score and classification one-hot keys
                      of each anchor box. Generally, X should be the output of
                      YOLOv3 network. The data type is float32 or float64. 
        img_size (Tensor): The image size tensor of YoloBox operator, This is a
                           2-D tensor with shape of [N, 2]. This tensor holds
                           height and width of each input image used for resizing
                           output box in input image scale. The data type is int32. 
        anchors (list|tuple): The anchor width and height, it will be parsed pair
                              by pair.
        class_num (int): The number of classes.
        conf_thresh (float): The confidence scores threshold of detection boxes.
                             Boxes with confidence scores under threshold should
                             be ignored.
        downsample_ratio (int): The downsample ratio from network input to
                                :attr:`yolo_box` operator input, so 32, 16, 8
                                should be set for the first, second, and thrid
                                :attr:`yolo_box` layer.
        clip_bbox (bool): Whether clip output bonding box in :attr:`img_size`
                          boundary. Default true."
        "
        scale_x_y (float): Scale the center point of decoded bounding box.
                           Default 1.0
        name (string): The default value is None.  Normally there is no need 
                       for user to set this property.  For more information, 
                       please refer to :ref:`api_guide_Name`

    Returns:
        Tensor: A 3-D tensor with shape [N, M, 4], the coordinates of boxes,
        and a 3-D tensor with shape [N, M, :attr:`class_num`], the classification 
        scores of boxes.

    Raises:
        TypeError: Input x of yolov_box must be Tensor
        TypeError: Attr anchors of yolo box must be list or tuple
        TypeError: Attr class_num of yolo box must be an integer
        TypeError: Attr conf_thresh of yolo box must be a float number

    Examples:

    .. code-block:: python

        import paddle
        import numpy as np

	x = np.random.random([2, 14, 8, 8]).astype('float32')
        img_size = np.ones((2, 2)).astype('int32')

        x = paddle.to_tensor(x)
        img_size = paddle.to_tensor(img_size)

        boxes, scores = paddle.vision.ops.yolo_box(x,
                                                   img_size=img_size,
                                                   anchors=[10, 13, 16, 30],
                                                   class_num=2,
                                                   conf_thresh=0.01,
                                                   downsample_ratio=8,
                                                   clip_bbox=True,
                                                   scale_x_y=1.)
    """
    if in_dygraph_mode():
        boxes, scores = core.ops.yolo_box(
            x, img_size, 'anchors', anchors, 'class_num', class_num,
            'conf_thresh', conf_thresh, 'downsample_ratio', downsample_ratio,
            'clip_bbox', clip_bbox, 'scale_x_y', scale_x_y)
        return boxes, scores

    helper = LayerHelper('yolo_box', **locals())

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'yolo_box')
    check_variable_and_dtype(img_size, 'img_size', 'int32', 'yolo_box')
    check_type(anchors, 'anchors', (list, tuple), 'yolo_box')
    check_type(conf_thresh, 'conf_thresh', float, 'yolo_box')

    boxes = helper.create_variable_for_type_inference(dtype=x.dtype)
    scores = helper.create_variable_for_type_inference(dtype=x.dtype)

    attrs = {
        "anchors": anchors,
        "class_num": class_num,
        "conf_thresh": conf_thresh,
        "downsample_ratio": downsample_ratio,
        "clip_bbox": clip_bbox,
        "scale_x_y": scale_x_y,
    }

    helper.append_op(
        type='yolo_box',
        inputs={
            "X": x,
            "ImgSize": img_size,
        },
        outputs={
            'Boxes': boxes,
            'Scores': scores,
        },
        attrs=attrs)
    return boxes, scores
