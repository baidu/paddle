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
from ..fluid.layers import nn, utils
from ..nn import Layer
from ..fluid.initializer import Normal

from paddle.common_ops_import import *

__all__ = [ #noqa
    'yolo_loss',
    'yolo_box',
    'deform_conv2d',
    'DeformConv2D',
    'read_file',
    'decode_jpeg'
]


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
             scale_x_y=1.,
             iou_aware=False,
             iou_aware_factor=0.5):
    r"""

    This operator generates YOLO detection boxes from output of YOLOv3 network.
    
    The output of previous network is in shape [N, C, H, W], while H and W
    should be the same, H and W specify the grid size, each grid point predict 
    given number boxes, this given number, which following will be represented as S,
    is specified by the number of anchors. In the second dimension(the channel
    dimension), C should be equal to S * (5 + class_num) if :attr:`iou_aware` is false,
    otherwise C should be equal to S * (6 + class_num). class_num is the object
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

    where the confidence scores follow the formula bellow

    .. math::

        score_{conf} = \begin{case}
                         obj, \text{if } iou_aware == flase \\
                         obj^{1 - iou_aware_factor} * iou^{iou_aware_factor}, \text{otherwise}
                       \end{case}

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
                          boundary. Default true.
        scale_x_y (float): Scale the center point of decoded bounding box.
                           Default 1.0
        name (string): The default value is None.  Normally there is no need 
                       for user to set this property.  For more information, 
                       please refer to :ref:`api_guide_Name`
        iou_aware (bool): Whether use iou aware. Default false
        iou_aware_factor (float): iou aware factor. Default 0.5

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
            'clip_bbox', clip_bbox, 'scale_x_y', scale_x_y, 'iou_aware',
            iou_aware, 'iou_aware_factor', iou_aware_factor)
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
        "iou_aware": iou_aware,
        "iou_aware_factor": iou_aware_factor
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


def deform_conv2d(x,
                  offset,
                  weight,
                  bias=None,
                  stride=1,
                  padding=0,
                  dilation=1,
                  deformable_groups=1,
                  groups=1,
                  mask=None,
                  name=None):
    r"""
    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          x shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          weight shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          offset shape: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

          mask shape: :math:`(N, H_f * W_f, H_{out}, W_{out})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        x (Tensor): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Tensor): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        weight (Tensor): The convolution kernel with shape [M, C/g, kH, kW], where M is
            the number of output channels, g is the number of groups, kH is the filter's
            height, kW is the filter's width.
        bias (Tensor, optional): The bias with shape [M,].
        stride (int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|list|tuple, optional): The padding size. If padding is a list/tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        groups (int, optonal): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        mask (Tensor, optional): The input mask of deformable convolution layer.
            A Tensor with type float32, float64. It should be None when you use
            deformable convolution v1.
        name(str, optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        Tensor: The tensor variable storing the deformable convolution \
                  result. A Tensor with type float32, float64.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python

          #deformable conv v2:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          weight = paddle.rand((16, 1, kh, kw))
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # mask shape should be [bs, hw * hw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          mask = paddle.rand((8, kh * kw, 26, 26))
          out = paddle.vision.ops.deform_conv2d(input, offset, weight, mask=mask)
          print(out.shape)
          # returns
          [8, 16, 26, 26]

          #deformable conv v1:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          weight = paddle.rand((16, 1, kh, kw))
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          out = paddle.vision.ops.deform_conv2d(input, offset, weight)
          print(out.shape)
          # returns
          [8, 16, 26, 26]
    """
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    use_deform_conv2d_v1 = True if mask is None else False

    if in_dygraph_mode():
        attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
                 'deformable_groups', deformable_groups, 'groups', groups,
                 'im2col_step', 1)
        if use_deform_conv2d_v1:
            op_type = 'deformable_conv_v1'
            pre_bias = getattr(core.ops, op_type)(x, offset, weight, *attrs)
        else:
            op_type = 'deformable_conv'
            pre_bias = getattr(core.ops, op_type)(x, offset, mask, weight,
                                                  *attrs)
        if bias is not None:
            out = nn.elementwise_add(pre_bias, bias, axis=1)
        else:
            out = pre_bias
    else:
        check_variable_and_dtype(x, "x", ['float32', 'float64'],
                                 'deform_conv2d')
        check_variable_and_dtype(offset, "offset", ['float32', 'float64'],
                                 'deform_conv2d')

        num_channels = x.shape[1]

        helper = LayerHelper('deformable_conv', **locals())
        dtype = helper.input_dtype()

        stride = utils.convert_to_list(stride, 2, 'stride')
        padding = utils.convert_to_list(padding, 2, 'padding')
        dilation = utils.convert_to_list(dilation, 2, 'dilation')

        pre_bias = helper.create_variable_for_type_inference(dtype)

        if use_deform_conv2d_v1:
            op_type = 'deformable_conv_v1'
            inputs = {
                'Input': x,
                'Filter': weight,
                'Offset': offset,
            }
        else:
            op_type = 'deformable_conv'
            inputs = {
                'Input': x,
                'Filter': weight,
                'Offset': offset,
                'Mask': mask,
            }

        outputs = {"Output": pre_bias}
        attrs = {
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'deformable_groups': deformable_groups,
            'im2col_step': 1,
        }
        helper.append_op(
            type=op_type, inputs=inputs, outputs=outputs, attrs=attrs)

        if bias is not None:
            out = helper.create_variable_for_type_inference(dtype)
            helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [bias]},
                outputs={'Out': [out]},
                attrs={'axis': 1})
        else:
            out = pre_bias
    return out


class DeformConv2D(Layer):
    r"""
    Compute 2-D deformable convolution on 4-D input.
    Given input image x, output feature map y, the deformable convolution operation can be expressed as follow:


    Deformable Convolution v2:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k) * \Delta m_k}

    Deformable Convolution v1:

    .. math::

        y(p) = \sum_{k=1}^{K}{w_k * x(p + p_k + \Delta p_k)}

    Where :math:`\Delta p_k` and :math:`\Delta m_k` are the learnable offset and modulation scalar for the k-th location,
    Which :math:`\Delta m_k` is one in deformable convolution v1. Please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168v2>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Example:
        - Input:

          x shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          weight shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          offset shape: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`

          mask shape: :math:`(N, H_f * W_f, H_{out}, W_{out})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1


    Parameters:
        in_channels(int): The number of input channels in the input image.
        out_channels(int): The number of output channels produced by the convolution.
        kernel_size(int|list|tuple): The size of the convolving kernel.
        stride(int|list|tuple, optional): The stride size. If stride is a list/tuple, it must
            contain three integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. The default value is 1.
        padding (int|list|tuple, optional): The padding size. If padding is a list/tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation(int|list|tuple, optional): The dilation size. If dilation is a list/tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. The default value is 1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        groups(int, optional): The groups number of the Conv3D Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. The default value is 1.
        weight_attr(ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. The default value is None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. The default value is None.
    Attribute:
        **weight** (Parameter): the learnable weights of filter of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Shape:
        - x: :math:`(N, C_{in}, H_{in}, W_{in})`
        - offset: :math:`(N, 2 * H_f * W_f, H_{out}, W_{out})`
        - mask: :math:`(N, H_f * W_f, H_{out}, W_{out})`
        - output: :math:`(N, C_{out}, H_{out}, W_{out})`
        Where
        ..  math::
           H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (kernel\_size[0] - 1) + 1))}{strides[0]} + 1
           W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (kernel\_size[1] - 1) + 1))}{strides[1]} + 1
    Examples:
        .. code-block:: python

          #deformable conv v2:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # mask shape should be [bs, hw * hw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          mask = paddle.rand((8, kh * kw, 26, 26))
          deform_conv = paddle.vision.ops.DeformConv2D(
              in_channels=1,
              out_channels=16,
              kernel_size=[kh, kw])
          out = deform_conv(input, offset, mask)
          print(out.shape)
          # returns
          [8, 16, 26, 26]

          #deformable conv v1:

          import paddle
          input = paddle.rand((8, 1, 28, 28))
          kh, kw = 3, 3
          # offset shape should be [bs, 2 * kh * kw, out_h, out_w]
          # mask shape should be [bs, hw * hw, out_h, out_w]
          # In this case, for an input of 28, stride of 1
          # and kernel size of 3, without padding, the output size is 26
          offset = paddle.rand((8, 2 * kh * kw, 26, 26))
          deform_conv = paddle.vision.ops.DeformConv2D(
              in_channels=1,
              out_channels=16,
              kernel_size=[kh, kw])
          out = deform_conv(input, offset)
          print(out.shape)
          # returns
          [8, 16, 26, 26]
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None):
        super(DeformConv2D, self).__init__()
        assert weight_attr is not False, "weight_attr should not be False in Conv."
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._deformable_groups = deformable_groups
        self._groups = groups
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._channel_dim = 1

        self._stride = utils.convert_to_list(stride, 2, 'stride')
        self._dilation = utils.convert_to_list(dilation, 2, 'dilation')
        self._kernel_size = utils.convert_to_list(kernel_size, 2, 'kernel_size')

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")

        self._padding = utils.convert_to_list(padding, 2, 'padding')

        filter_shape = [out_channels, in_channels // groups] + self._kernel_size

        def _get_default_param_initializer():
            filter_elem_num = np.prod(self._kernel_size) * self._in_channels
            std = (2.0 / filter_elem_num)**0.5
            return Normal(0.0, std, 0)

        self.weight = self.create_parameter(
            shape=filter_shape,
            attr=self._weight_attr,
            default_initializer=_get_default_param_initializer())
        self.bias = self.create_parameter(
            attr=self._bias_attr, shape=[self._out_channels], is_bias=True)

    def forward(self, x, offset, mask=None):
        out = deform_conv2d(
            x=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self._stride,
            padding=self._padding,
            dilation=self._dilation,
            deformable_groups=self._deformable_groups,
            groups=self._groups,
            mask=mask)
        return out


def read_file(filename, name=None):
    """
    Reads and outputs the bytes contents of a file as a uint8 Tensor
    with one dimension.

    Args:
        filename (str): Path of the file to be read.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        A uint8 tensor.

    Examples:
        .. code-block:: python

            import cv2
            import paddle

            fake_img = (np.random.random(
                        (400, 300, 3)) * 255).astype('uint8')

            cv2.imwrite('fake.jpg', fake_img)

            img_bytes = paddle.vision.ops.read_file('fake.jpg')
            
            print(img_bytes.shape)

    """

    if in_dygraph_mode():
        return core.ops.read_file('filename', filename)

    inputs = dict()
    attrs = {'filename': filename}

    helper = LayerHelper("read_file", **locals())
    out = helper.create_variable_for_type_inference('uint8')
    helper.append_op(
        type="read_file", inputs=inputs, attrs=attrs, outputs={"Out": out})

    return out


def decode_jpeg(x, mode='unchanged', name=None):
    """
    Decodes a JPEG image into a 3 dimensional RGB Tensor or 1 dimensional Gray Tensor. 
    Optionally converts the image to the desired format. 
    The values of the output tensor are uint8 between 0 and 255.

    Args:
        x (Tensor): A one dimensional uint8 tensor containing the raw bytes 
            of the JPEG image.
        mode (str): The read mode used for optionally converting the image. 
            Default: 'unchanged'.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Tensor: A decoded image tensor with shape (imge_channels, image_height, image_width)

    Examples:
        .. code-block:: python
            import cv2
            import paddle

            fake_img = (np.random.random(
                        (400, 300, 3)) * 255).astype('uint8')

            cv2.imwrite('fake.jpg', fake_img)

            img_bytes = paddle.vision.ops.read_file('fake.jpg')
            img = paddle.vision.ops.decode_jpeg(img_bytes)

            print(img.shape)
    """

    if in_dygraph_mode():
        return core.ops.decode_jpeg(x, "mode", mode)

    inputs = {'X': x}
    attrs = {"mode": mode}

    helper = LayerHelper("decode_jpeg", **locals())
    out = helper.create_variable_for_type_inference('uint8')
    helper.append_op(
        type="decode_jpeg", inputs=inputs, attrs=attrs, outputs={"Out": out})

    return out
