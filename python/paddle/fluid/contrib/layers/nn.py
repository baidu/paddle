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
"""
Contrib layers just related to the neural network.
"""

from __future__ import print_function

import numpy as np
import six
import os
import inspect
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import utils
from ... import unique_name
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from paddle.fluid.framework import Variable, convert_np_dtype_to_dtype_
from paddle.fluid.layers import slice, reshape
import warnings

__all__ = [
    'fused_elemwise_activation', 'sequence_topk_avg_pooling', 'var_conv_2d',
    'match_matrix_tensor', 'tree_conv', 'fused_embedding_seq_pool',
    'multiclass_nms2', 'search_pyramid_hash', 'shuffle_batch', 'partial_concat',
    'partial_sum', 'tdm_child', 'rank_attention', 'tdm_sampler'
]


def fused_elemwise_activation(x,
                              y,
                              functor_list,
                              axis=-1,
                              scale=0.0,
                              save_intermediate_out=True):
    """
    **Fused elementwise_add/mul and activation layers**

    This function computes an elementwise_add/mul cooperated with an activation.

    .. math::

        out = Unary(Binary(x, y))

    or

    .. math::

        out = Binary(x, Unary(y))

    Unary operators can be: `scale`, `relu`, `tanh`. Binary operators can be:
    `elementwise_add`, `elementwise_mul`.

    Args:
        x (Variable): left operation of the binary operator.
        y (Variable): right operator of the binary operator.
        functor_list (list of str): types of operator which will be executed
            by this layer. For example, ['elementwise_add', 'relu']
            (out = elementwise_add(x, relu(y))),
            or ['relu', 'elemmentwise_add'] (out = relu(elementwise_add(x, y))).
        axis (int32, default -1): axis of elementwise op.
        scale (float32, default 0): parameter of scale op.
        save_intermediate_out (bool, default True): whether to save the
            intermediate result, Unary(y) or Binary(x, y).

    Returns:
        Variable: The computation result.
    """
    if isinstance(functor_list, str):
        functor_list = functor_list.split(',')

    if not isinstance(functor_list, list) or len(functor_list) != 2:
        raise ValueError(
            'functor_list should be a list of str, and the length should be 2.')

    helper = LayerHelper('fused_elemwise_activation', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    intermediate_out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='fused_elemwise_activation',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out,
                 'IntermediateOut': intermediate_out},
        attrs={
            'axis': axis,
            'scale': scale,
            'save_intermediate_out': save_intermediate_out,
            'functor_list': functor_list
        })
    return out


def var_conv_2d(input,
                row,
                col,
                input_channel,
                output_channel,
                filter_size,
                stride=1,
                param_attr=None,
                act=None,
                dtype='float32',
                name=None):
    """
    The var_conv_2d layer calculates the output base on the :attr:`input` with variable length,
    row, col, input channel, filter size and strides. Both :attr:`input`, :attr:`row`,
    and :attr:`col` are 1-level LodTensor. The convolution operation is same as conv2d layer with
    padding. Besides, input.dims[1] should be 1.

    .. code-block:: text

            If input_channel is 2 and given row lodTensor and col lodTensor as follows:
                row.lod = [[5, 4]]
                col.lod = [[6, 7]]
            input is a lodTensor:
                input.lod = [[60, 56]]	# where 60 = input_channel * 5 * 6
                input.dims = [116, 1]	# where 116 = 60 + 56

            If set output_channel is 3, filter_size is [3, 3], stride is [1, 1]:
                # where 90 = output_channel * [(5-1)/stride + 1] * [(6-1)/stride + 1]
                output.lod = [[90, 84]]
                output.dims = [174, 1]  # where 174 = 90 + 84

    Args:
        input (Variable): The input should be 1-level LodTensor with dims[1] equals 1.
        row (Variable): The row should be 1-level LodTensor to provide height information.
        col (Variable): The col should be 1-level LodTensor to provide width information.
        input_channel (int): The number of input channel.
        output_channel (int): The number of output channel.
        filter_size (int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of var_conv2d. If it is set to None or one attribute of ParamAttr, var_conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        dtype ('float32'): The data type of parameter and output.
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically. Default: None

    Returns:
        Variable: Output variable with LoD specified by this layer.

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.fluid import layers
            from paddle.fluid import contrib

            x_lod_tensor = layers.data(name='x', shape=[1], lod_level=1)
            row_lod_tensor = layers.data(name='row', shape=[6], lod_level=1)
            col_lod_tensor = layers.data(name='col', shape=[6], lod_level=1)
            out = contrib.var_conv_2d(input=x_lod_tensor,
                                     row=row_lod_tensor,
                                     col=col_lod_tensor,
                                     input_channel=3,
                                     output_channel=5,
                                     filter_size=[3, 3],
                                     stride=1)
    """
    helper = LayerHelper('var_conv_2d', **locals())
    x_shape = list(input.shape)
    assert len(x_shape) == 2

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')

    filter_shape = [
        int(output_channel),
        int(input_channel) * filter_size[0] * filter_size[1]
    ]
    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype, )

    conv_res = helper.create_variable_for_type_inference(dtype)
    tmp_res = helper.create_variable_for_type_inference(
        dtype, stop_gradient=True)

    helper.append_op(
        type='var_conv_2d',
        inputs={
            'X': input,
            'ROW': row,
            'COLUMN': col,
            'W': filter_param,
        },
        outputs={"Out": conv_res,
                 "Col": tmp_res},
        attrs={
            'InputChannel': input_channel,
            'OutputChannel': output_channel,
            'StrideH': stride[0],
            'StrideW': stride[1],
            'KernelH': filter_size[0],
            'KernelW': filter_size[1],
        })

    return helper.append_activation(conv_res)


def match_matrix_tensor(x,
                        y,
                        channel_num,
                        act=None,
                        param_attr=None,
                        dtype='float32',
                        name=None):
    """
    Calculate the semantic matching matrix of two word sequences with variable length.
    Given a query A of length `n` and a title B of length `m`, the input shape are respectively
    [n, h] and [m, h], which h is hidden_size. If :attr:`channel_num` is set to 3,
    it will generate a learnable parameter matrix W with shape [h, 3, h].
    Then the semantic matching matrix of query A and title B is calculated by
    A * W * B.T = [n, h]*[h, 3, h]*[h, m] = [n, 3, m]. The learnable parameter matrix `W`
    is equivalent to a fully connected layer in the calculation process. If :attr:`act` is provided,
    the corresponding activation function will be applied to output matrix.
    The :attr:`x` and :attr:`y` should be LodTensor and only one level LoD is supported.

    .. code-block:: text

            Given a 1-level LoDTensor x:
                x.lod =  [
                    [2,                     3,                               ]]
                x.data = [[0.3, 0.1], [0.2, 0.3], [
                    0.5, 0.6], [0.7, 0.1], [0.3, 0.4]]
                x.dims = [5, 2]
            y is a Tensor:
                y.lod =  [[3,                                 1,       ]]
                y.data = [[0.1, 0.2], [0.3, 0.7], [0.9, 0.2], [0.4, 0.1]]
                y.dims = [4, 2]
            set channel_num 2, then we get a 1-level LoDTensor:
                # where 12 = channel_num * x.lod[0][0] * y.lod[0][0]
                out.lod =  [[12, 6]]
                out.dims = [18, 1]     # where 18 = 12 + 6

    Args:
        x (Variable): Input variable x which should be 1-level LodTensor.
        y (Variable): Input variable y which should be 1-level LodTensor.
        channel_num (int): The channel number of learnable parameter W.
        act (str, default None): Activation to be applied to the output of this layer.
        param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
            parameters/weights of this layer.
        dtype ('float32'): The data type of w data.
        name (str|None): A name for this layer(optional). If set None, the layer will be named automatically. Default: None

    Returns:
        Variable: output with LoD specified by this layer.

    Examples:
        .. code-block:: python

            import numpy as np
            from paddle.fluid import layers
            from paddle.fluid import contrib

            x_lod_tensor = layers.data(name='x', shape=[10], lod_level=1)
            y_lod_tensor = layers.data(name='y', shape=[10], lod_level=1)
            out, out_tmp = contrib.match_matrix_tensor(
                x=x_lod_tensor, y=y_lod_tensor, channel_num=3)
    """
    helper = LayerHelper('match_matrix_tensor', **locals())

    x_shape = list(x.shape)
    y_shape = list(y.shape)
    assert len(x_shape) == 2 and len(y_shape) == 2 and x_shape[-1] == y_shape[
        -1]

    weight_shape = [x_shape[-1], channel_num, y_shape[-1]]
    w = helper.create_parameter(
        attr=helper.param_attr, shape=weight_shape, dtype=dtype, is_bias=False)
    mm_res = helper.create_variable_for_type_inference(dtype)
    tmp_res = helper.create_variable_for_type_inference(
        dtype, stop_gradient=True)
    helper.append_op(
        type='match_matrix_tensor',
        inputs={
            'X': x,
            'Y': y,
            'W': w,
        },
        outputs={"Out": mm_res,
                 "Tmp": tmp_res},
        attrs={'dim_t': channel_num})

    return helper.append_activation(mm_res), tmp_res


def sequence_topk_avg_pooling(input, row, col, topks, channel_num):
    """
    The :attr:`topks` is a list with incremental values in this function. For each topk,
    it will average the topk features as an output feature for each channel of every
    input sequence. Both :attr:`row` and :attr:`col` are LodTensor, which provide height
    and width information for :attr:`input` tensor. If feature size of input sequence is less
    than topk, it will padding 0 at the back.

    .. code-block:: text

            If channel_num is 2 and given row LoDTensor and col LoDTensor as follows:
                row.lod = [[5, 4]]
                col.lod = [[6, 7]]

            input is a LoDTensor with input.lod[0][i] = channel_num * row.lod[0][i] * col.lod[0][i]
                input.lod = [[60, 56]]  # where 60 = channel_num * 5 * 6
                input.dims = [116, 1]   # where 116 = 60 + 56

            If topks is [1, 3, 5], then we get a 1-level LoDTensor:
                out.lod =  [[5, 4]] 	# share Lod info with row LodTensor
                out.dims = [9, 6]   	# where 6 = len(topks) * channel_num

    Args:
        input (Variable): The input should be 2D LodTensor with dims[1] equals 1.
        row (Variable): The row should be 1-level LodTensor to provide the height information
                        of the input tensor data.
        col (Variable): The col should be 1-level LodTensor to provide the width information
                        of the input tensor data.
        topks (list): A list of incremental value to average the topk feature.
        channel_num (int): The number of input channel.

    Returns:
        Variable: output LodTensor specified by this layer.

    Examples:

        .. code-block:: python

            import numpy as np
            from paddle.fluid import layers
            from paddle.fluid import contrib

            x_lod_tensor = layers.data(name='x', shape=[1], lod_level=1)
            row_lod_tensor = layers.data(name='row', shape=[6], lod_level=1)
            col_lod_tensor = layers.data(name='col', shape=[6], lod_level=1)
            out = contrib.sequence_topk_avg_pooling(input=x_lod_tensor,
                                                   row=row_lod_tensor,
                                                   col=col_lod_tensor,
                                                   topks=[1, 3, 5],
                                                   channel_num=5)
    """
    helper = LayerHelper('sequence_topk_avg_pooling', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    pos = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='sequence_topk_avg_pooling',
        inputs={'X': input,
                'ROW': row,
                'COLUMN': col},
        outputs={'Out': out,
                 'pos': pos},
        attrs={'topks': topks,
               'channel_num': channel_num})

    return out


def tree_conv(nodes_vector,
              edge_set,
              output_size,
              num_filters=1,
              max_depth=2,
              act='tanh',
              param_attr=None,
              bias_attr=None,
              name=None):
    """
    ${comment}

    Args:
        nodes_vector(${nodes_vector_type}): ${nodes_vector_comment}
        edge_set(${edge_set_type}): ${edge_set_comment}
        output_size(int): output feature width
        num_filters(int): number of filters, Default 1
        max_depth(int): max depth of filters, Default 2
        act(str): activation function, Default tanh
        param_attr(ParamAttr): the parameter attribute for the filters, Default None
        bias_attr(ParamAttr): the parameter attribute for the bias of this layer, Default None
        name(str): a name of this layer(optional). If set None, the layer will be named automatically, Default None

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # 10 for max_node_size of dataset, 5 for vector width
          nodes_vector = fluid.layers.data(
              name='vectors', shape=[10, 5], dtype='float32')
          # 10 for max_node_size of dataset, 2 for every edge has two nodes
          # edges must be directional
          edge_set = fluid.layers.data(name='edge_set', shape=[
                                       10, 2], dtype='float32')
          # the shape of output will be [10, 6, 1],
          # 10 for max_node_size of dataset, 6 for output size, 1 for 1 filter
          out_vector = fluid.layers.tree_conv(nodes_vector, edge_set, 6, 1, 2)
          # After reshape, output tensor could be nodes_vector for next tree convolution
          out_vector = fluid.layers.reshape(out_vector, shape=[-1, 10, 6])
          out_vector_2 = fluid.layers.tree_conv(out_vector, edge_set, 3, 4, 2)
          # also output tensor could be pooling(the pooling in paper called global pooling)
          pooled = fluid.layers.reduce_max(out_vector, dim=2) # global pooling
    """
    check_type(nodes_vector, 'nodes_vector', (Variable), 'tree_conv')
    check_type(edge_set, 'edge_set', (Variable), 'tree_conv')

    helper = LayerHelper("tree_conv", **locals())
    dtype = helper.input_dtype('nodes_vector')
    feature_size = nodes_vector.shape[2]
    W_shape = [feature_size, 3, output_size, num_filters]
    W = helper.create_parameter(
        attr=param_attr, shape=W_shape, dtype=dtype, is_bias=False)
    out = helper.create_variable_for_type_inference(dtype=dtype)
    helper.append_op(
        type='tree_conv',
        inputs={'NodesVector': nodes_vector,
                'EdgeSet': edge_set,
                'Filter': W},
        outputs={'Out': out, },
        attrs={'max_depth': max_depth})
    if helper.bias_attr:
        pre_activation = helper.append_bias_op(out)
    else:
        pre_activation = out
    return helper.append_activation(pre_activation)


def fused_embedding_seq_pool(input,
                             size,
                             is_sparse=False,
                             padding_idx=None,
                             combiner='sum',
                             param_attr=None,
                             dtype='float32'):
    """
    **Embedding Sequence pool**

    This layer is the fusion of lookup table and sequence_pool.

    Args:
        input (Variable): Input is a Tensor<int64> Variable, which contains the IDs' information.
            The value of the input IDs should satisfy :math:`0<= id < size[0]`.
        size (tuple|list): The shape of the lookup_table parameter. It should
            have two elements which indicate the size of the dictionary of
            embedding and the size of each embedding vector respectively.
        is_sparse (bool): The flag indicating whether to use sparse update.
            Default: False.
        padding_idx (int|long|None): It will output all-zero padding data whenever
            lookup encounters :math:`padding\_idx` in Ids. If set :attr:`None`, it makes
            no effect to output. If :math:`padding\_idx < 0`, the :math:`padding\_idx`
            will automatically be converted to :math:`size[0] + padding\_idx` to use.
            Default: None.
        combiner (str): The pooling type of sequence_pool, and only support `sum`.
            Default: sum.
        param_attr (ParamAttr): Parameters for this layer.
        dtype (np.dtype|core.VarDesc.VarType|str): The dtype refers to the data type of output
            tensor. It can be float32, float_16, int etc.
    Returns:
        The sequence pooling variable which is a Tensor.
    Examples:
        .. code-block:: python
            import numpy as np
            import paddle.fluid as fluid

            dict_size = 20
            data_t = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=1)
            padding_idx = np.random.randint(1, 10)
            out = fluid.contrib.fused_embedding_seq_pool(
                input=data_t,
                size=[dict_size, 32],
                param_attr='w',
                padding_idx=padding_idx,
                is_sparse=False)
    """
    helper = LayerHelper('fused_embedding_seq_pool', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    out = helper.create_variable_for_type_inference(dtype)
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        size[0] + padding_idx)
    helper.append_op(
        type='fused_embedding_seq_pool',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': out},
        attrs={
            'is_sparse': is_sparse,
            'combiner': combiner,
            'padding_idx': padding_idx
        })
    return out


def multiclass_nms2(bboxes,
                    scores,
                    score_threshold,
                    nms_top_k,
                    keep_top_k,
                    nms_threshold=0.3,
                    normalized=True,
                    nms_eta=1.,
                    background_label=0,
                    return_index=False,
                    name=None):
    """
    **Multiclass NMS2**

    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.

    Args:
        bboxes (Variable): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Variable): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        name(str): Name of the multiclass nms op. Default: None.

    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.


    Examples:
        .. code-block:: python


            import paddle.fluid as fluid
            boxes = fluid.layers.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = fluid.layers.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = fluid.layers.multiclass_nms2(bboxes=boxes,
                                              scores=scores,
                                              background_label=0,
                                              score_threshold=0.5,
                                              nms_top_k=400,
                                              nms_threshold=0.3,
                                              keep_top_k=200,
                                              normalized=False,
                                              return_index=True)
    """
    helper = LayerHelper('multiclass_nms2', **locals())

    output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
    index = helper.create_variable_for_type_inference(dtype='int')
    helper.append_op(
        type="multiclass_nms2",
        inputs={'BBoxes': bboxes,
                'Scores': scores},
        attrs={
            'background_label': background_label,
            'score_threshold': score_threshold,
            'nms_top_k': nms_top_k,
            'nms_threshold': nms_threshold,
            'nms_eta': nms_eta,
            'keep_top_k': keep_top_k,
            'nms_eta': nms_eta,
            'normalized': normalized
        },
        outputs={'Out': output,
                 'Index': index})
    output.stop_gradient = True
    index.stop_gradient = True

    if return_index:
        return output, index
    return output


def search_pyramid_hash(input,
                        num_emb,
                        space_len,
                        pyramid_layer,
                        rand_len,
                        drop_out_percent,
                        is_training,
                        use_filter,
                        white_list_len,
                        black_list_len,
                        seed,
                        lr,
                        param_attr=None,
                        param_attr_wl=None,
                        param_attr_bl=None,
                        name=None,
                        distribute_update_vars=None,
                        dtype='float32'):
    """
    **Pyramid hash embedding**

    Args:
        input (Variable): LoDTensor<int32> Variable contained the IDs' information.
        num_emb (int): The embedding size of output.
        space_len (int): The length of pyramid hash embedding space.
        pyramid_layer (int): The number of pyramid layers. It should be greater than 2.
        rand_len (int): The minimum length of pyramid hash cell.
        drop_out_percent (float): The probability of dropping out the input token randomly.
            It should satisfy: [0., 1.]
        is_training (bool): Whether in training or testing phrase.
        use_filter(bool): If set True, the white filter and black filter should be given by
            :attr:`param_attr_wl` and :attr:`param_attr_bl` .
        white_list_len(int): If set :math:`white_list_len>0` , white filter with shape [white_list_len, 1]
            should be provided by param_attr_wl.
        black_list_len(int): If set :math:`black_list_len>0` , black filter with shape [black_list_len, 1]
            should be provided by param_attr_bl.
        seed(int): The number of random seed.
        lr(float): The learning rate of weight created by :attr:`param_attr` with shape [space_len+rand_len, 1]
            in this layer.
        param_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        param_attr_wl(ParamAttr): Specified parameters of white filter.
        param_attr_bl(ParamAttr): Specified parameters of black filter.
        distribute_update_vars(list[ParamAttr.name]): Decided which params should be updated in distribute training.
            Used in Distribute Transpiler to create a trainer/server program.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
        dtype(str): The data type of output variable, float32.
    Returns:
        Variable: LoDTensor of pyramid hash embedding.
    """
    helper = LayerHelper('search_pyramid_hash', **locals())

    w_shape = [space_len + rand_len, 1]
    w = helper.create_parameter(
        attr=param_attr, shape=w_shape, dtype=dtype, is_bias=False)
    w.stop_gradient = True

    input_vars = {'X': input, 'W': w}
    if white_list_len > 0:
        wl_shape = [white_list_len, 1]
        white_list = helper.create_parameter(
            attr=param_attr_wl, shape=wl_shape, dtype=dtype, is_bias=False)
        white_list.stop_gradient = True
        input_vars['WhiteList'] = white_list

    if black_list_len >= 0:
        bl_shape = [black_list_len, 1]
        black_list = helper.create_parameter(
            attr=param_attr_bl, shape=bl_shape, dtype=dtype, is_bias=False)
        black_list.stop_gradient = True
        input_vars['BlackList'] = black_list

    distribute_update_vars_str = ""
    if distribute_update_vars:
        assert isinstance(distribute_update_vars, list)
        special_name_list = []
        if param_attr:
            special_name_list.append(param_attr.name)
        if param_attr_wl:
            special_name_list.append(param_attr_wl.name)
        if param_attr_bl:
            special_name_list.append(param_attr_bl.name)
        for param in distribute_update_vars:
            if param not in special_name_list:
                raise ValueError(
                    "Pyramid Hash layer didn't have parameter {}".format(param))
        distribute_update_vars_str = ",".join(distribute_update_vars)

    res = helper.create_variable_for_type_inference(dtype)
    drop_pos = helper.create_variable_for_type_inference(dtype)
    x_temp_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='pyramid_hash',
        inputs=input_vars,
        outputs={"Out": res,
                 "X_Temp_Out": x_temp_out,
                 'DropPos': drop_pos},
        attrs={
            'num_emb': num_emb,
            'space_len': space_len,
            'pyramid_layer': pyramid_layer,
            'rand_len': rand_len,
            'drop_out_percent': drop_out_percent,
            'is_training': is_training,
            'use_filter': use_filter,
            'white_list_len': white_list_len,
            'black_list_len': black_list_len,
            'seed': seed,
            'lr': lr,
            'distribute_update_vars': distribute_update_vars_str
        })

    return res


def shuffle_batch(x, seed=None):
    """
    This layer shuffle input tensor :attr:`x` . Normally, :attr:`x` is 2-D LoDTensor.

    :attr:`x` is a LoDTensor to be shuffled with shape :math:`[N_1, N_2, ..., N_k, D]` . Note that the last dim of input will not be shuffled.
    :math:`N_1 * N_2 * ... * N_k` numbers of elements with length :math:`D` will be shuffled randomly.

    For Example:

    .. code-block:: text

      Input:
        x.data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        x.dims = [4, 2]

      Attrs:
        seed = 2019

      Output:
        Out.data =[[7, 8], [1, 2], [3, 4], [5, 6]]
        Out.dims = [4, 2]

    Args:
        x (Variable): The input variable. The input variable is a N-D LoDTensor with type int, float32 or float64.
        seed (None|int|Variable): The start up seed. If set, seed will be set as the start up seed of shuffle engine.
                If not set(Default), start up seed of shuffle engine will be generated randomly.

    Returns:
        Variables: The shuffled LoDTensor with the same shape and lod as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name="x", shape=[-1, 4])
            out = fluid.contrib.layers.shuffle_batch(x)
    """
    helper = LayerHelper('shuffle_batch', **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    shuffle_idx = helper.create_variable_for_type_inference(dtype=np.int64)
    if seed is None and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed
    if seed is None:
        seed = np.random.randint(-65536, 65535)
    op_attrs = {}
    if isinstance(seed, int):
        op_attrs["startup_seed"] = seed
        seed = helper.create_variable(
            name=unique_name.generate("shuffle_batch_seed"),
            dtype="int64",
            persistable=True)
    helper.append_op(
        type='shuffle_batch',
        inputs={'X': x,
                'Seed': seed},
        outputs={'Out': out,
                 'ShuffleIdx': shuffle_idx,
                 'SeedOut': seed},
        attrs=op_attrs)
    return out


def partial_concat(input, start_index=0, length=-1):
    """
    **Partial Concat**
    This OP concatenates the inputs according to the start index and length. This
    OP exists in contrib, which means that it is not shown to the public.
    Only 2-D Tensor or LodTensor input is supported. Slice and concat can only be
    performed along the second dimension.

    .. code-block:: text

        Given:
            x = [[0, 1, 2],
                 [3, 4, 5]]
            y = [[6, 7 ,8],
                 [9, 10, 11]]
            output = partial_concat([x, y], start_index=0, length=2)

          we get:

            output = [[0, 1, 6, 7],
                      [3, 4, 9, 10]]

    Args:
        input(list): List of input Tensors with data type float32, float64, int32,
            int64.
        start_index(int32): The start index of each instance for partial concatenation.
            Default is 0.
        length(int32): The length of each instance for partial concatenation. Default is -1.
            Negative values for all elements after start_index.
    Returns:
        Variable: A Tensor with the same data type as input's.
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            x = fluid.data(name="x", shape=[None,3], dtype="float32")
            y = fluid.data(name="y", shape=[None,3], dtype="float32")
            concat = fluid.contrib.layers.partial_concat(
                [x, y], start_index=0, length=2)
    """
    if not isinstance(input, list):
        warnings.warn(
            "The type of input in partial_concat should be list, but received %s."
            % (type(input)))
        input = [input]
    for id, x in enumerate(input):
        check_variable_and_dtype(
            x, 'input[' + str(id) + ']',
            ['float16', 'float32', 'float64', 'int32', 'int64'],
            'partial_concat')
    check_type(start_index, 'start_index', (int), 'partial_concat')
    check_type(length, 'length', (int), 'partial_concat')
    inputs = {'X': input}
    attrs = {'start_index': start_index, 'length': length}
    helper = LayerHelper('partial_concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='partial_concat',
        inputs=inputs,
        outputs={'Out': [out]},
        attrs=attrs)
    return out


def partial_sum(input, start_index=0, length=-1):
    """
    **PartialSum**
    This Op can sum the vars by specifying the initial position(start_index) and length(length).
    This Op exists in contrib, which means that it is not shown to the public.
    Only 2-D Tensor or LodTensor input is supported. Slice and concat can only be
    performed along the second dimension.
    .. code-block:: text

        Given:
            x = [[0, 1, 2],
                 [3, 4, 5]]
            y = [[6, 7 ,8],
                 [9, 10, 11]]
            output = partial_sum([x, y], start_index=0, length=2)
          we get:

            output = [[6, 8],
                      [12, 14]]
    Args:
        input(list): List of input Tensors with data type float32, float64, int32,
            int64.
    Returns:
        Variable: A Tensor with the same data type as input's.
    Examples:
        .. code-block:: python
        import paddle.fluid.layers as layers
        import paddle.fluid as fluid
        import numpy as np
        x = fluid.data(name="x", shape=[None, 3], dtype="float32")
        y = fluid.data(name="y", shape=[None, 3], dtype="float32")
        sum = layers.partial_sum([x,y], start_index=0, length=2)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        xx = np.array([1,2,3,4,5,6]).reshape((2,3)).astype("float32")
        yy = np.array([6,5,4,4,5,6]).reshape((2,3)).astype("float32")
        out = exe.run(feed={"x":xx, "y":yy}, fetch_list=[sum])
    """
    for id, x in enumerate(input):
        check_variable_and_dtype(x, 'input[' + str(id) + ']',
                                 ['float32', 'float64', 'int32', 'int64'],
                                 'partial_sum')

    inputs = {'X': input}
    attrs = {}
    attrs['start_index'] = start_index
    attrs['length'] = length
    helper = LayerHelper('partial_sum', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='partial_sum', inputs=inputs, outputs={'Out': [out]}, attrs=attrs)
    return out


def tdm_child(x, node_nums, child_nums, param_attr=None, dtype='int32'):
    """
    **Tdm Child**
     According to the input node_id on the given tree, return the corresponding child node_id and 
      whether child is a leaf node by leaf_mask value.
    .. code-block:: text

        Given:
            tree[[0], [1, 2], [3, 4], [5, 6]] # A binary tree with seven nodes
            x = [[2], [3]]
            node_nums = 7
            child_nums = 2

          we get:
            child = [[5, 6],
                     [0, 0]]
            leaf_mask = [[1, 1],
                         [0, 0]]
    Args:
        x(Variable): Variable contained the node_id information, dtype support int32/int64.
        node_nums(int): Number of total nodes.
        child_nums(int): Maximum number of child nodes per node.
        param_attr(ParamAttr): To specify the tdm-tree-info parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in: ref: `api_fluid_ParamAttr`, should
            has shape(node_nums, 3 + child_nums), dtype support int32/int64. 
            The dimension[1] of tdm-tree-info contains the following: 
            1. Item_id(int, shape(1)), if node is a leaf node, give its item_id corresponding to node_id, else give 0.
            2. Layer_id(int, shape(1)), indicates which layer the node is on.
            3. Parent_id(int, shape(1)), node's parent node.
            4. Child_id(int, shape(child_nums)), all child node's node_id of this node should be given. 
            If the number of child nodes is insufficient, padding 0 until child nums equal to child_nums
        dtype(str): The data type of output child and leaf_mask, support int32/int64.

    Returns:
        tuple: A tuple including input node's child(Variable) and leaf_mask(Variable). 
            If child is a leaf node, leaf_mask equal ot 1, otherwise equal to 0.

    Examples:
        .. code-block:: python
        import paddle.fluid as fluid
        import numpy as np
        x = fluid.data(name="x", shape=[None, 1], dtype="int32", lod_level=1)
        tree_info = [[0,0,0,1,2],
                     [0,1,0,3,4],[0,1,0,5,6],
                     [0,2,1,0,0],[1,2,1,0,0],[2,2,2,0,0],[3,2,2,0,0]]
        tree_info_np = np.array(tree_info)
        tree_info_np = np.reshape(tree_info_np, (7,5))
        node_nums = 7
        child_nums = 2
        child, leaf_mask  = fluid.contrib.layers.tdm_child(x, node_nums, child_nums,
                                param_attr=fluid.ParamAttr(
                                    initializer=fluid.initializer.NumpyArrayInitializer(
                                                                            tree_info_np)))
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        xx = np.array([[2],[3]]).reshape((2,1)).astype("int32")
        child_res, leaf_mask_res = exe.run(feed={"x":xx}, fetch_list=[child, leaf_mask])
     """
    helper = LayerHelper("tdm_child", **locals())
    check_dtype(dtype, 'dtype', ['int32', 'int64'],
                'fluid.contrib.layers.tdm_child')
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    tree_info = helper.create_parameter(
        attr=helper.param_attr,
        shape=[node_nums, 3 + child_nums],
        dtype=dtype,
        default_initializer=Constant(0))
    tree_info.stop_gradient = True

    child = helper.create_variable_for_type_inference(dtype=dtype)
    leaf_mask = helper.create_variable_for_type_inference(dtype=dtype)

    helper.append_op(
        type='tdm_child',
        inputs={'X': x,
                'TreeInfo': tree_info},
        outputs={'Child': child,
                 'LeafMask': leaf_mask},
        attrs={'child_nums': child_nums,
               'dtype': c_dtype},
        stop_gradient=True)
    return (child, leaf_mask)


def tdm_sampler(x,
                neg_samples_num_list,
                layer_node_num_list,
                leaf_node_num,
                neg_samples_radius_list=None,
                tree_travel_attr=None,
                tree_layer_attr=None,
                output_positive=True,
                output_list=True,
                seed=0,
                tree_dtype='int32',
                dtype='int32'):
    """
    **Tdm Sampler**
    According to the input positive samples at leaf node(x), do negative sampling layer by layer on the given tree.
    .. code-block:: text

        Given:
            tree[[0], [1, 2], [3, 4], [5, 6]] # A binary tree with seven nodes
            travel_list = [[1, 3], [1, 4], [2, 5], [2, 6]] # leaf node's travel path (exclude root node)
            layer_list = [[1, 2], [3, 4, 5, 6]] # two layer (exclude root node)

            x = [[0], [1], [2], [3]] # Corresponding to leaf node [[3], [4], [5], [6]]
            neg_samples_num_list = [0, 0] # negative sample nums = 0
            layer_node_num_list = [2, 4]
            leaf_node_num = 4
            output_list = False

          we get:
            out = [[1, 3], [1, 4], [2, 5], [2, 6]]
            labels = [[1, 1], [1, 1], [1, 1], [1, 1]]
            mask = [[1, 1], [1, 1], [1, 1], [1, 1]]

    Args:
        x (Variable): Variable contained the item_id(corresponding to leaf node) information, dtype support int32/int64.
        neg_samples_num_list (list(int)): Number of negative samples per layer.
        layer_node_num_list (list(int)): Number of nodes per layer, must has same shape with neg_samples_num_list.
        leaf_node_num (int): Number of leaf nodes.
        tree_travel_attr (ParamAttr): To specify the tdm-travel parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr`, should 
            has shape (leaf_node_num, len(layer_node_num_list)), dtype support int32/int64.
        tree_layer_attr (ParamAttr): To specify the tdm-layer parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr`, should 
            has shape (node_num, 1), dtype support int32/int64.
        output_positive (bool): Whether to output positive samples (includ label and mask )at the same time.
        output_list (bool): Whether to divide the output into layers and organize it into list format.
        seed (int): The number of random seed.
        tree_dtype(np.dtype|core.VarDesc.VarType|str): The dtype of tdm-travel and tdm-layer, support int32/int64
        dtype(np.dtype|core.VarDesc.VarType|str): The dtype of output(sampling results, labels and masks) 

    Returns:
        tuple: A tuple including sampling results, corresponding labels and masks. if output_positive = True, sampling
            result  will include both positive and negative samples. If sampling reseult is a positive sample, the label is 1, 
            and if it is a negative sample, it is 0. If the tree is unbalanced, in order to ensure the consistency of the 
            sampling result shape, the padding sample's mask = 0, the real sample's mask value = 1. 
            If output_list = True, the result will organize into list format specified by layer information.
            Output variable have same type with tdm-travel and tdm-layer parameter(tree_dtype).

    Examples:
        .. code-block:: python
        import paddle.fluid as fluid
        import numpy as np
        x = fluid.data(name="x", shape=[None, 1], dtype="int32", lod_level=1)
        travel_list = [[1, 3], [1, 4], [2, 5], [2, 6]] # leaf node's travel path, shape(leaf_node_num, layer_num)
        layer_list_flat = [[1], [2], [3], [4], [5], [6]] # shape(node_nums, 1)

        neg_samples_num_list = [0, 0] # negative sample nums = 0
        layer_node_num_list = [2, 4] #two layer (exclude root node)
        leaf_node_num = 4

        travel_array = np.array(travel_list)
        layer_array = np.array(layer_list_flat)

        sample, label, mask = fluid.contrib.layers.tdm_sampler(
            x,
            neg_samples_num_list,
            layer_node_num_list,
            leaf_node_num,
            tree_travel_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    travel_array)),
            tree_layer_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    layer_array)),
            output_positive=True,
            output_list=True,
            seed=0,
            tree_dtype='int32')

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        xx = np.array([[0],[1]]).reshape((2,1)).astype("int32")

        exe.run(feed={"x":xx})

    """
    helper = LayerHelper("tdm_sampler", **locals())
    check_dtype(tree_dtype, 'tree_dtype', ['int32', 'int64'],
                'fluid.contrib.layers.tdm_sampler')
    check_dtype(dtype, 'dtype', ['int32', 'int64'],
                'fluid.contrib.layers.tdm_sampler')
    c_dtype = convert_np_dtype_to_dtype_(dtype)

    if len(neg_samples_num_list) != len(layer_node_num_list):
        raise ValueError(
            "The shape of negative samples list must match the shape of layers. "
            "But received len of neg_samples_num_list: {},"
            "and len of layer_node_num_list: {}, please check your input.".
            format(len(neg_samples_num_list), len(layer_node_num_list)))
    assert leaf_node_num is not None, "leaf_node_num should not be None here."

    layer_nums = 0
    node_nums = 0
    tree_layer_offset_lod = [0]
    for layer_idx, layer_node_num in enumerate(layer_node_num_list):
        layer_nums += 1
        node_nums += layer_node_num
        tree_layer_offset_lod.append(node_nums)
        if neg_samples_num_list[layer_idx] >= layer_node_num_list[layer_idx]:
            raise ValueError(
                "The number of negative samples must be less than the number of nodes "
                "in the layer {}, But received negative nums {}, and num of node at layer {} "
                "is {}, please check your input.".format(
                    layer_idx, neg_samples_num_list[
                        layer_idx], layer_idx, layer_node_num_list[layer_idx]))
    assert leaf_node_num < node_nums, "leaf_node_num must be less than total node nums."

    travel_shape = [leaf_node_num, layer_nums]
    travel = helper.create_parameter(
        attr=tree_travel_attr,
        shape=travel_shape,
        dtype=tree_dtype,
        default_initializer=Constant(0))

    layer_shape = [node_nums, 1]
    layer = helper.create_parameter(
        attr=tree_layer_attr,
        shape=layer_shape,
        dtype=tree_dtype,
        default_initializer=Constant(0))

    out = helper.create_variable_for_type_inference(dtype=dtype)
    out.stop_gradient = True

    labels = helper.create_variable_for_type_inference(dtype=dtype)
    labels.stop_gradient = True

    mask = helper.create_variable_for_type_inference(dtype=dtype)
    mask.stop_gradient = True

    helper.append_op(
        type='tdm_sampler',
        inputs={"X": x,
                "Travel": travel,
                "Layer": layer},
        outputs={'Out': out,
                 'Labels': labels,
                 'Mask': mask},
        attrs={
            'neg_samples_num_list': neg_samples_num_list,
            'neg_samples_radius_list': neg_samples_radius_list,
            'output_positive': output_positive,
            'layer_offset_lod': tree_layer_offset_lod,
            'seed': seed,
            'dtype': c_dtype
        })

    if output_list:
        output_list = []
        labels_list = []
        mask_list = []
        start_offset = 0
        positive_flag = 1
        if not output_positive:
            positive_flag = 0

        for layer_sample_num in neg_samples_num_list:
            end_offset = start_offset + \
                layer_sample_num + positive_flag
            layer_samples = slice(
                out, axes=[1], starts=[start_offset], ends=[end_offset])
            layer_labels = slice(
                labels, axes=[1], starts=[start_offset], ends=[end_offset])
            layer_mask = slice(
                mask, axes=[1], starts=[start_offset], ends=[end_offset])

            layer_samples = reshape(layer_samples,
                                    [-1, layer_sample_num + positive_flag, 1])
            layer_samples.stop_gradient = True

            layer_labels = reshape(layer_labels,
                                   [-1, layer_sample_num + positive_flag, 1])
            layer_labels.stop_gradient = True

            layer_mask = reshape(layer_mask,
                                 [-1, layer_sample_num + positive_flag, 1])
            layer_mask.stop_gradient = True

            output_list.append(layer_samples)
            labels_list.append(layer_labels)
            mask_list.append(layer_mask)
            start_offset = end_offset

        out = output_list
        labels = labels_list
        mask = mask_list

    return (out, labels, mask)


def rank_attention(input,
                   rank_offset,
                   rank_param_shape,
                   rank_param_attr,
                   max_rank=3):
    """
    **Rank Attention layer**
    This Op can calculate rank attention between input and rank_param, and 
    rank_param gives the organization of data. Notice: It currently supports
    GPU device.
    This Op exists in contrib, which means that it is not shown to the public.
    Args:
        input: Tensor with data type float32, float64.
        rank_offset: Tensor with data type int32.
        rank_para_shape: The shape of rank_param.
        rank_param_attr: Attribute initializer of rank_param.
        max_rank: The max rank of input's ranks.
    Returns:
        Variable: A Tensor with the same data type as input's.
    Examples:
        .. code-block:: python
           import paddle.fluid as fluid
           import numpy as np

           input = fluid.data(name="input", shape=[None, 2], dtype="float32")
           rank_offset = fluid.data(name="rank_offset", shape=[None, 7], dtype="int32")
           out = fluid.contrib.layers.rank_attention(input=input,
                                                     rank_offset=rank_offset,
                                                     rank_param_shape=[18,3],
                                                     rank_param_attr=
                                                       fluid.ParamAttr(learning_rate=1.0,
                                                                     name="ubm_rank_param.w_0",
                                                                     initializer=
                                                                     fluid.initializer.Xavier(uniform=False)),
                                                      max_rank=3)
    """
    helper = LayerHelper('rank_attention', **locals())
    dtype = helper.input_dtype(input_param_name='input')
    input_shape = input.shape
    assert input_shape[1] * max_rank * max_rank == rank_param_shape[0]

    rank_param = helper.create_parameter(
        attr=rank_param_attr, shape=rank_param_shape, dtype=dtype)
    rank_param.stop_gradient = False

    output = helper.create_variable_for_type_inference(dtype)
    ins_rank = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    helper.append_op(
        type="rank_attention",
        inputs={
            "X": input,
            "RankOffset": rank_offset,
            "RankParam": rank_param
        },
        outputs={"Out": output},
        attrs={"MaxRank": max_rank})

    return output
