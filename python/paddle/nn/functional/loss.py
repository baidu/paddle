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

import paddle
from ...fluid.layer_helper import LayerHelper
from ...fluid.data_feeder import check_variable_and_dtype
import paddle.fluid as fluid

# TODO: define loss functions of neural network
import numpy as np
import paddle
import paddle.fluid as fluid
from ...fluid.framework import core, in_dygraph_mode
from ...fluid.layers.nn import _elementwise_op_in_dygraph
from ...fluid.layers import dice_loss  #DEFINE_ALIAS
from ...fluid.layers import log_loss  #DEFINE_ALIAS
from ...fluid.layers import npair_loss  #DEFINE_ALIAS
from ...fluid.layers import reshape
from ...fluid.layers import softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import square_error_cost  #DEFINE_ALIAS

from ...fluid.layers import edit_distance  #DEFINE_ALIAS
from ...fluid.layers import huber_loss
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode
from ...fluid.framework import _varbase_creator
from ...fluid.framework import Variable

__all__ = [
    'binary_cross_entropy',
    'binary_cross_entropy_with_logits',
    'cross_entropy',
    'dice_loss',
    'hsigmoid_loss',
    'kl_div',
    'l1_loss',
    'log_loss',
    'mse_loss',
    'margin_ranking_loss',
    #       'nce',
    'nll_loss',
    'npair_loss',
    'sigmoid_focal_loss',
    'smooth_l1_loss',
    'softmax_with_cross_entropy',
    'square_error_cost',
    'ctc_loss',
]


def binary_cross_entropy(input, label, weight=None, reduction='mean',
                         name=None):
    """
    This op measures the binary_cross_entropy loss between input predictions ``input``
    and target labels ``label`` . The binary_cross_entropy loss can be described as:

    If :attr:`weight` is set, the loss is:

    .. math::
        Out = -1 * weight * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`weight` is None, the loss is:

    .. math::
        Out = -1 * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`reduction` set to ``'none'``, the interface will return the original loss `Out`.

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(Out)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(Out)

    Note that the input predictions ``input`` always be the output of sigmoid, and the target labels ``label``
    should be numbers between 0 and 1.

    Parameters:
        input (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``input``
            should always be the output of sigmod.  Available dtype is float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``input``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float32, float64.
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, has to be a Tensor of size nbatch and the data type
            is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        output (Tensor): If ``reduction`` is ``'none'``, the shape of output is
            same as ``input`` , else the shape of output is scalar.

    Examples:
        .. code-block:: python

            import paddle

            input = paddle.to_tensor([0.5, 0.6, 0.7], 'float32')
            label = paddle.to_tensor([1.0, 0.0, 1.0], 'float32')
            output = paddle.nn.functional.binary_cross_entropy(input, label)
            print(output)  # [0.65537095]

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in binary_cross_entropy should be 'sum', "
            "'mean' or 'none', but received %s, which is not allowed." %
            reduction)

    if in_dygraph_mode():
        out = core.ops.bce_loss(input, label)
        if weight is not None:
            out = core.ops.elementwise_mul(out, weight, 'axis', -1)

        if reduction == 'sum':
            return core.ops.reduce_sum(out, 'dim', [0], 'keep_dim', False,
                                       "reduce_all", True)
        elif reduction == 'mean':
            return core.ops.mean(out)
        else:
            return out

    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'binary_cross_entropy')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'binary_cross_entropy')

    sub_name = name if weight is None and reduction is 'none' else None
    helper = LayerHelper("binary_cross_entropy", name=sub_name)
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='bce_loss',
        inputs={
            'X': [input],
            'Label': [label],
        },
        outputs={'Out': [out]})

    if weight is not None:
        if isinstance(weight, paddle.static.Variable):
            weight_name = name if reduction is 'none' else None
            out = paddle.multiply(out, weight, name=weight_name)
        else:
            raise ValueError(
                "The weight is not a Tensor, please convert to Tensor.")

    if reduction == 'sum':
        return paddle.sum(out, name=name)
    elif reduction == 'mean':
        return paddle.mean(out, name=name)
    else:
        return out


def binary_cross_entropy_with_logits(logit,
                                     label,
                                     weight=None,
                                     reduction='mean',
                                     pos_weight=None,
                                     name=None):
    r"""
    This operator combines the sigmoid layer and the :ref:`api_nn_loss_BCELoss` layer.
    Also, we can see it as the combine of ``sigmoid_cross_entropy_with_logits``
    layer and some reduce operations.

    This measures the element-wise probability error in classification tasks
    in which each class is independent.
    This can be thought of as predicting labels for a data-point, where labels
    are not mutually exclusive. For example, a news article can be about
    politics, technology or sports at the same time or none of these.

    First this operator calculate loss function as follows:

    .. math::
           Out = -Labels * \\log(\\sigma(Logit)) - (1 - Labels) * \\log(1 - \\sigma(Logit))

    We know that :math:`\\sigma(Logit) = \\frac{1}{1 + e^{-Logit}}`. By substituting this we get:

    .. math::
           Out = Logit - Logit * Labels + \\log(1 + e^{-Logit})

    For stability and to prevent overflow of :math:`e^{-Logit}` when Logit < 0,
    we reformulate the loss as follows:

    .. math::
           Out = \\max(Logit, 0) - Logit * Labels + \\log(1 + e^{-\|Logit\|})

    Then, if ``weight`` or ``pos_weight`` is not None, this operator multiply the
    weight tensor on the loss `Out`. The ``weight`` tensor will attach different
    weight on every items in the batch. The ``pos_weight`` will attach different
    weight on the positive label of each class.

    Finally, this operator applies reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, the operator will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target labels ``label`` should be numbers between 0 and 1.

    Args:
        logit (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``logit``
            is usually the output of Linear layer. Available dtype is float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``logit``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float32, float64.
        weight (Tensor, optional): A manual rescaling weight given to the loss of each
            batch element. If given, it has to be a 1D Tensor whose size is `[N, ]`,
            The data type is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.
        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector
            with length equal to the number of classes. The data type is float32, float64.
            Default is ``'None'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        output (Tensor): If ``reduction`` is ``'none'``, the shape of output is
            same as ``logit`` , else the shape of output is scalar.

    Examples:

        .. code-block:: python

            import paddle

            logit = paddle.to_tensor([5.0, 1.0, 3.0])
            label = paddle.to_tensor([1.0, 0.0, 1.0])
            output = paddle.nn.functional.binary_cross_entropy_with_logits(logit, label)
            print(output)  # [0.45618808]

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in binary_cross_entropy_with_logits "
            "should be 'sum', 'mean' or 'none', but received %s, which is not allowed."
            % reduction)

    if in_dygraph_mode():
        one = _varbase_creator(dtype=logit.dtype)
        core.ops.fill_constant(one, 'value',
                               float(1.0), 'force_cpu', False, 'dtype',
                               one.dtype, 'str_value', '1.0', 'shape', [1])
        out = core.ops.sigmoid_cross_entropy_with_logits(logit, label)
        if pos_weight is not None:
            log_weight = core.ops.elementwise_add(
                core.ops.elementwise_mul(
                    label, core.ops.elementwise_sub(pos_weight, one)), one)
            out = core.ops.elementwise_mul(out, log_weight)
        if weight is not None:
            out = core.ops.elementwise_mul(out, weight)

        if reduction == "sum":
            return core.ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == "mean":
            return core.ops.mean(out)
        else:
            return out

    fluid.data_feeder.check_variable_and_dtype(
        logit, 'logit', ['float32', 'float64'],
        'binary_cross_entropy_with_logits')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64'],
        'binary_cross_entropy_with_logits')
    sigmoid_name = None
    if reduction == 'none' and pos_weight is None and weight is None:
        sigmoid_name = name

    out = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
        logit, label, name=sigmoid_name)

    one = paddle.fluid.layers.fill_constant(
        shape=[1], value=1.0, dtype=logit.dtype)
    if pos_weight is not None:
        fluid.data_feeder.check_variable_and_dtype(
            pos_weight, 'pos_weight', ['float32', 'float64'],
            'binary_cross_entropy_with_logits')
        log_weight = paddle.add(
            paddle.multiply(
                label, paddle.fluid.layers.elementwise_sub(pos_weight, one)),
            one)
        pos_weight_name = name if reduction == 'none' and weight is None else None
        out = paddle.multiply(out, log_weight, name=pos_weight_name)

    if weight is not None:
        fluid.data_feeder.check_variable_and_dtype(
            weight, 'weight', ['float32', 'float64'],
            'binary_cross_entropy_with_logits')
        weight_name = name if reduction == 'none' else None
        out = paddle.multiply(out, weight, name=weight_name)

    if reduction == "sum":
        return paddle.sum(out, name=name)
    elif reduction == "mean":
        return paddle.mean(out, name=name)
    return out


def hsigmoid_loss(input,
                  label,
                  num_classes,
                  weight,
                  bias=None,
                  path_table=None,
                  path_code=None,
                  is_sparse=False,
                  name=None):
    """
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        input (Tensor): A tensor with the shape [N, D], where N is the size of mini-batch,
            and D is the feature size. Its data type supports float32 or float64.
        label (Tensor): A tensor contains the labels of training data. Its shape is [N, 1]
            and data type is int64.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (path_code and path_table is None are None), `num_classes`
            should not be None. If the custom tree is used (path_code and path_table is None are not None),
            `num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        weight (Tensor): A tensor with shape (num_classes - 1, D), with the same data type as `input`.
        bias (Tensor, optional): A tensor with shape (num_classes - 1, 1), with the same data type as `input`.
            If `bias` is None, no bias will be add. Default is None.
        path_table (Tensor, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. If `path_table` and `path_code` are None, the default tree will be used.
            Default is None.
        path_code (Tensor, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. If `path_table` and
            `path_code` are None, the default tree will be used. Default is None.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating. If `is_sparse` is True,
            the gradient of `weight` and `input` will be sparse. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as `input`.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F

            paddle.set_device('cpu')

            input = paddle.uniform([2, 3])
            # [[-0.8018668   0.8736385  -0.9064771 ] # random
            #  [-0.10228515 -0.87188244 -0.8783718 ]] # random
            label = paddle.to_tensor([0, 1, 4, 5])
            num_classes = 5
            weight=paddle.uniform([num_classes-1, 3])
            # [[-0.24148715  0.8449961  -0.7399121 ] # random
            #  [-0.9800559   0.43509364  0.9091208 ] # random
            #  [ 0.60194826  0.10430074 -0.4521166 ] # random
            #  [-0.4469818  -0.01536179 -0.604454  ]] # random

            out=F.hsigmoid_loss(input, label, num_classes, weight)
            # [[3.0159328]
            #  [2.2407534]]
    """

    if in_dygraph_mode():
        out, _, _ = core.ops.hierarchical_sigmoid(
            input, weight, label, path_table, path_code, bias, 'num_classes',
            num_classes, 'is_sparse', is_sparse, 'remote_prefetch', is_sparse)
        return out

    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'hsigmoid_loss')
    check_variable_and_dtype(label, 'label', ['int64'], 'hsigmoid_loss')
    check_variable_and_dtype(weight, 'weight', ['float32', 'float64'],
                             'hsigmoid_loss')
    if bias is not None:
        check_variable_and_dtype(bias, 'bias', ['float32', 'float64'],
                                 'hsigmoid_loss')
    if path_table is not None:
        check_variable_and_dtype(path_table, 'path_table', ['int64'],
                                 'hsigmoid_loss')
    if path_code is not None:
        check_variable_and_dtype(path_code, 'path_code', ['int64'],
                                 'hsigmoid_loss')

    attrs = {
        "num_classes": num_classes,
        "is_sparse": is_sparse,
        "remote_prefetch": is_sparse
    }

    inputs = {
        "X": input,
        "W": weight,
        "Bias": bias,
        "PathTable": path_table,
        "PathCode": path_code,
        "Label": label
    }

    helper = LayerHelper('hsigmoid_loss', **locals())
    out = helper.create_variable_for_type_inference(input.dtype)
    pre_out = helper.create_variable_for_type_inference(input.dtype)
    outputs = {"Out": out, "PreOut": pre_out, "W_Out": weight}

    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)
    return out


def smooth_l1_loss(input, label, reduction='mean', delta=1.0, name=None):
    r"""
    This operator calculates smooth_l1_loss. Creates a criterion that uses a squared
    term if the absolute element-wise error falls below 1 and an L1 term otherwise.
    In some cases it can prevent exploding gradients and it is more robust and less
    sensitivity to outliers. Also known as the Huber loss:

    .. math::

         loss(x,y) = \\frac{1}{n}\\sum_{i}z_i


    where z_i is given by:

    .. math::

         \\mathop{z_i} = \\left\\{\\begin{array}{rcl}
        0.5(x_i - y_i)^2 & & {if |x_i - y_i| < delta} \\\\
        delta * |x_i - y_i| - 0.5 * delta^2 & & {otherwise}
        \\end{array} \\right.

    Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is
            (N, C), where C is number of classes, and if shape is more than 2D, this
            is (N, C, D1, D2,..., Dk), k >= 1.
        label (Tensor): Label tensor, the data type is float32 or float64. The shape of label
            is the same as the shape of input.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        delta (float, optional): Specifies the hyperparameter delta to be used.
            The value determines how large the errors need to be to use L1. Errors
            smaller than delta are minimized with L2. Parameter is ignored for
            negative/zero values. Default = 1.0
        name (str, optional): Name for the operation (optional, default is
            None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        The tensor variable storing the smooth_l1_loss of input and label.

    Return type: Tensor.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            input_data = np.random.rand(3,3).astype("float32")
            label_data = np.random.rand(3,3).astype("float32")
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            output = paddle.nn.functional.smooth_l1_loss(input, label)
            print(output)
    """
    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'smooth_l1_loss')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'smooth_l1_loss')

    out = huber_loss(input=input, label=label, delta=delta)

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in smooth_l1_loss should be 'sum', 'mean' or"
            " 'none', but received %s, which is not allowed." % reduction)
    if reduction == 'none':
        return out
    elif reduction == 'mean':
        return fluid.layers.reduce_mean(out)
    elif reduction == 'sum':
        return fluid.layers.reduce_sum(out)


def margin_ranking_loss(input,
                        other,
                        label,
                        margin=0.0,
                        reduction='mean',
                        name=None):
    r"""

    This op the calcluate the the margin rank loss between the input, other and label, use the math function as follows.

    .. math::
        margin\_rank\_loss = max(0, -label * (input - other) + margin)

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(margin\_rank\_loss)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(margin\_rank\_loss)

    If :attr:`reduction` set to ``'none'``, just return the origin ``margin_rank_loss``.

    Parameters:
        input(Tensor): the first input tensor, it's data type should be float32, float64.
        other(Tensor): the second input tensor, it's data type should be float32, float64.
        label(Tensor): the label value corresponding to input, it's data type should be float32, float64.
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candicates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns: Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[1]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Examples:

        .. code-block:: python

            import paddle

            input = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
            other = paddle.to_tensor([[2, 1], [2, 4]], dtype='float32')
            label = paddle.to_tensor([[1, -1], [-1, -1]], dtype='float32')
            loss = paddle.nn.functional.margin_ranking_loss(input, other, label)
            print(loss) # [0.75]
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in MarginRankingLoss should be 'sum', 'mean' or 'none', but "
            "received %s, which is not allowed." % reduction)
    if fluid.framework.in_dygraph_mode():
        out = core.ops.elementwise_sub(other, input)
        out = core.ops.elementwise_mul(out, label)
        if margin != 0.0:
            margin = fluid.dygraph.base.to_variable([margin], dtype=out.dtype)
            out = core.ops.elementwise_add(out, margin)
        out = core.ops.relu(out)
        if reduction == 'sum':
            return core.ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == 'mean':
            return core.ops.mean(out)
        return out

    helper = LayerHelper("margin_ranking_loss", **locals())
    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        other, 'other', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'margin_rank_loss')

    out = paddle.fluid.layers.elementwise_sub(other, input)
    out = paddle.multiply(out, label)

    if margin != 0.0:
        margin_var = out.block.create_var(dtype=out.dtype)
        paddle.fluid.layers.fill_constant(
            [1], out.dtype, margin, out=margin_var)
        out = paddle.add(out, margin_var)

    result_out = helper.create_variable_for_type_inference(input.dtype)

    if reduction == 'none':
        helper.append_op(
            type="relu", inputs={"X": out}, outputs={"Out": result_out})
        return result_out
    elif reduction == 'sum':
        out = paddle.nn.functional.relu(out)
        attrs = {"dim": [0], "keep_dim": False, "reduce_all": True}
        helper.append_op(
            type="reduce_sum",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs=attrs)
        return result_out
    elif reduction == 'mean':
        out = paddle.nn.functional.relu(out)
        helper.append_op(
            type="mean",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs={})
        return result_out


def l1_loss(input, label, reduction='mean', name=None):
    r"""
    This operator computes the L1 Loss of Tensor ``input`` and ``label`` as follows.

    If `reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \\lvert input - label \\rvert

    If `reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\\lvert input - label \\rvert)

    If `reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\\lvert input - label\\rvert)


    Parameters:
        input (Tensor): The input tensor. The shapes is [N, `*`], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, `*`], same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, the L1 Loss of Tensor ``input`` and ``label``.
            If `reduction` is ``'none'``, the shape of output loss is [N, *], the same as ``input`` .
            If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1].

    Examples:
        .. code-block:: python

            import paddle

            paddle.disable_static()
            input = paddle.to_tensor([[1.5, 0.8], [0.2, 1.3]])
            label = paddle.to_tensor([[1.7, 1], [0.4, 0.5]])

            l1_loss = paddle.nn.functional.l1_loss(input, label)
            print(l1_loss.numpy())
            # [0.35]

            l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='none')
            print(l1_loss.numpy())
            # [[0.20000005 0.19999999]
            # [0.2        0.79999995]]

            l1_loss = paddle.nn.functional.l1_loss(input, label, reduction='sum')
            print(l1_loss.numpy())
            # [1.4]
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
            "received %s, which is not allowed." % reduction)

    if in_dygraph_mode():
        unreduced = _elementwise_op_in_dygraph(
            input, label, axis=-1, act='abs', op_name='elementwise_sub')
        if reduction == 'mean':
            return core.ops.mean(unreduced)
        elif reduction == 'sum':
            return core.ops.reduce_sum(unreduced, 'dim', [0], 'keep_dim', False,
                                       'reduce_all', True)
        else:
            return unreduced

    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')

    if reduction == 'sum':
        unreduced = paddle.fluid.layers.elementwise_sub(input, label, act='abs')
        return paddle.sum(unreduced, name=name)
    elif reduction == 'mean':
        unreduced = paddle.fluid.layers.elementwise_sub(input, label, act='abs')
        return paddle.mean(unreduced, name=name)
    else:
        return paddle.fluid.layers.elementwise_sub(
            input, label, act='abs', name=name)


def nll_loss(input,
             label,
             weight=None,
             ignore_index=-100,
             reduction='mean',
             name=None):
    """
    This api returns negative log likelihood.
    See more detail in :ref:`api_nn_loss_NLLLoss` .

    Parameters:
         input (Tensor): Input tensor, the shape is :math:`[N, C]`, `C` is the number of classes.
             But in K-dimension situation, the shape is :math:`[N, C, d_1, d_2, ..., d_K]`.
             The data type is float32, float64.
         label (Tensor): Label tensor, the shape is :math:`[N,]` or :math:`[N, d_1, d_2, ..., d_K]`.
             The data type is int64.
         weight (Tensor, optional): Weight tensor, a manual rescaling weight given
             to each class. If given, it has to be a 1D Tensor whose size is `[C, ]`. Otherwise,
             it treated as if having all ones. the data type is
             float32, float64, Default is ``'None'``.
         ignore_index (int64, optional): Specifies a target value that is ignored
             and does not contribute to the input gradient.
         reduction (str, optional): Indicate how to average the loss,
             the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
             If `reduction` is ``'mean'``, the reduced mean loss is returned;
             if `reduction` is ``'sum'``, the reduced sum loss is returned;
             if `reduction` is ``'none'``, no reduction will be apllied.
             Default is ``'mean'``.
         name (str, optional): Name for the operation (optional, default is None).
             For more information, please refer to :ref:`api_guide_Name`.

    Returns:
         `Tensor`, the value of negative log likelihood loss.

    Examples:
        .. code-block:: python

                import paddle
                from paddle.nn.functional import nll_loss
                log_softmax = paddle.nn.LogSoftmax(axis=1)

                input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
                          [0.53331435, 0.07999352, 0.8549948 ],
                          [0.25879037, 0.39530203, 0.698465  ],
                          [0.73427284, 0.63575995, 0.18827209],
                          [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
                log_out = log_softmax(input)
                label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
                result = nll_loss(log_out, label)
                print(result) # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True, [1.07202101])
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
            "'none', but received %s, which is not allowed." % reduction)

    input_shape = list(input.shape)
    input_dims = len(input_shape)
    if input_dims < 2:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(
            input_dims))
    n = input_shape[0]
    c = input_shape[1]
    if in_dygraph_mode():
        if input_dims != 2 and input_dims != 4:
            input, _ = core.ops.reshape2(input, 'shape', [n, c, 1, -1])
            label, _ = core.ops.reshape2(label, 'shape', [n, 1, -1])
            out_shape = [n] + input_shape[2:]
        out, total_weight = core.ops.nll_loss(input, label, weight,
                                              'ignore_index', ignore_index,
                                              'reduction', reduction)
        if input_dims != 2 and input_dims != 4 and reduction == 'none':
            out, _ = core.ops.reshape2(out, 'shape', out_shape)
        return out

    helper = LayerHelper('nll_loss', **locals())

    if input_dims != 2 and input_dims != 4:
        input = reshape(input, shape=[n, c, 1, -1])
        label = reshape(label, shape=[n, 1, -1])
        out_shape = [n] + input_shape[2:]

    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'nll_loss')
    fluid.data_feeder.check_variable_and_dtype(label, 'label', ['int64'],
                                               'nll_loss')
    inputs = {'X': input, 'Label': label}
    attrs = {'reduction': reduction, 'ignore_index': ignore_index}
    if weight is not None:
        if isinstance(weight, Variable):
            inputs['Weight'] = weight

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    total_weight = helper.create_variable_for_type_inference(dtype=input.dtype)
    outputs = {'Out': out, 'Total_weight': total_weight}

    helper.append_op(
        type='nll_loss', inputs=inputs, outputs=outputs, attrs=attrs)
    if input_dims != 2 and input_dims != 4 and reduction == 'none':
        out = reshape(out, shape=out_shape)

    return out


def kl_div(input, label, reduction='mean', name=None):
    r"""
    This operator calculates the Kullback-Leibler divergence loss
    between Input(X) and Input(Target). Notes that Input(X) is the
    log-probability and Input(Target) is the probability.

    KL divergence loss is calculated as follows:

    $$l(x, y) = y * (\log(y) - x)$$

    While :math:`x` is input and :math:`y` is label.

    While :attr:`reduction` is :attr:`none`, output loss is in
    the same shape as input, loss in each point is calculated
    seperately and no reduction is applied.

    While :attr:`reduction` is :attr:`mean`, output loss is in
    shape of [1] and loss value is the mean value of all losses.

    While :attr:`reduction` is :attr:`sum`, output loss is in
    shape of [1] and loss value is the sum value of all losses.

    While :attr:`reduction` is :attr:`batchmean`, output loss is
    in shape of [1] and loss value is the sum value of all losses
    divided by batch size.

    Args:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means
             any number of additional dimensions. It's data type should be float32, float64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64.
        reduction (Tensor): Indicate how to average the loss,
             the candicates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
             If `reduction` is ``'mean'``, the reduced mean loss is returned;
             If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
             if `reduction` is ``'sum'``, the reduced sum loss is returned;
             if `reduction` is ``'none'``, no reduction will be apllied.
             Default is ``'mean'``.
        name(str, optional): Name for the operation (optional, default is None). For more information,
            please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor: The KL divergence loss. The data type is same as input tensor

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            import paddle.nn.functional as F

            shape = (5, 20)
            input = np.random.uniform(-10, 10, shape).astype('float32')
            target = np.random.uniform(-10, 10, shape).astype('float32')

            # 'batchmean' reduction, loss shape will be [1]
            pred_loss = F.kl_div(paddle.to_tensor(input),
                                 paddle.to_tensor(target), reduction='batchmean')
            # shape=[1]

            # 'mean' reduction, loss shape will be [1]
            pred_loss = F.kl_div(paddle.to_tensor(input),
                                 paddle.to_tensor(target), reduction='mean')
            # shape=[1]

            # 'sum' reduction, loss shape will be [1]
            pred_loss = F.kl_div(paddle.to_tensor(input),
                                 paddle.to_tensor(target), reduction='sum')
            # shape=[1]

            # 'none' reduction, loss shape is same with input shape
            pred_loss = F.kl_div(paddle.to_tensor(input),
                                 paddle.to_tensor(target), reduction='none')
            # shape=[5, 20]

    """
    # ugly type promotion
    if fluid.data_feeder.convert_dtype(
            input.dtype) == 'float32' and fluid.data_feeder.convert_dtype(
                label.dtype) == 'float64':
        input = fluid.layers.cast(input, 'float64')
    elif fluid.data_feeder.convert_dtype(
            input.dtype) == 'float64' and fluid.data_feeder.convert_dtype(
                label.dtype) == 'float32':
        label = fluid.layers.cast(label, 'float64')

    if paddle.in_dynamic_mode():
        out = core.ops.kldiv_loss(input, label, 'reduction', reduction)
        return out

    helper = LayerHelper('kl_div', **locals())

    fluid.data_feeder.check_variable_and_dtype(input, 'input',
                                               ['float32', 'float64'], 'kl_div')
    fluid.data_feeder.check_variable_and_dtype(label, 'label',
                                               ['float32', 'float64'], 'kl_div')
    fluid.data_feeder.check_type(reduction, 'reduction', str, 'kl_div')

    loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='kldiv_loss',
        inputs={'X': input,
                'Target': label},
        outputs={'Loss': loss},
        attrs={'reduction': reduction})
    return loss


def mse_loss(input, label, reduction='mean', name=None):
    r"""
    This op accepts input predications and label and returns the mean square error.

    If :attr:`reduction` is set to ``'none'``, loss is calculated as:

    .. math::
        Out = (input - label)^2

    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:

    .. math::
        Out = \operatorname{mean}((input - label)^2)

    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:

    .. math::
        Out = \operatorname{sum}((input - label)^2)

    Parameters:
        input (Tensor): Input tensor, the data type should be float32 or float64.
        label (Tensor): Label tensor, the data type should be float32 or float64.
        reduction (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned.
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.


    Returns:
        Tensor: The tensor tensor storing the mean square error difference of input and label.

    Return type: Tensor.

    Examples:

        .. code-block:: python

            import paddle
            mse_loss = paddle.nn.loss.MSELoss()
            input = paddle.to_tensor(1.5)
            label = paddle.to_tensor(1.7)
            output = mse_loss(input, label)
            print(output)
            # [0.04000002]

    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "'reduction' in 'mse_loss' should be 'sum', 'mean' or 'none', "
            "but received {}.".format(reduction))

    if not paddle.fluid.framework.in_dygraph_mode():
        paddle.fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'mse_loss')
        paddle.fluid.data_feeder.check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'mse_loss')

    if reduction == 'none':
        return paddle.fluid.layers.square(
            paddle.fluid.layers.elementwise_sub(input, label), name=name)
    elif reduction == 'mean':
        return paddle.mean(
            paddle.fluid.layers.square(
                paddle.fluid.layers.elementwise_sub(input, label)),
            name=name)
    else:
        return paddle.sum(paddle.fluid.layers.square(
            paddle.fluid.layers.elementwise_sub(input, label)),
                          name=name)


def ctc_loss(log_probs,
             labels,
             input_lengths,
             label_lengths,
             blank=0,
             reduction='mean'):
    """

    An operator integrating the open source Warp-CTC library (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation
    is interated to the Warp-CTC library to normalize values for each row of the input tensor.

    Parameters:
        log_probs (Tensor): The unscaled probability sequence with padding, which is a 3-D Tensor. The tensor shape is [max_logit_length, batch_size, num_classes + 1], where max_logit_length is the longest length of input logit sequence. The data type should be float32 or float64.
        labels (Tensor): The ground truth sequence with padding, which must be a 3-D Tensor. The tensor shape is [batch_size, max_label_length], where max_label_length is the longest length of label sequence. The data type must be int32.
        input_lengths (Tensor): The length for each input sequence, it should have shape [batch_size] and dtype int64.
        label_lengths (Tensor): The length for each label sequence, it should have shape [batch_size] and dtype int64.
        blank (int, optional): The blank label index of Connectionist Temporal Classification (CTC) loss, which is in the half-opened interval [0, num_classes + 1). The data type must be int32. Default is 0.
        reduction (string, optional): Indicate how to average the loss, the candicates are ``'none'`` | ``'mean'`` | ``'sum'``. If :attr:`reduction` is ``'mean'``, the output loss will be divided by the label_lengths, and then return the mean of quotient; If :attr:`reduction` is ``'sum'``, return the sum of loss; If :attr:`reduction` is ``'none'``, no reduction will be applied. Default is ``'mean'``.

    Returns:
        Tensor, The Connectionist Temporal Classification (CTC) loss between ``log_probs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is [1]. Data type is the same as ``log_probs``.

    Examples:

        .. code-block:: python

            # declarative mode
            import paddle.nn.functional as F
            import numpy as np
            import paddle

            # length of the longest logit sequence
            max_seq_length = 4
            #length of the longest label sequence
            max_label_length = 3
            # number of logit sequences
            batch_size = 2
            # class num
            class_num = 3

            np.random.seed(1)
            log_probs = np.array([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
                                    [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],

                                    [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
                                    [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],

                                    [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
                                    [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],

                                    [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
                                    [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],

                                    [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
                                    [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]]).astype("float32")
            labels = np.array([[1, 2, 2],
                            [1, 2, 2]]).astype("int32")
            input_lengths = np.array([5, 5]).astype("int64")
            label_lengths = np.array([3, 3]).astype("int64")

            log_probs = paddle.to_tensor(log_probs)
            labels = paddle.to_tensor(labels)
            input_lengths = paddle.to_tensor(input_lengths)
            label_lengths = paddle.to_tensor(label_lengths)

            loss = F.ctc_loss(log_probs, labels,
                input_lengths,
                label_lengths,
                blank=0,
                reduction='none')
            print(loss)  #[3.9179852 2.9076521]

            loss = F.ctc_loss(log_probs, labels,
                input_lengths,
                label_lengths,
                blank=0,
                reduction='mean')
            print(loss)  #[1.1376063]

    """

    loss_out = fluid.layers.warpctc(log_probs, labels, blank, False,
                                    input_lengths, label_lengths)

    loss_out = fluid.layers.squeeze(loss_out, [-1])
    assert reduction in ['mean', 'sum', 'none']
    if reduction == 'mean':
        loss_out = paddle.mean(loss_out / label_lengths)
    elif reduction == 'sum':
        loss_out = paddle.sum(loss_out)
    return loss_out


def cross_entropy(input,
                  label,
                  weight=None,
                  ignore_index=-100,
                  reduction='mean',
                  soft_label=False,
                  axis=-1,
                  name=None):
    r"""
    This operator implements the cross entropy loss function with softmax. This function 
    combines the calculation of the softmax operation and the cross entropy loss function 
    to provide a more numerically stable gradient.
    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators 
    expects mutually exclusive hard labels, each sample in a batch is in exactly 
    one class with a probability of 1.0. Each sample in the batch will have a 
    single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::

        loss_j =  -\\text{logits}_{label_j} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{logits}_i)\\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::

        loss_j =  -\\sum_{i=0}^{K}\\text{label}_i
        \\left(\\text{logits}_i - \\log\\left(\\sum_{i=0}^{K}
        \\exp(\\text{logits}_i)\\right)\\right), j = 1,...,K

 
    It is useful when training a classification problem with ``C`` classes.


    Parameters:
        input (Tensor): Input tensor, the data type is float32, float64. Shape is
	    (N, C), where C is number of classes, and if shape is more than 2D, this
	    is (N, D1, D2,..., Dk, C), k >= 1.
        label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
	    value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
	    (N, D1, D2,..., Dk), k >= 1.
        weight (Tensor, optional):a manual rescaling weight given to each class. 
            If given, has to be a Tensor of size C and the data type is float32, float64. 
            Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default is ``-100``.
        soft_label (bool): indicate whether label is soft. Default False, meaning that
                the label is hard. If soft_label=True, the label is soft.
        axis (int, optional): The index of dimension to perform softmax calculations. It 
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.


    Returns:
        Tensor.The tensor storing the cross_entropy_loss of input and label.


    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            input_data = np.random.random([5, 100]).astype("float64")
            label_data = np.random.randint(0, 100, size=(5)).astype(np.int64)
            weight_data = np.random.random([100]).astype("float64")

            input =  paddle.to_tensor(input_data)
            label =  paddle.to_tensor(label_data)
            weight = paddle.to_tensor(weight_data)

            loss = paddle.nn.functional.cross_entropy(input=input, label=label, weight=weight)
            print(loss)
            # [4.28546723]
    """

    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in softmax_cross_entropy"
            "should be 'sum', 'mean' or 'none', but received %s, which is not allowed."
            % reduction)
    input_dims = len(list(input.shape))
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            'Expected nput_dims - 1 = label_dims or input_dims == label_dims\
             (got nput_dims{}, label_dims{})'.format(input_dims, label_dims))
    if input_dims - 1 == label_dims:
        label = paddle.unsqueeze(label, axis=axis)
    if in_dygraph_mode():
        out = softmax_with_cross_entropy(
            input,
            label,
            soft_label=soft_label,
            ignore_index=ignore_index,
            axis=axis)  #N,1
        if weight is not None:
            weight_gather = core.ops.gather_nd(
                weight, label)  #trans weight from class to sample, shape:N
            input_shape = list(label.shape)
            weight_gather_reshape, _ = core.ops.reshape2(weight_gather, 'shape',
                                                         input_shape)  #N,1
            out = core.ops.elementwise_mul(out, weight_gather_reshape)

        if reduction == "sum":
            #   because of softmax_with_cross_entropy op's inner logic, 
            #   in the out tensor of this op, the loss of sample with class_index==ignore_index is 0
            #   so, reduce_sum all directly is ok
            return core.ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == "mean":
            #1. if weight==none, 
            #    numerator: reduce_sum all loss directly is ok causeof softmax_with_cross_entropy's inner logic
            #    denominator: count sample num with class_index!=ignore_index
            #2. else
            #    numerator: loss's weighted sum 
            #    denominator: cal the sum of weight where the sample's class_index!=ignore_index
            out_sum = core.ops.reduce_sum(out, 'reduce_all', True)
            #for each label[i],set 1 or 0, according to ignore_index
            ones_t = paddle.ones_like(label)
            zeros_t = paddle.zeros_like(label)
            #mask[i]=0, if label[i]==ignore_index
            #mask[i]=1, otherwise 
            mask = paddle.where(label != ignore_index, ones_t, zeros_t)
            if (weight is None):
                count = core.ops.reduce_sum(mask, 'reduce_all', True)
                ret = out_sum / count
            else:
                mask = paddle.cast(mask, "float64")
                weight_gather_reshape = paddle.cast(weight_gather_reshape,
                                                    "float64")
                weight_ignored = core.ops.elementwise_mul(mask,
                                                          weight_gather_reshape)
                weight_sum = core.ops.reduce_sum(weight_ignored, 'reduce_all',
                                                 True)
                ret = out_sum / weight_sum
            return ret
        else:
            if input_dims - 1 == label_dims:
                out = paddle.squeeze(out, axis=axis)
            return out

    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'softmax_cross_entropy')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['int32', 'int64'], 'softmax_cross_entropy')
    out = softmax_with_cross_entropy(
        input,
        label,
        soft_label=soft_label,
        ignore_index=ignore_index,
        axis=axis)
    if weight is not None:
        fluid.data_feeder.check_variable_and_dtype(
            weight, 'weight', ['float32', 'float64'], 'softmax_cross_entropy')
        weight_name = name if reduction == 'none' else None
        weight_gather = paddle.gather_nd(
            weight, label)  #trans weight from class to sample, shape:N
        input_shape = list(label.shape)
        weight_gather_reshape = reshape(weight_gather, shape=input_shape)
        out = paddle.multiply(out, weight_gather_reshape, name=weight_name)

    if reduction == "sum":
        return paddle.sum(out, name=name)
    elif reduction == "mean":
        out_sum = paddle.sum(out, name=name)
        #for each label[i],set 1 or 0, according to ignore_index
        ones_t = paddle.ones_like(label)
        zeros_t = paddle.zeros_like(label)
        #mask[i]=0, if label[i]==ignore_index
        #mask[i]=1, otherwise 
        mask = paddle.where(label != ignore_index, ones_t, zeros_t)
        if (weight is None):
            count = paddle.sum(mask, name=name)
            ret = out_sum / count
        else:
            mask = paddle.cast(mask, "float64")
            weight_gather_reshape = paddle.cast(weight_gather_reshape,
                                                "float64")
            weight_ignored = paddle.multiply(mask, weight_gather_reshape)
            weight_sum = paddle.sum(weight_ignored, name=name)
            ret = out_sum / weight_sum
        return ret
    else:
        if input_dims - 1 == label_dims:
            out = paddle.squeeze(out, axis=axis)

        return out


def sigmoid_focal_loss(logit,
                       label,
                       normalizer=None,
                       alpha=0.25,
                       gamma=2.0,
                       reduction='sum',
                       name=None):
    r"""
    `Focal Loss <https://arxiv.org/abs/1708.02002>`_ is proposed to address the
    foreground-background class imbalance for classification tasks. It down-weights
    easily-classified examples and thus focuses training on hard examples. For example,
    it is used in one-stage object detection where the foreground-background class
    imbalance is extremely high.

    This operator measures focal loss function as follows: 

    .. math::
           Out = -Labels * alpha * {(1 - \\sigma(Logit))}^{gamma}\\log(\\sigma(Logit)) - (1 - Labels) * (1 - alpha) * {\\sigma(Logit)}^{gamma}\\log(1 - \\sigma(Logit))

    We know that :math:`\\sigma(Logit) = \\frac{1}{1 + \\exp(-Logit)}`. 

    Then, if :attr:`normalizer` is not None, this operator divides the
    normalizer tensor on the loss `Out`:

    .. math::
           Out = \\frac{Out}{normalizer}

    Finally, this operator applies reduce operation on the loss.
    If :attr:`reduction` set to ``'none'``, the operator will return the original loss `Out`.
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is :math:`Out = MEAN(Out)`.
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is :math:`Out = SUM(Out)`.

    Note that the target ``label`` is 0 for the negative class and is 1 for the positive class.

    Args:
        logit (Tensor): The input logit tensor. The shape is [N, *], where N is batch_size,
            `*` means any number of additional dimensions. The ``logit`` is usually the
            output of a convolution layer. Available dtype is float32, float64.
        label (Tensor): The target label tensor with the same shape as
            ``logit``. The target label whose value should be numbers between 0 and 1.
            Available dtype is float32, float64.
        normalizer (Tensor, optional): The number normalizes the focal loss. It has to be
            a 1-D Tensor whose shape is `[1, ]`. The data type is float32, float64.
            For object detection task, it is the the number of positive samples.
            If set to None, the focal loss will not be normalized. Default is None.
        alpha(int|float, optional): Hyper-parameter to balance the positive and negative example,
            it should be between 0 and 1.  Default value is set to 0.25. 
        gamma(int|float, optional): Hyper-parameter to modulate the easy and hard examples.
            Default value is set to 2.0.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'sum'``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[1]`, otherwise the shape is the same as ``logit``. The same dtype as ``logit`` tensor.

    Examples:

        .. code-block:: python

            import paddle

            logit = paddle.to_tensor([[0.97, 0.91, 0.03], [0.55, 0.43, 0.71]], dtype='float32')
            label = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32')
            one = paddle.to_tensor([1.], dtype='float32')
            fg_label = paddle.greater_equal(label, one)
            fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32'))
            output = paddle.nn.functional.sigmoid_focal_loss(logit, label, normalizer=fg_num)
            print(output)  # [0.65782464]

    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in sigmoid_focal_loss "
            "should be 'sum', 'mean' or 'none', but received %s, which is not allowed."
            % reduction)

    if normalizer is not None:
        fluid.data_feeder.check_variable_and_dtype(normalizer, 'normalizer',
                                                   ['float32', 'float64'],
                                                   'sigmoid_focal_loss')
        normalizer_shape = list(normalizer.shape)
        normalizer_dims = len(normalizer_shape)
        if normalizer_dims > 1:
            raise ValueError(
                "Expected one dimension of normalizer in sigmoid_focal_loss but got {}.".
                format(normalizer_dims))

    if in_dygraph_mode():
        one = _varbase_creator(dtype=logit.dtype)
        core.ops.fill_constant(one, 'value',
                               float(1.0), 'force_cpu', False, 'dtype',
                               one.dtype, 'str_value', '1.0', 'shape',
                               logit.shape)
        loss = core.ops.sigmoid_cross_entropy_with_logits(logit, label)
        pred = core.ops.sigmoid(logit)
        p_t = core.ops.elementwise_add(
            core.ops.elementwise_mul(pred, label),
            core.ops.elementwise_mul(
                core.ops.elementwise_sub(one, pred),
                core.ops.elementwise_sub(one, label)))

        alpha = fluid.dygraph.base.to_variable([alpha], dtype=loss.dtype)
        alpha_t = core.ops.elementwise_add(
            core.ops.elementwise_mul(alpha, label),
            core.ops.elementwise_mul(
                core.ops.elementwise_sub(one, alpha),
                core.ops.elementwise_sub(one, label)))
        loss = core.ops.elementwise_mul(alpha_t, loss)

        gamma = fluid.dygraph.base.to_variable([gamma], dtype=loss.dtype)
        gamma_t = core.ops.elementwise_pow(
            core.ops.elementwise_sub(one, p_t), gamma)
        loss = core.ops.elementwise_mul(gamma_t, loss)

        if normalizer is not None:
            loss = core.ops.elementwise_div(loss, normalizer)

        if reduction == "sum":
            return core.ops.reduce_sum(loss, 'reduce_all', True)
        elif reduction == "mean":
            return core.ops.mean(loss)

        return loss

    fluid.data_feeder.check_variable_and_dtype(
        logit, 'logit', ['float32', 'float64'], 'sigmoid_focal_loss')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64'], 'sigmoid_focal_loss')

    bce_name = None
    if reduction == 'none' and normalizer is None:
        bce_name = name
    loss = paddle.nn.functional.binary_cross_entropy_with_logits(
        logit, label, reduction='none', name=bce_name)

    pred = fluid.layers.sigmoid(logit)
    p_t = pred * label + (1 - pred) * (1 - label)

    alpha_t = alpha * label + (1 - alpha) * (1 - label)
    loss = paddle.multiply(alpha_t, loss)

    gamma_t = paddle.pow((1 - p_t), gamma)
    loss = paddle.multiply(gamma_t, loss)

    if normalizer is not None:
        normalizer_name = name if reduction == 'none' else None
        loss = paddle.divide(loss, normalizer, name=normalizer_name)

    if reduction == 'mean':
        loss = paddle.mean(loss, name=name)
    elif reduction == 'sum':
        loss = paddle.sum(loss, name=name)

    return loss
