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

# TODO: define loss functions of neural network
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle
from .. import functional as F
from paddle.fluid.framework import core, in_dygraph_mode, _varbase_creator

__all__ = [
    'BCEWithLogitsLoss',
    'CrossEntropyLoss',
    'HSigmoidLoss',
    'MSELoss',
    'L1Loss',
    'NLLLoss',
    'BCELoss',
    'KLDivLoss',
    'MarginRankingLoss',
    'CTCLoss',
    'SmoothL1Loss',
]


class BCEWithLogitsLoss(fluid.dygraph.Layer):
    """
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

    We know that :math:`\\sigma(Logit) = \\frac{1}{1 + \\e^{-Logit}}`. By substituting this we get:

    .. math::
           Out = Logit - Logit * Labels + \\log(1 + \\e^{-Logit})

    For stability and to prevent overflow of :math:`\\e^{-Logit}` when Logit < 0,
    we reformulate the loss as follows:

    .. math::
           Out = \\max(Logit, 0) - Logit * Labels + \\log(1 + \\e^{-\|Logit\|})

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

    Shapes:
        logit (Tensor): The input predications tensor. 2-D tensor with shape: [N, *],
            N is batch_size, `*` means number of additional dimensions. The ``logit``
            is usually the output of Linear layer. Available dtype is float32, float64.
        label (Tensor): The target labels tensor. 2-D tensor with the same shape as
            ``logit``. The target labels which values should be numbers between 0 and 1.
            Available dtype is float32, float64.
        output (Tensor): If ``reduction`` is ``'none'``, the shape of output is
            same as ``logit`` , else the shape of output is scalar.

    Returns:
        A callable object of BCEWithLogitsLoss.

    Examples:

        .. code-block:: python
            import paddle
            paddle.disable_static()
            logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
            label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
            bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
            output = bce_logit_loss(logit, label)
            print(output.numpy())  # [0.45618808]

    """

    def __init__(self,
                 weight=None,
                 reduction='mean',
                 pos_weight=None,
                 name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in BCEWithLogitsLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)

        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.name = name

    def forward(self, logit, label):
        out = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit, label, self.weight, self.reduction, self.pos_weight,
            self.name)
        return out


class CrossEntropyLoss(fluid.dygraph.Layer):
    """
	:alias_main: paddle.nn.CrossEntropyLoss
	:alias: paddle.nn.CrossEntropyLoss,paddle.nn.layer.CrossEntropyLoss,paddle.nn.layer.loss.CrossEntropyLoss

    This operator implements the cross entropy loss function. This OP combines ``LogSoftmax``,
    and ``NLLLoss`` together.

    It is useful when training a classification problem with ``C`` classes.
    If provided, the optional argument ``weight`` should be a 1D Variable assigning
    weight to each of the classes.

    For predictions label, and target label, the loss is calculated as follows.

    .. math::

        loss_j =  -\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right), j = 1,..., K

    If weight is not ``None``:

    .. math::

        loss_j =  \\text{weight[class]}(-\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right)), j = 1,..., K

    Parameters:
        input (Variable): Input tensor, the data type is float32, float64. Shape is
	    (N, C), where C is number of classes, and if shape is more than 2D, this
	    is (N, C, D1, D2,..., Dk), k >= 1.
        label (Variable): Label tensor, the data type is int64. Shape is (N), where each
	    value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
	    (N, D1, D2,..., Dk), k >= 1.
        weight (Variable, optional): Weight tensor, a manual rescaling weight given
            to each class and the shape is (C). It has the same dimensions as class
	    number and the data type is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default is ``-100``.

    Returns:
        The tensor variable storing the cross_entropy_loss of input and label.

    Return type: Variable.

    Examples:
        .. code-block:: python

            # declarative mode
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            input = fluid.data(name='input', shape=[5, 100], dtype='float64')
            label = fluid.data(name='label', shape=[5], dtype='int64')
            weight = fluid.data(name='weight', shape=[100], dtype='float64')
            ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
            output = ce_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            input_data = np.random.random([5, 100]).astype("float64")
            label_data = np.random.randint(0, 100, size=(5)).astype(np.int64)
            weight_data = np.random.random([100]).astype("float64")
            output = exe.run(fluid.default_main_program(),
                        feed={"input": input_data, "label": label_data,"weight": weight_data},
                        fetch_list=[output],
                        return_numpy=True)
            print(output)

            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                weight = dg.to_variable(weight_data)
                ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
                output = ce_loss(input, label)
                print(output.numpy())
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, label):
        fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'cross_entropy_loss')
        fluid.data_feeder.check_variable_and_dtype(label, 'label', ['int64'],
                                                   'cross_entropy_loss')

        if self.reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in cross_entropy_loss should be 'sum', 'mean' or"
                " 'none', but received %s, which is not allowed." %
                self.reduction)

        return paddle.nn.functional.cross_entropy(
            input,
            label,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)


class HSigmoidLoss(fluid.dygraph.Layer):
    """
    Hierarchical Sigmoid Layer.
    
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>_`. For the custom
    tree, you need to set :attr:`is_custom` to True, and do the following steps (take the language model as an example):

    1. Using a custom word dict to build a binary tree, each leaf node should be an word in the word dict.
    2. Creating a dict map word_id -> path that from the word to the root node, we call it path_table.
    3. Creating a dict map word_id -> code of path that from the word to the root node, we call it path_code.
       Code means the label of each binary classifier, 1 indicate true, 0 indicate false.
    4. Now, each word should has its path and code along the path, you can pass a batch of path and code related
       to the same batch of inputs.

    Parameters:
        feature_size (int): The number of features.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        weight_attr (ParamAttr, optional): The parameter attribute for the learnable weights
            of hsigmoid. If it is set to None or one attribute of ParamAttr, hsigmoid will create a
            ParamAttr as param_attr. If the Initializer of the param_attr is not set, the parameter is
            initialized with Xavier. Default is None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of hsigmoid. If it
            is set to False, no bias will be added. If it is set to None or one attribute of ParamAttr,
            hsigmoid will create a ParamAttr as bias_attr. If the Initializer of the bias_attr is not
            set, the bias is initialized zero. Default is None.
        is_custom (bool, optional): Whether use custom binary tree. If it's True, `path_table` and 
            `path_code` should be passed to its forward method, otherwise `path_table` and `path_code`
            should not be passed to its forward method. Default is False.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True,
            the gradient of weight and input will be sparse. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input (Tensor): The input tensor. The shapes is [N, D], where N is batch size and D is feature size. It's data type should be float32, float64.
        label (Tensor): It's shapes is [N, 1]. It's data type should be int64.
        output (Tensor): The HSigmoid Loss of ``input`` and ``label``. Shape is [N, 1]

    Examples:
        .. code-block:: python

            import paddle
            paddle.set_device('cpu')

            input = paddle.uniform([2, 3])
            # [[-0.2820413   0.9528898  -0.81638825] # random
            #  [-0.6733154  -0.33866507  0.25770962]] # random
            label = paddle.to_tensor([0, 1, 4, 5])
            m = paddle.nn.HSigmoidLoss(3, 5)
            out = m(input, label)
            # [[2.4543471]
            #  [1.9359267]]
    """

    def __init__(self,
                 feature_size,
                 num_classes,
                 weight_attr=None,
                 bias_attr=None,
                 is_custom=False,
                 is_sparse=False,
                 name=None):
        super(HSigmoidLoss, self).__init__()
        if (num_classes < 2) and (not is_custom):
            raise ValueError(
                "num_classes must not be less than 2 with default tree")

        if (not is_custom) and (is_sparse):
            print("Sparse mode should not be used without custom tree")
            is_sparse = False

        self._feature_size = feature_size
        self._num_classes = num_classes
        self._is_custom = is_custom
        self._is_sparse = is_sparse

        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        self._name = name
        self._dtype = paddle.get_default_dtype()

        remote_prefetch = is_sparse
        print("With sparse mode, if your models has only"
              " small parameter prefetch may cause speed down")

        C = self._num_classes if is_custom else self._num_classes - 1
        self.weight = self.create_parameter(
            [C, self._feature_size],
            attr=self._weight_attr,
            is_bias=False,
            dtype=self._dtype)
        self.bias = self.create_parameter(
            [C, 1], attr=self._bias_attr, is_bias=True, dtype=self._dtype)

    def forward(self, input, label, path_table=None, path_code=None):
        out = F.hsigmoid_loss(
            input,
            label,
            self._num_classes,
            self.weight,
            self.bias,
            path_table=path_table,
            path_code=path_code,
            is_sparse=self._is_sparse,
            name=self._name)
        return out


class MSELoss(fluid.dygraph.layers.Layer):
    """
    **Mean Square Error Loss**
    Computes the mean square error (squared L2 norm) of given input and label.

    If :attr:`reduction` is set to ``'none'``, loss is calculated as:

    .. math::
        Out = (input - label)^2

    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:

    .. math::
        Out = \operatorname{mean}((input - label)^2)

    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:

    .. math::
        Out = \operatorname{sum}((input - label)^2)

    where `input` and `label` are `float32` tensors of same shape.

    Parameters:
        reduction (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned.
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.

    Shape:
        input (Tensor): Input tensor, the data type is float32 or float64
        label (Tensor): Label tensor, the data type is float32 or float64
        output (Tensor): output tensor storing the MSE loss of input and label, the data type is same as input.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle

            input_data = np.array([1.5]).astype("float32")
            label_data = np.array([1.7]).astype("float32")

            paddle.disable_static()
            mse_loss = paddle.nn.loss.MSELoss()
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            output = mse_loss(input, label)
            print(output.numpy())
            # [0.04000002]
    """

    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MSELoss' should be 'sum', 'mean' or 'none', "
                "but received {}.".format(reduction))
        self.reduction = reduction

    def forward(self, input, label):
        if not fluid.framework.in_dygraph_mode():
            fluid.data_feeder.check_variable_and_dtype(
                input, 'input', ['float32', 'float64'], 'MSELoss')
            fluid.data_feeder.check_variable_and_dtype(
                label, 'label', ['float32', 'float64'], 'MSELoss')

        square_out = fluid.layers.square(
            fluid.layers.elementwise_sub(input, label))
        if self.reduction == 'none':
            return square_out

        reduce_op = 'reduce_mean'
        if self.reduction == 'sum':
            reduce_op = 'reduce_sum'

        return getattr(fluid.layers, reduce_op)(square_out)


class L1Loss(fluid.dygraph.Layer):
    """
    This interface is used to construct a callable object of the ``L1Loss`` class.
    The L1Loss layer calculates the L1 Loss of ``input`` and ``label`` as follows.

     If `reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \lvert input - label\rvert

    If `reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\lvert input - label\rvert)

    If `reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\lvert input - label\rvert)


    Parameters:
        reduction (str, optional): Indicate the reduction to apply to the loss,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If `reduction` is ``'none'``, the unreduced loss is returned;
            If `reduction` is ``'mean'``, the reduced mean loss is returned.
            If `reduction` is ``'sum'``, the reduced sum loss is returned.
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, *], same shape as ``input`` . It's data type should be float32, float64, int32, int64.
        output (Tensor): The L1 Loss of ``input`` and ``label``.
            If `reduction` is ``'none'``, the shape of output loss is [N, *], the same as ``input`` .
            If `reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1].

    Examples:
        .. code-block:: python
            import paddle
            import numpy as np

            paddle.disable_static()
            input_data = np.array([[1.5, 0.8], [0.2, 1.3]]).astype("float32")
            label_data = np.array([[1.7, 1], [0.4, 0.5]]).astype("float32")
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)

            l1_loss = paddle.nn.loss.L1Loss()
            output = l1_loss(input, label)
            print(output.numpy())
            # [0.35]

            l1_loss = paddle.nn.loss.L1Loss(reduction='sum')
            output = l1_loss(input, label)
            print(output.numpy())
            # [1.4]

            l1_loss = paddle.nn.loss.L1Loss(reduction='none')
            output = l1_loss(input, label)
            print(output.numpy())
            # [[0.20000005 0.19999999]
            # [0.2        0.79999995]]
    """

    def __init__(self, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        return paddle.nn.functional.l1_loss(
            input, label, self.reduction, name=self.name)


class BCELoss(fluid.dygraph.Layer):
    """
    This interface is used to construct a callable object of the ``BCELoss`` class.
    The BCELoss layer measures the binary_cross_entropy loss between input predictions ``input``
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

    Shape:
        input (Tensor): 2-D tensor with shape: [N, *], N is batch_size, `*` means
            number of additional dimensions. The input ``input`` should always
            be the output of sigmod.  Available dtype is float32, float64.
        label (Tensor): 2-D tensor with the same shape as ``input``. The target
            labels which values should be numbers between 0 and 1. Available
            dtype is float32, float64.
        output (Tensor): If ``reduction`` is ``'none'``, the shape of output is
            same as ``input`` , else the shape of output is scalar.

    Returns:
        A callable object of BCELoss.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            input_data = np.array([0.5, 0.6, 0.7]).astype("float32")
            label_data = np.array([1.0, 0.0, 1.0]).astype("float32")

            paddle.disable_static()
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            bce_loss = paddle.nn.loss.BCELoss()
            output = bce_loss(input, label)
            print(output.numpy())  # [0.65537095]

    """

    def __init__(self, weight=None, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in bce_loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)

        super(BCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.name = name

    def forward(self, input, label):
        out = paddle.nn.functional.binary_cross_entropy(
            input, label, self.weight, self.reduction, self.name)
        return out


class NLLLoss(fluid.dygraph.Layer):
    """
	:alias_main: paddle.nn.NLLLoss
	:alias: paddle.nn.NLLLoss,paddle.nn.layer.NLLLoss,paddle.nn.layer.loss.NLLLoss

    This class accepts input and target label and returns negative log likelihood
    cross error. It is useful to train a classification problem with C classes.

    The input for the loss is epected to contain log-probabilities of
    each classes. It has to be a Tensor of size either (batch_size, C) or
    (batch_size, C, d1, d2, ..., dK) with K >= 1 for the K-dimensional case.
    The label for the loss should be a class index in the range [0, C-1]
    where C is the number of classes. If ignore_index is specified, the
    specified target value does not contribute to the input gradient.

    If the optional argument `weight` is provided, it should be a 1D Tensor
    assigning weight to each of the classed. This is particularly useful
    when you have an unbalanced training set.

    The loss is calculated as follows.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \\text{weight}[c] \cdot \mathbb{1}\{c \\not= \\text{ignore\\_index}\},

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \\begin{cases}
            \\sum_{n=1}^N \\frac{1}{\\sum_{n=1}^N w_{y_n}} l_n, &
            \\text{if reduction} = \\text{'mean';}\\\\
            \\sum_{n=1}^N l_n,  &
            \\text{if reduction} = \\text{'sum'.}
        \\end{cases}

    Parameters:
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

    Shape:
        input (Tensor): Input tensor, the shape is :math:`[N, C]`, `C` is the number of classes.
            But in K-dimension situation, the shape is :math:`[N, C, d_1, d_2, ..., d_K]`.
            The data type is float32, float64.
        label (Tensor): Label tensor, the shape is :math:`[N,]` or :math:`[N, d_1, d_2, ..., d_K]`.
            The data type is int64.
        output (Tensor): the `negative log likelihood loss` between input `x` and `label`.
            If `reduction` is `'none'`, the shape is `[N, *]`.
            If `reduction` is `'sum'` or `'mean'`, the shape is `[1]`.

    Examples:
        .. code-block:: python

                import paddle
                import numpy as np

                nll_loss = paddle.nn.layer.NLLLoss()
                log_softmax = paddle.nn.LogSoftmax(axis=1)

                input_np = np.array([[0.88103855, 0.9908683 , 0.6226845 ],
                                 [0.53331435, 0.07999352, 0.8549948 ],
                                 [0.25879037, 0.39530203, 0.698465  ],
                                 [0.73427284, 0.63575995, 0.18827209],
                                 [0.05689114, 0.0862954 , 0.6325046 ]]).astype(np.float32)
                label_np = np.array([0, 2, 1, 1, 0]).astype(np.int64)

                place = paddle.CPUPlace()
                paddle.disable_static(place)
                input = paddle.to_tensor(input_np)
                log_out = log_softmax(input)
                label = paddle.to_tensor(label_np)
                result = nll_loss(log_out, label)
                print(result.numpy()) # [1.0720209]

    """

    def __init__(self,
                 weight=None,
                 ignore_index=-100,
                 reduction='mean',
                 name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in nll_loss should be 'sum', 'mean' or "
                "'none', but received %s, which is not allowed." % reduction)
        super(NLLLoss, self).__init__()
        self._weight = weight
        self._ignore_index = ignore_index
        self._reduction = reduction
        self._name = name

    def forward(self, input, label):
        return F.nll_loss(
            input,
            label,
            weight=self._weight,
            ignore_index=self._ignore_index,
            reduction=self._reduction,
            name=self._name)


class KLDivLoss(fluid.dygraph.Layer):
    """
    This interface calculates the Kullback-Leibler divergence loss
    between Input(X) and Input(Target). Notes that Input(X) is the
    log-probability and Input(Target) is the probability.

    KL divergence loss is calculated as follows:

    $$l(x, y) = y * (\log(y) - x)$$

    Parameters:
        reduction (Tensor): Indicate how to average the loss,
             the candicates are ``'none'`` | ``'batchmean'`` | ``'mean'`` | ``'sum'``.
             If `reduction` is ``'mean'``, the reduced mean loss is returned;
             If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
             if `reduction` is ``'sum'``, the reduced sum loss is returned;
             if `reduction` is ``'none'``, no reduction will be apllied.
             Default is ``'mean'``.

    Shape:

        - input (Tensor): (N, *), where * means, any number of additional dimensions.

        - label (Tensor): (N, *), same shape as input.

        - output (Tensor): tensor with shape: [1] by default.


    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            import paddle.nn as nn

            paddle.disable_static()

            shape = (5, 20)
            x = np.random.uniform(-10, 10, shape).astype('float32')
            target = np.random.uniform(-10, 10, shape).astype('float32')

            # 'batchmean' reduction, loss shape will be [1]
            kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
            pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                        paddle.to_tensor(target))
            # shape=[1]

            # 'mean' reduction, loss shape will be [1]
            kldiv_criterion = nn.KLDivLoss(reduction='mean')
            pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                        paddle.to_tensor(target))
            # shape=[1]

            # 'sum' reduction, loss shape will be [1]
            kldiv_criterion = nn.KLDivLoss(reduction='sum')
            pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                        paddle.to_tensor(target))
            # shape=[1]

            # 'none' reduction, loss shape is same with X shape
            kldiv_criterion = nn.KLDivLoss(reduction='none')
            pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                        paddle.to_tensor(target))
            # shape=[5, 20]
    """

    def __init__(self, reduction='mean'):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label):
        out = F.kl_div(input, label, self.reduction)
        return out


class MarginRankingLoss(fluid.dygraph.Layer):
    """

    This interface is used to construct a callable object of the ``MarginRankingLoss`` class.
    The MarginRankingLoss layer calculates the margin rank loss between the input, other and label
    , use the math function as follows.

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
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candicates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        input: N-D Tensor, the shape is [N, *], N is batch size and `*` means any number of additional dimensions., available dtype is float32, float64.
        other: N-D Tensor, `other` have the same shape and dtype as `input`.
        label: N-D Tensor, label have the same shape and dtype as `input`.
        output: If :attr:`reduction` is ``'mean'`` or ``'sum'`` , the out shape is :math:`[1]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Returns:
        A callable object of MarginRankingLoss.

    Examples:

        .. code-block:: python

            import paddle
            paddle.disable_static()

            input = paddle.to_tensor([[1, 2], [3, 4]]), dtype="float32")
            other = paddle.to_tensor([[2, 1], [2, 4]]), dtype="float32")
            label = paddle.to_tensor([[1, -1], [-1, -1]], dtype="float32")
            margin_rank_loss = paddle.nn.MarginRankingLoss()
            loss = margin_rank_loss(input, other, label)
            print(loss.numpy()) # [0.75]
    """

    def __init__(self, margin=0.0, reduction='mean', name=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in MarginRankingLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.name = name

    def forward(self, input, other, label):
        out = paddle.nn.functional.margin_ranking_loss(
            input, other, label, self.margin, self.reduction, self.name)
        return out


class CTCLoss(fluid.dygraph.Layer):
    """
	:alias_main: paddle.nn.CTCLoss
	:alias: paddle.nn.CTCLoss, paddle.nn.layer.CTCLoss, paddle.nn.layer.loss.CTCLoss

    An operator integrating the open source Warp-CTC library (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation
    is interated to the Warp-CTC library to normalize values for each row of the input tensor.

    Parameters:
        blank (int, optional): The blank label index of Connectionist Temporal Classification (CTC) loss, which is in the half-opened interval [0, num_classes + 1). The data type must be int32. Default is 0.
        reduction (string, optional): Indicate how to average the loss, the candicates are ``'none'`` | ``'mean'`` | ``'sum'``. If :attr:`reduction` is ``'mean'``, the output loss will be divided by the label_lengths, and then return the mean of quotient; If :attr:`reduction` is ``'sum'``, return the sum of loss; If :attr:`reduction` is ``'none'``, no reduction will be applied. Default is ``'mean'``.

    Shape:
        log_probs (Tensor): The unscaled probability sequence with padding, which is a 3-D Tensor. The tensor shape is [max_logit_length, batch_size, num_classes + 1], where max_logit_length is the longest length of input logit sequence. The data type should be float32 or float64.
        labels (Tensor): The ground truth sequence with padding, which must be a 3-D Tensor. The tensor shape is [batch_size, max_label_length], where max_label_length is the longest length of label sequence. The data type must be int32.
        input_lengths (Tensor): The length for each input sequence, it should have shape [batch_size] and dtype int64.
        label_lengths (Tensor): The length for each label sequence, it should have shape [batch_size] and dtype int64.

    Returns:
        Tensor, The Connectionist Temporal Classification (CTC) loss between ``log_probs`` and  ``labels``. If attr:`reduction` is ``'none'``, the shape of loss is [batch_size], otherwise, the shape of loss is [1]. Data type is the same as ``log_probs``.

    Examples:

        .. code-block:: python

            # declarative mode
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

            paddle.disable_static()
            log_probs = paddle.to_tensor(log_probs)
            labels = paddle.to_tensor(labels)
            input_lengths = paddle.to_tensor(input_lengths)
            label_lengths = paddle.to_tensor(label_lengths)

            loss = paddle.nn.CTCLoss(blank=0, reduction='none')(log_probs, labels,
                input_lengths,
                label_lengths)
            print(loss.numpy())  #[3.9179852 2.9076521]

            loss = paddle.nn.CTCLoss(blank=0, reduction='mean')(log_probs, labels,
                input_lengths,
                label_lengths)
            print(loss.numpy())  #[1.1376063]
    """

    def __init__(self, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, log_probs, labels, input_lengths, label_lengths):
        return paddle.nn.functional.ctc_loss(log_probs, labels, input_lengths,
                                             label_lengths, self.blank,
                                             self.reduction)


class SmoothL1Loss(fluid.dygraph.Layer):
    """
    This operator calculates smooth_l1_loss. Creates a criterion that uses a squared
    term if the absolute element-wise error falls below 1 and an L1 term otherwise.
    In some cases it can prevent exploding gradients and it is more robust and less
    sensitivity to outliers. Also known as the Huber loss:

    .. math::

         loss(x,y)=\\frac{1}{n}\\sum_{i}z_i

    where z_i is given by:

    .. math::

         \\mathop{z_i}=\\left\\{\\begin{array}{rcl}
        0.5(x_i - y_i)^2 & & {if |x_i - y_i| < delta} \\\\
        delta * |x_i - y_i| - 0.5 * delta^2 & & {otherwise}
        \\end{array} \\right.

    Parameters:
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

    Call Parameters:
        input (Tensor): Input tensor, the data type is float32 or float64. Shape is
            (N, C), where C is number of classes, and if shape is more than 2D, this
            is (N, C, D1, D2,..., Dk), k >= 1.
        label (Tensor): Label tensor, the data type is float32 or float64. The shape of label
            is the same as the shape of input.

    Returns:
        The tensor variable storing the smooth_l1_loss of input and label.

    Return type: Tensor.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(3,3).astype("float32")
            label_data = np.random.rand(3,3).astype("float32")
            input = paddle.to_tensor(input_data)
            label = paddle.to_tensor(label_data)
            loss = paddle.nn.SmoothL1Loss()
            output = loss(input, label)
            print(output.numpy())
    """

    def __init__(self, reduction='mean', delta=1.0, name=None):
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.delta = delta
        self.name = name

    def forward(self, input, label):
        return F.smooth_l1_loss(
            input,
            label,
            reduction=self.reduction,
            delta=self.delta,
            name=self.name)
