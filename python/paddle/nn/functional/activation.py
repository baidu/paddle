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

# TODO: define activation functions of neural network
from ...fluid.layers import brelu  #DEFINE_ALIAS
from ...fluid.layers import erf  #DEFINE_ALIAS
from ...fluid.layers import hard_sigmoid  #DEFINE_ALIAS
from ...fluid.layers import hard_swish  #DEFINE_ALIAS
from ...fluid.layers import maxout  #DEFINE_ALIAS
from ...fluid.layers import soft_relu  #DEFINE_ALIAS
from ...fluid.layers import swish  #DEFINE_ALIAS
from ...fluid.layers import sigmoid  #DEFINE_ALIAS
from ...fluid.layers import thresholded_relu  #DEFINE_ALIAS
from ...tensor.math import tanh  #DEFINE_ALIAS

__all__ = [
    'brelu',
    'elu',
    'erf',
    'gelu',
    'hardshrink',
    'hardtanh',
    'hard_sigmoid',
    'hard_swish',
    'hsigmoid',
    'leaky_relu',
    'logsigmoid',
    'maxout',
    'prelu',
    'relu',
    'relu6',
    'selu',
    'soft_relu',
    'softmax',
    'softplus',
    'softshrink',
    'softsign',
    'sigmoid',
    'swish',
    'tanh',
    'tanhshrink',
    'thresholded_relu',
    'log_softmax',
]

import warnings
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import in_dygraph_mode, convert_np_dtype_to_dtype_
from ...fluid import core
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
import paddle


def elu(x, alpha=1.0, name=None):
    """
    elu activation.

    .. math::

        elu(x) = max(0, x) + min(0, \\alpha * (e^{x}-1))

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        alpha (float, optional): The 'alpha' value of the ELU formulation. Default is 1.0.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        A Tensor with the same data type and shape as ``x`` .
    
    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([[-1,6],[1,15.6]]))
            out = F.elu(x, alpha=0.2) 
            # [[-0.12642411  6.        ]
            #  [ 1.          15.6      ]]
    """

    if in_dygraph_mode():
        return core.ops.elu(x, 'alpha', alpha)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'elu')
    helper = LayerHelper("elu", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='elu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'alpha': alpha})
    return out


def gelu(x, approximate=False, name=None):
    """
    gelu activation.

    if approximate is True

    .. math::

        gelu(x) = 0.5 * x * (1 + tanh(\\sqrt{\\frac{2}{\\pi}} * (x + 0.044715x^{3})))

    else

    .. math::

        gelu(x) = 0.5 * x * (1 + erf(\\frac{x}{\\sqrt{2}}))
    
    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        approximate (bool, optional): Wether to enable approximation. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        A Tensor with the same data type and shape as ``x`` .
    
    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([[-1, 0.5],[1, 1.5]]))
            out1 = F.gelu(x) # [-0.158655 0.345731 0.841345 1.39979]
            out2 = F.gelu(x, True) # [-0.158808 0.345714 0.841192 1.39957]
    """

    if in_dygraph_mode():
        return core.ops.gelu(x, 'approximate', approximate)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'gelu')
    helper = LayerHelper("gelu", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='gelu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'approximate': approximate})
    return out


def hardshrink(x, threshold=0.5, name=None):
    """
    hard shrinkage activation

    .. math::

        hardshrink(x)=
            \left\{
            \begin{aligned}
            &x, & & if \ x > threshold \\
            &x, & & if \ x < -threshold \\
            &0, & & if \ others
            \end{aligned}
            \right.

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        threshold (float, optional): The value of threshold for hardthrink. Default is 0.5
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([-1, 0.3, 2.5]))
            out = F.hardshrink(x) # [-1., 0., 2.5]

    """
    if in_dygraph_mode():
        return core.ops.hard_shrink(x, 'threshold', threshold)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'hardshrink')
    helper = LayerHelper('hardshrink', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='hard_shrink',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


def hardtanh(x, min=-1.0, max=1.0, name=None):
    """
    hardtanh activation

    .. math::

        hardtanh(x)= \\begin{cases}
                        max, \\text{if } x > max \\\\
                        min, \\text{if } x < min \\\\
                        x,  \\text{otherwise}
                      \\end{cases}

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        min (float, optional): The minimum value of the linear region range. Default is -1.
        max (float, optional): The maximum value of the linear region range. Default is 1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([-1.5, 0.3, 2.5]))
            out = F.hardtanh(x) # [-1., 0.3, 1.]
    """

    if in_dygraph_mode():
        return core.ops.brelu(x, 't_min', min, 't_max', max)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'hardtanh')

    helper = LayerHelper('hardtanh', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='brelu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'t_min': min,
               't_max': max})
    return out


def hsigmoid(input,
             label,
             weight,
             bias,
             num_classes,
             path_table=None,
             path_code=None,
             is_sparse=False):
    """
	:alias_main: paddle.nn.functional.hsigmoid
	:alias: paddle.nn.functional.hsigmoid,paddle.nn.functional.activation.hsigmoid

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
        input (Variable): A tensor with the shape [N, D], where N is the size of mini-batch,
            and D is the feature size. Its data type supports float32 and float64.
        label (Variable): A tensor contains the labels of training data. Its shape is [N, 1]
            and data type is int64.
        weight (Variable): A tensor with shape (num_classes - 1, D) if not using custom tree(path_code and path_table is None), or (num_classes, D) if using custom tree.
        bias (Variable): A tensor with shape (num_classes - 1, 1) if not using custom tree(path_code and path_table is None), or (num_classes, 1) if using custom tree.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        path_table (Variable, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. Default: None.
        path_code (Variable, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. Default: None.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True, the
            gradient of W and input will be sparse. Default: False.

    Returns:
        Variable: A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as :attr:`input`.

    Examples:
        .. code-block:: python

            from paddle import fluid, nn
            import paddle.fluid.dygraph as dg
            import paddle.nn.functional as F
            import numpy as np

            main = fluid.Program()
            start = fluid.Program()
            feature_size = 6
            num_classes = 8
            with fluid.unique_name.guard():
                with fluid.program_guard(main, start):
                    x = fluid.data("input", [-1, feature_size],
                                  dtype="float32")
                    label = fluid.data("labels", [-1, 1], dtype="int64")
                    w = fluid.data("weight", (num_classes -1, feature_size), dtype="float32")
                    b = fluid.data("bias", (num_classes -1, ), dtype="float32")
                    y = F.hsigmoid(x, label, w, b, num_classes)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(start)
            feed_dict = {
                "input": np.random.randn(4, feature_size).astype(np.float32),
                "labels": np.random.randint(0, num_classes, (4, 1)).astype(np.int64),
                "weight": np.random.randn(num_classes - 1, feature_size).astype(np.float32),
                "bias": np.random.randn(num_classes - 1, ).astype(np.float32),
            }
            y_np, = exe.run(main, feed=feed_dict, fetch_list=[y])
            print(y_np.shape)

          # (4, 1)
    """

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

    helper = LayerHelper('hierarchical_sigmoid', **locals())
    dtype = helper.input_dtype()

    out = helper.create_variable_for_type_inference(dtype)
    pre_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": out, "PreOut": pre_out, "W_Out": weight}

    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs=outputs,
        attrs=attrs)
    return out


def leaky_relu(x, negative_slope=0.01, name=None):
    """
    leaky_relu activation

    .. math:
        leaky_relu(x)=
            \left\{
            \begin{aligned}
            &x, & & if \ x >= 0 \\
            &negative\_slope * x, & & otherwise \\
            \end{aligned}
            \right. \\

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([-2, 0, 1]))
            out = F.leaky_relu(x) # [-0.02, 0., 1.]

    """
    if in_dygraph_mode():
        return core.ops.leaky_relu(x, 'alpha', negative_slope)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'leaky_relu')
    helper = LayerHelper('leaky_relu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='leaky_relu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'alpha': negative_slope})
    return out


def prelu(x, weight, name=None):
    """
    prelu activation.

    .. math::

        prelu(x) = max(0, x) + weight * min(0, x)

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        weight (Tensor): The learnable parameter with data type same as ``x``.
            The weight shape is [1] or [in], where `in` is the input channel of ``x``.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            data = np.array([[[[-2.0,  3.0, -4.0,  5.0],
                            [ 3.0, -4.0,  5.0, -6.0],
                            [-7.0, -8.0,  8.0,  9.0]],
                            [[ 1.0, -2.0, -3.0,  4.0],
                            [-5.0,  6.0,  7.0, -8.0],
                            [ 6.0,  7.0,  8.0,  9.0]]]], 'float32')
            x = paddle.to_tensor(data)
            w = paddle.to_tensor(np.array([0.25]).astype('float32'))
            out = F.prelu(x, w)
            # [[[[-0.5 ,  3.  , -1.  ,  5.  ],
            #    [ 3.  , -1.  ,  5.  , -1.5 ],
            #    [-1.75, -2.  ,  8.  ,  9.  ]],
            #   [[ 1.  , -0.5 , -0.75,  4.  ],
            #    [-1.25,  6.  ,  7.  , -2.  ],
            #    [ 6.  ,  7.  ,  8.  ,  9.  ]]]]
    """
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'prelu')
    check_variable_and_dtype(weight, 'weight',
                             ['float16', 'float32', 'float64'], 'prelu')

    helper = LayerHelper('prelu', **locals())
    assert len(weight.shape
               ) == 1, "The dim count of weight shape should be 1 in prelu()."

    # NOTE(): The input of this API should be ``N,C,...`` format, 
    # which means x.shape[0] is batch_size and x.shape[0] is channel.
    mode = 'all'
    if weight.shape[0] > 1:
        assert len(
            x.shape
        ) > 1, "The dim count of x should be equal or larger than 2 in prelu() when weight shape is not [1]."
        assert weight.shape[0] == x.shape[
            1], "The weight size should be equal to x input channel in prelu() when weight shape is not [1]."
        mode = 'channel'

    if in_dygraph_mode():
        return core.ops.prelu(x, weight, 'mode', mode)

    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x,
                "Alpha": weight},
        outputs={"Out": out},
        attrs={"mode": mode})
    return out


def relu(x, name=None):
    """
    relu activation.

    .. math::

        out = max(x, 0)

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([-2, 0, 1]).astype('float32'))
            out = F.relu(x) # [0., 0., 1.]
    """

    if in_dygraph_mode():
        return core.ops.relu(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'relu')
    helper = LayerHelper('relu', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='relu', inputs={'X': x}, outputs={'Out': out})
    return out


def logsigmoid(x, name=None):
    """
    logsigmoid activation.

    .. math::

        logsigmoid(x) = log \\frac{1}{1 + e^{-x}}
    
    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Returns:
        A Tensor with the same data type and shape as ``x`` .
    
    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([1.0, 2.0, 3.0, 4.0]))
            out = F.logsigmoid(x) # [-0.313262 -0.126928 -0.0485874 -0.0181499]
    """

    if in_dygraph_mode():
        return core.ops.logsigmoid(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'logsigmoid')
    helper = LayerHelper("logsigmoid", **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='logsigmoid', inputs={'X': x}, outputs={'Out': out})
    return out


def relu6(x, name=None):
    """
    relu6 activation

    .. math::

        \text{relu6}(x) = \min(\max(0,x), 6)

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:

        .. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-1, 0.3, 6.5]))
        out = F.relu6(x) # [0, 0.3, 6]

    """
    threshold = 6.0
    if in_dygraph_mode():
        return core.ops.relu6(x, 'threshold', threshold)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'relu6')
    helper = LayerHelper('relu6', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='relu6',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


def selu(x,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717,
         name=None):
    """
    selu activation

    .. math::

        \text{selu}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1))), \\
        with\,alpha=1.6732632423543772848170429916717 and \\
        scale=1.0507009873554804934193349852946

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        scale (float, optional): The value of scale for selu. Default is 1.0507009873554804934193349852946
        alpha (float, optional): The value of alpha for selu. Default is 1.6732632423543772848170429916717
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:

        .. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([[0, 1],[2, 3]]))
        out = F.selu(x) # [[0, 1.050701],[2.101402, 3.152103]]

    """
    if in_dygraph_mode():
        return core.ops.selu(x, 'scale', scale, 'alpha', alpha)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'], 'selu')
    helper = LayerHelper('selu', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='selu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'scale': scale,
               'alpha': alpha})
    return out


def softmax(x, axis=-1, dtype=None, name=None):
    """
    This operator implements the softmax layer. The calculation process is as follows:

    1. The dimension :attr:`axis` of ``x`` will be permuted to the last.

    2. Then ``x`` will be logically flattened to a 2-D matrix. The matrix's second
    dimension(row length) is the same as the dimension :attr:`axis` of ``x``,
    and the first dimension(column length) is the product of all other dimensions
    of ``x``. For each row of the matrix, the softmax operator squashes the
    K-dimensional(K is the width of the matrix, which is also the size of ``x``'s
    dimension :attr:`axis`) vector of arbitrary real values to a K-dimensional
    vector of real values in the range [0, 1] that add up to 1.

    3. After the softmax operation is completed, the inverse operations of steps 1 and 2
    are performed to restore the two-dimensional matrix to the same dimension as the ``x`` .

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in the matrix, we have:

    .. math::

        softmax[i, j] = \\frac{\\exp(x[i, j])}{\\sum_j(exp(x[i, j])}

    Example:

    .. code-block:: text

        Case 1:
          Input:
            x.shape = [2, 3, 4]
            x.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]

          Attrs:
            axis = -1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                        [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]

        Case 2:
          Input:
            x.shape = [2, 3, 4]
            x.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]
          Attrs:
            axis = 1

          Output:
            out.shape = [2, 3, 4]
            out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                         [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                         [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                        [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                         [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                         [0.72747516, 0.72747516, 0.72747516, 0.72747516]]]

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` < 0, it works the same way as
            :math:`axis + D` . Default is -1.
        dtype (str|np.dtype|core.VarDesc.VarType, optional): The desired data
            type of the output tensor. If dtype is specified, ``x`` is casted
            to ``dtype`` before the operation is performed. This is useful for 
            preventing data type overflows. Supported dtype: float32, float64.
            If ``dtype`` is None, the output Tensor has the same dtype as x.
            Default is None.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same shape and data type (use ``dtype`` if it is
        specified) as x.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = np.array([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]], 'float32')
            x = paddle.to_tensor(x)
            out1 = F.softmax(x)
            out2 = F.softmax(x, dtype='float64')
            # out1's data type is float32; out2's data type is float64
            # out1 and out2's value is as follows:
            # [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
            # [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]
    """

    if (dtype is not None) and (not isinstance(dtype, core.VarDesc.VarType)):
        dtype = convert_np_dtype_to_dtype_(dtype)
    use_cudnn = True if axis is -1 else False

    if in_dygraph_mode():
        outs_cast = x if dtype is None \
            else core.ops.cast(x, 'in_dtype', x.dtype, 'out_dtype', dtype)
        return core.ops.softmax(outs_cast, 'axis', axis, 'use_cudnn', use_cudnn)

    if dtype is None:
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'softmax')
    else:
        check_dtype(dtype, 'dtype', ['float32', 'float64'], 'softmax',
                    'If dtype is not None, it only support float32 or float64.')

    helper = LayerHelper("softmax", **locals())
    outs_cast = x
    if dtype is not None:
        outs_cast = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='cast',
            inputs={'X': x},
            outputs={'Out': outs_cast},
            attrs={'in_dtype': x.dtype,
                   'out_dtype': dtype})

    outs_softmax = helper.create_variable_for_type_inference(outs_cast.dtype)
    helper.append_op(
        type='softmax',
        inputs={'X': outs_cast},
        outputs={'Out': outs_softmax},
        attrs={'axis': axis,
               'use_cudnn': use_cudnn})

    return outs_softmax


def softplus(x, beta=1, threshold=20, name=None):
    """
    softplus activation

    .. math::

        \text{softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) \\
        \text{For numerical stability, the implementation reverts to the linear function when :}\,x \times \beta > threshold.

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        beta (float, optional): The value of beta for softplus. Default is 1
        threshold (float, optional): The value of threshold for softplus. Default is 20
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:

        .. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
        out = F.softplus(x) # [0.513015, 0.598139, 0.744397, 0.854355]

    """
    if in_dygraph_mode():
        return core.ops.softplus(x, 'beta', beta, 'threshold', threshold)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'softplus')
    helper = LayerHelper('softplus', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='softplus',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'beta': beta,
               'threshold': threshold})
    return out


def softshrink(x, threshold=0.5, name=None):
    """
    softshrink activation

    .. math::

        \text{softshrink}(x) =
        \begin{cases}
        x - threshold, & \text{ if } x > threshold \\
        x + threshold, & \text{ if } x < -threshold \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        threshold (float, optional): The value of threshold(must be no less than zero) for softplus. Default is 0.5
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:

        .. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.9, -0.2, 0.1, 0.8]))
        out = F.softshrink(x) # [-0.4, 0, 0, 0.3]

    """
    if in_dygraph_mode():
        return core.ops.softshrink(x, 'lambda', threshold)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'softshrink')
    helper = LayerHelper('softshrink', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='softshrink',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'lambda': threshold})
    return out


def softsign(x, name=None):
    """
    softsign activation

    .. math::

        \text{softsign}(x) = \frac{x}{1 + |x|}

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:

        .. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
        out = F.softsign(x) # [-0.285714, -0.166667, 0.0909091, 0.230769]

    """
    if in_dygraph_mode():
        return core.ops.softsign(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'softsign')
    helper = LayerHelper('softsign', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='softsign', inputs={'X': x}, outputs={'Out': out})
    return out


def tanhshrink(x, name=None):
    """
    tanhshrink activation

    .. math::

        \text{tanhshrink}(x) = x - \text{tanh}(x)

    Args:
        x (Tensor): The input Tensor with data type float32, float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor with the same data type and shape as ``x`` .

    Examples:

        .. code-block:: python

        import paddle
        import paddle.nn.functional as F
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
        out = F.tanhshrink(x) # [-0.020051, -0.00262468, 0.000332005, 0.00868739]

    """
    if in_dygraph_mode():
        return core.ops.tanh_shrink(x)

    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'tanhshrink')
    helper = LayerHelper('tanh_shrink', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='tanh_shrink', inputs={'X': x}, outputs={'Out': out})
    return out


def log_softmax(x, axis=-1, dtype=None, name=None):
    """
    This operator implements the log_softmax layer. The calculation process is
    as follows:

    .. math::

        Out[i, j] = log(softmax(x)) 
                  = log(\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

    Parameters:
        x (Tensor): The input Tensor with data type float32, float64.
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of ``x`` . If ``axis`` < 0, it works the same way as
            :math:`axis + D` . Default is -1.
        dtype (str|np.dtype|core.VarDesc.VarType, optional): The desired data
            type of the output tensor. If dtype is specified, ``x`` is casted
            to ``dtype`` before the operation is performed. This is useful for 
            preventing data type overflows. Supported dtype: float32, float64.
            If ``dtype`` is None, the output Tensor has the same dtype as x.
            Default is None.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
 
    Returns:
        A Tensor with the same shape and data type (use ``dtype`` if it is
        specified) as x.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn.functional as F
            import numpy as np

            paddle.disable_static()

            x = np.array([[[-2.0, 3.0, -4.0, 5.0],
                            [3.0, -4.0, 5.0, -6.0],
                            [-7.0, -8.0, 8.0, 9.0]],
                            [[1.0, -2.0, -3.0, 4.0],
                            [-5.0, 6.0, 7.0, -8.0],
                            [6.0, 7.0, 8.0, 9.0]]], 'float32')
            x = paddle.to_tensor(x)
            out1 = F.log_softmax(x)
            out2 = F.log_softmax(x, dtype='float64')
            # out1's data type is float32; out2's data type is float64
            # out1 and out2's value is as follows:
            # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
            #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
            #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
            #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
            #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
            #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    if (dtype is not None) and (not isinstance(dtype, core.VarDesc.VarType)):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if in_dygraph_mode():
        if dtype is not None:
            x = core.ops.cast(x, 'in_dtype', x.dtype, 'out_dtype', dtype)
        return core.ops.log_softmax(x, 'axis', axis)

    if dtype is None:
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'log_softmax')
    else:
        check_dtype(dtype, 'dtype', ['float32', 'float64'], 'log_softmax',
                    'If dtype is not None, it only support float32 or float64.')

    helper = LayerHelper("log_softmax", **locals())
    out_cast = x
    if dtype is not None:
        out_cast = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type='cast',
            inputs={'X': x},
            outputs={'Out': out_cast},
            attrs={'in_dtype': x.dtype,
                   'out_dtype': dtype})

    out = helper.create_variable_for_type_inference(out_cast.dtype)
    helper.append_op(
        type='log_softmax',
        inputs={'X': out_cast},
        outputs={'Out': out},
        attrs={'axis': axis})

    return out
