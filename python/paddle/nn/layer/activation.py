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

__all__ = [
    'ELU',
    'GELU',
    'Hardshrink',
    'Tanh',
    'Hardtanh',
    'PReLU',
    'ReLU',
    'ReLU6',
    'SELU',
    'LeakyReLU',
    'Sigmoid',
    'Softmax',
    'Softplus',
    'Softshrink',
    'Softsign',
    'Tanhshrink',
    'LogSigmoid',
    'LogSoftmax',
    'HSigmoid',
]

from ...fluid.dygraph import layers
from ...fluid import core
from ...fluid.framework import in_dygraph_mode
from ...fluid.param_attr import ParamAttr
from ...fluid.initializer import Constant
from .. import functional as F


class ELU(layers.Layer):
    """
    ELU Activation.

    .. math::
    
        ELU(x) = max(0, x) + min(0, \\alpha * (e^{x}-1))

    Parameters:
        alpha (float, optional): The 'alpha' value of the ELU formulation. Default is 1.0.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([[-1,6],[1,15.6]]))
            m = paddle.nn.ELU(0.2)
            out = m(x)
            # [[-0.12642411  6.        ]
            #  [ 1.          15.6      ]]
    """

    def __init__(self, alpha=1.0, name=None):
        super(ELU, self).__init__()
        self._alpha = alpha
        self._name = name

    def forward(self, x):
        return F.elu(x, self._alpha, self._name)


class GELU(layers.Layer):
    """
    GELU Activation.

    If approximate is True

    .. math::

        GELU(x) = 0.5 * x * (1 + tanh(\\sqrt{\\frac{2}{\\pi}} * (x + 0.044715x^{3})))

    else

    .. math::

        GELU(x) = 0.5 * x * (1 + erf(\\frac{x}{\\sqrt{2}}))

    Parameters:
        approximate (bool, optional): Wether to enable approximation. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([[-1, 0.5],[1, 1.5]]))
            
            m = paddle.nn.GELU()
            out = m(x) # [-0.158655 0.345731 0.841345 1.39979]

            m = paddle.nn.GELU(True)
            out = m(x) # [-0.158808 0.345714 0.841192 1.39957]
    """

    def __init__(self, approximate=False, name=None):
        super(GELU, self).__init__()
        self._approximate = approximate
        self._name = name

    def forward(self, x):
        return F.gelu(x, self._approximate, self._name)


class Hardshrink(layers.Layer):
    """
    Hardshrink Activation

    .. math::

        hardshrink(x)=
            \left\{
            \begin{aligned}
            &x, & & if \ x > threshold \\
            &x, & & if \ x < -threshold \\
            &0, & & if \ others
            \end{aligned}
            \right.

    Parameters:
        threshold (float, optional): The value of threshold for hardthrink. Default is 0.5
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-1, 0.3, 2.5]))
        m = paddle.nn.Hardshrink()
        out = m(x) # [-1., 0., 2.5]
    """

    def __init__(self, threshold=0.5, name=None):
        super(Hardshrink, self).__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.hardshrink(x, self._threshold, self._name)


class Tanh(layers.Layer):
    """

    .. math::

        Tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_variable(np.array([-1, 0.5, 1]))
        m = paddle.nn.Tanh()
        out = m(x) # [-0.761594, 0.462117, 0.761594]
    """

    def __init__(self, name=None):
        super(Tanh, self).__init__()
        self._name = name

    def forward(self, x):
        return F.tanh(x, self._name)


class Hardtanh(layers.Layer):
    """
    Hardtanh Activation

    .. math::

        Hardtanh(x)= \\begin{cases}
                        max, \\text{if } x > max \\\\
                        min, \\text{if } x < min \\\\
                        x,  \\text{otherwise}
                      \\end{cases}

    Parameters:
        min (float, optional): The value of min for Hardtanh. Default is -1.
        max (float, optional): The value of max for Hardtanh. Default is 1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([-1.5, 0.3, 2.5]))
            m = paddle.nn.Hardtanh()
            out = m(x) # # [-1., 0.3, 1.]
    """

    def __init__(self, min=-1.0, max=1.0, name=None):
        super(Hardtanh, self).__init__()
        self._min = min
        self._max = max
        self._name = name

    def forward(self, x):
        return F.hardtanh(x, self._min, self._max, self._name)


class HSigmoid(layers.Layer):
    """
	:alias_main: paddle.nn.HSigmoid
	:alias: paddle.nn.HSigmoid,paddle.nn.layer.HSigmoid,paddle.nn.layer.activation.HSigmoid

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
        feature_size (int): The feature size.
        num_classes (int): The number of classes or the size of word dict, must be greater than 2.
            If the default tree is used (:attr:`is_custom` is set to False), :attr:`num_classes`
            should not be None. If the custom tree is used (:attr:`is_custom` is set to True),
            :attr:`num_classes` should be the number of non-leaf nodes, which indicates the num of
            classes using by the binary classifier.
        param_attr (ParamAttr, optional): The parameter attribute for the learnable parameters/weights
            of hsigmoid. If it is set to None or one attribute of ParamAttr, hsigmoid will create a
            ParamAttr as param_attr. If the Initializer of the param_attr is not set, the parameter is
            initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of hsigmoid. If it
            is set to False, no bias will be added. If it is set to None or one attribute of ParamAttr,
            hsigmoid will create a ParamAttr as bias_attr. If the Initializer of the bias_attr is not
            set, the bias is initialized zero. Default: None.
        is_custom (bool, optional): Whether use custom binary tree. If it's True, `path_table` and 
            `path_code` should be passed to its forward method, otherwise `path_table` and `path_code`
            should not be passed to its forward method. Default: False.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True, the
            gradient of W and input will be sparse. Default: False.

    Returns:
        None

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
                  hsm = nn.HSigmoid(feature_size, num_classes)
                  y = hsm(x, label)

          place = fluid.CPUPlace()
          exe = fluid.Executor(place)
          exe.run(start)
          feed_dict = {
              "input": np.random.randn(4, feature_size).astype(np.float32),
              "labels": np.random.randint(0, num_classes, (4, 1)).astype(np.int64),
          }
          y_np, = exe.run(main, feed=feed_dict, fetch_list=[y])
          print(y_np.shape)

          # (4, 1)
    """

    def __init__(self,
                 feature_size,
                 num_classes,
                 param_attr=None,
                 bias_attr=None,
                 is_custom=False,
                 is_sparse=False,
                 dtype="float32"):
        super(HSigmoid, self).__init__()
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

        self._param_attr = param_attr
        self._bias_attr = bias_attr

        self._dtype = dtype

        remote_prefetch = is_sparse
        print("With sparse mode, if your models has only"
              " small parameter prefetch may cause speed down")

        C = self._num_classes if is_custom else self._num_classes - 1
        self.weight = self.create_parameter(
            [C, self._feature_size],
            attr=self._param_attr,
            is_bias=False,
            dtype=self._dtype)
        self.bias = self.create_parameter(
            [C, 1], attr=self._bias_attr, is_bias=True, dtype=self._dtype)

    def forward(self, input, label, path_table=None, path_code=None):
        out = F.hsigmoid(
            input,
            label,
            self.weight,
            self.bias,
            self._num_classes,
            path_table=path_table,
            path_code=path_code,
            is_sparse=self._is_sparse)
        return out


class PReLU(layers.Layer):
    """
    PReLU Activation.

    .. math::

        PReLU(x) = max(0, x) + weight * min(0, x)

    Parameters:
        num_parameters (int, optional): Number of `weight` to learn. The supported values are:
            1 - a single parameter `alpha` is used for all input channels; 
            Number of channels - a seperate `alpha` is used for each input channel.
            Default is 1.
        init (float, optional): Init value of learnable `weight`. Default is 0.25.
        weight_attr(ParamAttr, optional): The parameter attribute for the learnable `weight`. 
            Default is None. For more information, please refer to :ref:`api_fluid_ParamAttr`.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            data = np.array([[[[-2.0,  3.0, -4.0,  5.0],
                            [ 3.0, -4.0,  5.0, -6.0],
                            [-7.0, -8.0,  8.0,  9.0]],
                            [[ 1.0, -2.0, -3.0,  4.0],
                            [-5.0,  6.0,  7.0, -8.0],
                            [ 6.0,  7.0,  8.0,  9.0]]]], 'float32')
            x = paddle.to_tensor(data)
            m = paddle.nn.PReLU(1, 0.25)
            out = m(x)
            # [[[[-0.5 ,  3.  , -1.  ,  5.  ],
            #    [ 3.  , -1.  ,  5.  , -1.5 ],
            #    [-1.75, -2.  ,  8.  ,  9.  ]],
            #   [[ 1.  , -0.5 , -0.75,  4.  ],
            #    [-1.25,  6.  ,  7.  , -2.  ],
            #    [ 6.  ,  7.  ,  8.  ,  9.  ]]]]
    """

    def __init__(self, num_parameters=1, init=0.25, weight_attr=None,
                 name=None):
        super(PReLU, self).__init__()
        self._num_parameters = num_parameters
        self._init = init
        self._weight_attr = weight_attr
        self._name = name

        self._weight = self.create_parameter(
            attr=self._weight_attr,
            shape=[num_parameters],
            dtype='float32',
            is_bias=False,
            default_initializer=Constant(init))

    def forward(self, x):
        return F.prelu(x, self._weight)


class ReLU(layers.Layer):
    """
    ReLU Activation.

    .. math::

        ReLU(x) = max(x, 0)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([-2, 0, 1]).astype('float32'))
            m = paddle.nn.ReLU()
            out = m(x) # [0., 0., 1.]
    """

    def __init__(self, name=None):
        super(ReLU, self).__init__()
        self._name = name

    def forward(self, x):
        return F.relu(x, self._name)


class ReLU6(layers.Layer):
    """
    ReLU6 Activation

    .. math::

        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-1, 0.3, 6.5]))
        m = paddle.nn.ReLU6()
        out = m(x) # [0, 0.3, 6]
    """

    def __init__(self, name=None):
        super(ReLU6, self).__init__()
        self._name = name

    def forward(self, x):
        return F.relu6(x, self._name)


class SELU(layers.Layer):
    """
    SELU Activation

    .. math::

        \text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1))), \\
        with\,alpha=1.6732632423543772848170429916717 and \\
        scale=1.0507009873554804934193349852946

    Parameters:
        scale (float, optional): The value of scale for SELU. Default is 1.0507009873554804934193349852946
        alpha (float, optional): The value of alpha for SELU. Default is 1.6732632423543772848170429916717
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([[0, 1],[2, 3]]))
        m = paddle.nn.SELU()
        out = m(x) # [[0, 1.050701],[2.101402, 3.152103]]
    """

    def __init__(self,
                 scale=1.0507009873554804934193349852946,
                 alpha=1.6732632423543772848170429916717,
                 name=None):
        super(SELU, self).__init__()
        self._scale = scale
        self._alpha = alpha
        self._name = name

    def forward(self, x):
        return F.selu(x, self._scale, self._alpha, self._name)


class LeakyReLU(layers.Layer):
    """
    Leaky ReLU Activation.

    .. math:

        LeakyReLU(x)=
            \left\{
            \begin{aligned}
            &x, & & if \ x >= 0 \\
            &negative\_slope * x, & & otherwise \\
            \end{aligned}
            \right. \\

    Parameters:
        negative_slope (float, optional): Slope of the activation function at
            :math:`x < 0` . Default is 0.01.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            m = paddle.nn.LeakyReLU()
            x = paddle.to_tensor(np.array([-2, 0, 1]))
            out = m(x)  # [-0.02, 0., 1.]
    """

    def __init__(self, negative_slope=0.01, name=None):
        super(LeakyReLU, self).__init__()
        self._negative_slope = negative_slope
        self._name = name

    def forward(self, x):
        return F.leaky_relu(x, self._negative_slope, self._name)


class Sigmoid(layers.Layer):
    """
    this interface is used to construct a callable object of the ``Sigmoid`` class. This layer calcluate the `sigmoid` of input x.
    
    .. math::

        Sigmoid(x) = \frac{1}{1 + e^{-x}}
    
    Parameters:
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        x: N-D tensor, available dtype is float16, float32, float64.

    Returns:
        A callable object of Sigmoid.
    
    Examples:

        .. code-block:: python

          import numpy as np
          import paddle

          paddle.disable_static()
          input_data = np.array([1.0, 2.0, 3.0, 4.0]).astype('float32')
          m = paddle.nn.Sigmoid()
          x = paddle.to_tensor(input_data)
          output = m(x)
          print(output.numpy()) # [0.7310586, 0.880797, 0.95257413, 0.98201376]
    """

    def __init__(self, name=None):
        super(Sigmoid, self).__init__()
        self.name = name

    def forward(self, x):
        return F.sigmoid(x, self.name)


class Softplus(layers.Layer):
    """
    Softplus Activation

    .. math::

        \text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x)) \\
        \text{For numerical stability, the implementation reverts to the linear function when :}\,x \times \beta > threshold.

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
        m = paddle.nn.Softplus()
        out = m(x) # [0.513015, 0.598139, 0.744397, 0.854355]

    """

    def __init__(self, beta=1, threshold=20, name=None):
        super(Softplus, self).__init__()
        self._beta = beta
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.softplus(x, self._beta, self._threshold, self._name)


class Softshrink(layers.Layer):
    """
    Softshrink Activation

    .. math::

        \text{Softshrink}(x) =
        \begin{cases}
        x - threshold, & \text{ if } x > threshold \\
        x + threshold, & \text{ if } x < -threshold \\
        0, & \text{ otherwise }
        \end{cases}

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.9, -0.2, 0.1, 0.8]))
        m = paddle.nn.Softshrink()
        out = m(x) # [-0.4, 0, 0, 0.3]
    """

    def __init__(self, threshold=0.5, name=None):
        super(Softshrink, self).__init__()
        self._threshold = threshold
        self._name = name

    def forward(self, x):
        return F.softshrink(x, self._threshold, self._name)


class Softsign(layers.Layer):
    """
    Softsign Activation

    .. math::

        \text{Softsign}(x) = \frac{x}{1 + |x|}

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:

        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
        m = paddle.nn.Softsign()
        out = m(x) # [-0.285714, -0.166667, 0.0909091, 0.230769]
    """

    def __init__(self, name=None):
        super(Softsign, self).__init__()
        self._name = name

    def forward(self, x):
        return F.softsign(x, self._name)


class Tanhshrink(layers.Layer):
    """
    Tanhshrink Activation

    .. math::

        \text{Tanhshrink}(x) = x - \text{Tanh}(x)

    Parameters:
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
        m = paddle.nn.Tanhshrink()
        out = m(x) # [-0.020051, -0.00262468, 0.000332005, 0.00868739]
    """

    def __init__(self, name=None):
        super(Tanhshrink, self).__init__()
        self._name = name

    def forward(self, x):
        return F.tanhshrink(x, self._name)


class LogSigmoid(layers.Layer):
    """
    LogSigmoid Activation.
    
    .. math::

        LogSigmoid(x) = log \\frac{1}{1 + e^{-x}}

    Parameters:
        x (Tensor): The input Tensor with data type float32, or float64.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
    
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.
    
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            x = paddle.to_tensor(np.array([1.0, 2.0, 3.0, 4.0]))
            m = paddle.nn.LogSigmoid()
            out = m(x) # [-0.313262 -0.126928 -0.0485874 -0.0181499]
    """

    def __init__(self, name=None):
        super(LogSigmoid, self).__init__()
        self._name = name

    def forward(self, x):
        return F.logsigmoid(x, self._name)


class Softmax(layers.Layer):
    """
    Softmax Activation.

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

        Softmax[i, j] = \\frac{\\exp(x[i, j])}{\\sum_j(exp(x[i, j])}

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

    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            paddle.disable_static()

            x = np.array([[[2.0, 3.0, 4.0, 5.0],
                        [3.0, 4.0, 5.0, 6.0],
                        [7.0, 8.0, 8.0, 9.0]],
                        [[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [6.0, 7.0, 8.0, 9.0]]], 'float32')
            x = paddle.to_tensor(x)
            m = paddle.nn.Softmax()
            out = m(x)
            # [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
            # [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
            #   [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]
    """

    def __init__(self, axis=-1, name=None):
        super(Softmax, self).__init__()
        self._axis = axis
        self._dtype = None
        self._name = name

    def forward(self, x):
        return F.softmax(x, self._axis, self._dtype, self._name)


class LogSoftmax(layers.Layer):
    """
    This operator implements the log_softmax layer. The calculation process is as follows:

    .. math::

        Out[i, j] = log(softmax(x)) 
                  = log(\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])})

    Parameters:
        axis (int, optional): The axis along which to perform log_softmax
            calculations. It should be in range [-D, D), where D is the
            dimensions of the input Tensor . If ``axis`` < 0, it works the
            same way as :math:`axis + D` . Default is -1.
        name (str, optional): Name for the operation (optional, default is None).
            For more information, please refer to :ref:`api_guide_Name`.
 
    Shape:
        - input: Tensor with any shape.
        - output: Tensor with the same shape as input.

    Examples:
        .. code-block:: python

        import paddle
        import numpy as np

        paddle.disable_static()

        x = np.array([[[-2.0, 3.0, -4.0, 5.0],
                        [3.0, -4.0, 5.0, -6.0],
                        [-7.0, -8.0, 8.0, 9.0]],
                        [[1.0, -2.0, -3.0, 4.0],
                        [-5.0, 6.0, 7.0, -8.0],
                        [6.0, 7.0, 8.0, 9.0]]])
        m = paddle.nn.LogSoftmax()
        x = paddle.to_tensor(x)
        out = m(x)
        # [[[ -7.1278396   -2.1278396   -9.127839    -0.12783948]
        #   [ -2.1270514   -9.127051    -0.12705144 -11.127051  ]
        #   [-16.313261   -17.313261    -1.3132617   -0.31326184]]
        #  [[ -3.0518122   -6.051812    -7.051812    -0.051812  ]
        #   [-12.313267    -1.3132664   -0.3132665  -15.313267  ]
        #   [ -3.4401896   -2.4401896   -1.4401896   -0.44018966]]]
    """

    def __init__(self, axis=-1, name=None):
        super(LogSoftmax, self).__init__()
        self._axis = axis
        self._name = name

    def forward(self, x):
        return F.log_softmax(x, self._axis)
