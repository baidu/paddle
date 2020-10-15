# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['RowConv']

from ...fluid.dygraph import layers
from .. import functional as F


class RowConv(layers.Layer):
    """
	:alias_main: paddle.nn.RowConv
	:alias: paddle.nn.RowConv,paddle.nn.layer.RowConv,paddle.nn.layer.extension.RowConv

    **Row-convolution operator**

    The row convolution is called lookahead convolution.  This operator was 
    introduced in the following paper for 
    `DeepSpeech2 <http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf>`_.

    The main motivation is that a bidirectional RNN, useful in DeepSpeech like 
    speech models, learns representation for a sequence by performing a
    forward and a backward pass through the entire sequence. However, unlike
    unidirectional RNNs, bidirectional RNNs are challenging to deploy in an online
    and low-latency setting. The lookahead convolution incorporates information
    from future subsequences in a computationally efficient manner to improve
    unidirectional recurrent neural networks. The row convolution operator is
    different from the 1D sequence convolution, and is computed as follows:

    Given an input sequence X of length t and input dimension D, and a filter 
    (W) of size context * D.

    More details about row_conv please refer to the design document 
    `<https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645>`_ .

    Parameters:
        num_channels (int): input data's feature size.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc. Default: None.
        act (str): Non-linear activation to be applied to output variable. Default: None.
        dtype (str, optional): Data type, it can be "float32". Default: "float32".

    Attributes:
        weight (Parameter): shape [future_context_size + 1, D], the learnable 
            weight (convolution kernel) of this layer.

    Returns:
        None

    Examples:
        .. code-block:: python

          from paddle import fluid, nn
          import paddle.fluid.dygraph as dg
          import paddle.nn.functional as F
          import numpy as np

          batch_size = 4
          time_steps = 8
          feature_size = 6
          context_size = 4

          x = np.random.randn(batch_size, time_steps, feature_size).astype(np.float32)

          place = fluid.CPUPlace()
          with dg.guard(place):
              x_var = dg.to_variable(x)
              conv = nn.RowConv(feature_size, context_size)
              y_var = conv(x_var)
              y_np = y_var.numpy()
          print(y_np.shape)

          # (4, 8, 6)
    """

    def __init__(self,
                 num_channels,
                 future_context_size,
                 param_attr=None,
                 act=None,
                 dtype="float32"):
        super(RowConv, self).__init__()
        self._dtype = dtype
        self._param_attr = param_attr
        self._act = act

        filter_shape = [future_context_size + 1, num_channels]
        self.weight = self.create_parameter(
            filter_shape, attr=param_attr, dtype=dtype)

    def forward(self, input):
        out = F.extension.row_conv(input, self.weight, act=self._act)
        return out
