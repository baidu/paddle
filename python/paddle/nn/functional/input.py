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

from __future__ import print_function
import warnings
from ...fluid.framework import Variable, in_dygraph_mode
from ...fluid.layer_helper import LayerHelper
from ...fluid.layers import core
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype

__all__ = ['one_hot']


def one_hot(x, num_classes, name=None):
    """

    The operator converts each id in the input 'x' to an one-hot vector with a
    num_classes length. The value in the vector dimension corresponding to the id
    is 1, and the value in the remaining dimension is 0.

    The shape of output Tensor is generated by appending num_classes dimension
    behind the last dimension of the 'x' shape.

    .. code-block:: text

        Example 1:

        input:
            x.shape = [4]
            x.data = [1, 1, 3, 0]
            num_classes = 4

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.],
                        [1., 0., 0., 0.]]

        Example 2:

        input:
            x.shape = [4]
            x.data = [1, 1, 5, 0]
            num_classes = 4

        output: Throw an exception for Illegal value
            The second dimension in X is 5, which is greater than num_classes,
            so it throws an exception.


    Args:
        x(Tensor): Tensor with shape :math:`[N_1, N_2, ..., N_k]` ,
            which contains at least one dimension. The data type is int32 or int64.
        num_classes(int): An integer defining the num_classes of the one hot dimension. If input 'x'
            is word id, num_classes is generally the dictionary size.

    Returns:
        Tensor: The one-hot representations of 'x'. A Tensor with type float32.

    Examples:
        .. code-block:: python

            import paddle
            # Correspond to the first example above, where label.shape is 4 and one_hot_label.shape is [4, 4].
            label = paddle.data(name="label", shape=[4, 1], dtype="int64")
            # label.shape = [4]
            # label.data = [1, 1, 3, 0]
            one_hot_label = paddle.nn.functional.one_hot(x=label, num_classes=4)
            # one_hot_label.shape = [4, 4]
            # one_hot_label.data = [[0., 1., 0., 0.],
            #                       [0., 1., 0., 0.],
            #                       [0., 0., 0., 1.],
            #                       [1., 0., 0., 0.]]
    """

    if in_dygraph_mode():
        return core.ops.one_hot_v2(x, 'depth', num_classes,
                                   'allow_out_of_range', False)
    else:
        check_variable_and_dtype(x, 'input', ['int32', 'int64'], 'one_hot_v2')
        helper = LayerHelper("one_hot_v2", **locals())

        one_hot_out = helper.create_variable_for_type_inference(dtype='float32')
        if not isinstance(num_classes, Variable):
            # user attribute 
            inputs = {'X': x}
            attrs = {'depth': num_classes, 'allow_out_of_range': False}
        else:
            num_classes.stop_gradient = True
            inputs = {'X': x, 'depth_tensor': num_classes}
            attrs = {'allow_out_of_range': False}
        helper.append_op(
            type="one_hot_v2",
            inputs=inputs,
            attrs=attrs,
            outputs={'Out': one_hot_out},
            stop_gradient=True)
        return one_hot_out
