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

# TODO: define the extention functions

__all__ = [
    'diag_embed',
    'row_conv'
]

import numpy as np
from ...fluid.data_feeder import check_dtype
from ...fluid.layer_helper import LayerHelper
from ...fluid.framework import Variable, in_dygraph_mode
from ...fluid.layers.tensor import assign
from ...fluid import core, dygraph_utils
from ...fluid.layers.layer_function_generator import templatedoc


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    """
	:alias_main: paddle.nn.functional.diag_embed
	:alias: paddle.nn.functional.diag_embed,paddle.nn.functional.extension.diag_embed

    This OP creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) 
    are filled by ``input``. By default, a 2D plane formed by the last two dimensions 
    of the returned tensor will be selected.

    The argument ``offset`` determines which diagonal is generated:

    - If offset = 0, it is the main diagonal.
    - If offset > 0, it is above the main diagonal.
    - If offset < 0, it is below the main diagonal.

    Args:
        input(Variable|numpy.ndarray): The input tensor. Must be at least 1-dimensional. The input data type should be float32, float64, int32, int64.
        offset(int, optional): Which diagonal to consider. Default: 0 (main diagonal).
        dim1(int, optional): The first dimension with respect to which to take diagonal. Default: -2.
        dim2(int, optional): The second dimension with respect to which to take diagonal. Default: -1.
    
    Returns:
        Variable, the output data type is the same as input data type.
    
    Examples:
        .. code-block:: python

            import paddle.nn.functional as F
            import paddle.fluid.dygraph as dg
            import numpy as np
            
            diag_embed = np.random.randn(2, 3).astype('float32')
            # [[ 0.7545889 , -0.25074545,  0.5929117 ],
            #  [-0.6097662 , -0.01753256,  0.619769  ]]
            with dg.guard():
                data1 = F.diag_embed(diag_embed)
                data1.numpy()
                # [[[ 0.7545889 ,  0.        ,  0.        ],
                #  [ 0.        , -0.25074545,  0.        ],
                #   [ 0.        ,  0.        ,  0.5929117 ]],

                # [[-0.6097662 ,  0.        ,  0.        ],
                #  [ 0.        , -0.01753256,  0.        ],
                #  [ 0.        ,  0.        ,  0.619769  ]]]

                data2 = F.diag_embed(diag_embed, offset=-1, dim1=0, dim2=2)
                data2.numpy()
                # [[[ 0.        ,  0.        ,  0.        ,  0.        ],
                #   [ 0.7545889 ,  0.        ,  0.        ,  0.        ],
                #   [ 0.        , -0.25074545,  0.        ,  0.        ],
                #   [ 0.        ,  0.        ,  0.5929117 ,  0.        ]],
                #
                #  [[ 0.        ,  0.        ,  0.        ,  0.        ],
                #   [-0.6097662 ,  0.        ,  0.        ,  0.        ],
                #   [ 0.        , -0.01753256,  0.        ,  0.        ],
                #   [ 0.        ,  0.        ,  0.619769  ,  0.        ]]]

                data3 = F.diag_embed(diag_embed, offset=1, dim1=0, dim2=2)
                data3.numpy()
                # [[[ 0.        ,  0.7545889 ,  0.        ,  0.        ],
                #   [ 0.        , -0.6097662 ,  0.        ,  0.        ]],
                #
                #  [[ 0.        ,  0.        , -0.25074545,  0.        ],
                #   [ 0.        ,  0.        , -0.01753256,  0.        ]],
                #
                #  [[ 0.        ,  0.        ,  0.        ,  0.5929117 ],
                #   [ 0.        ,  0.        ,  0.        ,  0.619769  ]],
                #
                #  [[ 0.        ,  0.        ,  0.        ,  0.        ],
                #   [ 0.        ,  0.        ,  0.        ,  0.        ]]]
    """
    inputs = {'Input': [input]}
    attrs = {'offset': offset, 'dim1': dim1, 'dim2': dim2}

    if not isinstance(input, Variable):
        input = assign(input)

    def __check_input(input, offset, dim1, dim2):
        check_dtype(input.dtype, 'Input',
                    ['int32', 'int64', 'float16', 'float32', 'float64'],
                    'diag_embed')

        input_shape = list(input.shape)
        assert len(input_shape) >= 1,                     \
                "Input must be at least 1-dimensional, "   \
                "But received Input's dimensional: %s.\n" %  \
                len(input_shape)

        assert np.abs(dim1) <= len(input_shape),    \
            "Dim1 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape) + 1), len(input_shape), dim1)

        assert np.abs(dim2) <= len(input_shape),      \
            "Dim2 is out of range (expected to be in range of [%d, %d], but got %d).\n"  \
            % (-(len(input_shape) + 1), len(input_shape), dim2)

        dim1_ = dim1 if dim1 >= 0 else len(input_shape) + dim1 + 1
        dim2_ = dim2 if dim2 >= 0 else len(input_shape) + dim2 + 1
        assert dim1_ != dim2_,       \
               "dim1 and dim2 cannot be the same dimension." \
                "But received dim1 = %d, dim2 = %d\n"%(dim1, dim2)

    if not in_dygraph_mode():
        __check_input(input, offset, dim1, dim2)
    helper = LayerHelper("diag_embed", **locals())

    out = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='diag_embed',
        inputs={'Input': [input]},
        attrs={'offset': offset,
               'dim1': dim1,
               'dim2': dim2},
        outputs={'Out': [out]})
    out.stop_gradient = True
    return out


@templatedoc()
def row_conv(input, weight, act=None):
    """

    ${comment}

    Args:
        input (Variable):  the input(X) is a LodTensor or tensor, LodTensor(X) 
            supports variable  time-length input sequences. The underlying 
            tensor in this LoDTensor is a matrix with shape (T, D), where 
            T is the total time steps in this mini-batch and D is the input 
            data dimension. 
            If the input is a padded minibatch, the shape of the input is 
            (N, T, D), N is batch size, T is the max time steps in the batch,
             D is the input data dimension.
        weight (Variable): The weight. A Tensor with shape 
            (future_context_size + 1, D), where future_context_size is the 
            context size of the RowConv operator.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        ${out_comment}.

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
            weight = np.random.randn(context_size + 1, feature_size).astype(np.float32)

            place = fluid.CPUPlace()
            with dg.guard(place):
                x_var = dg.to_variable(x)
                w_var = dg.to_variable(weight)
                y_var = F.extension.row_conv(x_var, w_var)
                y_np = y_var.numpy()

            print(y_np.shape)

            # (4, 8, 6)
    """

    if in_dygraph_mode():
        pre_act = core.ops.row_conv(input, weight)
        out = dygraph_utils._append_activation_in_dygraph(pre_act, act)
        return out
    else:
        helper = LayerHelper('row_conv', **locals())
        dtype = helper.input_dtype()

        inputs = {'X': [input], 'Filter': [weight]}
        pre_act = helper.create_variable_for_type_inference(dtype)
        outputs = {'Out': [pre_act]}
        helper.append_op(type='row_conv', inputs=inputs, outputs=outputs)
        out = helper.append_activation(pre_act)
    return out
