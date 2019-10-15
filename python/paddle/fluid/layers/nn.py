# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
All layers just related to the neural network.
"""

from __future__ import print_function

import numpy as np
import warnings
import six
import os
import inspect
from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant, NumpyArrayInitializer
from ..framework import Variable, OpProtoHolder, in_dygraph_mode
from ..dygraph import base
from ..param_attr import ParamAttr
from .layer_function_generator import autodoc, templatedoc, _generate_doc_string_
from .tensor import concat, assign, fill_constant, zeros
from . import utils
from .. import unique_name
from functools import reduce
from .. import core
from ..dygraph import layers
from ..data_feeder import convert_dtype

__all__ = [
    'fc',
    'center_loss',
    'embedding',
    'dynamic_lstm',
    'dynamic_lstmp',
    'dynamic_gru',
    'gru_unit',
    'linear_chain_crf',
    'crf_decoding',
    'cos_sim',
    'cross_entropy',
    'bpr_loss',
    'square_error_cost',
    'chunk_eval',
    'sequence_conv',
    'conv2d',
    'conv3d',
    'sequence_pool',
    'sequence_softmax',
    'softmax',
    'pool2d',
    'pool3d',
    'adaptive_pool2d',
    'adaptive_pool3d',
    'batch_norm',
    'instance_norm',
    'data_norm',
    'beam_search_decode',
    'conv2d_transpose',
    'conv3d_transpose',
    'sequence_expand',
    'sequence_expand_as',
    'sequence_pad',
    'sequence_unpad',
    'lstm_unit',
    'reduce_sum',
    'reduce_mean',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'reduce_all',
    'reduce_any',
    'sequence_first_step',
    'sequence_last_step',
    'sequence_slice',
    'dropout',
    'split',
    'ctc_greedy_decoder',
    'edit_distance',
    'l2_normalize',
    'matmul',
    'topk',
    'warpctc',
    'sequence_reshape',
    'transpose',
    'im2sequence',
    'nce',
    'sampled_softmax_with_cross_entropy',
    'hsigmoid',
    'beam_search',
    'row_conv',
    'multiplex',
    'layer_norm',
    'group_norm',
    'spectral_norm',
    'softmax_with_cross_entropy',
    'smooth_l1',
    'one_hot',
    'autoincreased_step_counter',
    'reshape',
    'squeeze',
    'unsqueeze',
    'lod_reset',
    'lod_append',
    'lrn',
    'pad',
    'pad_constant_like',
    'label_smooth',
    'roi_pool',
    'roi_align',
    'dice_loss',
    'image_resize',
    'image_resize_short',
    'resize_bilinear',
    'resize_trilinear',
    'resize_nearest',
    'gather',
    'gather_nd',
    'scatter',
    'scatter_nd_add',
    'scatter_nd',
    'sequence_scatter',
    'random_crop',
    'mean_iou',
    'relu',
    'selu',
    'log',
    'crop',
    'crop_tensor',
    'rank_loss',
    'margin_rank_loss',
    'elu',
    'relu6',
    'pow',
    'stanh',
    'hard_sigmoid',
    'swish',
    'prelu',
    'brelu',
    'leaky_relu',
    'soft_relu',
    'flatten',
    'sequence_mask',
    'stack',
    'pad2d',
    'unstack',
    'sequence_enumerate',
    'unique',
    'unique_with_counts',
    'expand',
    'expand_as',
    'sequence_concat',
    'scale',
    'elementwise_add',
    'elementwise_div',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'elementwise_mod',
    'elementwise_floordiv',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'sampling_id',
    'gaussian_random_batch_size_like',
    'sum',
    'slice',
    'strided_slice',
    'shape',
    'rank',
    'size',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
    'clip',
    'clip_by_norm',
    'mean',
    'mul',
    'sigmoid_cross_entropy_with_logits',
    'maxout',
    'space_to_depth',
    'affine_grid',
    'sequence_reverse',
    'affine_channel',
    'similarity_focus',
    'hash',
    'grid_sampler',
    'log_loss',
    'add_position_encoding',
    'bilinear_tensor_product',
    'merge_selected_rows',
    'get_tensor_from_selected_rows',
    'lstm',
    'shuffle_channel',
    'temporal_shift',
    'py_func',
    'psroi_pool',
    'prroi_pool',
    'teacher_student_sigmoid_loss',
    'huber_loss',
    'kldiv_loss',
    'npair_loss',
    'pixel_shuffle',
    'fsp_matrix',
    'continuous_value_model',
    'where',
    'sign',
    'deformable_conv',
    'unfold',
    'deformable_roi_pooling',
    'filter_by_instag',
    'shard_index',
    'hard_swish',
    'gather_tree',
    'mse_loss',
    'uniform_random',
]

kIgnoreIndex = -100


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None):
    """
    **Fully Connected Layer**

    This operator creates a fully connected layer in the network. It can take
    a Tensor(or LoDTensor) or a list of Tensor(or LoDTensor) as its inputs(see
    Args in detail). It creates a variable called weight for each input Tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input Tensor
    with its corresponding weight to produce an output Tensor with shape :math:`[M, size]` ,
    where M is batch size. If a list of Tensor is given, the results of
    multiple output Tensors with shape :math:`[M, size]` will be summed up. If :attr:`bias_attr`
    is not None, a bias variable will be created and added to the output.
    Finally, if :attr:`act` is not None, it will be applied to the output as well.

    When the input is a single Tensor(or LoDTensor):

    .. math::

        Out = Act({XW + b})

    When the input is a list of Tensor(or LoDTensor):

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    In the above equation:

    * :math:`N`: Number of the input. N equals to len(input) if input is list of Variable.
    * :math:`X_i`: The i-th input tensor.
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output Tensor.

    .. code-block:: text

        Case 1:
        Given a single Tensor data_1, and num_flatten_dims = 2:
            data_1.data = [[[0.1, 0.2],
                            [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            out = fluid.layers.fc(input=data_1, size=1, num_flatten_dims=2)

        Then output is:
            out.data = [[0.83234344], [0.34936576]]
            out.shape = (1, 2, 1)

        Case 2:
        Given a list of Tensor:
            data_1.data = [[[0.1, 0.2],
                           [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size

            data_2 = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3)

            out = fluid.layers.fc(input=[data_1, data_2], size=2)

        Then:
            out.data = [[0.18669507, 0.1893476]]
            out.shape = (1, 2)

    Args:
        input (Variable|list of Variable): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` or
            a list of Tensor(or LoDTensor). The dimensions of the input Tensor is at least 2 and the data
            type should be float32 or float64.
        size(int): The number of output units in this layer, which also means the feature size of ouput
            Tensor(or LoDTensor).
        num_flatten_dims (int): The fc layer can accept an input Tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-D matrix. The parameter :attr:`num_flatten_dims` determines how the input
            Tensor is flattened: the first :attr:`num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest :math:`rank(X) - num\_flatten\_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, assuming that
            X is a 5-dimensional Tensor with a shape [2, 3, 4, 5, 6], and :attr:`num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1.
        param_attr (ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr): To specify the bias parameter property. Default: None, which means the
            default bias parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        act (str): Activation to be applied to the output of this layer, such as tanh, softmax,
            sigmoid, relu. For more information, please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: Tensor or LoDTensor calculated by fc layer. The data type is same with input.

    Raises:
        ValueError: If dimensions of the input Tensor is less than 2.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          # when input is single tensor
          data = fluid.data(name="data", shape=[-1, 32], dtype="float32")
          fc = fluid.layers.fc(input=data, size=1000, act="tanh")

          # when input are multiple tensors
          data_1 = fluid.data(name="data_1", shape=[-1, 32], dtype="float32")
          data_2 = fluid.data(name="data_2", shape=[-1, 36], dtype="float32")
          fc = fluid.layers.fc(input=[data_1, data_2], size=1000, act="tanh")
    """
    helper = LayerHelper("fc", **locals())
    if isinstance(input, (list, tuple)):
        for i, input_x in enumerate(input):
            if not isinstance(input_x, Variable):
                raise TypeError(
                    "The type of input[%d] in fc must be Variable, but received %s"
                    % (i, type(input_x)))
    else:
        if not isinstance(input, Variable):
            raise TypeError(
                "The type of 'input' in fc must be Variable, but received %s" %
                (type(input)))
    dtype = helper.input_dtype()
    if convert_dtype(dtype) in ['float16']:
        warnings.warn(
            "The data type of 'input' in fc only support float16 in GPU now.")
    if convert_dtype(dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in fc must be float16, float32 or float64, but received %s."
            % (convert_dtype(dtype)))

    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="mul",
            inputs={"X": input_var,
                    "Y": w},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims,
                   "y_num_col_dims": 1})
        mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


def center_loss(input,
                label,
                num_classes,
                alpha,
                param_attr,
                update_center=True):
    """
    **Center loss Cost layer**
    
    This OP accepts input (deep features,the output of the last hidden layer)
    and target label and return the center loss cost. The average of the 
    distances of each sample in the mini-batch from the center of the 
    corresponding category is calculated as the center loss.
    
    For deep features, :math:`X`, and target labels, :math:`Y`, the equation is:
    
    .. math::

        Out = \\frac{1}{2}(X - Y)^2

    Args:
        input (Variable): a 2-D tensor with shape[N x M]. Its dtype should be float32 or float64.
        label (Variable): the groud truth which is a 2-D tensor
                         with shape[N x 1],where N is the batch size. Its dtype should be int32.
        num_classes (int): the number of classification categories.
        alpha (float|Variable): learning rate of centers.
        param_attr (ParamAttr): Attribute initializer of centers. 
        update_center (bool): whether to update value of center.

    Returns:
        Variable: 2-D tensor with shape [N * 1] 

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid 

          input = fluid.data(name='x',shape=[20,30],dtype='float32')
          label = fluid.data(name='y',shape=[20,1],dtype='int64')
          num_classes = 1000
          alpha = 0.01
          param_attr = fluid.initializer.Xavier(uniform=False)
          center_loss=fluid.layers.center_loss(input=input,
                 label=label,
                 num_classes=1000,
                 alpha=alpha,
                 param_attr=fluid.initializer.Xavier(uniform=False),
                 update_center=True)
    """
    helper = LayerHelper('center_loss', **locals())
    dtype = helper.input_dtype()
    centers_shape = [num_classes, input.shape[1]]
    centers_param = helper.create_parameter(
        attr=param_attr, shape=centers_shape, dtype=dtype)
    centers_param.stop_gradient = True

    if isinstance(alpha, Variable):
        alpha_param = alpha
    else:
        assert isinstance(alpha, float)
        alpha_param = helper.create_variable(
            name="centerloss_alpha",
            shape=[1],
            dtype="float32",
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=True,
            stop_gradient=True,
            initializer=Constant(alpha))

    centersdiff = helper.create_variable_for_type_inference(dtype=input.dtype)
    loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='center_loss',
        inputs={
            'X': [input],
            'Label': [label],
            'Centers': [centers_param],
            'CenterUpdateRate': [alpha_param]
        },
        outputs={
            'SampleCenterDiff': [centersdiff],
            'Loss': [loss],
            'CentersOut': [centers_param]
        },
        attrs={'cluster_num': num_classes,
               'need_update': update_center})
    return loss


def embedding(input,
              size,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype='float32'):
    """

    **WARING:** This OP will be deprecated in a future release. This OP requires the
    last dimension of Tensor shape must be equal to 1. It is recommended to use
    fluid. :ref:`api_fluid_embedding` .

    The operator is used to lookup embeddings vector of ids provided by :attr:`input` .
    It automatically constructs a 2D embedding matrix based on the
    input :attr:`size` (vocab_size, emb_size) and :attr:`dtype` .

    This OP requires the last dimension of Tensor shape must be equal to 1. The shape
    of output Tensor is generated by replacing the last dimension of the input Tensor shape
    with emb_size.

    **Note:** The id in :attr:`input` must satisfy :math:`0 =< id < size[0]` , 
    otherwise the program will throw an exception and exit.

    .. code-block:: text

        Case 1:

        input is a Tensor. padding_idx = -1
            input.data = [[[1], [3]], [[2], [4]], [[4], [127]]]
            input.shape = [3, 2, 1]
        Given size = [128, 16]
        output is a Tensor:
            out.shape = [3, 2, 16]
            out.data = [[[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654]],

                        [[0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365]],
                        
                        [[0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]]  # padding data
        The input padding_idx is less than 0, it is automatically converted to padding_idx = -1 + 128 = 127
        It will pad all-zero data when ids is 127.
        
        Case 2:

        input is a LoDTensor with 1-level LoD. padding_idx = 0
            input.lod = [[2, 3]]
            input.data = [[1], [3], [2], [4], [0]]
            input.shape = [5, 1]
        Given size = [128, 16]
        output is a LoDTensor:
            out.lod = [[2, 3]]
            out.shape = [5, 16]
            out.data = [[0.129435295, 0.244512452, ..., 0.436322452],
                        [0.345421456, 0.524563927, ..., 0.144534654],
                        [0.345249859, 0.124939536, ..., 0.194353745],
                        [0.945345345, 0.435394634, ..., 0.435345365],
                        [0.0,         0.0,         ..., 0.0        ]]  # padding data
        It will pad all-zero data when ids is 0.

    Args:
        input(Variable): A Tensor or LoDTensor with type int64, which contains the id information.
            The last dimension of Tensor shape must be equal to 1. The value of the input id should
            satisfy :math:`0<= id < size[0]` .
        size(tuple|list): The shape of lookup table parameter. It should have two elements which
            indicates the size of the dictionary of embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set 
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_fluid_optimizer_AdadeltaOptimizer` , :ref:`api_fluid_optimizer_AdamaxOptimizer` , 
            :ref:`api_fluid_optimizer_DecayedAdagradOptimizer` , :ref:`api_fluid_optimizer_FtrlOptimizer` ,
            :ref:`api_fluid_optimizer_LambOptimizer` and :ref:`api_fluid_optimizer_LarsMomentumOptimizer` .
            In these case, is_sparse must be False. Default: False.
        is_distributed(bool): Whether to store the embedding matrix in a distributed manner. Only used
            in multi-machine distributed CPU training. Default: False.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-vocab_size, vocab_size). 
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        param_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter. 
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector shoud be consistent with :attr:`size` . Then :ref:`api_fluid_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example 2 for details.
        dtype(str|core.VarDesc.VarType): It refers to the data type of output Tensor.
            It must be float32 or float64. Default: float32.

    Returns:
        Variable: Embedding Tensor or LoDTensor mapped by input. The data type is the same as :attr:`dtype` .

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          data = fluid.data(name='x', shape=[None, 1], dtype='int64')

          # exampel 1
          emb_1 = fluid.embedding(input=data, size=[128, 64])

          # example 2: load custom or pre-trained word vectors
          weight_data = np.random.random(size=(128, 100))  # word vectors with numpy format
          w_param_attrs = fluid.ParamAttr(
              name="emb_weight",
              learning_rate=0.5,
              initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
              trainable=True)
          emb_2 = fluid.layers.embedding(input=data, size=(128, 100), param_attr=w_param_attrs, dtype='float32')   
    """

    helper = LayerHelper('embedding', **locals())
    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in layers.embedding must be Variable, but received %s"
            % (type(input)))
    if convert_dtype(input.dtype) not in ['int64']:
        raise TypeError(
            "The data type of 'input' in layers.embedding must be int64, but received %s."
            % (convert_dtype(input.dtype)))
    if convert_dtype(dtype) in ['float16']:
        warnings.warn(
            "The 'dtype' of layers.embedding only support float16 in GPU now.")
    if convert_dtype(dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The 'dtype' of layers.embedding must be float16, float32 or float64, but received %s."
            % (convert_dtype(dtype)))
    remote_prefetch = is_sparse and (not is_distributed)
    if remote_prefetch:
        assert is_sparse is True and is_distributed is False
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_variable_for_type_inference(dtype)
    padding_idx = -1 if padding_idx is None else padding_idx if padding_idx >= 0 else (
        size[0] + padding_idx)
    helper.append_op(
        type='lookup_table',
        inputs={'Ids': input,
                'W': w},
        outputs={'Out': tmp},
        attrs={
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'remote_prefetch': remote_prefetch,
            'padding_idx': padding_idx
        })
    return tmp


def _pull_box_sparse(input, size, dtype='float32'):
    """
    **Pull Box Sparse Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    BoxPS lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    Args:
        input(Variable|list of Variable): Input is a Tensor<int64> Variable, which 
            contains the IDs information.
        size(int): The embedding size parameter, which indicates the size of 
            each embedding vector respectively.
        dtype(str): The dtype refers to the data type of output tensor. Only supports 
	    float32 now.

    Returns:
        Variable|list of Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.layers.data(name='sequence', shape=[1], dtype='int64', lod_level=1)
          emb = fluid.layers.pull_box_sparse(input=data, size=[11])    
    """
    helper = LayerHelper('pull_box_sparse', **locals())
    if dtype != 'float32':
        raise ValueError(
            "BoxPS only support float type embedding now, and your type is: " +
            dtype)
    helper.input_dtype()
    inputs = helper.multiple_input()
    outs = [
        helper.create_variable_for_type_inference(dtype)
        for i in range(len(inputs))
    ]
    helper.append_op(
        type='pull_box_sparse',
        inputs={'Ids': inputs},
        outputs={'Out': outs},
        attrs={'size': size})
    if len(outs) == 1:
        return outs[0]
    return outs


def dynamic_lstm(input,
                 size,
                 h_0=None,
                 c_0=None,
                 param_attr=None,
                 bias_attr=None,
                 use_peepholes=True,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 cell_activation='tanh',
                 candidate_activation='tanh',
                 dtype='float32',
                 name=None):
    """
    **Note**:
        1. This OP only supports LoDTensor as inputs. If you need to deal with Tensor, please use :ref:`api_fluid_layers_lstm` .
        2. In order to improve efficiency, users must first map the input of dimension [T, hidden_size] to input of [T, 4 * hidden_size], and then pass it to this OP.

    The implementation of this OP include diagonal/peephole connections.
    Please refer to `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ .
    If you do not need peephole connections, please set use_peepholes to False .

    This OP computes each timestep as follows:

    .. math::
      i_t = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_{x_i} + b_{h_i})
    .. math::
      f_t = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_{x_f} + b_{h_f})
    .. math::
      o_t = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_{x_o} + b_{h_o})
    .. math::
      \widetilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + b{x_c} + b_{h_c})
    .. math::
      c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
    .. math::
      h_t = o_t \odot tanh(c_t)

    The symbolic meanings in the formula are as follows:

    - :math:`x_{t}` represents the input at timestep :math:`t`
    - :math:`h_{t}` represents the hidden state at timestep :math:`t`
    - :math:`h_{t-1}, c_{t-1}` represent the hidden state and cell state at timestep :math:`t-1` , respectively
    - :math:`\widetilde{c_t}` represents the candidate cell state
    - :math:`i_t` , :math:`f_t` and :math:`o_t` represent input gate, forget gate, output gate, respectively
    - :math:`W` represents weight (e.g., :math:`W_{ix}` is the weight of a linear transformation of input :math:`x_{t}` when calculating input gate :math:`i_t` )
    - :math:`b` represents bias (e.g., :math:`b_{i}` is the bias of input gate)
    - :math:`\sigma` represents nonlinear activation function for gate, default sigmoid
    - :math:`\odot` represents the Hadamard product of a matrix, i.e. multiplying the elements of the same position for two matrices with the same dimension to get another matrix with the same dimension

    Parameters:
        input ( :ref:`api_guide_Variable_en` ): LSTM input tensor, multi-dimensional LODTensor of shape :math:`[T, 4*hidden\_size]` . Data type is float32 or float64.
        size (int): must be 4 * hidden_size.
        h_0( :ref:`api_guide_Variable_en` , optional): The initial hidden state of the LSTM, multi-dimensional Tensor of shape :math:`[batch\_size, hidden\_size]` .
                       Data type is float32 or float64. If set to None, it will be a vector of all 0. Default: None.
        c_0( :ref:`api_guide_Variable_en` , optional): The initial hidden state of the LSTM, multi-dimensional Tensor of shape :math:`[batch\_size, hidden\_size]` .
                       Data type is float32 or float64. If set to None, it will be a vector of all 0. `h_0` and `c_0` can be None but only at the same time. Default: None.
        param_attr(ParamAttr, optional): Parameter attribute of weight. If it is None, the default weight parameter attribute is used. Please refer to ref:`api_fluid_ParamAttr' .
                              If the user needs to set this parameter, the dimension must be :math:`[hidden\_size, 4*hidden\_size]` . Default: None.

                              - Weights = :math:`\{ W_{cr},W_{ir},W_{fr},W_{or} \}` , the shape is [hidden_size, 4*hidden_size].

        bias_attr (ParamAttr, optional): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.
                              Please refer to ref:`api_fluid_ParamAttr' . Default: None.

                              1. `use_peepholes = False`
                                 - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                 - The shape is [1, 4*hidden_size].
                              2. `use_peepholes = True`
                                 - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                 - The shape is [1, 7*hidden_size].
                                 
        use_peepholes (bool, optional): Whether to use peephole connection or not. Default: True.
        is_reverse (bool, optional): Whether to calculate reverse LSTM. Default: False.
        gate_activation (str, optional): The activation for input gate, forget gate and output gate. Default: "sigmoid".
        cell_activation (str, optional): The activation for cell output. Default: "tanh".
        candidate_activation (str, optional): The activation for candidate hidden state. Default: "tanh".
        dtype (str, optional): Data type, can be "float32" or "float64". Default: "float32".
        name (str, optional): A name for this layer. Please refer to :ref:`api_guide_Name` . Default: None.

    Returns:
        tuple ( :ref:`api_guide_Variable` , :ref:`api_guide_Variable` ) :

            The hidden state and cell state of LSTM

                - hidden: LoDTensor with shape of :math:`[T, hidden\_size]` , and its lod and dtype is the same as the input.
                - cell: LoDTensor with shape of :math:`[T, hidden\_size]` , and its lod and dtype is the same as the input.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            emb_dim = 256
            vocab_size = 10000
            hidden_dim = 512
            
            data = fluid.data(name='x', shape=[None], dtype='int64', lod_level=1)
            emb = fluid.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)

            forward_proj = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                           bias_attr=False)

            forward, cell = fluid.layers.dynamic_lstm(
                input=forward_proj, size=hidden_dim * 4, use_peepholes=False)
            forward.shape  # (-1, 512)
            cell.shape  # (-1, 512)
    """
    assert in_dygraph_mode(
    ) is not True, "please use lstm instead of dynamic_lstm in dygraph mode!"
    assert bias_attr is not False, "bias_attr should not be False in dynamic_lstmp."
    helper = LayerHelper('lstm', **locals())
    size = size // 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 4 * size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    hidden = helper.create_variable_for_type_inference(dtype)
    cell = helper.create_variable_for_type_inference(dtype)
    batch_gate = helper.create_variable_for_type_inference(dtype)
    batch_cell_pre_act = helper.create_variable_for_type_inference(dtype)
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    batch_size = input.shape[0]
    if h_0:
        assert h_0.shape == (batch_size, size), \
            'The shape of h0 should be (batch_size, %d)' % size
        inputs['H0'] = h_0
    if c_0:
        assert c_0.shape == (batch_size, size), \
            'The shape of c0 should be (batch_size, %d)' % size
        inputs['C0'] = c_0

    helper.append_op(
        type='lstm',
        inputs=inputs,
        outputs={
            'Hidden': hidden,
            'Cell': cell,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act
        },
        attrs={
            'use_peepholes': use_peepholes,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation
        })
    return hidden, cell


def lstm(input,
         init_h,
         init_c,
         max_len,
         hidden_size,
         num_layers,
         dropout_prob=0.0,
         is_bidirec=False,
         is_test=False,
         name=None,
         default_initializer=None,
         seed=-1):
    """
    **Note**:
        This OP only supports running on GPU devices.

    This OP implements LSTM operation - `Hochreiter, S., & Schmidhuber, J. (1997) <http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf>`_ .

    The implementation of this OP does not include diagonal/peephole connections.
    Please refer to `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ .
    If you need peephole connections, please use :ref:`api_fluid_layers_dynamic_lstm` .

    This OP computes each timestep as follows:

    .. math::
      i_t = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + b_{x_i} + b_{h_i})
    .. math::
      f_t = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + b_{x_f} + b_{h_f})
    .. math::
      o_t = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + b_{x_o} + b_{h_o})
    .. math::
      \widetilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + b{x_c} + b_{h_c})
    .. math::
      c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
    .. math::
      h_t = o_t \odot tanh(c_t)

    The symbolic meanings in the formula are as follows:

    - :math:`x_{t}` represents the input at timestep :math:`t`
    - :math:`h_{t}` represents the hidden state at timestep :math:`t`
    - :math:`h_{t-1}, c_{t-1}` represent the hidden state and cell state at timestep :math:`t-1` , respectively
    - :math:`\widetilde{c_t}` represents the candidate cell state
    - :math:`i_t` , :math:`f_t` and :math:`o_t` represent input gate, forget gate, output gate, respectively
    - :math:`W` represents weight (e.g., :math:`W_{ix}` is the weight of a linear transformation of input :math:`x_{t}` when calculating input gate :math:`i_t` )
    - :math:`b` represents bias (e.g., :math:`b_{i}` is the bias of input gate)
    - :math:`\sigma` represents nonlinear activation function for gate, default sigmoid
    - :math:`\odot` represents the Hadamard product of a matrix, i.e. multiplying the elements of the same position for two matrices with the same dimension to get another matrix with the same dimension

    Parameters:
        input ( :ref:`api_guide_Variable_en` ): LSTM input tensor, 3-D Tensor of shape :math:`[batch\_size, seq\_len, input\_dim]` . Data type is float32 or float64
        init_h( :ref:`api_guide_Variable_en` ): The initial hidden state of the LSTM, 3-D Tensor of shape :math:`[num\_layers, batch\_size, hidden\_size]` .
                       If is_bidirec = True, shape should be :math:`[num\_layers*2, batch\_size, hidden\_size]` . Data type is float32 or float64.
        init_c( :ref:`api_guide_Variable_en` ): The initial cell state of the LSTM, 3-D Tensor of shape :math:`[num\_layers, batch\_size, hidden\_size]` .
                       If is_bidirec = True, shape should be :math:`[num\_layers*2, batch\_size, hidden\_size]` . Data type is float32 or float64.
        max_len (int): max length of LSTM. the first dim of input tensor CAN NOT greater than max_len.
        hidden_size (int): hidden size of the LSTM.
        num_layers (int): total layers number of the LSTM.
        dropout_prob(float, optional): dropout prob, dropout ONLY work between rnn layers, NOT between time steps
                             There is NO dropout work on rnn output of the last RNN layers.
                             Default: 0.0.
        is_bidirec (bool, optional): If it is bidirectional. Default: False.
        is_test (bool, optional): If it is in test phrase. Default: False.
        name (str, optional): A name for this layer. If set None, the layer
                         will be named automatically. Default: None.
        default_initializer(Initializer, optional): Where use initializer to initialize the Weight
                         If set None, defaule initializer will be used. Default: None.
        seed(int, optional): Seed for dropout in LSTM, If it's -1, dropout will use random seed. Default: 1.


    Returns:
        tuple ( :ref:`api_guide_Variable_en` , :ref:`api_guide_Variable_en` , :ref:`api_guide_Variable_en` ) :

                        Three tensors, rnn_out, last_h, last_c:

                        - rnn_out is result of LSTM hidden, shape is :math:`[seq\_len, batch\_size, hidden\_size]` \
                          if is_bidirec set to True, shape will be :math:`[seq\_len, batch\_size, hidden\_size*2]`
                        - last_h is the hidden state of the last step of LSTM \
                          shape is :math:`[num\_layers, batch\_size, hidden\_size]` \
                          if is_bidirec set to True, shape will be :math:`[num\_layers*2, batch\_size, hidden\_size]`
                        - last_c(Tensor): the cell state of the last step of LSTM \
                          shape is :math:`[num\_layers, batch\_size, hidden\_size]` \
                          if is_bidirec set to True, shape will be :math:`[num\_layers*2, batch\_size, hidden\_size]`


    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            emb_dim = 256
            vocab_size = 10000
            data = fluid.data(name='x', shape=[None, 100], dtype='int64')
            emb = fluid.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
            batch_size = 20
            max_len = 100
            dropout_prob = 0.2
            input_size = 100
            hidden_size = 150
            num_layers = 1
            init_h = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
            init_c = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
            rnn_out, last_h, last_c = layers.lstm( emb, init_h, init_c, \
                    max_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)
            rnn_out.shape  # (-1, 100, 150)
            last_h.shape  # (1, 20, 150)
            last_c.shape  # (1, 20, 150)
    """

    helper = LayerHelper('cudnn_lstm', **locals())

    dtype = input.dtype
    input_shape = list(input.shape)
    input_size = input_shape[-1]
    weight_size = 0
    for i in range(num_layers):
        if i == 0:
            input_weight_size = (input_size * hidden_size) * 4
        else:
            if is_bidirec:
                input_weight_size = (hidden_size * 2 * hidden_size) * 4
            else:
                input_weight_size = (hidden_size * hidden_size) * 4

        hidden_weight_size = (hidden_size * hidden_size) * 4

        if is_bidirec:
            weight_size += (input_weight_size + hidden_weight_size) * 2
            weight_size += hidden_size * 8 * 2
        else:
            weight_size += input_weight_size + hidden_weight_size
            weight_size += hidden_size * 8

    weight = helper.create_parameter(
        attr=helper.param_attr,
        shape=[weight_size],
        dtype=dtype,
        default_initializer=default_initializer)

    out = helper.create_variable_for_type_inference(dtype)
    last_h = helper.create_variable_for_type_inference(dtype)
    last_c = helper.create_variable_for_type_inference(dtype)

    cache = helper.create_variable(
        persistable=True, type=core.VarDesc.VarType.RAW, stop_gradient=True)

    helper.append_op(
        type='cudnn_lstm',
        inputs={
            'Input': input,
            'InitH': init_h,
            'InitC': init_c,
            'W': weight,
            'Cache': cache,
        },
        outputs={
            'Out': out,
            'last_h': last_h,
            'last_c': last_c,
        },
        attrs={
            'max_len': max_len,
            'is_bidirec': is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'is_test': is_test,
            'dropout_prob': dropout_prob,
            'seed': seed,
        })
    return out, last_h, last_c


def dynamic_lstmp(input,
                  size,
                  proj_size,
                  param_attr=None,
                  bias_attr=None,
                  use_peepholes=True,
                  is_reverse=False,
                  gate_activation='sigmoid',
                  cell_activation='tanh',
                  candidate_activation='tanh',
                  proj_activation='tanh',
                  dtype='float32',
                  name=None,
                  h_0=None,
                  c_0=None,
                  cell_clip=None,
                  proj_clip=None):
    """
    **Note**:
        1. In order to improve efficiency, users must first map the input of dimension [T, hidden_size] to input of [T, 4 * hidden_size], and then pass it to this OP.

    This OP implements the LSTMP (LSTM Projected) layer.
    The LSTMP layer has a separate linear mapping layer behind the LSTM layer. -- `Sak, H., Senior, A., & Beaufays, F. (2014) <https://ai.google/research/pubs/pub43905.pdf>`_ .

    Compared with the standard LSTM layer, LSTMP has an additional linear mapping layer,
    which is used to map from the original hidden state :math:`h_t` to the lower dimensional state :math:`r_t` .
    This reduces the total number of parameters and computational complexity, especially when the output unit is relatively large.

    The default implementation of the OP contains diagonal/peephole connections,
    please refer to `Gers, F. A., & Schmidhuber, J. (2000) <ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf>`_ .
    If you need to disable the peephole connections, set use_peepholes to False.

    This OP computes each timestep as follows:

    .. math::
      i_t = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)
    .. math::
          f_t = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)
    .. math::
          o_t = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_{t-1} + b_o)
    .. math::
          \widetilde{c_t} = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)
    .. math::
          c_t = f_t \odot c_{t-1} + i_t \odot \widetilde{c_t}
    .. math::
          h_t = o_t \odot act_h(c_t)
    .. math::
          r_t = \overline{act_h}(W_{rh}h_t)

    The symbolic meanings in the formula are as follows:

    - :math:`x_{t}` represents the input at timestep :math:`t`
    - :math:`h_{t}` represents the hidden state at timestep :math:`t`
    - :math:`r_{t}` : represents the state of the projected output of the hidden state :math:`h_{t}`
    - :math:`h_{t-1}, c_{t-1}, r_{t-1}` represent the hidden state, cell state and projected output at timestep :math:`t-1` , respectively
    - :math:`\widetilde{c_t}` represents the candidate cell state
    - :math:`i_t` , :math:`f_t` and :math:`o_t` represent input gate, forget gate, output gate, respectively
    - :math:`W` represents weight (e.g., :math:`W_{ix}` is the weight of a linear transformation of input :math:`x_{t}` when calculating input gate :math:`i_t` )
    - :math:`b` represents bias (e.g., :math:`b_{i}` is the bias of input gate)
    - :math:`\sigma` represents nonlinear activation function for gate, default sigmoid
    - :math:`\odot` represents the Hadamard product of a matrix, i.e. multiplying the elements of the same position for two matrices with the same dimension to get another matrix with the same dimension

    Parameters:
        input( :ref:`api_guide_Variable_en` ): The input of dynamic_lstmp layer, which supports
                         variable-time length input sequence.
                         It is a multi-dimensional LODTensor of shape :math:`[T, 4*hidden\_size]` . Data type is float32 or float64.
        size(int): must be 4 * hidden_size.
        proj_size(int): The size of projection output.
        param_attr(ParamAttr, optional): Parameter attribute of weight. If it is None, the default weight parameter attribute is used. Please refer to ref:`api_fluid_ParamAttr' .
                              If the user needs to set this parameter, the dimension must be :math:`[hidden\_size, 4*hidden\_size]` . Default: None.

                              - Weights = :math:`\{ W_{cr},W_{ir},W_{fr},W_{or} \}` , the shape is [P, 4*hidden_size] , where P is the projection size.
                              - Projection weight  = :math:`\{ W_{rh} \}` , the shape is [hidden_size, P].

        bias_attr (ParamAttr, optional): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.
                              Please refer to ref:`api_fluid_ParamAttr' . Default: None.

                              1. `use_peepholes = False`
                                 - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                 - The shape is [1, 4*hidden_size].
                              2. `use_peepholes = True`
                                 - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                 - The shape is [1, 7*hidden_size].

        use_peepholes (bool, optional): Whether to use peephole connection or not. Default True.
        is_reverse (bool, optional): Whether to calculate reverse LSTM. Default False.
        gate_activation (str, optional): The activation for input gate, forget gate and output gate. Default "sigmoid".
        cell_activation (str, optional): The activation for cell output. Default "tanh".
        candidate_activation (str, optional): The activation for candidate hidden state. Default "tanh".
        proj_activation(str, optional): The activation for projection output. Default "tanh".
        dtype (str, optional): Data type, can be "float32" or "float64". Default "float32".
        name (str, optional): A name for this layer. Please refer to :ref:`api_guide_Name` . Default: None.
        h_0( :ref:`api_guide_Variable` , optional): The initial hidden state is an optional input, default is zero.
                       This is a tensor with shape :math:`[batch\_size, P]` , where P is the projection size. Default: None.
        c_0( :ref:`api_guide_Variable` , optional): The initial cell state is an optional input, default is zero.
                       This is a tensor with shape :math:`[batch\_size, P]` , where P is the projection size.
                       `h_0` and `c_0` can be None but only at the same time. Default: None.
        cell_clip(float, optional): If not None, the cell state is clipped
                             by this value prior to the cell output activation. Default: None.
        proj_clip(float, optional): If `num_proj > 0` and `proj_clip` is
                            provided, then the projected values are clipped elementwise to within
                            `[-proj_clip, proj_clip]`. Default: None.

    Returns:
        tuple ( :ref:`api_guide_Variable` , :ref:`api_guide_Variable` ) :

                The hidden state and cell state of LSTMP

                - hidden: LoDTensor with shape of :math:`[T, P]` , and its lod and dtype is the same as the input.
                - cell: LoDTensor with shape of :math:`[T, hidden\_size]` , and its lod and dtype is the same as the input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='sequence', shape=[None], dtype='int64', lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim, proj_dim = 512, 256
            fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                    act=None, bias_attr=None)
            proj_out, last_c = fluid.layers.dynamic_lstmp(input=fc_out,
                                                    size=hidden_dim * 4,
                                                    proj_size=proj_dim,
                                                    use_peepholes=False,
                                                    is_reverse=True,
                                                    cell_activation="tanh",
                                                    proj_activation="tanh")
            proj_out.shape  # (-1, 256)
            last_c.shape  # (-1, 512)
    """

    assert in_dygraph_mode(
    ) is not True, "please use lstm instead of dynamic_lstmp in dygraph mode!"

    assert bias_attr is not False, "bias_attr should not be False in dynamic_lstmp."
    helper = LayerHelper('lstmp', **locals())
    size = size // 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[proj_size, 4 * size], dtype=dtype)
    proj_weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, proj_size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    projection = helper.create_variable_for_type_inference(dtype)
    cell = helper.create_variable_for_type_inference(dtype)
    ordered_proj0 = helper.create_variable_for_type_inference(dtype)
    batch_hidden = helper.create_variable_for_type_inference(dtype)
    batch_gate = helper.create_variable_for_type_inference(dtype)
    batch_cell_pre_act = helper.create_variable_for_type_inference(dtype)
    inputs = {
        'Input': input,
        'Weight': weight,
        'ProjWeight': proj_weight,
        'Bias': bias
    }
    batch_size = input.shape[0]
    if h_0:
        assert h_0.shape == (batch_size, proj_size), \
            'The shape of h0 should be (batch_size, %d)' % proj_size
        inputs['H0'] = h_0
    if c_0:
        assert c_0.shape == (batch_size, size), \
            'The shape of c0 should be (batch_size, %d)' % size
        inputs['C0'] = c_0

    if cell_clip:
        assert cell_clip >= 0, "cell_clip should not be negtive."
    if proj_clip:
        assert proj_clip >= 0, "proj_clip should not be negtive."

    helper.append_op(
        type='lstmp',
        inputs=inputs,
        outputs={
            'Projection': projection,
            'Cell': cell,
            'BatchHidden': batch_hidden,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act
        },
        attrs={
            'use_peepholes': use_peepholes,
            'cell_clip': cell_clip,
            'proj_clip': proj_clip,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation,
            'proj_activation': proj_activation
        })
    return projection, cell


def dynamic_gru(input,
                size,
                param_attr=None,
                bias_attr=None,
                is_reverse=False,
                gate_activation='sigmoid',
                candidate_activation='tanh',
                h_0=None,
                origin_mode=False):
    """
    **Note: The input type of this must be LoDTensor. If the input type to be
    processed is Tensor, use** :ref:`api_fluid_layers_StaticRNN` .

    This operator is used to perform the calculations for a single layer of
    Gated Recurrent Unit (GRU) on full sequences step by step. The calculations
    in one time step support these two modes:

    If ``origin_mode`` is True, then the formula used is from paper
    `Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ .

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    if ``origin_mode`` is False, then the formula used is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling  <https://arxiv.org/pdf/1412.3555.pdf>`_

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \\tilde{h_t}

    :math:`x_t` is the input of current time step, but it is not from ``input`` .
    This operator does not include the calculations :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` ,
    **Note** thus a fully-connect layer whose size is 3 times of ``size`` should
    be used before this operator, and the output should be used as ``input`` here.
    :math:`h_{t-1}` is the hidden state from previous time step. 
    :math:`u_t` , :math:`r_t` , :math:`\\tilde{h_t}` and :math:`h_t` stand for
    update gate, reset gate, candidate hidden and hidden output separately.
    :math:`W_{uh}, b_u` , :math:`W_{rh}, b_r` and :math:`W_{ch}, b_c` stand for
    the weight matrix and bias used in update gate, reset gate, candidate hidden
    calculations. For implementation, the three weight matrix are merged into a
    tensor shaped :math:`[D, D \\times 3]` , the three bias are concatenated as
    a tensor shaped :math:`[1, D \\times 3]` , where :math:`D` stands for the
    hidden size; The data layout of weight tensor is: :math:`W_{uh}` and :math:`W_{rh}`
    are concatenated with shape :math:`[D, D  \\times 2]` lying on the first part,
    and :math:`W_{ch}` lying on the latter part with shape :math:`[D, D]` .


    Args:
        input(Variable): A LoDTensor whose lod level is 1, representing the input
            after linear projection. Its shape should be :math:`[T, D \\times 3]` ,
            where :math:`T` stands for the total sequence lengths in this mini-batch,
            :math:`D` for the hidden size. The data type should be float32 or float64.
        size(int): Indicate the hidden size.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        is_reverse(bool, optional): Whether to compute in the reversed order of
            input sequences. Default False.
        gate_activation(str, optional): The activation fuction corresponding to
            :math:`act_g` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "sigmoid".
        candidate_activation(str, optional): The activation fuction corresponding to
            :math:`act_c` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "tanh".
        h_0 (Variable, optional): A Tensor representing the initial hidden state.
            It not provided, the default initial hidden state is 0. The shape is
            :math:`[N, D]` , where :math:`N` is the number of sequences in the
            mini-batch, :math:`D` for the hidden size. The data type should be
            same as ``input`` . Default None.

    Returns:
        Variable: A LoDTensor whose lod level is 1 and shape is :math:`[T, D]` , \
            where :math:`T` stands for the total sequence lengths in this mini-batch \
            :math:`D` for the hidden size. It represents GRU transformed sequence output, \
            and has the same lod and data type with ``input`` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='sequence',
                      shape=[None],
                      dtype='int64',
                      lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            hidden = fluid.layers.dynamic_gru(input=x, size=hidden_dim)
    """

    assert in_dygraph_mode(
    ) is not True, "please use gru instead of dynamic_gru in dygraph mode!"

    helper = LayerHelper('gru', **locals())
    dtype = helper.input_dtype()

    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=[1, 3 * size], dtype=dtype, is_bias=True)
    batch_size = input.shape[0]
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    if h_0:
        assert h_0.shape == (
            batch_size, size
        ), 'The shape of h0 should be(batch_size, %d)' % size
        inputs['H0'] = h_0

    hidden = helper.create_variable_for_type_inference(dtype)
    batch_gate = helper.create_variable_for_type_inference(dtype)
    batch_reset_hidden_prev = helper.create_variable_for_type_inference(dtype)
    batch_hidden = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='gru',
        inputs=inputs,
        outputs={
            'Hidden': hidden,
            'BatchGate': batch_gate,
            'BatchResetHiddenPrev': batch_reset_hidden_prev,
            'BatchHidden': batch_hidden
        },
        attrs={
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'activation': candidate_activation,
            'origin_mode': origin_mode
        })
    return hidden


def gru_unit(input,
             hidden,
             size,
             param_attr=None,
             bias_attr=None,
             activation='tanh',
             gate_activation='sigmoid',
             origin_mode=False):
    """
    Gated Recurrent Unit (GRU) RNN cell. This operator performs GRU calculations for
    one time step and it supports these two modes:

    If ``origin_mode`` is True, then the formula used is from paper
    `Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ .

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    if ``origin_mode`` is False, then the formula used is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling  <https://arxiv.org/pdf/1412.3555.pdf>`_

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \\tilde{h_t}

    :math:`x_t` is the input of current time step, but it is not ``input`` .
    This operator does not include the calculations :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` ,
    **Note** thus a fully-connect layer whose size is 3 times of GRU hidden size should
    be used before this operator, and the output should be used as ``input`` here.
    :math:`h_{t-1}` is the hidden state from previous time step. 
    :math:`u_t` , :math:`r_t` , :math:`\\tilde{h_t}` and :math:`h_t` stand for
    update gate, reset gate, candidate hidden and hidden output separately.
    :math:`W_{uh}, b_u` , :math:`W_{rh}, b_r` and :math:`W_{ch}, b_c` stand for
    the weight matrix and bias used in update gate, reset gate, candidate hidden
    calculations. For implementation, the three weight matrix are merged into a
    tensor shaped :math:`[D, D \\times 3]` , the three bias are concatenated as
    a tensor shaped :math:`[1, D \\times 3]` , where :math:`D` stands for the
    hidden size; The data layout of weight tensor is: :math:`W_{uh}` and :math:`W_{rh}`
    are concatenated with shape :math:`[D, D  \\times 2]` lying on the first part,
    and :math:`W_{ch}` lying on the latter part with shape :math:`[D, D]` .


    Args:
        input(Variable): A 2D Tensor representing the input after linear projection
            after linear projection. Its shape should be :math:`[N, D \\times 3]` ,
            where :math:`N` stands for batch size, :math:`D` for the hidden size.
            The data type should be float32 or float64.
        hidden(Variable): A 2D Tensor representing the hidden state from previous step.
            Its shape should be :math:`[N, D]` , where :math:`N` stands for batch size,
            :math:`D` for the hidden size. The data type should be same as ``input`` .
        size(int): Indicate the hidden size.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        activation(str, optional): The activation fuction corresponding to
            :math:`act_c` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "tanh".
        gate_activation(str, optional): The activation fuction corresponding to
            :math:`act_g` in the formula. "sigmoid", "tanh", "relu" and "identity"
            are supported. Default "sigmoid".

    Returns:
        tuple: The tuple contains three Tensor variables with the same data type \
            as ``input`` . They represent the hidden state for next time step ( :math:`h_t` ), \
            reseted previous hidden state ( :math:`r_t \odot h_{t-1}` ), and the \
            concatenation of :math:`h_t, r_t, \\tilde{h_t}` . And they have shape \
            :math:`[N, D]` , :math:`[N, D]` , :math:`[N, D \times 3]` separately. \
            Usually only the hidden state for next time step ( :math:`h_t` ) is used \
            as output and state, the other two are intermediate results of calculations.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='step_data', shape=[None], dtype='int64')
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            pre_hidden = fluid.data(
                name='pre_hidden', shape=[None, hidden_dim], dtype='float32')
            hidden = fluid.layers.gru_unit(
                input=x, hidden=pre_hidden, size=hidden_dim * 3)

    """
    activation_dict = dict(
        identity=0,
        sigmoid=1,
        tanh=2,
        relu=3, )
    activation = activation_dict[activation]
    gate_activation = activation_dict[gate_activation]

    helper = LayerHelper('gru_unit', **locals())
    dtype = helper.input_dtype()
    size = size // 3

    # create weight
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)

    gate = helper.create_variable_for_type_inference(dtype)
    reset_hidden_pre = helper.create_variable_for_type_inference(dtype)
    updated_hidden = helper.create_variable_for_type_inference(dtype)
    inputs = {'Input': input, 'HiddenPrev': hidden, 'Weight': weight}
    # create bias
    if helper.bias_attr:
        bias_size = [1, 3 * size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias

    helper.append_op(
        type='gru_unit',
        inputs=inputs,
        outputs={
            'Gate': gate,
            'ResetHiddenPrev': reset_hidden_pre,
            'Hidden': updated_hidden,
        },
        attrs={
            'activation': 2,  # tanh
            'gate_activation': 1,  # sigmoid
            'origin_mode': origin_mode
        })

    return updated_hidden, reset_hidden_pre, gate


@templatedoc()
def linear_chain_crf(input, label, param_attr=None, length=None):
    """
    Linear Chain CRF.

    ${comment}

    Args:
        input(${emission_type}): ${emission_comment} 
        label(${label_type}): ${label_comment}
        Length(${length_type}): ${length_comment}
        param_attr(ParamAttr): The attribute of the learnable parameter for transition parameter.

    Returns:
        output(${emission_exps_type}): ${emission_exps_comment} \n
        output(${transition_exps_type}): ${transition_exps_comment} \n
        output(${log_likelihood_type}): ${log_likelihood_comment} \n

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            #define net structure, using LodTensor
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data = fluid.data(name='input_data', shape=[-1,10], dtype='float32')
                label = fluid.data(name='label', shape=[-1,1], dtype='int')
                emission= fluid.layers.fc(input=input_data, size=10, act="tanh")
                crf_cost = fluid.layers.linear_chain_crf(
                    input=emission,
                    label=label,
                    param_attr=fluid.ParamAttr(
                    name='crfw',
                    learning_rate=0.01)) 
            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)    
            #define data, using LoDTensor
            a = fluid.create_lod_tensor(np.random.rand(12,10).astype('float32'), [[3,3,4,2]], place)
            b = fluid.create_lod_tensor(np.array([[1],[1],[2],[3],[1],[1],[1],[3],[1],[1],[1],[1]]),[[3,3,4,2]] , place)
            feed1 = {'input_data':a,'label':b}
            loss= exe.run(train_program,feed=feed1, fetch_list=[crf_cost])
            print(loss) 

            #define net structure, using padding
            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                input_data2 = fluid.data(name='input_data2', shape=[-1,10,10], dtype='float32')
                label2 = fluid.data(name='label2', shape=[-1,10,1], dtype='int')
                label_length = fluid.data(name='length', shape=[-1,1], dtype='int')
                emission2= fluid.layers.fc(input=input_data2, size=10, act="tanh", num_flatten_dims=2)
                crf_cost2 = fluid.layers.linear_chain_crf(
                    input=emission2,
                    label=label2,
                    length=label_length,
                    param_attr=fluid.ParamAttr(
                     name='crfw',
                     learning_rate=0.01))

            use_cuda = False
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_program)

            #define data, using padding
            cc=np.random.rand(4,10,10).astype('float32')
            dd=np.random.rand(4,10,1).astype('int64')
            ll=np.array([[3],[3],[4],[2]])
            feed2 = {'input_data2':cc,'label2':dd,'length':ll}
            loss2= exe.run(train_program,feed=feed2, fetch_list=[crf_cost2])
            print(loss2) 
            #[array([[ 7.8902354],
            #        [ 7.3602567],
            #        [ 10.004011],
            #        [ 5.86721  ]], dtype=float32)]

            #you can use find_var to get transition parameter.
            transition=np.array(fluid.global_scope().find_var('crfw').get_tensor())
            print(transition)
            
    """
    helper = LayerHelper('linear_chain_crf', **locals())
    size = input.shape[2] if length else input.shape[1]
    transition = helper.create_parameter(
        attr=helper.param_attr,
        shape=[size + 2, size],
        dtype=helper.input_dtype())
    alpha = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    emission_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    transition_exps = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    log_likelihood = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    this_inputs = {
        "Emission": [input],
        "Transition": transition,
        "Label": [label]
    }
    if length:
        this_inputs['Length'] = [length]
    helper.append_op(
        type='linear_chain_crf',
        inputs=this_inputs,
        outputs={
            "Alpha": [alpha],
            "EmissionExps": [emission_exps],
            "TransitionExps": transition_exps,
            "LogLikelihood": log_likelihood
        })

    return log_likelihood


@templatedoc()
def crf_decoding(input, param_attr, label=None, length=None):
    """
    ${comment}

    Args:
        input(${emission_type}): ${emission_comment}

        param_attr (ParamAttr|None): To specify the weight parameter attribute. 
            Default: None, which means the default weight parameter property is 
            used. See usage for details in :ref:`api_fluid_ParamAttr` .

        label(${label_type}, optional): ${label_comment}
        
        length(${length_type}, optional): ${length_comment}

    Returns:
        Variable: ${viterbi_path_comment}

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid

           # LoDTensor-based example
           num_labels = 10
           feature = fluid.data(name='word_emb', shape=[-1, 784], dtype='float32', lod_level=1)
           label = fluid.data(name='label', shape=[-1, 1], dtype='int64', lod_level=1)
           emission = fluid.layers.fc(input=feature, size=num_labels)
           
           crf_cost = fluid.layers.linear_chain_crf(input=emission, label=label, 
                     param_attr=fluid.ParamAttr(name="crfw"))
           crf_decode = fluid.layers.crf_decoding(input=emission, 
                     param_attr=fluid.ParamAttr(name="crfw"))

           # Common tensor example
           num_labels, max_len = 10, 20
           feature = fluid.data(name='word_emb_pad', shape=[-1, max_len, 784], dtype='float32')
           label = fluid.data(name='label_pad', shape=[-1, max_len, 1], dtype='int64')
           length = fluid.data(name='length', shape=[-1, 1], dtype='int64')
           emission = fluid.layers.fc(input=feature, size=num_labels,
                                      num_flatten_dims=2)
           
           crf_cost = fluid.layers.linear_chain_crf(input=emission, label=label, length=length, 
                     param_attr=fluid.ParamAttr(name="crfw_pad"))
           crf_decode = fluid.layers.crf_decoding(input=emission, length=length,
                     param_attr=fluid.ParamAttr(name="crfw_pad"))
    """
    helper = LayerHelper('crf_decoding', **locals())
    transition = helper.get_parameter(param_attr.name)
    viterbi_path = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    inputs = {"Emission": [input], "Transition": transition, "Label": label}
    if length:
        inputs['Length'] = length
    helper.append_op(
        type='crf_decoding',
        inputs=inputs,
        outputs={"ViterbiPath": [viterbi_path]})

    return viterbi_path


@templatedoc()
def cos_sim(X, Y):
    """
    ${comment}

    Args:
        X (Variable): ${x_comment}.
        Y (Variable): ${y_comment}.

    Returns:
        A Variable holding LoDTensor representing the output of cosine(X, Y).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[3, 7], dtype='float32')
            y = fluid.data(name='y', shape=[1, 7], dtype='float32')
            out = fluid.layers.cos_sim(x, y)
    """
    helper = LayerHelper('cos_sim', **locals())
    out = helper.create_variable_for_type_inference(dtype=X.dtype)
    xnorm = helper.create_variable_for_type_inference(dtype=X.dtype)
    ynorm = helper.create_variable_for_type_inference(dtype=X.dtype)
    helper.append_op(
        type='cos_sim',
        inputs={'X': [X],
                'Y': [Y]},
        outputs={'Out': [out],
                 'XNorm': [xnorm],
                 'YNorm': [ynorm]})
    return out


def dropout(x,
            dropout_prob,
            is_test=False,
            seed=None,
            name=None,
            dropout_implementation="downgrade_in_infer"):
    """
    Computes dropout.

    Drop or keep each element of `x` independently. Dropout is a regularization
    technique for reducing overfitting by preventing neuron co-adaption during
    training. The dropout operator randomly sets (according to the given dropout
    probability) the outputs of some units to zero, while others are remain
    unchanged.

    dropout op can be removed from the program to make the program more efficient.

    Args:
        x (Variable): The input tensor variable. The data type is float16 or float32 or float64.
        dropout_prob (float): Probability of setting units to zero.
        is_test (bool): A flag indicating whether it is in test phrase or not.
        seed (int): A Python integer used to create random seeds. If this
                    parameter is set to None, a random seed is used.
                    NOTE: If an integer seed is given, always the same output
                    units will be dropped. DO NOT use a fixed seed in training.Default: None.
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.
        dropout_implementation(string): ['downgrade_in_infer'(default)|'upscale_in_train']

                                        1. downgrade_in_infer(default), downgrade the outcome at inference

                                           - train: out = input * mask
                                           - inference: out = input * (1.0 - dropout_prob)

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)
                                        2. upscale_in_train, upscale the outcome at training time

                                           - train: out = input * mask / ( 1.0 - dropout_prob )
                                           - inference: out = input

                                           (mask is a tensor same shape with input, value is 0 or 1
                                           ratio of 0 is dropout_prob)


    Returns:
        A Variable holding Tensor representing the dropout, has same shape and data type with `x`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name="data", shape=[None, 32, 32], dtype="float32")
            droped = fluid.layers.dropout(x, dropout_prob=0.5)
    """

    helper = LayerHelper('dropout', **locals())

    if not isinstance(x, Variable):
        raise TypeError(
            "The type of 'input' in dropout must be Variable, but received %s" %
            (type(x)))
    if convert_dtype(x.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'input' in dropout only support float16 on GPU now."
        )
    if convert_dtype(x.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in dropout must be float16 or float32 or float64, but received %s."
            % (convert_dtype(x.dtype)))

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mask = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)

    if (seed is None or seed == 0) and helper.main_program.random_seed != 0:
        seed = helper.main_program.random_seed

    helper.append_op(
        type='dropout',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'Mask': [mask]},
        attrs={
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0,
            'dropout_implementation': dropout_implementation,
        })
    return out


def cross_entropy(input, label, soft_label=False, ignore_index=kIgnoreIndex):
    """
    This operator computes the cross entropy between input and label. It
    supports both hard-label and and soft-label cross entropy computation.

    1. Hard-label cross entropy: if soft_label=False, :math:`label[i_1, i_2, ..., i_k]`
       is the hard label of each sample.

        .. math::

           output[i_1, i_2, ..., i_k]=-log(input[i_1, i_2, ..., i_k, j]), label[i_1, i_2, ..., i_k] = j, j != ignore\_index

    2. Soft-label cross entropy: if soft_label=True,  :math:`label[i_1, i_2, ..., i_k, j]`
       is the soft label of each sample corresponding to the j-th class.

        .. math::

           output[i_1, i_2, ..., i_k]= -\sum_{j}label[i_1,i_2,...,i_k,j]*log(input[i_1, i_2, ..., i_k,j])

    Args:
        input (Variable): a multidimensional Tensor with shape
                :math:`[N_1, N_2, ..., N_k, D]`, where the last dimension D is
                the class number. The data type should be float32 or float64.
        label (Variable): label value corresponding to input. If
                soft_label=False, the dimension of label should be :math:`[N_1, N_2, ..., N_k]`
                or :math:`[N_1, N_2, ..., N_k, 1]` , and its data type should be int64,
                and the value must be inside [0, D). If soft_label=True, the shape,
                data type of label should be the same with input, and the sum of
                soft label value of each sample should be 1.
        soft_label (bool): indicate whether label is soft. Default False, meaning that
                the label is hard. If soft_label=True, the label is soft.
        ignore_index (int): specify an ignorable label value. The ignored label would be
                omitted when computing. If it is a negative integer, no label would
                be ignored. Only valid when soft_label=False. Default -100.

    Returns:
         A Variable holding Tensor representing the cross entropy, whose data type is the same with input.
         If soft_label=False, the shape of output is the same with label.
         If soft_label=True, the shape of output is :math:`[N_1, N_2, ..., N_k, 1]` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            class_num = 7
            x = fluid.data(name='x', shape=[None, 3, 10], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            predict = fluid.layers.fc(input=x, size=class_num, act='softmax')
            cost = fluid.layers.cross_entropy(input=predict, label=label)
    """
    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in cross_entropy must be Variable, but received %s"
            % (type(input)))
    if convert_dtype(input.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'input' in cross_entropy only support float16 on GPU now."
        )
    if convert_dtype(input.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in cross_entropy must be float16 or float32 or float64, but received %s."
            % (convert_dtype(input.dtype)))

    if not soft_label:
        return cross_entropy2(input, label, ignore_index)
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs={"soft_label": soft_label,
               "ignore_index": ignore_index})
    return out


def cross_entropy2(input, label, ignore_index=kIgnoreIndex):
    helper = LayerHelper('cross_entropy2', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    xshape = helper.create_variable_for_type_inference(dtype=input.dtype)
    match_x = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy2',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out],
                 'MatchX': [match_x],
                 'XShape': [xshape]},
        attrs={'ignore_index': ignore_index})
    return out


def bpr_loss(input, label, name=None):
    """
    **Bayesian Personalized Ranking Loss Operator**

    This operator belongs to pairwise ranking loss. Label is the desired item.
    The loss at a given point in one session is defined as:

    .. math::
        Y[i] = 1/(N[i] - 1) * \sum_j{\log(\sigma(X[i, Label[i]]-X[i, j]))}

    Learn more details by reading paper <session-based recommendations with recurrent
    neural networks>.

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x D], where N is the
                                batch size and D is the number of positive classes and negative classes
                                This input is not probability but logits.
        label (Variable|list):  the ground truth which is a 2-D tensor.  `label`
                                is a tensor<int64> with shape [N x 1].
        name (str|None):        A name for this layer(optional). If set None, the
                                layer will be named automatically. Default: None.
    Returns:
        A 2-D tensor with shape [N x 1], the bpr loss.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          neg_size = 10
          label = fluid.data(
                    name="label", shape=[3, 1], dtype="int64")
          predict = fluid.data(
                    name="predict", shape=[3, neg_size + 1], dtype="float32")
          cost = fluid.layers.bpr_loss(input=predict, label=label)
    """
    helper = LayerHelper('bpr_loss', **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='bpr_loss',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]})
    return out


def square_error_cost(input, label):
    """
    This op accepts input predictions and target label and returns the
    squared error cost.

    For predictions label, and target label, the equation is:

    .. math::

        Out = (input - label)^2

    Parameters:
        input (Variable): Input tensor, the data type should be float32.
        label (Variable): Label tensor, the data type should be float32.

    Returns:
        The tensor variable storing the element-wise squared error \
                  difference between input and label.

    Return type: Variable.

    Examples:

        .. code-block:: python

	    # declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[1])
	    label = fluid.data(name="label", shape=[1])
	    output = fluid.layers.square_error_cost(input,label)
	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.array([1.5]).astype("float32")
	    label_data = np.array([1.7]).astype("float32")
	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data, "label":label_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data)
	    # [array([0.04000002], dtype=float32)]
	    
	    # imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		label = dg.to_variable(label_data)
    		output = fluid.layers.square_error_cost(input, label)
    		print(output.numpy())
	        
	        # [0.04000002]
    """
    helper = LayerHelper('square_error_cost', **locals())
    minus_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='square', inputs={'X': [minus_out]},
        outputs={'Out': [square_out]})
    return square_out


@templatedoc()
def chunk_eval(input,
               label,
               chunk_scheme,
               num_chunk_types,
               excluded_chunk_types=None,
               seq_length=None):
    """
    This operator computes the precision, recall and F1-score for chunk detection.
    It is often used in sequence tagging tasks, such as Named Entity Recognition(NER).

    For some basics of chunking, please refer to
    `Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>`_ .

    This operator supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.
    Here is a NER example for the usage of these tagging schemes:

    .. code-block:: python

       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
              Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========
       IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
       IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
       IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
       IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC
       ====== ====== ======  =====  ==  ============   =====  ===== =====  ==  =========

    There are three chunk types(named entity types) including PER(person), ORG(organization)
    and LOC(location), and we can see that the labels have the form `<tag type>-<chunk type>` .

    Since the implementation of this operator actually uses label ids rather than
    label strings, to make it work, there should be a way to map label ids to
    tag types and chunk types. This operator uses the following way to do mapping:

    .. code-block:: python

       tag_type = label % num_tag_type
       chunk_type = label / num_tag_type

    where `num_tag_type` is the num of tag types in the tagging scheme, `num_chunk_type`
    is the num of chunk types, and `tag_type` get its value from the following table.

    .. code-block:: python

       Scheme Begin Inside End   Single
        plain   0     -      -     -
        IOB     0     1      -     -
        IOE     -     0      1     -
        IOBES   0     1      2     3

    Accordingly, in the above NER example, if the tagging scheme is IOB and chunk
    types are ORG, PER and LOC, then the label ids would be as follows:

    .. code-block:: python

       B-ORG  0
       I-ORG  1
       B-PER  2
       I-PER  3
       B-LOC  4
       I-LOC  5
       O      6

    With which we can map each label id to the corresponding tag type and chunk
    type correctly.

    Args:
        input (Variable): A Tensor or LoDTensor, representing the predicted labels
            from the network. When it is a Tensor, its shape would be `[N, M, 1]`,
            where `N` stands for batch size, `M` for sequence length; When it is
            a LoDTensor, its shape would be `[N, 1]` where `N` stands for the total
            sequence lengths in this mini-batch. The data type should be int64.
        label (Variable): A Tensor or LoDTensor representing the ground-truth labels.
            It shoud have the same shape, lod and data type as ``input`` .
        chunk_scheme (str): Indicate the tagging schemes used here. The value must
            be IOB, IOE, IOBES or plain.
        num_chunk_types (int): The number of chunk types.
        excluded_chunk_types (list, optional): Indicate the chunk types shouldn't
            be taken into account. It should be a list of chunk type ids(integer).
            Default None.
        seq_length(Variable, optional): A 1D Tensor containing the length of each
            sequence when ``input`` and ``label`` are Tensor. It needn't be
            provided if ``input`` and ``label`` are LoDTensor. Default None.

    Returns:
        tuple: A tuple including precision, recall, F1-score, chunk number detected, \
            chunk number in ground-truth, chunk number correctly detected. Each \
            is a Tensor with shape `[1]`. The data type of precision, recall and \
            F1-score all is float32, and the others' data type all is int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            dict_size = 10000
            label_dict_len = 7
            sequence = fluid.data(
                name='id', shape=[-1, 1], lod_level=1, dtype='int64')
            embedding = fluid.embedding(
                input=sequence, size=[dict_size, 512])
            hidden = fluid.layers.fc(input=embedding, size=512)
            label = fluid.layers.data(
                name='label', shape=[1], lod_level=1, dtype='int32')
            crf = fluid.layers.linear_chain_crf(
                input=hidden, label=label, param_attr=fluid.ParamAttr(name="crfw"))
            crf_decode = fluid.layers.crf_decoding(
                input=hidden, param_attr=fluid.ParamAttr(name="crfw"))
            fluid.layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) / 2)
    """
    helper = LayerHelper("chunk_eval", **locals())

    # prepare output
    precision = helper.create_variable_for_type_inference(dtype="float32")
    recall = helper.create_variable_for_type_inference(dtype="float32")
    f1_score = helper.create_variable_for_type_inference(dtype="float32")
    num_infer_chunks = helper.create_variable_for_type_inference(dtype="int64")
    num_label_chunks = helper.create_variable_for_type_inference(dtype="int64")
    num_correct_chunks = helper.create_variable_for_type_inference(
        dtype="int64")

    this_input = {"Inference": [input], "Label": [label]}

    if seq_length:
        this_input["SeqLength"] = [seq_length]

    helper.append_op(
        type="chunk_eval",
        inputs=this_input,
        outputs={
            "Precision": [precision],
            "Recall": [recall],
            "F1-Score": [f1_score],
            "NumInferChunks": [num_infer_chunks],
            "NumLabelChunks": [num_label_chunks],
            "NumCorrectChunks": [num_correct_chunks]
        },
        attrs={
            "num_chunk_types": num_chunk_types,
            "chunk_scheme": chunk_scheme,
            "excluded_chunk_types": excluded_chunk_types or []
        })
    return (precision, recall, f1_score, num_infer_chunks, num_label_chunks,
            num_correct_chunks)


@templatedoc()
def sequence_conv(input,
                  num_filters,
                  filter_size=3,
                  filter_stride=1,
                  padding=True,
                  padding_start=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None,
                  name=None):
    """
    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use conv2d Op.(fluid.layers.** :ref:`api_fluid_layers_conv2d` ).

    This operator receives input sequences with variable length and other convolutional
    configuration parameters(num_filters, filter_size) to apply the convolution operation.
    It fills all-zero padding data on both sides of the sequence by default to ensure that
    the output is the same length as the input. You can customize the padding behavior by
    configuring the parameter :attr:`padding\_start` .
    
    **Warning:** the parameter :attr:`padding` take no effect and will be deprecated in the future.

    .. code-block:: text

            Here we will illustrate the details of the padding operation:
            For a mini-batch of 2 variable lengths sentences, containing 3, and 1 time-steps:
            Assumed input (X) is a [4, N] float LoDTensor, and for the sake of simplicity, we assume N=2.
            input.data = [[1, 1],
                          [2, 2],
                          [3, 3],
                          [4, 4]]

            This is to say that input (X) has 4 words and the dimension of each word
            representation is 2.

            * Case1:

                If padding_start is -1 and filter_size is 3.
                The length of padding data is calculated as follows:
                up_pad_len = max(0, -padding_start) = 1
                down_pad_len = max(0, filter_size + padding_start - 1) = 1

                The output of the input sequence after padding is:
                data_aftet_padding = [[0, 0, 1, 1, 2, 2],
                                      [1, 1, 2, 2, 3, 3],
                                      [2, 2, 3, 3, 0, 0],
                                      [0, 0, 4, 4, 0, 0]]

                It will be multiplied by the filter weight to get the final output.
                Assume num_filters = 3
                output.data = [[ 0.3234, -0.2334,  0.7433],
                               [ 0.5646,  0.9464, -0.1223],
                               [-0.1343,  0.5653,  0.4555],
                               [ 0.9954, -0.1234, -0.1234]]
                output.shape = [4, 3]     # 3 = num_filters
                output.lod = [[0, 3, 4]]  # Remain the same


    Args:
        input (Variable): LoDTensor with shape :math:`(M, K)`, where M is the total time-step of mini-batch
            and K is hidden_size of input. Only lod_level of 1 is supported. The data type should be float32 or
            float64.
        num_filters (int): the number of filters.
        filter_size (int): the height of filter. Specified filter width is not supported, the width is
            hidden_size by default. Default: 3.
        filter_stride (int): stride of the filter. Currently only supports :attr:`stride` = 1.
        padding (bool): the parameter :attr:`padding` take no effect and will be discarded in the
            future. Currently, it will always pad input to make sure the length of the output is
            the same as input whether :attr:`padding` is set true or false. Because the length of
            input sequence may be shorter than :attr:`filter\_size`, which will cause the convolution
            result to not be computed correctly. These padding data will not be trainable or updated
            while trainnig. Default: True.
        padding_start (int): It is used to indicate the start index for padding the input
            sequence, which can be negative. The negative number means to pad
            :attr:`|padding_start|` time-steps of all-zero data at the beginning of each instance.
            The positive number means to skip :attr:`padding_start` time-steps of each instance,
            and it will pad :math:`filter\_size + padding\_start - 1` time-steps of all-zero data
            at the end of the sequence to ensure that the output is the same length as the input.
            If set None, the same length :math:`\\frac{filter\_size}{2}` of data will be filled
            on both sides of the sequence. If set 0, the length of :math:`filter\_size - 1` data
            is padded at the end of each input sequence. Default: None.
        bias_attr (ParamAttr): To specify the bias parameter property. Default: None, which means the
            default bias parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        param_attr (ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        act (str): Activation to be applied to the output of this layer, such as tanh, softmax,
            sigmoid, relu. For more information, please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: LoDTensor with the same length as input. The data type is float32 or float64, which is same as input.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid

             x = fluid.data(name='x', shape=[-1, 10], dtype='float32', lod_level=1)
             x_conved = fluid.layers.sequence_conv(input=x, num_filters=2, filter_size=3, padding_start=-1)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [filter_size * input.shape[1], num_filters]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_variable_for_type_inference(dtype)
    if padding_start is None:
        padding_start = -int(filter_size // 2)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter_param],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': padding_start,
            'contextLength': filter_size,
        })
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


def sequence_softmax(input, use_cudnn=False, name=None):
    """
    **Note**:
    
    **The input type of the OP must be LoDTensor. For Tensor, use:** :ref:`api_fluid_layers_softmax` 

    A LoD-tensor can be regarded as several sequences, and this op apply softmax algo on each sequence.
    The shape of input Tensor can be :math:`[N, 1]` or :math:`[N]`, where :math:`N`
    is the sum of the length of all sequences. Recommended usage: :math:`[N]`.

    For i-th sequence in a mini-batch:

    .. math::

        Out(X[lod[i]:lod[i+1]], :) = \\frac{\exp(X[lod[i]:lod[i+1], :])}{\sum(\exp(X[lod[i]:lod[i+1], :]))}

    For example, for a LoD-Tensor with 6 sequences ([3, 2, 4, 1, 2, 3] - sequence length list in order), 
    the lod in the runtime is [[0, 3, 5, 9, 10, 12, 15]],
    then softmax will be computed among :math:`X[0:3,:],X[3:5,:],X[5:9,:],X[9:10,:],X[10:12,:],X[12:15,:]`,
    and :math:`N` turns out to be 15.

    .. code-block:: text

        *Case 1:

            Given:
                input.data = [0.7, 1, 0.6,
                              1.5, 1.1,
                              1.2, 0.2, 0.6, 1.9,
                              3.1,
                              2.5, 0.8,
                              0.1, 2.4, 1.3]
                input.lod = [[0, 3, 5, 9, 10, 12, 15]]
            then:
                 output.data = [0.30724832, 0.41474187, 0.2780098,
                                0.59868765, 0.40131235,
                                0.2544242, 0.09359743, 0.13963096, 0.5123474, 
                                1.,
                                0.84553474, 0.15446526,
                                0.06995796, 0.69777346, 0.23226859]
                 output.lod = [[0, 3, 5, 9, 10, 12, 15]]    
    

    Args:
        input (Variable):A LoDTensor with shape of  :math:`[N, 1]` or  :math:`[N]`, Recommended usage: :math:`[N]`. 
                         Supported data types: float32, float64. 
        use_cudnn (bool, optional): Use cudnn kernel or not. Effective only when the cudnn version of the paddle 
                                    library is installed and GPU is used for training or reasoning. Default: False.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. 
                              For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A LoD-Tensor which has the same shape and data type with input.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             x = fluid.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_sequence_softmax_1 = fluid.layers.sequence_softmax(input=x)  

             y = fluid.data(name='y', shape=[7],
                 dtype='float32', lod_level=1)
             x_sequence_softmax_2 = fluid.layers.sequence_softmax(input=y)  
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_softmax', **locals())
    dtype = helper.input_dtype()
    softmax_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="sequence_softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn})
    return softmax_out


def softmax(input, use_cudnn=False, name=None, axis=-1):
    """
    This operator implements the softmax layer. The calculation process is as follows:

    1. The dimension :attr:`axis` of the ``input`` will be permuted to the last.
    
    2. Then the input tensor will be logically flattened to a 2-D matrix. The matrix's
    second dimension(row length) is the same as the dimension :attr:`axis` of the input
    tensor, and the first dimension(column length) is the product of all other
    dimensions of the input tensor. For each row of the matrix, the softmax operator
    squashes the K-dimensional(K is the width of the matrix, which is also the size
    of the input tensor's dimension :attr:`axis`) vector of arbitrary real values to a
    K-dimensional vector of real values in the range [0, 1] that add up to 1.

    3. After the softmax operation is completed, the inverse operations of steps 1 and 2 
    are performed to restore the two-dimensional matrix to the same dimension as the ``input``.

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in the matrix, we have:

    .. math::

        Out[i, j] = \\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])}

    Example:

    .. code-block:: text

        Case 1:
          Input:
            X.shape = [2, 3, 4]
            X.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]

          Attrs:
            axis = -1

          Output:
            Out.shape = [2, 3, 4]
            Out.data = [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
                        [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
                         [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]

        Case 2:
          Input:
            X.shape = [2, 3, 4]
            X.data = [[[2.0, 3.0, 4.0, 5.0],
                       [3.0, 4.0, 5.0, 6.0],
                       [7.0, 8.0, 8.0, 9.0]],
                      [[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [6.0, 7.0, 8.0, 9.0]]]
          Attrs:
            axis = 1

          Output:
            Out.shape = [2, 3, 4]
            Out.data = [[[0.00657326, 0.00657326, 0.01714783, 0.01714783],
                         [0.01786798, 0.01786798, 0.04661262, 0.04661262],
                         [0.97555875, 0.97555875, 0.93623955, 0.93623955]],
                        [[0.00490169, 0.00490169, 0.00490169, 0.00490169],
                         [0.26762315, 0.26762315, 0.26762315, 0.26762315],
                         [0.72747516, 0.72747516, 0.72747516, 0.72747516]]] 

    Args:
        input (Variable): The input variable. A multi-dimension ``Tensor`` with type float32 or float64.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn \
            library is installed. To improve numerical stablity, set use_cudnn to \
            False by default.
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` . Default: None.
            will be named automatically. Default: None.
        axis (int, optional): The index of dimension to perform softmax calculations, it should
            be in range :math:`[-1, rank - 1]`, while :math:`rank` is the rank of
            input variable. Default: -1. -1 means the last dimension.

    Returns:
        Variable: ``Tensor`` indicates the output of softmax. The data type and shape are the same as ``input`` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            data = fluid.data(name="input", shape=[-1, 3],dtype="float32")
            result = fluid.layers.softmax(data,axis=1)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            x = np.random.rand(3, 3).astype("float32")
            output= exe.run(feed={"input": x},
                             fetch_list=[result[0]])
            print(output)
    """
    helper = LayerHelper('softmax', **locals())
    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in softmax must be Variable, but received %s" %
            (type(input)))
    if convert_dtype(input.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'input' in softmax only support float16 in GPU now."
        )
    if convert_dtype(input.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in softmax must be float16, float32 or float64, but received %s."
            % (convert_dtype(input.dtype)))

    dtype = helper.input_dtype()
    softmax_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"axis": axis,
               "use_cudnn": use_cudnn})
    return softmax_out


def conv2d(input,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           name=None,
           data_format="NCHW"):
    """
    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW or NHWC format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Variable): The input is 4-D Tensor with shape [N, C, H, W], the data type 
            of input is float16 or float32 or float64.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size 
            is a tuple, it must contain two integers, (filter_size_height, 
            filter_size_width). Otherwise, filter_size_height = filter_size_width =\
            filter_size.
        stride (int|tuple): The stride size. It means the stride in convolution. 
            If stride is a tuple, it must contain two integers, (stride_height, stride_width). 
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings
            on both sides for each dimention.If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when 
            `data_format` is `"NCHW"`, `padding` can be in the form `[[0,0], [0,0], 
            [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|tuple): The dilation size. It means the spacing between the kernel
            points. If dilation is a tuple, it must contain two integers, (dilation_height, 
            dilation_width). Otherwise, dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups (int): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None
        name(str|None): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.
        data_format (str): The data format of the input and output data. An optional string from: `"NCHW"`, `"NHWC"`.
            The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        A Variable holding Tensor representing the conv2d, whose data type is the 
        same with input. If act is None, the tensor variable storing the convolution 
        result, and if act is not None, the tensor variable storing convolution 
        and non-linearity activation result.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
    """

    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in conv2d must be Variable, but received %s" %
            (type(input)))
    if convert_dtype(input.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'input' in conv2d only support float16 on GPU now."
        )
    if convert_dtype(input.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in conv2d must be float16 or float32 or float64, but received %s."
            % (convert_dtype(input.dtype)))

    num_channels = input.shape[1]
    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): %s. " % str(use_cudnn))

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    channel_last = (data_format == "NHWC")
    num_channels = input.shape[3] if channel_last else input.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input(%s) should be defined. "
            "Received: %s." % (str(input.shape), str(num_channels)))
    assert param_attr is not False, "param_attr should not be False here."

    l_type = 'conv2d'
    if (num_channels == groups and num_filters % num_channels == 0 and
            not use_cudnn):
        l_type = 'depthwise_conv2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(num_channels, input.shape, groups))
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    # padding
    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')
        else:
            padding = utils.convert_to_list(padding, 2, 'padding')
            padding = [padding[0], padding[0], padding[1], padding[1]]

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            'fuse_relu_before_depthwise_conv': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)

    return helper.append_activation(pre_act)


def conv3d(input,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=None,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           name=None,
           data_format="NCDHW"):
    """
    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW or NDHWC format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convlution3D is similar with Convlution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, D_f, H_f, W_f)`

        - Output:
          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

            D_{out}&= \\frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{strides[0]} + 1 \\\\
            H_{out}&= \\frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{strides[1]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{strides[2]} + 1

    Args:
        input (Variable): The input is 5-D Tensor with shape [N, C, D, H, W], the data 
            type of input is float16 or float32 or float64.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height, 
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size.
        stride (int|tuple): The stride size. It means the stride in convolution. If stride is a 
            tuple, it must contain three integers, (stride_depth, stride_height, stride_width). 
            Otherwise, stride_depth = stride_height = stride_width = stride. Default: stride = 1.
        padding (string|int|list|tuple): The padding size. It means the number of zero-paddings 
            on both sides for each dimention. If `padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `pool_padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation (int|tuple): The dilation size. It means the spacing between the kernel points. 
            If dilation is a tuple, it must contain three integers, (dilation_depth, dilation_height,
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups (int): The groups number of the Conv3d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        param_attr (ParamAttr|None): The parameter attribute for learnable parameters/weights
            of conv3d. If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as param_attr. If it is set to None, the parameter
            is initialized with :math:`Normal(0.0, std)`, and the :math:`std` is
            :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool|None): The parameter attribute for the bias of conv3d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str|None): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.
        data_format (str): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
            The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_depth, input_height, input_width]`.

    Returns:
        A Variable holding Tensor representing the conv3d, whose data type is 
        the same with input. If act is None, the tensor variable storing the 
        convolution result, and if act is not None, the tensor variable storing 
        convolution and non-linearity activation result.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 12, 32, 32], dtype='float32')
          conv3d = fluid.layers.conv3d(input=data, num_filters=2, filter_size=3, act="relu")
    """

    l_type = 'conv3d'
    assert param_attr is not False, "param_attr should not be False here."
    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): %s. " % str(use_cudnn))

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    channel_last = (data_format == "NDHWC")
    num_channels = input.shape[4] if channel_last else input.shape[1]
    if num_channels < 0:
        raise ValueError(
            "The channel dimmention of the input(%s) should be defined. "
            "Received: %s." % (str(input.shape), str(num_channels)))

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError(
                "The number of input channels must be divisible by Attr(groups). "
                "Received: number of channels(%s), groups(%s)." %
                (str(num_channels), str(groups)))
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 3, 'filter_size')
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 5:
            if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:5]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
                if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:4]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 6, 'padding')

        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')
        else:
            padding = utils.convert_to_list(padding, 3, 'padding')
            padding = [
                padding[0], padding[0], padding[1], padding[1], padding[2],
                padding[2]
            ]

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * filter_size[
            2] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=l_type,
        inputs={
            'Input': input,
            'Filter': filter_param,
        },
        outputs={"Output": pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'use_mkldnn': False,
            "padding_algorithm": padding_algorithm,
            "data_format": data_format,
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)

    return helper.append_activation(pre_act)


def sequence_pool(input, pool_type, is_test=False, pad_value=0.0):
    """
    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use pool2d Op.(fluid.layers.** :ref:`api_fluid_layers_pool2d` ).

    This operator only supports LoDTensor as input. It will apply specified pooling
    operation on the input LoDTensor. It pools features of all time-steps of each
    sequence at the last lod_level using :attr:`pool_type` mentioned in the parameters,
    such as sum, average, sqrt, etc.

    It supports six pool_type:

    - average: :math:`Out[i] = \\frac{\sum_i X_i}{N}`
    - sum:     :math:`Out[i] = \sum_jX_{ij}`
    - sqrt:    :math:`Out[i] = \\frac{\sum_jX_{ij}}{\sqrt{len(X_i)}}`
    - max:     :math:`Out[i] = max(X_i)`
    - last:    :math:`Out[i] = X_{N_i}`
    - first:   :math:`Out[i]` = X_0

    where :math:`N_i` is the length of i-th input sequence.

    .. code-block:: text

        Case 1:
        input is a 1-level LoDTensor and pad_value = 0.0:
            input.lod = [[0, 2, 5, 7, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is LoDTensor:
            out.shape = [4, 1]
            with condition out.shape[0] == len(x.lod[-1]) == 4

        for different pool_type:
            average: out.data = [[2.], [4.], [3.], [0.0]], where 2.=(1. + 3.)/2, 4.=(2. + 4. + 6.)/3, 3.=(5. + 1.)/2
            sum    : out.data = [[4.], [12.], [6.], [0.0]], where 4.=1. + 3., 12.=2. + 4. + 6., 6.=5. + 1.
            sqrt   : out.data = [[2.82], [6.93], [4.24], [0.0]], where 2.82=(1. + 3.)/sqrt(2), 6.93=(2. + 4. + 6.)/sqrt(3), 4.24=(5. + 1.)/sqrt(2)
            max    : out.data = [[3.], [6.], [5.], [0.0]], where 3.=max(1., 3.), 6.=max(2., 4., 6.), 5.=max(5., 1.)
            last   : out.data = [[3.], [6.], [1.], [0.0]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)
            first  : out.data = [[1.], [2.], [5.], [0.0]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

            and all above [0.0] at last of out.data is padding data.

        Case 2:
        input is a 2-level LoDTensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        If pool_typ = sum, it will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is LoDTensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            where out.shape[0] == len(x.lod[-1]) == 5
            sum: out.data = [[1.], [5.], [4.], [0.0], [12.]]
            where 1.=1., 5.=3. + 2., 4.=4., 0.0=pad_value, 12.=6. + 5. + 1.

    Args:
        input (variable): LoDTensor with lod_level no more than 2. The data type should be float32.
        pool_type (str): The pooling type that supports average, sum, sqrt, max, last or first.
        is_test (bool): Only works when :attr:`pool_type` is max. If set False, a temporary Tenosr maxIndex is
            created to record the index information corresponding to the maximum value, which is used for backward
            gradient calculation in the training phase. Default: False.
        pad_value (float): Used to pad the pooling result for empty input sequence. Default: 0.0

    Returns:
        Variable: LoDTensor after pooling with data type float32.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
            avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
            sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
            sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
            max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
            last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
            first_x = fluid.layers.sequence_pool(input=x, pool_type='first')
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    max_index = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "MaxIndex": max_index},
        attrs={
            "pooltype": pool_type.upper(),
            "is_test": is_test,
            "pad_value": pad_value
        })

    # when pool_type is max, variable max_index is initialized,
    # so we stop the gradient explicitly here
    if pool_type == 'max':
        max_index.stop_gradient = True

    return pool_out


@templatedoc()
def sequence_concat(input, name=None):
    """
    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use concat Op.(fluid.layers.** :ref:`api_fluid_layers_concat` ).

    This operator only supports LoDTensor as input. It concatenates the multiple LoDTensor from input by the LoD information,
    and outputs the concatenated LoDTensor.

    .. code-block:: text

        input is a list of LoDTensor:
            input = [x1, x2]
        where:
            x1.lod = [[0, 3, 5]]
            x1.data = [[1], [2], [3], [4], [5]]
            x1.shape = [5, 1]

            x2.lod = [[0, 2, 4]]
            x2.data = [[6], [7], [8], [9]]
            x2.shape = [4, 1]
        and should satisfy: len(x1.lod[0]) == len(x2.lod[0])

        output is LoDTensor:
            out.lod = [[0, 3+2, 5+4]]
            out.data = [[1], [2], [3], [6], [7], [4], [5], [8], [9]]
            out.shape = [9, 1]

    Args:
        input(list of Variable): List of LoDTensor to be concatenated. The length of each LoDTensor should be same.
            The data type can be float32, float64 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: Output the concatenated LoDTensor. The data type is same as input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[-1, 10], dtype='float32', lod_level=1)
            y = fluid.data(name='y', shape=[-1, 10], dtype='float32', lod_level=1)
            out = fluid.layers.sequence_concat(input=[x, y])
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_concat', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='sequence_concat', inputs={'X': input}, outputs={'Out': [out]})
    return out


def sequence_first_step(input):
    """
    This operator only supports LoDTensor as input. Given the input LoDTensor, it will
    select first time-step feature of each sequence as output.

    .. code-block:: text

       Case 1:
        input is 1-level LoDTensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is a LoDTensor:
            out.shape = [3, 1]
            out.shape[0] == len(x.lod[-1]) == 3
            out.data = [[1.], [2.], [5.]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

        Case 2:
        input is a 2-level LoDTensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        It will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is a LoDTensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            out.shape[0] == len(x.lod[-1]) == 5
            out.data = [[1.], [3.], [4.], [0.0], [6.]]
            where 1.=first(1.), 3.=first(3., 2.), 4.=first(4.), 0.0 = pad_value, 6.=first(6., 5., 1.)

    Args:
        input(Variable): LoDTensor with lod_level no more than 2. The data type should be float32.

    Returns:
        Variable: LoDTensor consist of the sequence's first step vector. The data type is float32.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             x = fluid.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
             x_first_step = fluid.layers.sequence_first_step(input=x)
    """
    return sequence_pool(input=input, pool_type="first")


def sequence_last_step(input):
    """
    This operator only supports LoDTensor as input. Given the input LoDTensor, it will
    select last time-step feature of each sequence as output.

    .. code-block:: text

        Case 1:
        input is 1-level LoDTensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        output is a LoDTensor:
            out.shape = [3, 1]
            out.shape[0] == len(x.lod[-1]) == 3
            out.data = [[3.], [6.], [1.]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)

        Case 2:
        input is a 2-level LoDTensor containing 3 sequences with length info [2, 0, 3],
        where 0 means empty sequence.
        The first sequence contains 2 subsequence with length info [1, 2];
        The last sequence contains 3 subsequence with length info [1, 0, 3].
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        It will apply pooling on last lod_level [0, 1, 3, 4, 4, 7]. pad_value = 0.0
        output is a LoDTensor:
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            out.shape[0] == len(x.lod[-1]) == 5
            out.data = [[1.], [2.], [4.], [0.0], [1.]]
            where 1.=last(1.), 2.=last(3., 2.), 4.=last(4.), 0.0 = pad_value, 1=last(6., 5., 1.)


    Args:
        input(Variable): LoDTensor with lod_level no more than 2. The data type should be float32.

    Returns:
        Variable: LoDTensor consist of the sequence's last step vector. The data type is float32.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             x = fluid.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
             x_last_step = fluid.layers.sequence_last_step(input=x)
    """
    return sequence_pool(input=input, pool_type="last")


def sequence_slice(input, offset, length, name=None):
    """
    **Sequence Slice Layer**

    The layer crops a subsequence from given sequence with given start
    offset and subsequence length.

    It only supports sequence data (LoDTensor with lod_level equal to 1).

    .. code-block:: text

              - Case:

            Given the input Variable **input**:

                input.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]],
                input.lod = [[3, 2]],
                input.dims = (5, 2),

            with offset.data = [[0], [1]] and length.data = [[2], [1]],

            the output Variable will be

                out.data = [[a1, a2], [b1, b2], [e1, e2]],
                out.lod = [[2, 1]],
                out.dims = (3, 2).

    Note:
          The first dimension size of **input**, **offset** and **length**
          should be equal. The **offset** should start from 0.

    Args:
        input(Variable): LoDTensor, The input Variable which consists of the complete
                         sequences.The data type is float32 or float64.
        offset(Variable): LoDTensor, The offset to slice each sequence.The data
                         type is int32 or int64.
        length(Variable): LoDTensor, The length of each subsequence.The data
                         type is int32 or int64.
        name(str|None): The default value is None.  Normally there is no need
                        for user to set this property.  For more information,
                        please refer to :ref:`api_guide_Name`

    Returns:
        Variable: The output subsequences.

    Examples:

        .. code-block:: python

             import paddle.fluid as fluid
             import numpy as np
             seqs = fluid.data(name='x', shape=[10, 5],
                              dtype='float32', lod_level=1)
             offset = fluid.layers.assign(input=np.array([[0, 1]]).astype("int32"))
             length = fluid.layers.assign(input=np.array([[2, 1]]).astype("int32"))
             subseqs = fluid.layers.sequence_slice(input=seqs, offset=offset,
                                                   length=length)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper("sequence_slice", **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)

    offset.stop_gradient = True
    length.stop_gradient = True

    helper.append_op(
        type="sequence_slice",
        inputs={"X": input,
                "Offset": offset,
                "Length": length},
        outputs={"Out": out})

    return out


@templatedoc()
def pool2d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None,
           exclusive=True,
           data_format="NCHW"):
    """
    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        pool_type: ${pooling_type_comment}
        pool_stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (pool_stride_Height, pool_stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        pool_padding (string|int|list|tuple): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and when `data_format` is `"NCHW"`,
            `pool_padding` can be in the form `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Otherwise, the pool padding size will be a square of an int.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NDHW"`.
                The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                `[batch_size, input_channels, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `pool_type` is not "max" nor "avg"
        ValueError: If `global_pooling` is False and `pool_size` is -1
        ValueError: If `use_cudnn` is not a bool value.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid

          data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')

          # max pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "max",
            pool_stride = 1,
            global_pooling=False)

          # average pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=False)

          # global average pool2d
          pool2d = fluid.layers.pool2d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=True)

          # Attr(pool_padding) is a list with 4 elements, Attr(data_format) is "NCHW".
          out_1 = fluid.layers.pool2d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = [1, 2, 1, 0],
            data_format = "NCHW")

          # Attr(pool_padding) is a string, Attr(data_format) is "NCHW".
          out_2 = fluid.layers.pool2d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = "VALID",
            data_format = "NCHW")
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown Attr(pool_type): '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When Attr(global_pooling) is False, Attr(pool_size) must be passed "
            "and be a valid value. Received pool_size: %s." % str(pool_size))

    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): %s." % str(use_cudnn))

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
    pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')

    def update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')

        else:
            padding = utils.convert_to_list(padding, 2, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(pool_padding, str):
        pool_padding = pool_padding.upper()
        if pool_padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(pool_padding))
        if pool_padding == "VALID":
            padding_algorithm = "VALID"
            pool_padding = [0, 0, 0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif pool_padding == "SAME":
            padding_algorithm = "SAME"
            pool_padding = [0, 0, 0, 0]

    pool_padding = update_padding(pool_padding, data_format)

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    return pool_out


@templatedoc()
def pool3d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           name=None,
           exclusive=True,
           data_format="NCDHW"):
    """
    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W]. The format of
                          input tensor is `"NCDHW"` or `"NDHWC"`, where `N` is batch size, `C` is
                          the number of channels, `D` is the depth of the feature,
                          `H` is the height of the feature, and `W` is the width
                          of the feature.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size 
            is a tuple or list, it must contain three integers, 
            (pool_size_Depth, pool_size_Height, pool_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        pool_type (string): ${pooling_type_comment}
        pool_stride (string|int|list|tuple)): The pool padding. If `pool_padding` is a string, either 'VALID' or
            'SAME' which is the padding algorithm. If pool stride size is a tuple or list,
            it must contain three integers, `[stride_Depth, stride_Height, stride_Width]`.
            Otherwise, the pool stride size will be a cube of an int.
        pool_padding (int|list|tuple): The pool padding size. If pool padding size is a tuple or list,
            it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `"NCDHW"`, `pool_padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `"NDHWC"`, `pool_padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
        exclusive (bool): Whether to exclude padding points in average pooling
                          mode, default is true.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                `[batch_size, input_channels, input_depth, input_height, input_width]`.

    Returns:
        Variable: The output tensor of pooling result. The data type is same as input tensor.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid

          data = fluid.data(name='data', shape=[None, 3, 32, 32, 32], dtype='float32')

          # max pool3d
          pool3d = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "max",
            pool_stride = 1,
            global_pooling=False)

          # average pool3d
          pool3d = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=False)

          # global average pool3d
          pool3d = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            global_pooling=True)

          # example 1:
          # Attr(pool_padding) is a list with 6 elements, Attr(data_format) is "NCDHW".
          out_1 = fluid.layers.pool3d(
            input = data,
            pool_size = 2,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = [1, 2, 1, 0, 1, 2],
            global_pooling = False,
            data_format = "NCDHW")

          # example 2:
          # Attr(pool_padding) is a string, Attr(data_format) is "NCDHW".
          out_2 = fluid.layers.pool3d(
            input = data,
            pool_size = 3,
            pool_type = "avg",
            pool_stride = 1,
            pool_padding = "VALID",
            global_pooling = False,
            data_format = "NCDHW")

    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown Attr(pool_type): '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When Attr(global_pooling) is False, Attr(pool_size) must be passed "
            "and be a valid value. Received Attr(pool_size): %s." %
            str(pool_size))

    if not isinstance(use_cudnn, bool):
        raise ValueError("Attr(use_cudnn) should be True or False. Received "
                         "Attr(use_cudnn): %s. " % str(use_cudnn))

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s" % str(data_format))

    pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')
    pool_stride = utils.convert_to_list(pool_stride, 3, 'pool_stride')

    def update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, (list, tuple)):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 5:
            if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:5]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
                if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:4]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 6, 'padding')

        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')

        else:
            padding = utils.convert_to_list(padding, 3, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(pool_padding, str):
        pool_padding = pool_padding.upper()
        if pool_padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(pool_padding))
        if pool_padding == "VALID":
            padding_algorithm = "VALID"
            pool_padding = [0, 0, 0, 0, 0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", ceil_mode must be False. "
                    "Received ceil_mode: True.")
        elif pool_padding == "SAME":
            padding_algorithm = "SAME"
            pool_padding = [0, 0, 0, 0, 0, 0]

    pool_padding = update_padding(pool_padding, data_format)

    op_type = "pool3d"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    return pool_out


@templatedoc(op_type="pool2d")
def adaptive_pool2d(input,
                    pool_size,
                    pool_type="max",
                    require_index=False,
                    name=None):
    """
    This operation calculates the output based on the input, pool_size,
    pool_type parameters. Input(X) and output(Out) are in NCHW format, where N is batch
    size, C is the number of channels, H is the height of the feature, and W is
    the width of the feature. Parameters(pool_size) should contain two elements which
    represent height and width, respectively. Also the H and W dimensions of output(Out)
    is same as Parameter(pool_size). The output tensor shape will be [N, C, pool_size[0], pool_size[1]]

    For average adaptive pool2d:

    ..  math::

       hstart &= floor(i * H_{in} / H_{out})

       hend &= ceil((i + 1) * H_{in} / H_{out})

       wstart &= floor(j * W_{in} / W_{out})

       wend &= ceil((j + 1) * W_{in} / W_{out})

       Output(i ,j) &= \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}

    Args:
        input (Variable): The input tensor of pooling operator, which is a 4-D tensor
                          with shape [N, C, H, W].  The format of input tensor is NCHW,
                          where N is batch size, C is the number of channels, H is the
                          height of the feature, and W is the width of the feature.
                          The data type is float32 or float64.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (pool_size_Height, pool_size_Width).
        pool_type: ${pooling_type_comment}
        require_index (bool): If true, the index of max pooling point will be returned along
            with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Variable: The output tensor of adaptive pooling result. The data type is same 
                  as input tensor.

    Raises:
        ValueError: 'pool_type' is not 'max' nor 'avg'.
        ValueError: invalid setting 'require_index' true when 'pool_type' is 'avg'.
        ValueError: 'pool_size' should be a list or tuple with length as 2.

    Examples:
        .. code-block:: python

          # average adaptive pool2d
          # suppose input data in shape of [N, C, H, W], `pool_size` is [m, n],
          # output shape is [N, C, m, n], adaptive pool divide H and W dimentions
          # of input data into m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(m):
          #         for j in range(n):
          #             hstart = floor(i * H / m)
          #             hend = ceil((i + 1) * H / m)
          #             wstart = floor(i * W / n)
          #             wend = ceil((i + 1) * W / n)
          #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
          #
          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          pool_out = fluid.layers.adaptive_pool2d(
                            input=data,
                            pool_size=[3, 3],
                            pool_type='avg')

          # max adaptive pool2d
          # suppose input data in shape of [N, C, H, W], `pool_size` is [m, n],
          # output shape is [N, C, m, n], adaptive pool divide H and W dimentions
          # of input data into m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(m):
          #         for j in range(n):
          #             hstart = floor(i * H / m)
          #             hend = ceil((i + 1) * H / m)
          #             wstart = floor(i * W / n)
          #             wend = ceil((i + 1) * W / n)
          #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
          #
          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          pool_out = fluid.layers.adaptive_pool2d(
                            input=data,
                            pool_size=[3, 3],
                            pool_type='max')
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if pool_type == "avg" and require_index:
        raise ValueError(
            "invalid setting 'require_index' true when 'pool_type' is 'avg'.")

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')

    if pool_type == "max":
        l_type = 'max_pool2d_with_index'
    else:
        l_type = "pool2d"

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    if pool_type == "max":
        mask = helper.create_variable_for_type_inference(dtype)
        outputs["Mask"] = mask

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (pool_out, mask) if require_index else pool_out


@templatedoc(op_type="pool3d")
def adaptive_pool3d(input,
                    pool_size,
                    pool_type="max",
                    require_index=False,
                    name=None):
    """
    This operation calculates the output based on the input, pool_size,
    pool_type parameters. Input(X) and output(Out) are in NCDHW format, where N is batch
    size, C is the number of channels, D is the depth of the feature, H is the height of
    the feature, and W is the width of the feature. Parameters(pool_size) should contain
    three elements which represent height and width, respectively. Also the D, H and W
    dimensions of output(Out) is same as Parameter(pool_size). The output tensor shape
    will be [N, C, pool_size[0], pool_size[1], pool_size[2]]

    For average adaptive pool3d:

    ..  math::

      dstart &= floor(i * D_{in} / D_{out})

      dend &= ceil((i + 1) * D_{in} / D_{out})

      hstart &= floor(j * H_{in} / H_{out})

      hend &= ceil((j + 1) * H_{in} / H_{out})

      wstart &= floor(k * W_{in} / W_{out})

      wend &= ceil((k + 1) * W_{in} / W_{out})

      Output(i ,j, k) &= \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}

    Args:
        input (Variable): The input tensor of pooling operator, which is a 5-D tensor with 
                          shape [N, C, D, H, W]. The format of input tensor is NCDHW, where
                          N is batch size, C is the number of channels, D is the depth of the feature,
                          H is the height of the feature, and W is the width of the feature.
                          The data type is float32 or float64.
        pool_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three integers, (Depth, Height, Width).
        pool_type: ${pooling_type_comment}
        require_index (bool): If true, the index of max pooling point will be returned along
            with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Variable: The output tensor of adaptive pooling result. The data type is same as input tensor.

    Raises:
        ValueError: 'pool_type' is not 'max' nor 'avg'.
        ValueError: invalid setting 'require_index' true when 'pool_type' is 'avg'.
        ValueError: 'pool_size' should be a list or tuple with length as 2.

    Examples:
        .. code-block:: python

          # average adaptive pool3d
          # suppose input data in shape of [N, C, D, H, W], `pool_size` is [l, m, n],
          # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimentions
          # of input data into l * m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(l):
          #         for j in range(m):
          #             for k in range(n):
          #                 dstart = floor(i * D / l)
          #                 dend = ceil((i + 1) * D / l)
          #                 hstart = floor(j * H / m)
          #                 hend = ceil((j + 1) * H / m)
          #                 wstart = floor(k * W / n)
          #                 wend = ceil((k + 1) * W / n)
          #                 output[:, :, i, j, k] =
          #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
          #

          import paddle.fluid as fluid

          data = fluid.data(
              name='data', shape=[None, 3, 32, 32, 32], dtype='float32')
          pool_out = fluid.layers.adaptive_pool3d(
                            input=data,
                            pool_size=[3, 3, 3],
                            pool_type='avg')

          # max adaptive pool3d
          # suppose input data in shape of [N, C, D, H, W], `pool_size` is [l, m, n],
          # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimentions
          # of input data into l * m * n grids averagely and performs poolings in each
          # grid to get output.
          # adaptive average pool performs calculations as follow:
          #
          #     for i in range(l):
          #         for j in range(m):
          #             for k in range(n):
          #                 dstart = floor(i * D / l)
          #                 dend = ceil((i + 1) * D / l)
          #                 hstart = floor(j * H / m)
          #                 hend = ceil((j + 1) * H / m)
          #                 wstart = floor(k * W / n)
          #                 wend = ceil((k + 1) * W / n)
          #                 output[:, :, i, j, k] =
          #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
          #

          import paddle.fluid as fluid

          data = fluid.data(
              name='data', shape=[None, 3, 32, 32, 32], dtype='float32')
          pool_out = fluid.layers.adaptive_pool3d(
                            input=data,
                            pool_size=[3, 3, 3],
                            pool_type='max')
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if pool_type == "avg" and require_index:
        raise ValueError(
            "invalid setting 'require_index' true when 'pool_type' is 'avg'.")

    pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')

    if pool_type == "max":
        l_type = 'max_pool3d_with_index'
    else:
        l_type = "pool3d"

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    if pool_type == "max":
        mask = helper.create_variable_for_type_inference(dtype)
        outputs["Mask"] = mask

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (pool_out, mask) if require_index else pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               in_place=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=True,
               fuse_with_relu=False,
               use_global_stats=False):
    """
    **Batch Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

        moving\_mean = moving\_mean * momentum + mini-batch\_mean * (1. - momentum) \\\\
        moving\_var = moving\_var * momentum + mini-batch\_var * (1. - momentum) 


    moving_mean is global mean and moving_var is global variance.

    When use_global_stats = True, the :math:`\\mu_{\\beta}`
    and :math:`\\sigma_{\\beta}^{2}` are not the statistics of one mini-batch.
    They are global (or running) statistics. (It usually got from the
    pre-trained model.)
    The training and testing (or inference) have the same behavior:

    ..  math::

        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}}  \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta

    Note:
        if build_strategy.sync_batch_norm=True, the batch_norm in network will use 
        sync_batch_norm automatically.

    Args:
        input(variable): The rank of input variable can be 2, 3, 4, 5. The data type 
            is float16 or float32 or float64.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test (bool, Default False): A flag indicating whether it is in
            test phrase or not.
        momentum(float, Default 0.9): The value used for the moving_mean and
            moving_var computation. The updated formula is:
            :math:`moving\_mean = moving\_mean * momentum + new\_mean * (1. - momentum)`
            :math:`moving\_var = moving\_var * momentum + new\_var * (1. - momentum)`
            Default is 0.9.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized 
	     with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
	     Default: None.
        data_layout(str, default NCHW): the data_layout of input, is NCHW or NHWC.
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(str|None): For detailed information, please refer to :ref:`api_guide_Name`. 
            Usually name is no need to set and None by default. 
        moving_mean_name(str, Default None): The name of moving_mean which store the global Mean. If it 
            is set to None, batch_norm will save global mean with a random name, otherwise, batch_norm 
            will save global mean with the string.
        moving_variance_name(str, Default None): The name of the moving_variance which store the global Variance.
            If it is set to None, batch_norm will save global variance with a random name, otherwise, batch_norm 
            will save global variance with the string.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance should do model
            average when model average is enabled.
        fuse_with_relu (bool): if True, this OP performs relu after batch norm.
        use_global_stats(bool, Default False): Whether to use global mean and
            variance. In inference or test mode, set use_global_stats to true
            or is_test to true, and the behavior is equivalent.
            In train mode, when setting use_global_stats True, the global mean
            and variance are also used during train period.

    Returns:
        A Variable holding Tensor which is the result after applying batch normalization on the input, 
        has same shape and data type with input. 

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            hidden2 = fluid.layers.batch_norm(input=hidden1)
    """
    assert bias_attr is not False, "bias_attr should not be False in batch_norm."
    helper = LayerHelper('batch_norm', **locals())

    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in batch_norm must be Variable, but received %s"
            % (type(input)))
    if convert_dtype(input.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'input' in batch_norm only support float16 on GPU now."
        )
    if convert_dtype(input.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in batch_norm must be float16 or float32 or float64, but received %s."
            % (convert_dtype(input.dtype)))

    dtype = helper.input_dtype()
    # use fp32 for bn parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)

    mean = helper.create_parameter(
        attr=ParamAttr(
            name=moving_mean_name,
            initializer=Constant(0.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    batch_norm_out = input if in_place else helper.create_variable_for_type_inference(
        dtype)

    helper.append_op(
        type="batch_norm",
        inputs={
            "X": input,
            "Scale": scale,
            "Bias": bias,
            "Mean": mean,
            "Variance": variance
        },
        outputs={
            "Y": batch_norm_out,
            "MeanOut": mean_out,
            "VarianceOut": variance_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={
            "momentum": momentum,
            "epsilon": epsilon,
            "is_test": is_test,
            "data_layout": data_layout,
            "use_mkldnn": False,
            "fuse_with_relu": fuse_with_relu,
            "use_global_stats": use_global_stats
        })

    return helper.append_activation(batch_norm_out)


def instance_norm(input,
                  epsilon=1e-05,
                  param_attr=None,
                  bias_attr=None,
                  name=None):
    """
    **Instance Normalization Layer**

    Can be used as a normalizer function for convolution or fully_connected operations.
    The required data format for this layer is one of the following:

    DataLayout: NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Instance Normalization: The Missing Ingredient for 
    Fast Stylization <https://arxiv.org/pdf/1607.08022.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW} x_i \\qquad &//\\
        \\ mean\ of\ one\  feature\ map\ in\ mini-batch \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{HW} \\sum_{i=1}^{HW}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ variance\ of\ one\ feature\ map\ in\ mini-batch \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Note:
        `H` means height of feature map, `W` means width of feature map.

    Args:
        input(variable): The rank of input variable can be 2, 3, 4, 5. 
            The data type is float32 or float64.
        epsilon(float, Default 1e-05): A value added to the denominator for
            numerical stability. Default is 1e-5.
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of instance_norm. If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as param_attr, the name of scale can be set in ParamAttr.
	     If the Initializer of the param_attr is not set, the parameter is initialized 
	     with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of instance_norm.
             If it is set to None or one attribute of ParamAttr, instance_norm
	     will create ParamAttr as bias_attr, the name of bias can be set in ParamAttr. 
	     If the Initializer of the bias_attr is not set, the bias is initialized zero. 
	     Default: None.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        A Variable holding Tensor which is the result after applying instance normalization on the input, 
        has same shape and data type with input. 

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[3, 7, 3, 7], dtype='float32')
            hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            hidden2 = fluid.layers.instance_norm(input=hidden1)
    """
    assert bias_attr is not False, "bias_attr should not be False in instance_norm."
    helper = LayerHelper('instance_norm', **locals())
    dtype = helper.input_dtype()

    # use fp32 for in parameter
    if dtype == core.VarDesc.VarType.FP16:
        dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    channel_num = input_shape[1]

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))
    bias = helper.create_parameter(
        attr=helper.bias_attr,
        shape=param_shape,
        dtype=dtype,
        is_bias=True,
        default_initializer=Constant(0.0))

    # create output
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    instance_norm_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="instance_norm",
        inputs={
            "X": input,
            "Scale": scale,
            "Bias": bias,
        },
        outputs={
            "Y": instance_norm_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={"epsilon": epsilon, })

    return instance_norm_out


def data_norm(input,
              act=None,
              epsilon=1e-05,
              param_attr=None,
              data_layout='NCHW',
              in_place=False,
              name=None,
              moving_mean_name=None,
              moving_variance_name=None,
              do_model_average_for_mean_and_var=True):
    """
    **Data Normalization Layer**

    This op can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Args:
        input(variable): The input variable which is a LoDTensor.
        act(string, Default None): Activation type, linear|relu|prelu|...
        epsilon(float, Default 1e-05):
        param_attr(ParamAttr): The parameter attribute for Parameter `scale`.
        data_layout(string, default NCHW): NCHW|NHWC
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.
        moving_mean_name(string, Default None): The name of moving_mean which store the global Mean.
        moving_variance_name(string, Default None): The name of the moving_variance which store the global Variance.
        do_model_average_for_mean_and_var(bool, Default True): Whether parameter mean and variance
            should do model average when model average is enabled.

    Returns:
        Variable: A tensor variable which is the result after applying data normalization on the input.

    Examples:

        .. code-block:: python
            
            import paddle.fluid as fluid

            hidden1 = fluid.data(name="hidden1", shape=[64, 200])
            hidden2 = fluid.layers.data_norm(name="hidden2", input=hidden1)
    """
    helper = LayerHelper('data_norm', **locals())
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    batch_size_default = 1e4
    batch_sum_default = 0.0
    batch_square_sum_default = 1e4

    if param_attr and isinstance(param_attr, dict):
        batch_size_default = param_attr.get("batch_size", 1e4)
        batch_sum_default = param_attr.get("batch_sum", 0.0)
        batch_square_sum_default = param_attr.get("batch_square", 1e4)

    # create parameter
    batch_size = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_size',
            initializer=Constant(value=float(batch_size_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    batch_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_sum',
            initializer=Constant(value=float(batch_sum_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    batch_square_sum = helper.create_parameter(
        attr=ParamAttr(
            name=name + '.batch_square_sum',
            initializer=Constant(value=float(batch_square_sum_default)),
            trainable=True),
        shape=param_shape,
        dtype=input.dtype)

    means = helper.create_variable(dtype=dtype, stop_gradient=True)
    scales = helper.create_variable(dtype=dtype, stop_gradient=True)

    data_norm_out = input if in_place else helper.create_variable(dtype=dtype)

    helper.append_op(
        type="data_norm",
        inputs={
            "X": input,
            "BatchSize": batch_size,
            "BatchSum": batch_sum,
            "BatchSquareSum": batch_square_sum
        },
        outputs={"Y": data_norm_out,
                 "Means": means,
                 "Scales": scales},
        attrs={"epsilon": epsilon})

    return helper.append_activation(data_norm_out)


@templatedoc()
def layer_norm(input,
               scale=True,
               shift=True,
               begin_norm_axis=1,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               act=None,
               name=None):
    """
    **Layer Normalization Layer**

    The API implements the function of the Layer Normalization Layer and can be applied to mini-batch input data.
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_

    The formula is as follows:

    ..  math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} x_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}{(x_i - \\mu)^2} + \\epsilon}

        y & = f(\\frac{g}{\\sigma}(x - \\mu) + b)

    - :math:`x`: the vector representation of the summed inputs to the neurons in that layer.
    - :math:`H`: the number of hidden units in a layers
    - :math:`\\epsilon`: the small value added to the variance to prevent division by zero.
    - :math:`g`: the trainable scale parameter.
    - :math:`b`: the trainable bias parameter.

    Args:
        input(Variable): A multi-dimension ``Tensor`` , and the data type is float32 or float64.
        scale(bool, optional): Whether to learn the adaptive gain :math:`g` after
            normalization. Default: True.
        shift(bool, optional): Whether to learn the adaptive bias :math:`b` after
            normalization. Default: True.
        begin_norm_axis(int, optional): The normalization will be performed along
            dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
            Default: 1.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            gain :math:`g`. If :attr:`scale` is False, :attr:`param_attr` is
            omitted. If :attr:`scale` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr, optional): The parameter attribute for the learnable
            bias :math:`b`. If :attr:`shift` is False, :attr:`bias_attr` is
            omitted. If :attr:`shift` is True and :attr:`param_attr` is None,
            a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        act(str, optional): Activation to be applied to the output of layer normalizaiton.
                  Default: None.
        name(str): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: ``Tensor``  indicating the normalized result, the data type is the same as  ``input`` , and the return dimension is the same as  ``input`` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            x = fluid.data(name='x', shape=[-1, 32, 32], dtype='float32')
            hidden1 = fluid.layers.layer_norm(input=x, begin_norm_axis=1)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            np_x = np.random.random(size=(8, 3, 32, 32)).astype('float32')
            output = exe.run(feed={"x": np_x}, fetch_list = [hidden1])
            print(output)
    """
    assert in_dygraph_mode(
    ) is not True, "please use FC instead of fc in dygraph mode!"
    helper = LayerHelper('layer_norm', **locals())
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    param_shape = [reduce(lambda x, y: x * y, input_shape[begin_norm_axis:])]
    if scale:
        assert param_attr is not False, "param_attr should not be False when using scale."
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        inputs['Scale'] = scale
    else:
        if param_attr:
            warnings.warn("param_attr is only avaliable with scale is True.")
    if shift:
        assert bias_attr is not False, "bias_attr should not be False when using shift."
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias
    else:
        if bias_attr:
            warnings.warn("bias_attr is only avaliable with shift is True.")

    # create output
    mean_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    layer_norm_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type="layer_norm",
        inputs=inputs,
        outputs={
            "Y": layer_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={"epsilon": epsilon,
               "begin_norm_axis": begin_norm_axis})

    return helper.append_activation(layer_norm_out)


@templatedoc()
def group_norm(input,
               groups,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               act=None,
               data_layout='NCHW',
               name=None):
    """
    **Group Normalization Layer**

    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_ .

    Parameters:
        input(Variable): 4-D Tensor, the data type is float32 or float64.
        groups(int): The number of groups that divided from channels, the data type
            is int32.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero, the data type is float32. Default: 1e-05.
        param_attr(ParamAttr|bool, optional): ParamAttr object that specifies weight parameter
            attribute. If a bool type, only False is supported, which means there is no weight parameter.
            Default: None, the default weight parameter attribute is used. For more information, please
            refer to :ref:`api_guide_ParamAttr` .
        bias_attr(ParamAttr|bool, optional): ParamAttr object that specifies bias parameter
            attribute. If a bool type, only False is supported, which means there is no bias parameter.
            Default: None, the default bias parameter attribute is used. For more information, please
            refer to :ref:`api_guide_ParamAttr` .
        act(str, optional): Activation to be applied to the output of group normalizaiton.
        data_layout(str, optional): The data format of the input and output data. An optional string
            from: `"NCHW"`, `"NHWC"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, channels, height, width]`. Default: "NCHW".
        name (str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A 4-D Tensor has same data type and data format with `input`.

    Raises:
        ValueError: If `data_layout` is neither 'NCHW' nor 'NHWC'.

    Examples:
       .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.data(name='data', shape=[None, 8, 32, 32], dtype='float32')
            x = fluid.layers.group_norm(input=data, groups=4)
    """
    helper = LayerHelper('group_norm', **locals())
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    if data_layout != 'NCHW' and data_layout != 'NHWC':
        raise ValueError(
            "Param(data_layout) of Op(fluid.layers.group_norm) got wrong value: received "
            + data_layout + " but only NCHW or NHWC supported.")
    channel_num = input_shape[1] if data_layout == 'NCHW' else input_shape[-1]
    param_shape = [channel_num]
    if param_attr:
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        inputs['Scale'] = scale
    if bias_attr:
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias

    # create output
    mean_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    variance_out = helper.create_variable(dtype=dtype, stop_gradient=True)
    group_norm_out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="group_norm",
        inputs=inputs,
        outputs={
            "Y": group_norm_out,
            "Mean": mean_out,
            "Variance": variance_out,
        },
        attrs={
            "epsilon": epsilon,
            "groups": groups,
            "data_layout": data_layout
        })

    return helper.append_activation(group_norm_out)


@templatedoc()
def spectral_norm(weight, dim=0, power_iters=1, eps=1e-12, name=None):
    """
    **Spectral Normalization Layer**

    This operation calculates the spectral normalization value of weight parameters of
    fc, conv1d, conv2d, conv3d layers which should be 2-D, 3-D, 4-D, 5-D
    Parameters. Output tensor will be in same shape with input tensor.
    Calculations are showed as follows.

    Step 1:
    Generate vector U in shape of [H], and V in shape of [W].
    While H is the :attr:`dim` th dimension of the input weights,
    and W is the product result of remaining dimensions.

    Step 2:
    :attr:`power_iters` shoule be a positive interger, do following
    calculations with U and V for :attr:`power_iters` rounds. Calculations
    as follows:

    .. math:: 

        \mathbf{v} := \\frac{\mathbf{W}^{T} \mathbf{u}}{\|\mathbf{W}^{T} \mathbf{u}\|_2}

        \mathbf{u} := \\frac{\mathbf{W}^{T} \mathbf{v}}{\|\mathbf{W}^{T} \mathbf{v}\|_2}

    Step 3:
    Calculate :math:`\sigma(\mathbf{W})` and normalize weight values.

    .. math::

        \sigma(\mathbf{W}) = \mathbf{u}^{T} \mathbf{W} \mathbf{v}

        \mathbf{W} = \\frac{\mathbf{W}}{\sigma(\mathbf{W})}
                

    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .

    Args:
        weight(${weight_type}): ${weight_comment}
        dim(int): ${dim_comment}
        power_iters(int): ${power_iters_comment}
        eps(float): ${eps_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Variable: A tensor variable of weight parameters after spectral normalization.
                  The data type and shape is same as input tensor.

    Examples:
       .. code-block:: python

            import paddle.fluid as fluid

            weight = fluid.data(name='weight', shape=[2, 8, 32, 32], dtype='float32')
            x = fluid.layers.spectral_norm(weight=weight, dim=1, power_iters=2)
    """
    helper = LayerHelper('spectral_norm', **locals())
    dtype = weight.dtype

    # create intput and parameters
    inputs = {'Weight': weight}
    input_shape = weight.shape
    h = input_shape[dim]
    w = np.prod(input_shape) // h

    u = helper.create_parameter(
        attr=ParamAttr(),
        shape=[h],
        dtype=dtype,
        default_initializer=Normal(0., 1.))
    u.stop_gradient = True
    inputs['U'] = u
    v = helper.create_parameter(
        attr=ParamAttr(),
        shape=[w],
        dtype=dtype,
        default_initializer=Normal(0., 1.))
    inputs['V'] = v
    v.stop_gradient = True

    # create output
    out = helper.create_variable(dtype=dtype)

    helper.append_op(
        type="spectral_norm",
        inputs=inputs,
        outputs={"Out": out, },
        attrs={
            "dim": dim,
            "power_iters": power_iters,
            "eps": eps,
        })

    return out


def conv2d_transpose(input,
                     num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=None,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     act=None,
                     name=None,
                     data_format='NCHW'):
    """
    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW or NHWC format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a 4-D Tensor with NCHW or NHWC format.
    * :math:`W`: Filter value, a 4-D Tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, a 4-D Tensor with data format 'NCHW' or 'NHWC', the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H^\prime_{out} &= (H_{in} - 1) * strides[0] - pad_height_top - pad_height_bottom + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - pad_width_left - pad_width_right + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] ]

    Note:
          The conv2d_transpose can be seen as the backward of the conv2d. For conv2d, 
          when stride > 1, conv2d maps multiple input shape to the same output shape, 
          so for conv2d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, W_{out} = W^\prime_{out}`; 
          else, the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[0]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[1]`, 
          conv2d_transpose can compute the kernel size automatically.

    Args:
        input(Variable): 4-D Tensor with [N, C, H, W] or [N, H, W, C] format,
                         its data type is float32 or float64.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_height, image_width). None if use
            filter_size, padding, and stride to calculate output_size.
            If output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None. output_size and filter_size 
            should not be None at the same time.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_height, filter_size_width).
            Otherwise, filter_size_height = filter_size_width = filter_size. None if 
            use output size to calculate filter_size. Default: None. filter_size and 
            output_size should not be None at the same time.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution. 
            If stride is a tuple, it must contain two integers, (stride_height, stride_width). 
            Otherwise, stride_height = stride_width = stride. Default: stride = 1.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively adds
             `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a
             string, either 'VALID' or 'SAME' supported, which is the padding algorithm.
             If `padding` is a tuple or list, it could be in three forms:
             `[pad_height, pad_width]` or
            `[pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`, and
            when `data_format` is `'NCHW'`,
            `padding` can be in the form `[[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `'NHWC'`, `padding` can be in the form
            `[[0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        dilation(int|tuple, optional): The dilation size. It means the spacing between the kernel points. 
            If dilation is a tuple, it must contain two integers, (dilation_height, dilation_width). 
            Otherwise, dilation_height = dilation_width = dilation. Default: dilation = 1.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_height, filter_size_width).
            Otherwise, filter_size_height = filter_size_width = filter_size. None if 
            use output size to calculate filter_size. Default: None.
        groups(int, optional): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups = 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.
        data_format(str, optional): The data format of the input and output data. An optional string
            from: `"NCHW"`, `"NHWC"`. When it is `"NCHW"`, the data is stored in the order of:
            `[batch_size, input_channels, input_height, input_width]`. Default: 'NCHW'.

    Returns:
        A Variable holding Tensor representing the conv2d_transpose, whose 
        data type is the same with input and shape is (num_batches, channels, out_h, 
        out_w) or (num_batches, out_h, out_w, channels). If act is None, the tensor variable 
        storing the transposed convolution result, and if act is not None, the 
        tensor variable storing transposed convolution and non-linearity activation 
        result.

    Raises:
        ValueError: If the shapes of output, input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 32, 32], dtype='float32')
          conv2d_transpose = fluid.layers.conv2d_transpose(input=data, num_filters=2, filter_size=3)
    """
    assert param_attr is not False, "param_attr should not be False in conv2d_transpose."
    if data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Attr(data_format) of Op(fluid.layers.conv2d_transpose) got wrong value: received "
            + data_format + " but only NCHW or NHWC supported.")

    input_channel = input.shape[1] if data_format == 'NCHW' else input.shape[-1]
    op_type = 'conv2d_transpose'
    if (input_channel == groups and num_filters == input_channel and
            not use_cudnn):
        op_type = 'depthwise_conv2d_transpose'

    helper = LayerHelper(op_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv2d_transpose must be Variable")

    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]) and (data_format == "NCHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:4]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NHWC"):
                if not (padding[0] == [0, 0] and padding[3] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:3]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 4, 'padding')
        else:
            padding = utils.convert_to_list(padding, 2, 'padding')
            padding = [padding[0], padding[0], padding[1], padding[1]]
        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        h_in = input.shape[2] if data_format == 'NCHW' else input.shape[1]
        w_in = input.shape[3] if data_format == 'NCHW' else input.shape[2]

        filter_size_h = (output_size[0] - (h_in - 1) * stride[0] + padding[0] +
                         padding[1] - 1) // dilation[0] + 1
        filter_size_w = (output_size[1] - (w_in - 1) * stride[1] + padding[2] +
                         padding[3] - 1) // dilation[1] + 1
        filter_size = [filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 2,
                                            'conv2d_transpose.filter_size')

    if output_size is None:
        output_size = []
    elif isinstance(output_size, list) or isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        raise ValueError("output_size should be list or int")
    groups = 1 if groups is None else groups
    filter_shape = [input_channel, num_filters // groups] + filter_size

    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'output_size': output_size,
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    out = helper.append_activation(pre_act)
    return out


def conv3d_transpose(input,
                     num_filters,
                     output_size=None,
                     filter_size=None,
                     padding=0,
                     stride=1,
                     dilation=1,
                     groups=None,
                     param_attr=None,
                     bias_attr=None,
                     use_cudnn=True,
                     act=None,
                     name=None,
                     data_format='NCDHW'):
    """
    The convolution3D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW or NDHWC format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <https://arxiv.org/pdf/1603.07285.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a Tensor with NCDHW or NDHWC format.
    * :math:`W`: Filter value, a Tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D Tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, D_f, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`

        Where

        .. math::

           D^\prime_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\\\
           H^\prime_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1 \\\\
           D_{out} &\in [ D^\prime_{out}, D^\prime_{out} + strides[0] ] \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[1] ] \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[2] ]

    Note:
          The conv3d_transpose can be seen as the backward of the conv3d. For conv3d, 
          when stride > 1, conv3d maps multiple input shape to the same output shape, 
          so for conv3d_transpose, when stride > 1, input shape maps multiple output shape.
          If output_size is None, :math:`H_{out} = H^\prime_{out}, :math:`H_{out} = \
          H^\prime_{out}, W_{out} = W^\prime_{out}`; else, the :math:`D_{out}` of the output 
          size must between :math:`D^\prime_{out}` and :math:`D^\prime_{out} + strides[0]`, 
          the :math:`H_{out}` of the output size must between :math:`H^\prime_{out}` 
          and :math:`H^\prime_{out} + strides[1]`, and the :math:`W_{out}` of the output size must 
          between :math:`W^\prime_{out}` and :math:`W^\prime_{out} + strides[2]`, 
          conv3d_transpose can compute the kernel size automatically.

    Args:
        input(Variable): The input is 5-D Tensor with shape [N, C, D, H, W] or [N, D, H, W, C], the data type 
            of input is float32 or float64.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple, optional): The output image size. If output size is a
            tuple, it must contain three integers, (image_depth, image_height, image_width). This
            parameter only works when filter_size is None. If output_size and filter_size are 
            specified at the same time, They should follow the formula above. Default: None. 
            Output_size and filter_size should not be None at the same time.
        filter_size(int|tuple, optional): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_depth, filter_size_height,
            filter_size_width). Otherwise, filter_size_depth = filter_size_height = \
            filter_size_width = filter_size. None if use output size to
            calculate filter_size. Default: None. filter_size and output_size should not be 
            None at the same time.
        padding(int|list|str|tuple, optional): The padding size. The padding argument effectively
             adds `dilation * (kernel - 1)` amount of zero-padding on both sides of input. If `padding` is a string,
             either 'VALID' or 'SAME' supported, which is the padding algorithm. If `padding`
             is a tuple or list, it could be in three forms: `[pad_depth, pad_height, pad_width]` or
            `[pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]`,
            and when `data_format` is `'NCDHW'`, `padding` can be in the form
            `[[0,0], [0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]`.
            when `data_format` is `'NDHWC'`, `padding` can be in the form
            `[[0,0], [pad_depth_front, pad_depth_back], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0,0]]`.
            Default: padding = 0.
        stride(int|tuple, optional): The stride size. It means the stride in transposed convolution. 
            If stride is a tuple, it must contain three integers, (stride_depth, stride_height, 
            stride_width). Otherwise, stride_depth = stride_height = stride_width = stride. 
            Default: stride = 1.
        dilation(int|tuple, optional): The dilation size. It means the spacing between the kernel points. 
            If dilation is a tuple, it must contain three integers, (dilation_depth, dilation_height, 
            dilation_width). Otherwise, dilation_depth = dilation_height = dilation_width = dilation. 
            Default: dilation = 1.
        groups(int, optional): The groups number of the Conv3d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups=1
        param_attr (ParamAttr, optional): The parameter attribute for learnable parameters/weights
            of conv3d_transpose. If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr|bool, optional): The parameter attribute for the bias of conv3d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv3d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        name(str, optional): For detailed information, please refer 
           to :ref:`api_guide_Name`. Usually name is no need to set and 
           None by default.
        data_format(str, optional):The data format of the input and output data. An optional string from: `"NCHW"`, `"NHWC"`.
            When it is `"NCHW"`, the data is stored in the order of: `[batch_size, input_channels, input_height, input_width]`.
            Default: 'NCDHW'.

    Returns:
        A Variable holding Tensor representing the conv3d_transpose, whose data 
        type is the same with input and shape is (num_batches, channels, out_d, out_h, 
        out_w) or (num_batches, out_d, out_h, out_w, channels). If act is None, the tensor 
        variable storing the transposed convolution result, and if act is not None, the tensor 
        variable storing transposed convolution and non-linearity activation result.

    Raises:
        ValueError: If the shapes of output, input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 12, 32, 32], dtype='float32')
          conv3d_transpose = fluid.layers.conv3d_transpose(input=data, num_filters=2, filter_size=3)
    """
    assert param_attr is not False, "param_attr should not be False in conv3d_transpose."
    if data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Param(data_format) of Op(fluid.layers.conv3d_transpose) got wrong value: received "
            + data_format + " but only NCDHW or NDHWC supported.")
    l_type = "conv3d_transpose"
    helper = LayerHelper(l_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv3d_transpose must be Variable")
    input_channel = input.shape[1] if data_format == 'NCDHW' else input.shape[
        -1]

    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    def _update_padding(padding, data_format):
        def is_list_or_tuple(ele):
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        if is_list_or_tuple(padding) and len(padding) == 5:
            if is_list_or_tuple(padding[0]) and (data_format == "NCDHW"):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:5]
                padding = [ele for a_list in padding for ele in a_list]
            elif is_list_or_tuple(padding[0]) and (data_format == "NDHWC"):
                if not (padding[0] == [0, 0] and padding[4] == [0, 0]):
                    raise ValueError(
                        "Non-zero padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[1:4]
                padding = [ele for a_list in padding for ele in a_list]
            padding = utils.convert_to_list(padding, 6, 'padding')

        elif is_list_or_tuple(padding) and len(padding) == 6:
            padding = utils.convert_to_list(padding, 6, 'padding')
        else:
            padding = utils.convert_to_list(padding, 3, 'padding')
            padding = [
                padding[0], padding[0], padding[1], padding[1], padding[2],
                padding[2]
            ]

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '%s'. It can only be 'SAME' or 'VALID'." %
                str(padding))
        if padding == "VALID":
            padding_algorithm = "VALID"
            padding = [0, 0, 0, 0, 0, 0]
        elif padding == "SAME":
            padding_algorithm = "SAME"
            padding = [0, 0, 0, 0, 0, 0]

    padding = _update_padding(padding, data_format)

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        d_in = input.shape[2] if data_format == 'NCDHW' else input.shape[1]
        h_in = input.shape[3] if data_format == 'NCDHW' else input.shape[2]
        w_in = input.shape[4] if data_format == 'NCDHW' else input.shape[3]

        filter_size_d = (output_size[0] - (d_in - 1) * stride[0] + padding[0] +
                         padding[1] - 1) // dilation[0] + 1
        filter_size_h = (output_size[1] - (h_in - 1) * stride[1] + padding[2] +
                         padding[3] - 1) // dilation[1] + 1
        filter_size_w = (output_size[2] - (w_in - 1) * stride[2] + padding[4] +
                         padding[5] - 1) // dilation[2] + 1
        filter_size = [filter_size_d, filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 3,
                                            'conv3d_transpose.filter_size')

    groups = 1 if groups is None else groups
    filter_shape = [input_channel, num_filters // groups] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    if data_format == 'NCDHW':
        data_format = 'NCHW'
    if data_format == 'NDHWC':
        data_format = 'NHWC'

    pre_bias = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'padding_algorithm': padding_algorithm,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn,
            'data_format': data_format
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    out = helper.append_activation(pre_act)
    return out


def sequence_expand(x, y, ref_level=-1, name=None):
    """Sequence Expand Layer. This layer will expand the input variable ``x`` \
        according to specified level ``ref_level`` lod of ``y``. Please note that \
        the lod level of ``x`` is at most 1. If the lod level of ``x`` is 1, than \
        the size of lod of ``x`` must be equal to the length of ``ref_level`` lod \
        of ``y``. If the lod level of ``x`` is 0, then the first dim of ``x`` should \
        be equal to the size of ``ref_level`` of ``y``. The rank of **x** is at least 2. \
        When rank of ``x`` is greater than 2, then it would be viewed as a 2-D tensor.

    Please note that the input ``x`` should be LodTensor or Tensor, \
        and input ``y`` must be LodTensor.

    Following examples will explain how sequence_expand works:

    .. code-block:: text

        Case 1

        Consider 2 sequences [a][b] and [c][d], now we want to expand them to [a][b], [a][b], [c][d] and [c][d].
        Sequence [a][b] expand twice and [c][d] expands twice, so the lod which according to is [2, 2].

        Input x is a 1-level LoDTensor:
            x.lod  = [[2,        2]]    #lod based on length may be easier to understand
            x.data = [[a], [b], [c], [d]]
            x.dims = [4, 1]

        input y is a LoDTensor:
            y.lod = [[2,    2],    #the 0th level lod, according to this level
                     [3, 3, 1, 1]] #the 1st level lod, it has nothing to do with this level

        ref_level: 0

        then output is a 1-level LoDTensor out:
            out.lod =  [[2,        2,        2,        2]]    #lod based on offfset
            out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
            out.dims = [8, 1]


        Case 2

        Consider 3 sequences [a], [b], [c], now we want to expand them to [a][a], [c][c][c].
        It's obvious that the lod info of expanded sequences is [2, 0, 3].

        x is a Tensor:
            x.data = [[a], [b], [c]]
            x.dims = [3, 1]

        y is a LoDTensor:
            y.lod = [[2, 0, 3]]

        ref_level: -1

        then output is a 1-level LodTensor:
            out.data = [[a], [a], [c], [c], [c]]
            out.dims = [5, 1]

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor, with the \
            dims ``[M, K]``. The lod level is at most 1. The data type should be \
            float32, float64, int8, int32 or int64.
        y (Variable): The input variable which is a LoDTensor, the lod level is \
            at least 1.
        ref_level (int): Lod level of ``y`` to be referred by ``x``. If set to -1, \
                         refer the last level of lod.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default. 

    Returns: The expanded variable which is a LoDTensor, with dims ``[N, K]``. \
            ``N`` depends on the lod info of ``x`` and ``y``. \
            The data type is same as input.

    Return Type: Variable

    Examples:
        .. code-block:: python
	
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            x = fluid.data(name='x', shape=[4, 1], dtype='float32')
            y = fluid.data(name='y', shape=[8, 1],
                        dtype='float32', lod_level=1)
            out = layers.sequence_expand(x=x, y=y, ref_level=0)

            exe = fluid.Executor(fluid.CPUPlace())
            place = fluid.CPUPlace()

            np_data = np.array([[1], [2], [3], [4]]).astype('float32')
            x_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2]], place)
            print(x_lod_tensor)
            #lod: [[0, 2, 4]]
            #    dim: 4, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 2 3 4]

            np_data = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).astype('float32')
	    y_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2], [3,3,1,1]], place)
            print(y_lod_tensor)
            #lod: [[0, 2, 4][0, 3, 6, 7, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: int64_t
            #    data: [0 0 1 1 1 1 1 0]

            out_main = exe.run(fluid.default_main_program(),
                            feed={'x': x_lod_tensor, 'y': y_lod_tensor},
                            fetch_list=[out], return_numpy=False)
            print(out_main[0])
            #lod: [[0, 2, 4, 6, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 2 1 2 3 4 3 4]
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_expand', input=x, **locals())
    dtype = helper.input_dtype()
    tmp = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sequence_expand',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp},
        attrs={'ref_level': ref_level})
    return tmp


def sequence_expand_as(x, y, name=None):
    """Sequence Expand As Layer. This OP will expand the input variable ``x`` \
        according to the zeroth level lod of ``y``. Current implementation requires \
        the level number of ``y``'s lod must be 1, and the first dimension of \
        ``x`` should be equal to the size of ``y``'s zeroth level lod, thus \
        the expanded LodTensor has the same lod info as ``y``. The expanded result \
        has nothing to do with ``x``'s lod, so the lod of Input(X) is not considered.

    Please note that the input ``x`` should be LodTensor or Tensor, \
        and input ``y`` must be LodTensor.

    Following examples will explain how sequence_expand_as works:

    .. code-block:: text

        Case 1:

        Consider 4 sequences [a], [b], [c], [d], now we want to expand them to [a][a][a], [b][b][b], [c] and [d].
        It's obvious that the lod info of expanded sequences is [0, 3, 6, 7, 8].
        Given a 1-level LodTensor ``x``: 
            x.data = [[a], [b], [c], [d]]
            x.dims = [4, 1]
        and input ``y``
            y.lod = [[3, 3, 1, 1]] #lod based on length may be easier to understand

        then we get 1-level LoDTensor out:
            Out.lod =  [[0,            3,              6,  7,  8]] #based on offset
            Out.data = [[a], [a], [a], [b], [b], [b], [c], [d]]
            Out.dims = [8, 1]


        Case 2:

        Given a common Tensor ``x``:
            x.data = [[a, b], [c, d], [e, f]]
            x.dims = [3, 2]
        and input ``y``:
            y.lod = [[0, 2, 3, 6]]

        then we get a 1-level LoDTensor:
            out.lod =  [[0,             2,     3,                    6]]
            out.data = [[a, b], [a, b] [c, d], [e, f], [e, f], [e, f]]
            out.dims = [6, 2]

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor, with the \
            dims ``[M, K]``. The data type should be float32, float64, int8, int32 \
            or int64.
        y (Variable): The input variable which is a LoDTensor with 1-level lod.
        name (str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The expanded variable which is a LoDTensor with the dims ``[N, K]``. \
            ``N`` depends on the lod of ``y``, and the lod level must be 1. \
            The data type is same as input.

    Return Type: Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            x = fluid.data(name='x', shape=[4, 1], dtype='float32')
            y = fluid.data(name='y', shape=[8, 1], dtype='float32', lod_level=1)
            out = layers.sequence_expand_as(x=x, y=y)

            exe = fluid.Executor(fluid.CPUPlace())
            place = fluid.CPUPlace()

            np_data = np.array([[1], [2], [3], [4]]).astype('float32')
            x_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2]], place)
            print(x_lod_tensor)
            #lod: [[0, 2, 4]]
            #    dim: 4, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 2 3 4]

            np_data = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).astype('float32')
	    y_lod_tensor = fluid.create_lod_tensor(np_data, [[3,3,1,1]], place)
            print(y_lod_tensor)
            #lod: [[0, 3, 6, 7, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: int64_t
            #    data: [0 0 1 0 1 1 1 0]

            out_main = exe.run(fluid.default_main_program(),
                            feed={'x': x_lod_tensor, 'y': y_lod_tensor},
                            fetch_list=[out], return_numpy=False)
            print(out_main[0])
            #lod: [[0, 3, 6, 7, 8]]
            #    dim: 8, 1
            #    layout: NCHW
            #    dtype: float
            #    data: [1 1 1 2 2 2 3 4]
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_expand_as', input=x, **locals())
    dtype = helper.input_dtype()
    tmp = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sequence_expand_as',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp})
    return tmp


def sequence_pad(x, pad_value, maxlen=None, name=None):
    """
    This layer padding the sequences in a same batch to a common length (according \
         to ``maxlen``). The padding value is defined by ``pad_value``, and will be \
        appended to the tail of sequences. The result is a Python tuple ``(Out, Length)``: \
        the LodTensor ``Out`` is the padded sequences, and LodTensor ``Length`` is \
        the length information of input sequences. For removing paddding data (unpadding \
	operation), See :ref:`api_fluid_layers_sequence_unpad` .

    Please note that the input ``x`` should be LodTensor.

    .. code-block:: text

        Case 1:
        Given input 1-level LoDTensor x:
            x.lod = [[0,  2,   5]]
            x.data = [[a],[b],[c],[d],[e]]
        pad_value:
            pad_value.data = [0]
        maxlen = 4

        the output tuple (Out, Length):
            Out.data = [[[a],[b],[0],[0]],[[c],[d],[e],[0]]]
            Length.data = [2, 3]      #Original sequences length

        Case 2:
        Given input 1-level LoDTensor x:
            x.lod =  [[0,             2,                     5]]
            x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
        pad_value:
            pad_value.data = [0]
        defualt maxlen = None, (the virtual value is 3, according to the shape of x)

        the output tuple (Out, Length):
            Out.data = [[[a1,a2],[b1,b2],[0,0]],[[c1,c2],[d1,d2],[e1,e2]]]
            Length.data = [2, 3]

        Case 3:
        Given input 1-level LoDTensor x:
            x.lod =  [[0,             2,                     5]]
            x.data = [[a1,a2],[b1,b2],[c1,c2],[d1,d2],[e1,e2]]
        pad_value:
            pad_value.data = [p1,p2]
        defualt maxlen = None, (the virtual value is 3)

        get tuple (Out, Length):
            Out.data = [[[a1,a2],[b1,b2],[p1,p2]],[[c1,c2],[d1,d2],[e1,e2]]]
            Length.data = [2, 3]



    Args:
        x (Variable): Input 1-level LodTensor with dims ``[M, K]``. The batch \
            size is described by lod infor (the number of sequnces ). \
            The data type should be float32, float64, int8, int32 or int64.
        pad_value (Variable): Padding value. It can be a scalar or a 1D tensor \
            with length ``K``. If it's a scalar, it will be automatically broadcasted \
            to a Tensor. The data type should be as same as ``x``.
        maxlen (int, optional): The length of padded sequences, None by default. \
            When it is None, all sequences will be padded up to the length of the \
            longest one among them; when it a certain positive value, it must be \
            greater than the length of the longest original sequence.
        name (str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: A Python tuple (Out, Length): the 1st is a 0 level LodTensor \
            ``Out``, with the shape ``[batch_size, maxlen, K]``; the second is the original \
            sequences length infor ``Length``, which should be a 0-level 1D LodTensor. \
            The size of ``Length`` is equal to batch size, and the data type is int64.

    Return Type: tuple

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            x = fluid.data(name='x', shape=[10, 5], dtype='float32', lod_level=1)
            pad_value = fluid.layers.assign(
                input=numpy.array([0.0], dtype=numpy.float32))
            out = fluid.layers.sequence_pad(x=x, pad_value=pad_value)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_pad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    length = helper.create_variable_for_type_inference(dtype)

    pad_value.stop_gradient = True
    length.stop_gradient = True

    if maxlen is None:
        maxlen = -1
    helper.append_op(
        type='sequence_pad',
        inputs={'X': x,
                'PadValue': pad_value},
        outputs={'Out': out,
                 'Length': length},
        attrs={'padded_length': maxlen})
    return out, length


def sequence_unpad(x, length, name=None):
    """
    **Note**:
    
    **The input of the OP is Tensor and the output is LoDTensor.  For padding operation, See:**  :ref:`api_fluid_layers_sequence_pad`  
     
    The OP removes the padding data from the input based on the length information and returns a LoDTensor.

    .. code-block:: text

	Case 1:

	Given input Variable **x**:
	    x.data = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
		      [ 6.0,  7.0,  8.0,  9.0, 10.0],
		      [11.0, 12.0, 13.0, 14.0, 15.0]],

	in which there are 3 sequences padded to length 5, and the acutal length
	specified by input Variable **length**:

	    length.data = [2, 3, 4],

	after unpadding, the output Variable will be:

	    out.data = [[1.0, 2.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0]]
	    out.lod = [[0, 2, 5, 9]]

    Args:
        x(Variable): A Tensor which contains padding data, and its shape size can not be less than 2.
                     Supported data types: float32, float64, int32, int64.
        length(Variable): A 1D Tensor that stores the actual length of each sample, and the Tensor 
                          has the same shape with the 0th dimension of the X . Supported data types: int64.
        name(str|None):  The default value is None.  Normally there is no need for user to set this property.  
                         For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A LoDTensor whose recursive sequence length is consistent with the information of the length parameter and it has the same data type with input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # pad data
            x = fluid.data(name='x', shape=[10, 5], dtype='float32', lod_level=1)
            pad_value = fluid.layers.assign(input=numpy.array([0.0], dtype=numpy.float32))
            pad_data, len = fluid.layers.sequence_pad(x=x, pad_value=pad_value)
            
            # unpad data
            unpad_data = fluid.layers.sequence_unpad(x=pad_data, length=len)
    """

    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_unpad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)

    length.stop_gradient = True

    helper.append_op(
        type='sequence_unpad',
        inputs={'X': x,
                'Length': length},
        outputs={'Out': out})
    return out


def beam_search(pre_ids,
                pre_scores,
                ids,
                scores,
                beam_size,
                end_id,
                level=0,
                is_accumulated=True,
                name=None,
                return_parent_idx=False):
    """
    Beam search is a classical algorithm for selecting candidate words in a
    machine translation task.

    Refer to `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_
    for more details.

    **This operator only supports LoDTensor.** It is used after finishing
    scores calculation to perform beam search for one time step. Specifically,
    after ``ids`` and ``scores`` have been produced, it selects the top-K
    ( `k` is ``beam_size`` ) candidate word ids of current step from ``ids``
    according to the correspongding ``scores``. Additionally, ``pre_id`` and
    ``pre_scores`` are the output of `beam_search` at previous step, they
    are needed for special use to handle ended candidate translations.

    Note that if ``is_accumulated`` is True, the ``scores`` passed in should
    be accumulated scores. Otherwise, the ``scores`` are
    considered as the probabilities of single step and would be transformed to
    the log field and added up with ``pre_scores`` for final scores in this
    operator. Length penalty should be done with extra operators before calculating
    the accumulated scores if needed.

    Please see the following demo for a fully beam search usage example:

        fluid/tests/book/test_machine_translation.py

    Args:
        pre_ids(Variable): A LodTensor variable (lod level is 2), representing
            the selected ids of previous step. It is the output of beam_search
            at previous step. Its shape is `[batch_size, 1]` and its lod is
            `[[0, 1, ... , batch_size], [0, 1, ..., batch_size]]` at the
            first step. The data type should be int64.
        pre_scores(Variable): A LodTensor variable has the same shape and lod
            with ``pre_ids`` , representing the accumulated scores corresponding
            to the selected ids of previous step. It is the output of
            beam_search at previous step. The data type should be float32.
        ids(Variable|None): A LodTensor variable containing the candidates ids.
            It has the same lod with ``pre_ids`` and its shape should be
            `[batch_size * beam_size, K]`, where `K` supposed to be greater than
            ``beam_size`` and the first dimension size (decrease as samples reach
            to the end) should be same as that of ``pre_ids`` . The data type
            should be int64. It can be None, which use indice in ``scores`` as
            ids.
        scores(Variable): A LodTensor variable containing the accumulated
            scores corresponding to ``ids`` . Both its shape and lod are same as
            thoes of ``ids`` . The data type should be float32.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        level(int): **It can be ignored and mustn't change currently.**
            The 2 level lod used in this operator has the following
            meaning: The first level describes how many beams each sample has,
            which would change to 0 when beams of the sample all end (batch reduce);
            The second level describes how many times each beam is selected.
            Default 0, which shouldn't be changed currently.
        is_accumulated(bool): Whether the input ``score`` is accumulated scores.
            Default True.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.
        return_parent_idx(bool, optional): Whether to return an extra Tensor variable
            in output, which stores the selected ids' parent indice in
            ``pre_ids`` and can be used to update RNN's states by gather operator.
            Default False.

    Returns:
        tuple: The tuple contains two or three LodTensor variables. The two LodTensor, \
            representing the selected ids and the corresponding accumulated scores of \
            current step, have the same shape `[batch_size, beam_size]` and lod with 2 levels, \
            and have data types int64 and float32. If ``return_parent_idx`` is True, \
            an extra Tensor variable preserving the selected ids' parent indice \
            is included, whose shape is `[batch_size * beam_size]` and data type \
            is int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # Suppose `probs` contains predicted results from the computation
            # cell and `pre_ids` and `pre_scores` is the output of beam_search
            # at previous step.
            beam_size = 4
            end_id = 1
            pre_ids = fluid.data(
                name='pre_id', shape=[None, 1], lod_level=2, dtype='int64')
            pre_scores = fluid.data(
                name='pre_scores', shape=[None, 1], lod_level=2, dtype='float32')
            probs = fluid.data(
                name='probs', shape=[None, 10000], dtype='float32')
            topk_scores, topk_indices = fluid.layers.topk(probs, k=beam_size)
            accu_scores = fluid.layers.elementwise_add(
                x=fluid.layers.log(x=topk_scores),
                y=fluid.layers.reshape(pre_scores, shape=[-1]),
                axis=0)
            selected_ids, selected_scores = fluid.layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=end_id)
    """
    helper = LayerHelper('beam_search', **locals())
    score_type = pre_scores.dtype
    id_type = pre_ids.dtype

    inputs = {"pre_ids": pre_ids, "pre_scores": pre_scores, "scores": scores}
    if ids is not None:
        inputs["ids"] = ids

    selected_scores = helper.create_variable_for_type_inference(
        dtype=score_type)
    selected_ids = helper.create_variable_for_type_inference(dtype=id_type)
    # parent_idx is a tensor used to gather cell states at the next time
    # step. Though lod in selected_ids can also be used to gather by
    # sequence_expand, it is not efficient.
    # gather_op's index input only supports int32 dtype currently
    parent_idx = helper.create_variable_for_type_inference(dtype="int32")

    helper.append_op(
        type='beam_search',
        inputs=inputs,
        outputs={
            'selected_ids': selected_ids,
            'selected_scores': selected_scores,
            'parent_idx': parent_idx
        },
        attrs={
            # TODO(ChunweiYan) to assure other value support
            'level': level,
            'beam_size': beam_size,
            'end_id': end_id,
            'is_accumulated': is_accumulated,
        })
    if return_parent_idx:
        return selected_ids, selected_scores, parent_idx
    else:
        return selected_ids, selected_scores


def beam_search_decode(ids, scores, beam_size, end_id, name=None):
    """
    This operator is used after beam search has completed. It constructs the
    full predicted sequences for each sample by walking back along the search
    paths stored in lod of ``ids`` . The result sequences are stored in a
    LoDTensor, which uses the following way to parse:

    .. code-block:: text

        If lod = [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]

        The first level of lod stands for: There are 2 samples each having 3
        (beam width) predicted sequence.

        The second level of lod stands for: The lengths of the first sample's
        3 predicted sequences are 12, 12, 16; The lengths of the second sample's
        3 predicted sequences are 14, 13, 15.


    Please see the following demo for a fully beam search usage example:
        fluid/tests/book/test_machine_translation.py

    Args:
        ids(Variable): The LoDTensorArray variable containing the selected ids
            of all steps. Each LoDTensor in it has int64 data type and 2 level
            lod which can be used to get the search paths.
        scores(Variable): The LodTensorArray variable containing the accumulated
            scores corresponding to selected ids of all steps. It has the same size
            as ``ids`` . Each LoDTensor in it has the same shape and lod as the
            counterpart in ``ids`` , and has a float32 data type.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.

    Returns:
        tuple: The tuple contains two LodTensor variables. The two LodTensor, \
            containing the full sequences of ids and the correspongding accumulated \
            scores, have the same shape flattened to 1D and have the same 2 level \
            lod. The lod can be used to get how many predicted sequences each sample \
            has and how many ids each predicted sequence has.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # Suppose `ids` and `scores` are LodTensorArray variables reserving
            # the selected ids and scores of all steps
            ids = fluid.layers.create_array(dtype='int64')
            scores = fluid.layers.create_array(dtype='float32')
            finished_ids, finished_scores = fluid.layers.beam_search_decode(
                ids, scores, beam_size=5, end_id=0)
    """
    helper = LayerHelper('beam_search_decode', **locals())
    sentence_ids = helper.create_variable_for_type_inference(dtype=ids.dtype)
    sentence_scores = helper.create_variable_for_type_inference(dtype=ids.dtype)

    helper.append_op(
        type="beam_search_decode",
        inputs={"Ids": ids,
                "Scores": scores},
        outputs={
            "SentenceIds": sentence_ids,
            "SentenceScores": sentence_scores
        },
        attrs={"beam_size": beam_size,
               "end_id": end_id})

    return sentence_ids, sentence_scores


def lstm_unit(x_t,
              hidden_t_prev,
              cell_t_prev,
              forget_bias=0.0,
              param_attr=None,
              bias_attr=None,
              name=None):
    """
    Long-Short Term Memory (LSTM) RNN cell. This operator performs LSTM calculations for
    one time step, whose implementation is based on calculations described in `RECURRENT
    NEURAL NETWORK REGULARIZATION <http://arxiv.org/abs/1409.2329>`_  .

    We add forget_bias to the biases of the forget gate in order to
    reduce the scale of forgetting. The formula is as follows:
    
    .. math::

        i_{t} & = \sigma(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = \sigma(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} tanh (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = \sigma(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} tanh (c_{t})

    :math:`x_{t}` stands for ``x_t`` , corresponding to the input of current time step;
    :math:`h_{t-1}` and :math:`c_{t-1}` correspond to ``hidden_t_prev`` and ``cell_t_prev`` ,
    representing the output of from previous time step.
    :math:`i_{t}, f_{t}, c_{t}, o_{t}, h_{t}` are input gate, forget gate, cell, output gate
    and hidden calculation.

    Args:
        x_t(Variable): A 2D Tensor representing the input of current time step.
            Its shape should be :math:`[N, M]` , where :math:`N` stands for batch
            size, :math:`M` for the feature size of input. The data type should
            be float32 or float64.
        hidden_t_prev(Variable): A 2D Tensor representing the hidden value from
            previous step. Its shape should be :math:`[N, D]` , where :math:`N`
            stands for batch size, :math:`D` for the hidden size. The data type
            should be same as ``x_t`` .
        cell_t_prev(Variable): A 2D Tensor representing the cell value from
            previous step. It has the same shape and data type with ``hidden_t_prev`` .
        forget_bias (float, optional): :math:`forget\\_bias` added to the biases
            of the forget gate. Default 0.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.

    Returns:
        tuple: The tuple contains two Tensor variables with the same shape and \
            data type with ``hidden_t_prev`` , representing the hidden value and \
            cell value which correspond to :math:`h_{t}` and :math:`c_{t}` in \
            the formula.

    Raises:
        ValueError: Rank of x_t must be 2.
        ValueError: Rank of hidden_t_prev must be 2.
        ValueError: Rank of cell_t_prev must be 2.
        ValueError: The 1st dimensions of x_t, hidden_t_prev and cell_t_prev must be the same.
        ValueError: The 2nd dimensions of hidden_t_prev and cell_t_prev must be the same.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim, hidden_dim = 128, 64, 512
            data = fluid.data(name='step_data', shape=[None], dtype='int64')
            x = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            pre_hidden = fluid.data(
                name='pre_hidden', shape=[None, hidden_dim], dtype='float32')
            pre_cell = fluid.data(
                name='pre_cell', shape=[None, hidden_dim], dtype='float32')
            hidden = fluid.layers.lstm_unit(
                x_t=x,
                hidden_t_prev=pre_hidden,
                cell_t_prev=pre_cell)
    """
    helper = LayerHelper('lstm_unit', **locals())

    if len(x_t.shape) != 2:
        raise ValueError("Rank of x_t must be 2.")

    if len(hidden_t_prev.shape) != 2:
        raise ValueError("Rank of hidden_t_prev must be 2.")

    if len(cell_t_prev.shape) != 2:
        raise ValueError("Rank of cell_t_prev must be 2.")

    if x_t.shape[0] != hidden_t_prev.shape[0] or x_t.shape[
            0] != cell_t_prev.shape[0]:
        raise ValueError("The 1st dimensions of x_t, hidden_t_prev and "
                         "cell_t_prev must be the same.")

    if hidden_t_prev.shape[1] != cell_t_prev.shape[1]:
        raise ValueError("The 2nd dimensions of hidden_t_prev and "
                         "cell_t_prev must be the same.")

    if bias_attr is None:
        bias_attr = ParamAttr()

    size = cell_t_prev.shape[1]
    concat_out = concat(input=[x_t, hidden_t_prev], axis=1)
    fc_out = fc(input=concat_out,
                size=4 * size,
                param_attr=param_attr,
                bias_attr=bias_attr)
    dtype = x_t.dtype
    c = helper.create_variable_for_type_inference(dtype)
    h = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='lstm_unit',
        inputs={"X": fc_out,
                "C_prev": cell_t_prev},
        outputs={"C": c,
                 "H": h},
        attrs={"forget_bias": forget_bias})

    return h, c


def reduce_sum(input, dim=None, keep_dim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of summation operation on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Raises:
        TypeError, if out data type is different with the input data type.
    
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_sum(x)  # [3.5]
            fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
            fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_sum(y, dim=[1, 2]) # [10, 26]
            fluid.layers.reduce_sum(y, dim=[0, 1]) # [16, 20]

    """
    helper = LayerHelper('reduce_sum', **locals())
    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in reduce_sum must be Variable, but received %s"
            % (type(input)))
    if convert_dtype(
            input.dtype) not in ['float32', 'float64', 'int32', 'int64']:
        raise TypeError(
            "The data type of 'input' in reduce_sum  must be float32 or float64 or int32 or int64, but received %s."
            % (convert_dtype(input.dtype)))
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_mean(input, dim=None, keep_dim=False, name=None):
    """
    Computes the mean of the input tensor's elements along the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimension along which the mean is computed. If
            `None`, compute the mean over all elements of :attr:`input`
            and return a variable with a single element, otherwise it
            must be in the range :math:`[-rank(input), rank(input))`. If
            :math:`dim[i] < 0`, the dimension to reduce is
            :math:`rank(input) + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default 
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        Variable: Tensor, results of average on the specified dim of input tensor,
        it's data type is the same as input's Tensor.
    
    Raises:
        TypeError, if out data type is different with the input data type.
    
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_mean(x)  # [0.4375]
            fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
            fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
            fluid.layers.reduce_mean(x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_mean(y, dim=[1, 2]) # [2.5, 6.5]
            fluid.layers.reduce_mean(y, dim=[0, 1]) # [4.0, 5.0]
    """
    helper = LayerHelper('reduce_mean', **locals())
    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in reduce_mean must be Variable, but received %s"
            % (type(input)))
    if convert_dtype(
            input.dtype) not in ['float32', 'float64', 'int32', 'int64']:
        raise TypeError(
            "The data type of 'input' in reduce_mean  must be float32 or float64 or int32 or int64, but received %s."
            % (convert_dtype(input.dtype)))
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_mean',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_max(input, dim=None, keep_dim=False, name=None):
    """
    Computes the maximum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimension along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, results of maximum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_max(x)  # [0.9]
            fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
            fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
            fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_max(y, dim=[1, 2]) # [4.0, 8.0]
            fluid.layers.reduce_max(y, dim=[0, 1]) # [7.0, 8.0]
    """
    helper = LayerHelper('reduce_max', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_max',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_min(input, dim=None, keep_dim=False, name=None):
    """
    Computes the minimum of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, result of minimum on the specified dim of input tensor,
        it's data type is the same as input's Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_min(x)  # [0.1]
            fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
            fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
            fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_min(y, dim=[1, 2]) # [1.0, 5.0]
            fluid.layers.reduce_min(y, dim=[0, 1]) # [1.0, 2.0]
    """
    helper = LayerHelper('reduce_min', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_min',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_prod(input, dim=None, keep_dim=False, name=None):
    """
    Computes the product of tensor elements over the given dimension.

    Args:
        input (Variable): The input variable which is a Tensor, the data type is float32,
            float64, int32, int64.
        dim (list|int, optional): The dimensions along which the product is performed. If
            :attr:`None`, multipy all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Variable: Tensor, result of product on the specified dim of input tensor,
        it's data type is the same as input's Tensor.
    
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            x = fluid.data(name='x', shape=[2, 4], dtype='float32')
            fluid.layers.reduce_prod(x)  # [0.0002268]
            fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
            fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
            fluid.layers.reduce_prod(x, dim=1,
                                     keep_dim=True)  # [[0.027], [0.0084]]

            # y is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            y = fluid.data(name='y', shape=[2, 2, 2], dtype='float32')
            fluid.layers.reduce_prod(y, dim=[1, 2]) # [24.0, 1680.0]
            fluid.layers.reduce_prod(y, dim=[0, 1]) # [105.0, 384.0]
    """
    helper = LayerHelper('reduce_prod', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_prod',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_all(input, dim=None, keep_dim=False, name=None):
    """
    This OP computes the ``logical and`` of tensor elements over the given dimension, and output the result.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor, the input data type should be `bool`.
        dim (list|int|optional): The dimension along which the logical and is computed.
            If :attr:`None`, compute the logical and over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`. The default value is None. 
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true. The default value is False.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically. The default value is None. 

    Returns: 
        Variable, the output data type is bool. : The reduced tensor variable with ``logical and`` in given dims.

    Examples:
        .. code-block:: python
        
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            # x is a bool Tensor variable with following elements:
            #    [[True, False]
            #     [True, True]]
            x = layers.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
            x = layers.cast(x, 'bool')

            out = layers.reduce_all(x)  # False 
            out = layers.reduce_all(x, dim=0)  # [True, False]
            out = layers.reduce_all(x, dim=-1)  # [False, True]
            # keep_dim=False, x.shape=(2,2), out.shape=(2,)

            out = layers.reduce_all(x, dim=1, keep_dim=True)  # [[False], [True]]
            # keep_dim=True, x.shape=(2,2), out.shape=(2,1)

    """
    helper = LayerHelper('reduce_all', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_all',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def reduce_any(input, dim=None, keep_dim=False, name=None):
    """
    This OP computes the ``logical or`` of tensor elements over the given dimension, and output the result.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor, the input data type should be `bool`.
        dim (list|int|optional): The dimension along which the logical and is computed.
            If :attr:`None`, compute the logical and over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`. The default value is None. 
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true. The default value is False.
        name(str|None): A name for this layer(optional). If set None, the layer

    Returns: 
        Variable, the output data type is bool. : The reduced tensor variable with ``logical or`` in given dims.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            import numpy as np

            # x is a bool Tensor variable with following elements:
            #    [[True, False]
            #     [False, False]]
            x = layers.assign(np.array([[1, 0], [0, 0]], dtype='int32'))
            x = layers.cast(x, 'bool')

            out = layers.reduce_any(x)  # True
            out = layers.reduce_any(x, dim=0)  # [True, False]
            out = layers.reduce_any(x, dim=-1)  # [True, False]
            # keep_dim=False, x.shape=(2,2), out.shape=(2,)

            out = layers.reduce_any(x, dim=1,
                                     keep_dim=True)  # [[True], [False]]
            # keep_dim=True, x.shape=(2,2), out.shape=(2,1)

    """
    helper = LayerHelper('reduce_any', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    if dim is not None and not isinstance(dim, list):
        dim = [dim]
    helper.append_op(
        type='reduce_any',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={
            'dim': dim if dim != None else [0],
            'keep_dim': keep_dim,
            'reduce_all': True if dim == None else False
        })
    return out


def split(input, num_or_sections, dim=-1, name=None):
    """
    Split the input tensor into multiple sub-Tensors.

    Args:
        input (Variable): The input variable which is an N-D Tensor or LoDTensor, data type being float32, float64, int32 or int64.
        num_or_sections (int|list): Integer or list of Integers. If :attr:`num_or_sections` is an integer,
            then the integer indicates the number of equal sized sub-Tensors
            that the Tensor will be divided into. If :attr:`num_or_sections`
            is a list of integers, the length of list indicates the number of
            sub-Tensors and the integers indicate the sizes of sub-Tensors'
            :attr:`dim` dimension orderly. The the length of the list mustn't be larger than the Tensor's size of :attr:`dim` .
        dim (int): The dimension along which to split. If :math:`dim < 0`, the
            dimension to split along is :math:`rank(input) + dim`.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        list(Variable): The list of segmented Tensor variables.

    Example:
        .. code-block:: python

            import paddle.fluid as fluid

            # input is a variable which shape is [-1, 3, 9, 5]
            input = fluid.layers.data(
                 name="input", shape=[3, 9, 5], dtype="float32")

            x0, x1, x2 = fluid.layers.split(input, num_or_sections=3, dim=2)
            # x0.shape [-1, 3, 3, 5]
            # x1.shape [-1, 3, 3, 5]
            # x2.shape [-1, 3, 3, 5]

            x0, x1, x2 = fluid.layers.split(input, num_or_sections=[2, 3, 4], dim=2)
            # x0.shape [-1, 3, 2, 5]
            # x1.shape [-1, 3, 3, 5]
            # x2.shape [-1, 3, 4, 5]
    """
    helper = LayerHelper('split', **locals())
    input_shape = input.shape
    dim = (len(input_shape) + dim) if dim < 0 else dim
    if isinstance(num_or_sections, int):
        assert num_or_sections > 1, 'num_or_sections must be more than 1.'
        num = num_or_sections
    else:
        assert len(num_or_sections) <= input_shape[
            dim], 'len(num_or_sections) must not be more than input.shape[dim].'
        num = len(num_or_sections)
    outs = [
        helper.create_variable_for_type_inference(dtype=helper.input_dtype())
        for i in range(num)
    ]
    helper.append_op(
        type='split',
        inputs={'X': input},
        outputs={'Out': outs},
        attrs={
            'num': num_or_sections if isinstance(num_or_sections, int) else 0,
            'sections': num_or_sections
            if isinstance(num_or_sections, list) else [],
            'axis': dim
        })
    return outs


def l2_normalize(x, axis, epsilon=1e-12, name=None):
    """
    This op normalizes `x` along dimension `axis` using an L2
    norm. For a 1-D tensor (`dim` is fixed to 0), this layer computes

    .. math::

        y = \\frac{x}{ \sqrt{\sum {x^2} + epsion }}

    For `x` with more dimensions, this layer independently normalizes each 1-D
    slice along dimension `axis`.

    Args:
        x(Variable|list): The input tensor could be N-D tensor, and the input data type could be float32 or float64.
        axis(int): The axis on which to apply normalization. If `axis < 0`, \
            the dimension to normalization is rank(X) + axis. -1 is the
            last dimension.
        epsilon(float): The epsilon value is used to avoid division by zero, \
            the default value is 1e-12.
	name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    
    Returns:
        Variable: The output has the same shape and data type with `x`.

    Examples:

        .. code-block:: python
	    
	    # declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[2,3])
	    output = fluid.layers.l2_normalize(x=input,axis=0)
	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.random.rand(2,3).astype("float32")
	    print(input_data)

	    # [[0.5171216  0.12704141 0.56018186]
	    # [0.93251234 0.5382788  0.81709313]]
	
	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data)

	    # [array([[0.48496857, 0.22970329, 0.56545246],
	    # [0.8745316 , 0.9732607 , 0.82478094]], dtype=float32)]

	    # imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.l2_normalize(x=input, axis=-1)
    		print(output.numpy())
	    	
		# [[0.66907585 0.16437206 0.7247892 ]
		# [0.6899054  0.3982376  0.6045142 ]]
		
    """

    if len(x.shape) == 1:
        axis = 0
    helper = LayerHelper("l2_normalize", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    norm = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="norm",
        inputs={"X": x},
        outputs={"Out": out,
                 "Norm": norm},
        attrs={
            "axis": 1 if axis is None else axis,
            "epsilon": epsilon,
        })
    return out


def matmul(x, y, transpose_x=False, transpose_y=False, alpha=1.0, name=None):
    """
    Applies matrix multiplication to two tensors.

    Currently, the input tensors' rank can be any, but when the rank of any
    inputs is bigger than 3, this two inputs' rank should be equal.

    The actual behavior depends on the shapes of :math:`x`, :math:`y` and the
    flag values of :attr:`transpose_x`, :attr:`transpose_y`. Specifically:

    - If a transpose flag is specified, the last two dimensions of the tensor
      are transposed. If the tensor is rank-1 of shape :math:`[D]`, then for
      :math:`x` it is treated as :math:`[1, D]` in nontransposed form and as
      :math:`[D, 1]` in transposed form, whereas for :math:`y` it is the
      opposite: It is treated as :math:`[D, 1]` in nontransposed form and as
      :math:`[1, D]` in transposed form.

    - After transpose, the two tensors are 2-D or n-D and matrix multiplication
      performs in the following way.

      - If both are 2-D, they are multiplied like conventional matrices.
      - If either is n-D, it is treated as a stack of matrices residing in the
        last two dimensions and a batched matrix multiply supporting broadcast
        applies on the two tensors.

    Also note that if the raw tensor :math:`x` or :math:`y` is rank-1 and
    nontransposed, the prepended or appended dimension :math:`1` will be
    removed after matrix multiplication.

    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a Tensor or LoDTensor.
        transpose_x (bool): Whether to transpose :math:`x` before multiplication.
        transpose_y (bool): Whether to transpose :math:`y` before multiplication.
        alpha (float): The scale of output. Default 1.0.
        name(str|optional): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor (or LoDTensor) variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            # fluid.layers.matmul(x, y)
            # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            # fluid.layers.matmul(x, y)
            # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            # fluid.layers.matmul(x, y)
            # out: [B, M, N]

            # x: [M, K], y: [K, N]
            # fluid.layers.matmul(x, y)
            # out: [M, N]

            # x: [B, M, K], y: [K]
            # fluid.layers.matmul(x, y)
            # out: [B, M]

            # x: [K], y: [K]
            # fluid.layers.matmul(x, y)
            # out: [1]

            # x: [M], y: [N]
            # fluid.layers.matmul(x, y, True, True)
            # out: [M, N]

            import paddle.fluid as fluid
            import numpy

            # Graph Organizing
            x = fluid.data(name='x', shape=[2, 3], dtype='float32')
            y = fluid.data(name='y', shape=[3, 2], dtype='float32')
            output = fluid.layers.matmul(x, y, True, True)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            input_x = numpy.ones([2, 3]).astype(numpy.float32)
            input_y = numpy.ones([3, 2]).astype(numpy.float32)
            res, = exe.run(fluid.default_main_program(),
                           feed={'x':input_x, 'y':input_y},
                           fetch_list=[output])
            print(res)
            '''
            Output Value:
            [[2. 2. 2.]
             [2. 2. 2.]
             [2. 2. 2.]]
            '''
    """

    def __check_input(x, y):
        var_names = {'x': x, 'y': y}
        for name, val in var_names.items():
            if not isinstance(val, Variable):
                raise TypeError(
                    "The type of %s in matmul must be Variable, but received %s.\n"
                    % (name, (type(val))))
            if convert_dtype(val.dtype) in ['float16']:
                warnings.warn(
                    "The data type of %s in matmul only support float16 in GPU now."
                    % name)
            if convert_dtype(
                    val.dtype) not in ['float16', 'float32', 'float64']:
                raise TypeError(
                    "The data type of %s in matmul must be float16 or float32 or float64, but received %s.\n"
                    % (name, (convert_dtype(val.dtype))))

        x_shape = list(x.shape)
        y_shape = list(y.shape)
        if len(x_shape) == 1:
            x_shape = [1] + x_shape
        if len(y_shape) == 1:
            y_shape = y_shape + [1]

        # check the inner 2 dimensions
        if transpose_x:
            x_shape[-2], x_shape[-1] = x_shape[-1], x_shape[-2]
        if transpose_y:
            y_shape[-2], y_shape[-1] = y_shape[-1], y_shape[-2]
        if x_shape[-1] != y_shape[-2]:
            raise ValueError(
                "After performing an optional transpose, Input X's width should be "
                "equal to Y's width for multiplication "
                "prerequisites. But received X's shape: %s, Y's shape: %s\n" %
                (x_shape, y_shape))

        if len(y_shape) > 2 and len(x_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                # don't check neg shape
                if dim_x < 0 or y_shape[i] < 0:
                    continue
                if dim_x != y_shape[i]:
                    raise ValueError(
                        "When the matrix is larger than 2 dimensions, the higher "
                        "dimensional values of the two matrices need to be equal. "
                        "But received x_shape[%d] != y_shape[%d]. X's shape: %s, "
                        "Y's shape: %s.\n" % (i, i, x_shape, y_shape))

    __check_input(x, y)

    helper = LayerHelper('matmul', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='matmul',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={
            'transpose_X': transpose_x,
            'transpose_Y': transpose_y,
            'alpha': float(alpha),
        })
    return out


def topk(input, k, name=None):
    """
    This OP is used to find values and indices of the k largest entries
    for the last dimension.

    If the input is a 1-D Tensor, finds the k largest entries and outputs
    their values and indices.

    If the input is a Tensor with higher rank, this operator computes the top k
    entries along the last dimension.

    .. code-block:: text

        Case 1:

          Input:
            input.shape = [3, 4]
            input.data = [[5, 4, 2, 3],
                     [9, 7, 10, 25],
                     [6, 2, 10, 1]]
            k = 2

          Output:
            The first output:
            values.shape = [3, 2]
            values.data = [[5, 4],
                      [10, 25],
                      [6, 10]]

            The second output:
            indices.shape = [3, 2]
            indices.data = [[0, 1],
                       [2, 3],
                       [0, 2]]

    Args:
        input(Variable): The input tensor. Support data types: float32, float64.
        k(int | Variable): The number of top elements to look for along the last dimension
                           of input tensor.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Values (Variable): Input tensor's k largest elements along each last dimensional slice. The dimension is: :math:`input.shape[:-1]+[k]`.
        Indices (Variable): Indices of k largest elements alone the last dimension of input. The dimension is same as values.

    Raises:
        ValueError: If :math:`k < 1` or :math:`k > last dimension of input`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            # set batch size=None
            input = fluid.data(name="input", shape=[None, 13, 11], dtype='float32')
            top5_values, top5_indices = layers.topk(input, k=5) # top5_values.shape[None, 13, 5], top5_indices.shape=[None, 13, 5]

            # 1D Tensor
            input1 = fluid.data(name="input1", shape=[None, 13], dtype='float32')
            top5_values, top5_indices = layers.topk(input1, k=5) #top5_values.shape=[None, 5], top5_indices.shape=[None, 5]

            # k=Variable
            input2 = fluid.data(name="input2", shape=[None, 13, 11], dtype='float32')
            vk = fluid.data(name="vk", shape=[None, 1], dtype='int32') # save k in vk.data[0]
            vk_values, vk_indices = layers.topk(input2, k=vk) #vk_values.shape=[None, 13, k], vk_indices.shape=[None, 13, k]

    """
    helper = LayerHelper("top_k", **locals())
    values = helper.create_variable_for_type_inference(dtype=input.dtype)
    indices = helper.create_variable_for_type_inference(dtype="int64")
    inputs = {"X": [input]}
    attrs = None
    if isinstance(k, Variable):
        inputs['K'] = k
    else:
        attrs = {'k': k}
    helper.append_op(
        type="top_k",
        inputs=inputs,
        outputs={"Out": [values],
                 "Indices": [indices]},
        attrs=attrs)
    values.stop_gradient = True
    indices.stop_gradient = True
    return values, indices


def edit_distance(input,
                  label,
                  normalized=True,
                  ignored_tokens=None,
                  input_length=None,
                  label_length=None):
    """
    This op computes the edit distances between a batch of
    hypothesis strings and their references. Edit distance, also called
    Levenshtein distance, measures how dissimilar two strings are by counting
    the minimum number of operations to transform one string into anthor.
    Here the operations include insertion, deletion, and substitution.

    For example, given hypothesis string A = "kitten" and reference
    B = "sitting", the edit distance is 3 for A will be transformed into B
    at least after two substitutions and one insertion:

    "kitten" -> "sitten" -> "sittin" -> "sitting"

    The input is a LoDTensor/Tensor consisting of all the hypothesis strings with
    the total number denoted by `batch_size`, and the separation is specified
    by the LoD information or input_length. And the `batch_size` reference strings are arranged
    in order in the same way as `input`.

    The output contains the `batch_size` results and each stands for the edit
    distance for a pair of strings respectively. If Attr(normalized) is true,
    the edit distance will be divided by the length of reference string.

    Parameters:
        input(Variable): The indices for hypothesis strings, its rank should equals to 2 and its data type should be int64.
        label(Variable): The indices for reference strings, its rank should equals to 2 and its data type should be int64.
        normalized(bool, default True): Indicated whether to normalize the edit distance by
                          the length of reference string.
        ignored_tokens(list<int>, default None): Tokens that should be removed before
                                     calculating edit distance.
        input_length(Variable): The length for each sequence in `input` if it's of Tensor type, it should have shape `[batch_size]` and dtype int64.
        label_length(Variable): The length for each sequence in `label` if it's of Tensor type, it should have shape `[batch_size]` and dtype int64.

    Returns:
	Tuple:

        edit_distance_out(Variable): edit distance result in shape [batch_size, 1].
        sequence_num(Variable): sequence number in shape [].
        


    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid

            # using LoDTensor
            x_lod = fluid.data(name='x_lod', shape=[None,1], dtype='int64', lod_level=1)
            y_lod = fluid.data(name='y_lod', shape=[None,1], dtype='int64', lod_level=1)
            distance_lod, seq_num_lod = fluid.layers.edit_distance(input=x_lod, label=y_lod)

            # using Tensor
            x_seq_len = 5
            y_seq_len = 6
            x_pad = fluid.data(name='x_pad', shape=[None,x_seq_len], dtype='int64')
            y_pad = fluid.data(name='y_pad', shape=[None,y_seq_len], dtype='int64')
            x_len = fluid.data(name='x_len', shape=[None], dtype='int64')
            y_len = fluid.data(name='y_len', shape=[None], dtype='int64')
            distance_pad, seq_num_pad = fluid.layers.edit_distance(input=x_pad, label=y_pad, input_length=x_len, label_length=y_len)

    """
    helper = LayerHelper("edit_distance", **locals())

    # remove some tokens from input and labels
    if ignored_tokens is not None and len(ignored_tokens) > 0:
        erased_input = helper.create_variable_for_type_inference(dtype="int64")
        erased_label = helper.create_variable_for_type_inference(dtype="int64")

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [input]},
            outputs={"Out": [erased_input]},
            attrs={"tokens": ignored_tokens})
        input = erased_input

        helper.append_op(
            type="sequence_erase",
            inputs={"X": [label]},
            outputs={"Out": [erased_label]},
            attrs={"tokens": ignored_tokens})
        label = erased_label

    this_inputs = {"Hyps": [input], "Refs": [label]}
    if input_length and label_length:
        this_inputs['HypsLength'] = [input_length]
        this_inputs['RefsLength'] = [label_length]

    # edit distance op
    edit_distance_out = helper.create_variable_for_type_inference(dtype="int64")
    sequence_num = helper.create_variable_for_type_inference(dtype="int64")
    helper.append_op(
        type="edit_distance",
        inputs=this_inputs,
        outputs={"Out": [edit_distance_out],
                 "SequenceNum": [sequence_num]},
        attrs={"normalized": normalized})

    return edit_distance_out, sequence_num


def ctc_greedy_decoder(input,
                       blank,
                       input_length=None,
                       padding_value=0,
                       name=None):
    """
    This op is used to decode sequences by greedy policy by the following steps:

    1. Get the indexes of maximum value for each row in input. a.k.a.
       numpy.argmax(input, axis=0).
    2. For each sequence in result of step1, merge repeated tokens between two
       blanks and delete all blanks.

    This op is implemented in two modes: lod and padding, either of them can be used.
    The input can be either LoDTensor or Tensor, corresponding to lod and padding 
    mode respectively.

    A simple example as below:

    .. code-block:: text

        Given:
        (1) for lod mode:

        input.data = [[0.6, 0.1, 0.3, 0.1],
                      [0.3, 0.2, 0.4, 0.1],
                      [0.1, 0.5, 0.1, 0.3],
                      [0.5, 0.1, 0.3, 0.1],

                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.2, 0.4],
                      [0.2, 0.2, 0.1, 0.5],
                      [0.5, 0.1, 0.3, 0.1]]

        input.lod = [[4, 4]]

        Computation:

        step1: Apply argmax to first input sequence which is input.data[0:4]. Then we get:
               [[0], [2], [1], [0]]
        step2: merge repeated tokens and remove blank which is 0. Then we get first output sequence:
               [[2], [1]]

        Finally:

        output.data = [[2],
                       [1],
                       [3]]

        output.lod = [[2, 1]]

        (2) for padding mode:

         input.data = [[[0.6, 0.1, 0.3, 0.1],
                        [0.3, 0.2, 0.4, 0.1],
                        [0.1, 0.5, 0.1, 0.3],
                        [0.5, 0.1, 0.3, 0.1]],

                       [[0.5, 0.1, 0.3, 0.1],
                        [0.2, 0.2, 0.2, 0.4],
                        [0.2, 0.2, 0.1, 0.5],
                        [0.5, 0.1, 0.3, 0.1]]]

        input_length.data = [[4], [4]]
        input.shape = [2, 4, 4]

        step1: Apply argmax to first input sequence which is input.data[0:4]. Then we get:
               [[0], [2], [1], [0]], for input.data[4:8] is [[0], [3], [3], [0]], shape is [2,4,1]
        step2: Change the argmax result to use padding mode, then argmax result is 
                [[0, 2, 1, 0], [0, 3, 3, 0]], shape is [2, 4], lod is [], input_length is [[4], [4]]
        step3: Apply ctc_align to padding argmax result, padding_value is 0

        Finally:
        output.data = [[2, 1, 0, 0],
                       [3, 0, 0, 0]]
        output_length.data = [[2], [1]]


    Parameters:

        input(Variable): the probabilities of variable-length sequences. When in lod mode, 
                         it is a 2-D LoDTensor with LoD information. It's shape is [Lp, num_classes + 1] 
                         where Lp is the sum of all input sequences' length and
                         num_classes is the true number of classes. When in padding mode,
                         it is a 3-D Tensor with padding, It's shape is [batch_size, N, num_classes + 1].
                         (not including the blank label). The data type can be float32 or float64.
        blank(int): the blank label index of Connectionist Temporal
                    Classification (CTC) loss, which is in the half-opened
                    interval [0, num_classes + 1).
        input_length(Variable, optional): 2-D LoDTensor, shape is [batch_size, 1], data type is int64.
                                 It is used for padding mode. In lod mode, input_length is None.
        padding_value(int): padding value.
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name` 

    Returns:
        For lod mode, returns the result of CTC greedy decoder, 2-D LoDTensor, shape is [Lp, 1], \
        data type is int64. 'Lp' is the sum of all output sequences' length. If all the sequences \
        in result were empty, the result LoDTensor will be [-1] with  empty \
        LoD [[]].

        For padding mode, returns a tuple of (output, output_length), which was describled as below: 

        output, 2-D Tensor, shape is [batch_size, N], data type is int64.

        output_length, 2-D Tensor, shape is [batch_size, 1], data type is int64. It is the length of \
                           each sequence of output for padding mode.

    Return type:
        For lod mode: Variable

        For padding mode: tuple of two Variables (output, output_length).


    Examples:
        .. code-block:: python

            # for lod mode
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 8], dtype='float32', lod_level=1)
            cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)

            # for padding mode
            x_pad = fluid.data(name='x_pad', shape=[10, 4, 8], dtype='float32')
            x_pad_len = fluid.data(name='x_pad_len', shape=[10, 1], dtype='int64')
            out, out_len = fluid.layers.ctc_greedy_decoder(input=x_pad, blank=0,
                            input_length=x_pad_len)

    """
    helper = LayerHelper("ctc_greedy_decoder", **locals())
    _, topk_indices = topk(input, k=1)

    # ctc align op
    ctc_out = helper.create_variable_for_type_inference(dtype="int64")

    if input_length is None:
        helper.append_op(
            type="ctc_align",
            inputs={"Input": [topk_indices]},
            outputs={"Output": [ctc_out]},
            attrs={"merge_repeated": True,
                   "blank": blank})
        return ctc_out
    else:
        ctc_out_len = helper.create_variable_for_type_inference(dtype="int64")
        ctc_input = squeeze(topk_indices, [2])

        helper.append_op(
            type="ctc_align",
            inputs={"Input": [ctc_input],
                    "InputLength": [input_length]},
            outputs={"Output": [ctc_out],
                     "OutputLength": [ctc_out_len]},
            attrs={
                "merge_repeated": True,
                "blank": blank,
                "padding_value": padding_value
            })
        return ctc_out, ctc_out_len


def warpctc(input,
            label,
            blank=0,
            norm_by_times=False,
            input_length=None,
            label_length=None):
    """
    An operator integrating the open source Warp-CTC library
    (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation is
    interated to the Warp-CTC library to normlize values for each row of the
    input tensor.

    Args:
       input (Variable): The unscaled probabilities of variable-length sequences,
         which is a 2-D Tensor with LoD information, or a 3-D Tensor without Lod
         information. When it is a 2-D LodTensor, it's shape is 
         [Lp, num_classes + 1], where Lp is the sum of all input
         sequences' length and num_classes is the true number of classes.
         (not including the blank label). When it is a 3-D Tensor, it's shape 
         is [max_logit_length, batch_size, num_classes + 1],
         where max_logit_length is the length of the longest
         input logit sequence. The data type must be float32.
       label (Variable): The ground truth of variable-length sequence,
         which is a 2-D Tensor with LoD information or a 2-D Tensor without
         LoD information. When it is a 2-D LoDTensor or 2-D Tensor, 
         it is of the shape [Lg, 1], where Lg is th sum of all labels' length.
         The data type must be int32.
       blank (int, default 0): The blank label index of Connectionist
         Temporal Classification (CTC) loss, which is in the
         half-opened interval [0, num_classes + 1). The data type must be int32. 
       norm_by_times(bool, default false): Whether to normalize the gradients
         by the number of time-step, which is also the sequence's length.
         There is no need to normalize the gradients if warpctc layer was
         follewed by a mean_op.
       input_length(Variable): The length for each input sequence if it is 
         of Tensor type, it should have shape `[batch_size]` and dtype int64.
       label_length(Variable): The length for each label sequence if it is
         of Tensor type, it should have shape `[batch_size]` and dtype int64.

    Returns:
        Variable: The Connectionist Temporal Classification (CTC) loss,
        which is a 2-D Tensor with the shape [batch_size, 1].
        The date type is the same as input.

    Examples:

        .. code-block:: python

            # using LoDTensor
            import paddle.fluid as fluid
            import numpy as np
            
            predict = fluid.data(name='predict', 
                                        shape=[None, 5],
                                        dtype='float32',lod_level=1)
            label = fluid.data(name='label', shape=[None, 1],
                                      dtype='int32', lod_level=1)
            cost = fluid.layers.warpctc(input=predict, label=label)
            place = fluid.CPUPlace()
            x=fluid.LoDTensor()
            data = np.random.rand(8, 5).astype("float32")
            x.set(data, place)
            x.set_lod([[0,4,8]])
            y=fluid.LoDTensor()
            data = np.random.randint(0, 5, [4, 1]).astype("int32")
            y.set(data, place)
            y.set_lod([[0,2,4]])
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            output= exe.run(feed={"predict": x,"label": y},
                                         fetch_list=[cost.name])
            print output

        .. code-block:: python

            # using Tensor
            import paddle.fluid as fluid
            import numpy as np
            
            # length of the longest logit sequence
            max_seq_length = 5
            # number of logit sequences
            batch_size = None
            logits = fluid.data(name='logits', 
                                       shape=[max_seq_length, batch_size, 5],
                                       dtype='float32')
            logits_length = fluid.data(name='logits_length', shape=[None],
                                         dtype='int64')
            label = fluid.layers.data(name='label', shape=[None, 1],
                                       dtype='int32')
            label_length = fluid.layers.data(name='labels_length', shape=[None],
                                         dtype='int64')
            cost = fluid.layers.warpctc(input=logits, label=label,
                                        input_length=logits_length,
                                        label_length=label_length)
            place = fluid.CPUPlace()
            batch_size = 2
            x = np.random.rand(max_seq_length, batch_size, 5).astype("float32")
            y = np.random.randint(0, 5, [max_seq_length * batch_size, 1]).astype("int32")
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            output= exe.run(feed={"logits": x,
                                  "label": y,
                                  "logits_length": np.array([5, 4]).astype("int64"),
                                  "labels_length": np.array([3, 2]).astype("int64")},
                                  fetch_list=[cost.name])
            print(output)
    """
    helper = LayerHelper('warpctc', **locals())
    this_inputs = {'Logits': [input], 'Label': [label]}
    if input_length and label_length:
        this_inputs['LogitsLength'] = [input_length]
        this_inputs['LabelLength'] = [label_length]

    loss_out = helper.create_variable_for_type_inference(dtype=input.dtype)
    grad_out = helper.create_variable_for_type_inference(dtype=input.dtype)

    helper.append_op(
        type='warpctc',
        inputs=this_inputs,
        outputs={'WarpCTCGrad': [grad_out],
                 'Loss': [loss_out]},
        attrs={
            'blank': blank,
            'norm_by_times': norm_by_times,
        })
    return loss_out


def sequence_reshape(input, new_dim):
    """
    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use reshape Op.(fluid.layers.** :ref:`api_fluid_layers_reshape` ).

    This operator only supports LoDTensor as input. Given :attr:`new_dim` ,
    it will compute new shape according to original length of each sequence,
    original dimensions and :attr:`new_dim` . Then it will output a new LoDTensor
    containing :attr:`new_dim` . Currently it only supports 1-level LoDTensor.
    Please make sure that (original length * original dimensions) can be divided
    by the :attr:`new_dim` with no remainder for each sequence.

    .. code-block:: text

        input is a LoDTensor:
            input.lod  = [[0, 2, 6]]
            input.data = [[1,  2], [3,  4],
                          [5,  6], [7,  8],
                          [9, 10], [11, 12]]
            input.shape = [6, 2]

        set new_dim = 4
        out is a LoDTensor:
            out.lod  = [[0, 1, 3]]
            out.data = [[1,  2,  3,  4],
                        [5,  6,  7,  8],
                        [9, 10, 11, 12]]
            out.shape = [3, 4]


    Args:

       input (Variable): 1-level LoDTensor with shape :math:`[M, K]` . The data type should
            be int32, int64, float32 or float64.
       new_dim (int): New dimension that the input LoDTensor is reshaped to.

    Returns:
        Variable: Reshaped LoDTensor according to new dimension. The data type is same as input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 16], dtype='float32', lod_level=1)
            x_reshaped = fluid.layers.sequence_reshape(input=x, new_dim=4)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_reshape', **locals())
    out = helper.create_variable_for_type_inference(helper.input_dtype())
    helper.append_op(
        type='sequence_reshape',
        inputs={'X': [input]},
        outputs={'Out': [out]},
        attrs={'new_dim': new_dim})
    return out


# FIXME(wuyi): let docstring_checker.py understand @autodoc.
# For now, the comments in c++ use types like Tensor, but in python side
# the type is often "Variable", and arguments may vary.
@templatedoc(op_type="nce")
def nce(input,
        label,
        num_total_classes,
        sample_weight=None,
        param_attr=None,
        bias_attr=None,
        num_neg_samples=None,
        name=None,
        sampler="uniform",
        custom_dist=None,
        seed=0,
        is_sparse=False):
    """
    ${comment}

    Args:
        input (Variable): Input variable, 2-D tensor with shape [batch_size, dim], 
            and data type is float32 or float64.
        label (Variable): Input label, 2-D tensor with shape [batch_size, num_true_class],
            and data type is int64.
        num_total_classes (int):${num_total_classes_comment}.
        sample_weight (Variable|None): A Variable of shape [batch_size, 1]
            storing a weight for each sample. The default weight for each
            sample is 1.0.
        param_attr (ParamAttr|None): To specify the weight parameter attribute. 
            Default: None, which means the default weight parameter property is 
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr|None): To specify the bias parameter attribute. 
            Default: None, which means the default bias parameter property is 
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        num_neg_samples (int): ${num_neg_samples_comment}.
        name(str|None): For detailed information, please refer to 
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        sampler (str, optional): The sampler used to sample class from negtive classes.
                       It can be 'uniform', 'log_uniform' or 'custom_dist'.
                       default: 'uniform'.
        custom_dist (nd.array|None): A numpy ndarray with size=num_total_classes.
                       It is used when sampler is set to 'custom_dist'.
                       custom_dist[i] is the probsbility of i-th class to be sampled.
                       default: None.
        seed (int, optional): The seed used in sampler. Default 0, means no random seed.
        is_sparse(bool, optional): The flag indicating whether to use sparse update, 
            the weight@GRAD and bias@GRAD will be changed to SelectedRows. Default False.

    Returns:
        Variable: The output nce loss.

    Examples:
        .. code-block:: python


            import paddle.fluid as fluid
            import numpy as np

            window_size = 5
            words = []
            for i in xrange(window_size):
                words.append(fluid.data(
                    name='word_{0}'.format(i), shape=[-1, 1], dtype='int64'))

            dict_size = 10000
            label_word = int(window_size / 2) + 1

            embs = []
            for i in xrange(window_size):
                if i == label_word:
                    continue

                emb = fluid.layers.embedding(input=words[i], size=[dict_size, 32],
                                   param_attr='embed', is_sparse=True)
                embs.append(emb)

            embs = fluid.layers.concat(input=embs, axis=1)
            loss = fluid.layers.nce(input=embs, label=words[label_word],
                      num_total_classes=dict_size, param_attr='nce.w_0',
                      bias_attr='nce.b_0')

             #or use custom distribution
             dist = np.array([0.05,0.5,0.1,0.3,0.05])
             loss = fluid.layers.nce(input=embs, label=words[label_word],
                       num_total_classes=5, param_attr='nce.w_1',
                       bias_attr='nce.b_1',
                       num_neg_samples=3,
                       sampler="custom_dist",
                       custom_dist=dist)
    """
    helper = LayerHelper('nce', **locals())

    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in nce layer must be Variable, but received %s"
            % (type(input)))
    if not isinstance(label, Variable):
        raise TypeError(
            "The type of 'label' in nce layer must be Variable, but received %s"
            % (type(label)))
    if convert_dtype(input.dtype) not in ['float32', 'float64']:
        raise TypeError(
            "The data type of 'input' in nce layer must be float32 or float64, but received %s."
            % (convert_dtype(input.dtype)))
    if convert_dtype(label.dtype) not in ['int64']:
        raise TypeError(
            "The data type of 'label' in nce layer must be int64, but received %s."
            % (convert_dtype(label.dtype)))

    dim = input.shape[1]
    num_true_class = label.shape[1]
    w = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_total_classes, dim],
        is_bias=False,
        dtype=input.dtype)
    inputs = {}
    if helper.bias_attr:
        b = helper.create_parameter(
            attr=helper.bias_attr,
            shape=[num_total_classes, 1],
            is_bias=True,
            dtype=input.dtype)
        inputs['Bias'] = b
    cost = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_logits = helper.create_variable_for_type_inference(dtype=input.dtype)
    sample_labels = helper.create_variable_for_type_inference(dtype=label.dtype)

    inputs['Input'] = input
    inputs['Label'] = label
    inputs['Weight'] = w
    inputs['SampleWeight'] = sample_weight if sample_weight is not None else []

    if sampler == "uniform":
        sampler = 0
    elif sampler == "log_uniform":
        sampler = 1
    elif sampler == "custom_dist":
        assert custom_dist is not None
        # assert isinstance(custom_dist, Variable)

        custom_dist_len = num_total_classes
        alias_probs_ = [0] * custom_dist_len
        alias_ = [0] * custom_dist_len
        bigs = []
        littles = []
        for i in range(custom_dist_len):
            normal_prob = custom_dist[i] * custom_dist_len
            if normal_prob - 1.0 > 0:
                bigs.append((i, normal_prob))
            elif 1.0 - normal_prob > 0:
                littles.append((i, normal_prob))
            else:
                alias_probs_[i] = normal_prob
                alias_[i] = -1

        while len(bigs) and len(littles):
            big = bigs.pop(0)
            little = littles.pop(0)

            big_idx = big[0]
            big_prob = big[1]

            alias_probs_[little[0]] = little[1]
            alias_[little[0]] = big_idx
            big_left = big[1] + little[1] - 1
            if big_left - 1.0 > 0:
                bigs.append((big_idx, big_left))
            elif 1.0 - big_left > 0:
                littles.append((big_idx, big_left))
            else:
                alias_probs_[big_idx] = big_left
                alias_[big_idx] = -1

        if len(bigs):
            big = bigs.pop(0)
            alias_probs_[big[0]] = 1.0
            alias_[big[0]] = -1
        if len(littles):
            little = littles.pop(0)
            alias_probs_[little[0]] = 1.0
            alias_[little[0]] = -1

        def _init_by_numpy_array(numpy_array):
            ret = helper.create_parameter(
                attr=ParamAttr(),
                shape=numpy_array.shape,
                dtype=numpy_array.dtype,
                default_initializer=NumpyArrayInitializer(numpy_array))
            ret.stop_gradient = True
            return ret

        inputs['CustomDistProbs'] = _init_by_numpy_array(
            np.array(custom_dist).astype('float32'))
        inputs['CustomDistAlias'] = _init_by_numpy_array(
            np.array(alias_).astype('int32'))
        inputs['CustomDistAliasProbs'] = _init_by_numpy_array(
            np.array(alias_probs_).astype('float32'))
        sampler = 2
    else:
        raise Exception("Unsupported sampler type.")

    if num_neg_samples is None:
        num_neg_samples = 10
    else:
        num_neg_samples = int(num_neg_samples)

    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )

    attrs = {
        'num_total_classes': int(num_total_classes),
        'num_neg_samples': num_neg_samples,
        'seed': seed,
        'sampler': sampler,
        'is_sparse': is_sparse,
        'remote_prefetch': remote_prefetch
    }

    helper.append_op(
        type='nce',
        inputs=inputs,
        outputs={
            'Cost': cost,
            'SampleLogits': sample_logits,
            'SampleLabels': sample_labels
        },
        attrs=attrs)
    return cost / (num_neg_samples + 1)


def hsigmoid(input,
             label,
             num_classes,
             param_attr=None,
             bias_attr=None,
             name=None,
             path_table=None,
             path_code=None,
             is_custom=False,
             is_sparse=False):
    """
    The hierarchical sigmoid organizes the classes into a complete binary tree to reduce the computational complexity
    and speed up the model training, especially the training of language model.
    Each leaf node of the complete binary tree represents a class(word) and each non-leaf node acts as a binary classifier.
    For each class(word), there's a unique path from root to itself, hsigmoid calculate the cost for each non-leaf node on
    the path, and sum them to get a total cost.
    Comparing to softmax, the OP can reduce the computational complexity from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the number of classes or the size of word dict.

    The OP supports default tree and custom tree. For the default tree, you can refer to `Hierarchical Probabilistic Neural
    Network Language Model <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`. For the custom
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
        name (str, optional): Normally there is no need for user to set this property. For more information,
            please refer to :ref:`api_guide_Name`. Default: None.
        path_table (Variable, optional): A tensor that stores each batch of samples' path from leaf to root
            node, its shape is [N, L] and data type is int64, where L is the length of path. For each sample i,
            path_table[i] is a np.array like structure and each element in this array is the indexes in parent
            nodes' weight matrix. Default: None.
        path_code (Variable, optional): A tensor that stores each batch of samples' code of path from leaf
            to root node, its shape is [N, L] and data type is int64, which is the same as :attr:`path_table`.
            Each code of path is consisted with the code of nodes from leaf to root node. Default: None.
        is_custom (bool, optional): Whether use custom binary tree. If it's True, :attr:`path_table`,
            :attr:`path_code` and :attr:`num_classes` should be set, otherwise :attr:`num_classes` should
            be set. Default: False.
        is_sparse (bool, optional): Whether use sparse updating instead of dense updating, if it's True, the
            gradient of W and input will be sparse. Default: False.

    Returns:
        Variable: A tensor with the cost of hierarchical sigmoid, its shape is [N, 1] and data type is the same as :attr:`input`.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.fill_constant(shape=[4, 3], value=0.9, dtype='float32')
            # x = [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]
            y = fluid.layers.fill_constant(
                shape=[4, 1], value=1, dtype='int64')
            # y = [[1], [1], [1], [1]]
            out = fluid.layers.hsigmoid(input=x, label=y, num_classes=2, param_attr=fluid.initializer.Constant(
                value=0.05), bias_attr=fluid.initializer.Constant(value=.0))
            # out = [[0.62792355], [0.62792355], [0.62792355], [0.62792355]]
    """

    helper = LayerHelper('hierarchical_sigmoid', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    pre_out = helper.create_variable_for_type_inference(dtype)
    dim = input.shape[1]
    if ((num_classes is None) or (num_classes < 2)) and (not is_custom):
        raise ValueError(
            "num_classes must not be less than 2 with default tree")

    if (not is_custom) and (is_sparse):
        print("Sparse mode should not be used without custom tree")
        is_sparse = False

    if (not is_custom) and ((path_table is not None) or
                            (path_code is not None)):
        raise ValueError(
            "only num_classes should be passed without custom tree")

    if (is_custom) and (path_code is None):
        raise ValueError("path_code should not be None with custom tree")
    elif (is_custom) and (path_table is None):
        raise ValueError("path_table should not be None with custom tree")
    elif (is_custom) and (num_classes is None):
        raise ValueError("num_classes should not be None with custom tree")
    else:
        pass

    weights = None
    remote_prefetch = is_sparse
    print(
        "With sparse mode, if your models has only small parameter prefetch may cause speed down"
    )
    if not is_custom:
        weights = helper.create_parameter(
            attr=helper.param_attr,
            shape=[num_classes - 1, dim],
            is_bias=False,
            dtype=input.dtype)
    else:
        weights = helper.create_parameter(
            attr=helper.param_attr,
            shape=[num_classes, dim],
            is_bias=False,
            dtype=input.dtype)
    inputs = {
        "X": input,
        "W": weights,
        "PathTable": path_table,
        "PathCode": path_code,
        "Label": label
    }
    if helper.bias_attr:
        if not is_custom:
            bias = helper.create_parameter(
                attr=helper.bias_attr,
                shape=[num_classes - 1, 1],
                is_bias=True,
                dtype=input.dtype)
            inputs['Bias'] = bias
        else:
            bias = helper.create_parameter(
                attr=helper.bias_attr,
                shape=[num_classes, 1],
                is_bias=True,
                dtype=input.dtype)
            inputs['Bias'] = bias
    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs={"Out": out,
                 "PreOut": pre_out,
                 "W_Out": weights},
        attrs={
            "num_classes": num_classes,
            "is_sparse": is_sparse,
            "remote_prefetch": remote_prefetch
        })
    return out


def transpose(x, perm, name=None):
    """
    Permute the data dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
        x (Variable): The input Tensor. It is a N-D Tensor of data types float32, float64, int32.
        perm (list): Permute the input accoring to the data of perm.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: A transposed n-D Tensor, with data type being float32, float64, int32, int64.

    For Example:

        .. code-block:: text

         x = [[[ 1  2  3  4] [ 5  6  7  8] [ 9 10 11 12]]
             [[13 14 15 16] [17 18 19 20] [21 22 23 24]]]
         shape(x) =  [2,3,4]

         # Example 1
         perm0 = [1,0,2]
         y_perm0 = [[[ 1  2  3  4] [13 14 15 16]]
                   [[ 5  6  7  8]  [17 18 19 20]]
                   [[ 9 10 11 12]  [21 22 23 24]]]
         shape(y_perm0) = [3,2,4]

         # Example 2
         perm1 = [2,1,0]
         y_perm1 = [[[ 1 13] [ 5 17] [ 9 21]]
                   [[ 2 14] [ 6 18] [10 22]]
                   [[ 3 15]  [ 7 19]  [11 23]]
                   [[ 4 16]  [ 8 20]  [12 24]]]
         shape(y_perm1) = [4,3,2]

    Examples:

        .. code-block:: python

            # use append_batch_size=False to avoid prepending extra
            # batch size in shape
            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[2, 3, 4],
                            dtype='float32', append_batch_size=False)
            x_transposed = fluid.layers.transpose(x, perm=[1, 0, 2])
            print x_transposed.shape
            #(3L, 2L, 4L)

    """
    if not isinstance(x, Variable):
        raise TypeError(
            "The type of Input(x) in transpose must be Variable, but received %s"
            % (type(x)))
    if convert_dtype(x.dtype) not in [
            "float16", "float32", "float64", "int32", "int64"
    ]:
        raise TypeError(
            "The data type of Input(x) in transpose must be one of [float16, float32, float64, int32, int64], but received %s."
            % (convert_dtype(x.dtype)))
    if not isinstance(perm, list):
        raise TypeError(
            "The type of Input(perm) in transpose must be list, but received %s"
            % (type(perm)))
    if len(perm) != len(x.shape):
        raise ValueError(
            "Input(perm) is the permutation of dimensions of Input(x), "
            "its length should be equal to dimensions of Input(x), "
            "but received dimension of Input(x) is %s, "
            "the length of Input(perm) is %s." % (len(x.shape), len(perm)))
    for idx, dim in enumerate(perm):
        if dim >= len(x.shape):
            raise ValueError(
                "Each element in Input(perm) should be less than Input(x)'s dimension, "
                "but %d-th element in Input(perm) is %d which exceeds Input(x)'s "
                "dimension %d." % (idx, perm[idx], len(x.shape)))

    helper = LayerHelper('transpose', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='transpose2',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'XShape': [x_shape]},
        attrs={'axis': perm})
    return out


def im2sequence(input,
                filter_size=1,
                stride=1,
                padding=0,
                input_image_size=None,
                out_stride=1,
                name=None):
    """
    Extracts image patches from the input tensor to form a tensor of shape
    {input.batch_size * output_height * output_width, filter_size_height *
    filter_size_width * input.channels}. This op use filter to scan images
    and convert these images to sequences. After expanding, the number of time step are
    output_height * output_width for an image, in which output_height and
    output_width are calculated by below equation:

    .. math::

        output\_height  = 1 + \
            (padding\_up + padding\_down + input\_height  - filter\_size\_height  + stride\_height - 1) / stride\_height \\\\
        output\_width  = 1 + \
            (padding\_left + padding\_right + input\_width  - filter\_size\_width  + stride\_width - 1) / stride\_width

    And the dimension of each time step is filter_size_height * filter_size_width * input.channels.

    Parameters:
        input (Variable): The input should be a 4-D Tensor in :math:`NCHW` format. The data type is float32.

        filter_size(int32 | List[int32]): The filter size. If filter_size is a List,
            it must contain two integers, :math:`[filter\_size\_height, filter\_size\_width]` .
            Otherwise, the filter size will be a square :math:`[filter\_size, filter\_size]` . Default is 1.

        stride(int32 | List[int32]): The stride size. If stride is a List, it must
            contain two integers, :math:`[stride\_height, stride\_width]` . Otherwise, the stride size will be a square :math:`[stride\_size, stride\_size]` . Default is 1.

        padding(int32 | List[int32]): The padding size. If padding is a List, it can
            contain four integers like :math:`[padding\_up, padding\_left, padding\_down, padding\_right]` to indicate
            paddings of four direction.  Or it can contain two integers :math:`[padding\_height, padding\_width]` which means
            padding_up = padding_down = padding_height and
            padding_left = padding_right = padding_width. Otherwise, a scalar padding means
            padding_up = padding_down = padding_left = padding_right = padding. 
            Default is 0.

        input_image_size(Variable, optional): the input contains image real size.It's dim
            is :math:`[batchsize, 2]` . It is just for batch inference when not None. Default is None.

        out_stride(int32 | List[int32]): The scaling of image through CNN. It is valid only when input_image_size is not None.
            If out_stride is List,  it must contain two intergers,
            :math:`[out\_stride\_height, out\_stride\_W]` . Otherwise,
            the out_stride_height = out_stride_width = out_stride. Default is 1.

        name (str, optional): The default value is None.  Normally there is no need for
                    user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
    
    Returns: 
            The output is a 2-D LoDTensor with shape {input.batch\_size * output\_height * output\_width, \ 
            filter\_size\_height * filter\_size\_width * input.channels}. The data type is float32.

    Return Type: Variable

    Examples:

        .. code-block:: text

            Given:

            x = [[[[ 6.  2.  1.]
                   [ 8.  3.  5.]
                   [ 0.  2.  6.]]

                  [[ 2.  4.  4.]
                   [ 6.  3.  0.]
                   [ 6.  4.  7.]]]

                 [[[ 6.  7.  1.]
                   [ 5.  7.  9.]
                   [ 2.  4.  8.]]

                  [[ 1.  2.  1.]
                   [ 1.  3.  5.]
                   [ 9.  0.  8.]]]]

            x.dims = {2, 2, 3, 3}

            And:

            filter = [2, 2]
            stride = [1, 1]
            padding = [0, 0]

            Then:

            output.data = [[ 6.  2.  8.  3.  2.  4.  6.  3.]
                           [ 2.  1.  3.  5.  4.  4.  3.  0.]
                           [ 8.  3.  0.  2.  6.  3.  6.  4.]
                           [ 3.  5.  2.  6.  3.  0.  4.  7.]
                           [ 6.  7.  5.  7.  1.  2.  1.  3.]
                           [ 7.  1.  7.  9.  2.  1.  3.  5.]
                           [ 5.  7.  2.  4.  1.  3.  9.  0.]
                           [ 7.  9.  4.  8.  3.  5.  0.  8.]]

            output.dims = {8, 8}

            output.lod = [[4, 4]]

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.data(name='data', shape=[None, 3, 32, 32],
                                     dtype='float32')
            output = fluid.layers.im2sequence(
                input=data, stride=[1, 1], filter_size=[2, 2])


    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")

    if isinstance(filter_size, int):
        filter_size = [filter_size, filter_size]
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding]
    if len(padding) == 2:
        padding.append(padding[0])
        padding.append(padding[1])
    inputs = {"X": input}
    attrs = {"kernels": filter_size, "strides": stride, "paddings": padding}
    if input_image_size:
        if isinstance(out_stride, int):
            out_stride = [out_stride, out_stride]
        inputs["Y"] = input_image_size
        attrs["out_stride"] = out_stride
    helper = LayerHelper('im2sequence', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='im2sequence', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


@templatedoc()
def row_conv(input, future_context_size, param_attr=None, act=None):
    """
    ${comment}

    Args:
        input (${x_type}): ${x_comment}.
        future_context_size (int): Future context size. Please note, the shape
            of convolution kernel is [future_context_size + 1, D].
        param_attr (ParamAttr): Attributes of parameters, including
            name, initializer etc.
        act (str): Non-linear activation to be applied to output variable.

    Returns:
        ${out_comment}.

    Examples:
        >>>  # for LodTensor inputs
        >>> import paddle.fluid as fluid
        >>> x = fluid.data(name='x', shape=[9, 16],
        >>>                        dtype='float32', lod_level=1)
        >>> out = fluid.layers.row_conv(input=x, future_context_size=2)
        >>> # for Tensor inputs
        >>> x = fluid.data(name='x', shape=[9, 4, 16], dtype='float32')
        >>> out = fluid.layers.row_conv(input=x, future_context_size=2)
    """
    helper = LayerHelper('row_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [future_context_size + 1, input.shape[1]]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='row_conv',
        inputs={'X': [input],
                'Filter': [filter_param]},
        outputs={'Out': [out]})
    return helper.append_activation(out)


@templatedoc()
def multiplex(inputs, index):
    """

    Based on the given index parameter, the OP selects a specific row from each input Tensor to construct the output Tensor.

    If the input of this OP contains :math:`m` Tensors, where :math:`I_{i}` means the i-th input Tensor, :math:`i` between :math:`[0,m)` .

    And :math:`O` means the output, where :math:`O[i]` means the i-th row of the output, then the output satisfies that :math:`O[i] = I_{index[i]}[i]` .

    For Example:

            .. code-block:: text

                Given:

                inputs = [[[0,0,3,4], [0,1,3,4], [0,2,4,4], [0,3,3,4]],
                          [[1,0,3,4], [1,1,7,8], [1,2,4,2], [1,3,3,4]],
                          [[2,0,3,4], [2,1,7,8], [2,2,4,2], [2,3,3,4]],
                          [[3,0,3,4], [3,1,7,8], [3,2,4,2], [3,3,3,4]]]

                index = [[3],[0],[1],[2]]

                out = [[3,0,3,4],    # out[0] = inputs[index[0]][0] = inputs[3][0] = [3,0,3,4]
                       [0,1,3,4],    # out[1] = inputs[index[1]][1] = inputs[0][1] = [0,1,3,4]
                       [1,2,4,2],    # out[2] = inputs[index[2]][2] = inputs[1][2] = [1,2,4,2]
                       [2,3,3,4]]    # out[3] = inputs[index[3]][3] = inputs[2][3] = [2,3,3,4]


    Args:
       inputs (list): The input Tensor list. The list elements are N-D Tensors of data types float32, float64, int32, int64. All input Tensor shapes should be the same and rank must be at least 2.
       index (Variable): Used to select some rows in the input Tensor to construct an index of the output Tensor. It is a 2-D Tensor with data type int32 or int64 and shape [M, 1], where M is the number of input Tensors.

    Returns:
        Variable(Tensor): Output of multiplex OP, with data type being float32, float64, int32, int64.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            x1 = fluid.data(name='x1', shape=[None, 2], dtype='float32')
            x2 = fluid.data(name='x2', shape=[None, 2], dtype='float32')
            index = fluid.data(name='index', shape=[None, 1], dtype='int32')
            out = fluid.layers.multiplex(inputs=[x1, x2], index=index)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
            img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
            index = np.array([[1], [0]]).astype(np.int32)

            res = exe.run(fluid.default_main_program(), feed={'x1':img1, 'x2':img2, 'index':index}, fetch_list=[out])
            print(res) # [array([[5., 6.], [3., 4.]], dtype=float32)]

    """
    helper = LayerHelper('multiplex', **locals())

    if not isinstance(inputs, list) and len(inputs) < 2:
        raise ValueError("inputs should be a list object and contains at least "
                         "2 elements.")

    out = helper.create_variable_for_type_inference(inputs[0].dtype)
    helper.append_op(
        type='multiplex',
        inputs={'X': inputs,
                'Ids': index},
        outputs={'Out': [out]})
    return out


def softmax_with_cross_entropy(logits,
                               label,
                               soft_label=False,
                               ignore_index=kIgnoreIndex,
                               numeric_stable_mode=True,
                               return_softmax=False,
                               axis=-1):
    """
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

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::

        max_j &= \\max_{i=0}^{K}{\\text{logits}_i}

        log\\_max\\_sum_j &= \\log\\sum_{i=0}^{K}\\exp(logits_i - max_j)

        softmax_j &= \\exp(logits_j - max_j - {log\\_max\\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Variable): A multi-dimension ``Tensor`` , and the data type is float32 or float64. The input tensor of unscaled log probabilities.
        label (Variable): The ground truth  ``Tensor`` , data type is the same
            as the ``logits`` . If :attr:`soft_label` is set to :attr:`True`, 
            Label is a ``Tensor``  in the same shape with :attr:`logits`. 
            If :attr:`soft_label` is set to :attr:`True`, Label is a ``Tensor`` 
            in the same shape with :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool, optional): A flag to indicate whether to interpretate the given
            labels as soft labels. Default False.
        ignore_index (int, optional): Specifies a target value that is ignored and does
                                      not contribute to the input gradient. Only valid
                                      if :attr:`soft_label` is set to :attr:`False`. 
                                      Default: kIgnoreIndex(-100).
        numeric_stable_mode (bool, optional): A flag to indicate whether to use a more
                                              numerically stable algorithm. Only valid
                                              when :attr:`soft_label` is :attr:`False` 
                                              and GPU is used. When :attr:`soft_label` 
                                              is :attr:`True` or CPU is used, the 
                                              algorithm is always numerically stable.
                                              Note that the speed may be slower when use
                                              stable algorithm. Default: True.
        return_softmax (bool, optional): A flag indicating whether to return the softmax
                                         along with the cross entropy loss. Default: False.
        axis (int, optional): The index of dimension to perform softmax calculations. It 
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.

    Returns:
        ``Variable`` or Tuple of two ``Variable`` : Return the cross entropy loss if \
                                                    `return_softmax` is False, otherwise the tuple \
                                                    (loss, softmax), softmax is in the same shape \
                                                    with input logits and cross entropy loss is in \
                                                    the same shape with input logits except shape \
                                                    in dimension :attr:`axis` as 1.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            data = fluid.data(name='data', shape=[-1, 128], dtype='float32')
            label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
            fc = fluid.layers.fc(input=data, size=100)
            out = fluid.layers.softmax_with_cross_entropy(
                logits=fc, label=label)
    """
    helper = LayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': logits,
                'Label': label},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs={
            'soft_label': soft_label,
            'ignore_index': ignore_index,
            'numeric_stable_mode': numeric_stable_mode,
            'axis': axis
        })

    if return_softmax:
        return loss, softmax

    return loss


def sampled_softmax_with_cross_entropy(logits,
                                       label,
                                       num_samples,
                                       num_true=1,
                                       remove_accidental_hits=True,
                                       use_customized_samples=False,
                                       customized_samples=None,
                                       customized_probabilities=None,
                                       seed=0):
    """
    **Sampled Softmax With Cross Entropy Operator.**

    Cross entropy loss with sampled softmax is used as the output layer for 
    larger output classes extensively. This operator samples a number of samples
    for all examples, and computes the softmax normalized values for each 
    row of the sampled tensor, after which cross-entropy loss is computed. 

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.
    
    For examples with T true labels (T >= 1), we assume that each true label has
    a probability of 1/T. For each sample, S samples are generated using a
    log uniform distribution. True labels are concatenated with these samples to
    form T + S samples for each example. So, assume the shape of logits is
    [N x K], the shape for samples is [N x (T+S)]. For each sampled label, a 
    probability is calculated, which corresponds to the Q(y|x) in 
    [Jean et al., 2014](http://arxiv.org/abs/1412.2007).
    
    Logits are sampled according to the sampled labels. Then if 
    remove_accidental_hits is True, if a sample[i, j] accidentally hits true 
    labels, then the corresponding sampled_logits[i, j] is minus by 1e20 to 
    make its softmax result close to zero. Then sampled logits are subtracted by
    logQ(y|x), these sampled logits and re-indexed labels are used to compute 
    a softmax with cross entropy.

    Args:
        logits (Variable): The unscaled log probabilities, which is a 2-D tensor
            with shape [N x K]. N is the batch_size, and K is the class number.
        label (Variable): The ground truth which is a 2-D tensor. Label is a 
            Tensor<int64> with shape [N x T], where T is the number of true 
            labels per example. 
        num_samples (int): The number for each example, num_samples should be 
            less than the number of class.
        num_true(int): The number of target classes per training example.
        remove_accidental_hits (bool): A flag indicating whether to remove 
            accidental hits when sampling. If True and if a sample[i, j] 
            accidentally hits true labels, then the corresponding 
            sampled_logits[i, j] is minus by 1e20 to make its softmax result 
            close to zero. Default is True.
        use_customized_samples (bool): Whether to use custom samples and probabities to sample
            logits.
        customized_samples (Variable): User defined samples, which is a 2-D tensor
            with shape [N, T + S]. S is the num_samples, and T is the number of true 
            labels per example. 
        customized_probabilities (Variable): User defined probabilities of samples, 
            a 2-D tensor which has the same shape with customized_samples.
        seed (int): The random seed for generating random number, which is used
            in the process of sampling. Default is 0.

    Returns:
        Variable: Return the cross entropy loss which is a 2-D tensor with shape
                  [N x 1].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.layers.data(name='data', shape=[256], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            fc = fluid.layers.fc(input=input, size=100)
            out = fluid.layers.sampled_softmax_with_cross_entropy(
                      logits=fc, label=label, num_samples=25)
    """
    helper = LayerHelper('sample_logits', **locals())
    samples = helper.create_variable_for_type_inference(dtype='int64')
    probabilities = helper.create_variable_for_type_inference(
        dtype=logits.dtype)
    sampled_logits \
        = helper.create_variable_for_type_inference(dtype=logits.dtype)
    sampled_label = helper.create_variable_for_type_inference(dtype='int64')
    sampled_softlabel = helper.create_variable_for_type_inference(
        dtype=logits.dtype)
    logits_dim = helper.create_variable_for_type_inference(dtype=logits.dtype)
    labels_dim = helper.create_variable_for_type_inference(dtype=label.type)

    helper.append_op(
        type='sample_logits',
        inputs={
            'Logits': logits,
            'Labels': label,
            'CustomizedSamples': customized_samples,
            'CustomizedProbabilities': customized_probabilities
        },
        outputs={
            'Samples': samples,
            'Probabilities': probabilities,
            'SampledLabels': sampled_label,
            'SampledLogits': sampled_logits,
            'LogitsDim': logits_dim,
            'LabelsDim': labels_dim
        },
        attrs={
            'use_customized_samples': use_customized_samples,
            'uniq': True,
            'remove_accidental_hits': remove_accidental_hits,
            'num_samples': num_samples,
            'seed': seed
        })
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    helper.append_op(
        type='one_hot',
        inputs={'X': sampled_label},
        attrs={'depth': num_samples + 1},
        outputs={'Out': sampled_softlabel})

    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': sampled_logits,
                'Label': sampled_softlabel},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs={
            'soft_label': True,
            'ignore_index': False,
            'numeric_stable_mode': False
        })
    return loss / num_true


def smooth_l1(x, y, inside_weight=None, outside_weight=None, sigma=None):
    """
    This layer computes the smooth L1 loss for Variable :attr:`x` and :attr:`y`.
    It takes the first dimension of :attr:`x` and :attr:`y` as batch size.
    For each instance, it computes the smooth L1 loss element by element first
    and then sums all the losses. So the shape of ouput Variable is
    [batch_size, 1].

    Args:
        x (Variable): A tensor with rank at least 2. The input value of smooth
            L1 loss op with shape [batch_size, dim1, ..., dimN].
            A LoDTensor or Tensor with type float32.
        y (Variable): A tensor with rank at least 2. The target value of smooth
            L1 loss op with same shape as :attr:`x`.
            A LoDTensor or Tensor with type float32.
        inside_weight (Variable|None):  A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the result of (:attr:`x` - :attr:`y`) will be multiplied
            by this tensor element by element.
            A Tensor with type float32.
        outside_weight (Variable|None): A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the out smooth L1 loss will be multiplied by this tensor
            element by element.
            A Tensor with type float32.
        sigma (float|None): Hyper parameter of smooth L1 loss layer. A float
           scalar with default value 1.0.

    Returns:
        Variable: The output smooth L1 loss with shape [batch_size, 1].  A Tensor with type float32.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            data = fluid.data(name="x", shape=[-1, 3], dtype="float32")
            label = fluid.data(name="y", shape=[-1, 3], dtype="float32")
            result = fluid.layers.smooth_l1(data,label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            x = np.random.rand(3,3).astype("float32")
            y = np.random.rand(3,3).astype("float32")
            output= exe.run(feed={"x":x, "y":y},
                             fetch_list=[result])
            print(output)
        
            #[array([[0.08220536],
            #       [0.36652038],
            #      [0.20541131]], dtype=float32)]

    """

    helper = LayerHelper('smooth_l1_loss', **locals())
    diff = helper.create_variable_for_type_inference(dtype=x.dtype)
    loss = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='smooth_l1_loss',
        inputs={
            'X': x,
            'Y': y,
            'InsideWeight': inside_weight,
            'OutsideWeight': outside_weight
        },
        outputs={'Diff': diff,
                 'Out': loss},
        attrs={'sigma': sigma if sigma is not None else 1.0})
    return loss


def one_hot(input, depth, allow_out_of_range=False):
    """

    **WARING:** This OP requires the last dimension of Tensor shape must be equal to 1.
    This OP will be deprecated in a future release. It is recommended to use fluid. :ref:`api_fluid_one_hot` .

    The operator converts each id in the input to an one-hot vector with a
    :attr:`depth` length. The value in the vector dimension corresponding to the id
    is 1, and the value in the remaining dimension is 0.

    The shape of output Tensor or LoDTensor is generated by adding :attr:`depth` dimension
    behind the last dimension of the input shape.

    .. code-block:: text

        Example 1 (allow_out_of_range=False):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [3], [0]]
            depth = 4

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 0., 1.],
                        [1., 0., 0., 0.]]

        Example 2 (allow_out_of_range=True):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [5], [0]]
            depth = 4
            allow_out_of_range = True

        output:
            Out.shape = [4, 4]
            Out.data = [[0., 1., 0., 0.],
                        [0., 1., 0., 0.], 
                        [0., 0., 0., 0.], # This id is 5, which goes beyond depth, so set it all-zeros data.
                        [1., 0., 0., 0.]]

        Example 3 (allow_out_of_range=False):

        input:
            X.shape = [4, 1]
            X.data = [[1], [1], [5], [0]]
            depth = 4
            allow_out_of_range = False

        output: Throw an exception for Illegal value
            The second dimension in X is 5, which is greater than depth.  
            Allow_out_of_range =False means that does not allow the word id to exceed depth,
            so it throws an exception.

    Args:
        input(Variable): Tensor or LoDTensor with shape :math:`[N_1, N_2, ..., N_k, 1]` ,
            which contains at least one dimension and the last dimension must be 1.
            The data type is int32 or int64.
        depth(scalar): An integer defining the :attr:`depth` of the one hot dimension. If input 
            is word id, depth is generally the dictionary size.
        allow_out_of_range(bool): A bool value indicating whether the input
            indices could be out of range :math:`[0, depth)` . When input indices are
            out of range, exceptions :code:`Illegal value` is raised if :attr:`allow_out_of_range`
            is False, or zero-filling representations is created if it is set True.
            Default: False.

    Returns:
        Variable: The one-hot representations of input. A Tensor or LoDTensor with type float32.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            # Correspond to the first example above, where label.shape is [4, 1] and one_hot_label.shape is [4, 4].
            label = fluid.data(name="label", shape=[4, 1], dtype="int64")
            one_hot_label = fluid.layers.one_hot(input=label, depth=4)
    """
    helper = LayerHelper("one_hot", **locals())

    one_hot_out = helper.create_variable_for_type_inference(dtype='float32')

    if in_dygraph_mode():
        inputs = {'X': input}
        attrs = {'depth': depth}
    else:
        if not isinstance(depth, Variable):
            # user attribute
            inputs = {'X': input}
            attrs = {'depth': depth}
        else:
            depth.stop_gradient = True
            inputs = {'X': input, 'depth_tensor': depth}
            attrs = {}
    helper.append_op(
        type="one_hot",
        inputs=inputs,
        attrs=attrs,
        outputs={'Out': one_hot_out})
    one_hot_out.stop_gradient = True
    return one_hot_out


def autoincreased_step_counter(counter_name=None, begin=1, step=1):
    """
    Create an auto-increase variable. which will be automatically increased 
    by 1 in every iteration. By default, the first return of this counter is 1, 
    and the step size is 1.

    Args:
        counter_name(str, optional): The counter name. Default '@STEP_COUNTER@'.
        begin(int, optional): The first return value of this counter. Default 1.
        step(int, optional): The step size. Default 1.

    Returns:
        Variable: The auto-increased Variable with data type int64.

    Examples:
        .. code-block:: python

           import paddle.fluid as fluid
           global_step = fluid.layers.autoincreased_step_counter(
               counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)
    """
    helper = LayerHelper('global_step_counter')
    if counter_name is None:
        counter_name = '@STEP_COUNTER@'
    counter, is_new_var = helper.create_or_get_global_variable(
        name=counter_name,
        dtype='int64',
        shape=[1],
        persistable=True,
        belong_to_optimizer=True)
    if is_new_var:
        helper.set_variable_initializer(
            counter, initializer=Constant(
                value=begin - 1, force_cpu=True))
        helper.main_program.global_block()._prepend_op(
            type='increment',
            inputs={'X': [counter]},
            outputs={'Out': [counter]},
            attrs={'step': float(step)})
        counter.stop_gradient = True

    return counter


def reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None):
    """
    This operator changes the shape of ``x`` without changing its data.

    The target shape can be given by ``shape`` or ``actual_shape``.
    When ``shape`` and ``actual_shape`` are set at the same time,
    ``actual_shape`` has a higher priority than ``shape``
    but at this time ``shape`` can only be an integer list or tuple, and ``shape`` still should be set correctly to
    gurantee shape inference in compile-time.

    Some tricks exist when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The indice of 0s in shape can not exceed
    the dimension of x.

    Here are some examples to explain it.

    1. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [6, 8], the reshape operator will transform x into a 2-D tensor with
    shape [6, 8] and leaving x's data unchanged.

    2. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    specified is [2, 3, -1, 2], the reshape operator will transform x into a
    4-D tensor with shape [2, 3, 4, 2] and leaving x's data unchanged. In this
    case, one dimension of the target shape is set to -1, the value of this
    dimension is inferred from the total element number of x and remaining
    dimensions.

    3. Given a 3-D tensor x with a shape [2, 4, 6], and the target shape
    is [-1, 0, 3, 2], the reshape operator will transform x into a 4-D tensor
    with shape [2, 4, 3, 2] and leaving x's data unchanged. In this case,
    besides -1, 0 means the actual dimension value is going to be copied from
    the corresponding dimension of x.

    **Note**:
        The parameter ``actual_shape`` will be deprecated in the future and only use ``shape`` instead to represent the target shape.

    Args:
        x(Variable): A ``Tensor`` or ``LoDTensor`` . The data type is ``float32``, ``float64``, ``int32`` or ``int64``.
        shape(list|tuple|Variable): Define the target shape. At most one dimension of the target shape can be -1.
                        The data type is ``int32`` . If ``shape`` is a list or tuple, the elements of it should be integers or Tensors with shape [1].
                        If ``shape`` is an Variable, it should be an 1-D Tensor .
        actual_shape(variable, optional): An 1-D ``Tensor`` or ``LoDTensor`` . The data type is ``int32`` . If provided, reshape
                                according to this given shape rather than ``shape`` specifying shape.
                                That is to say ``actual_shape`` has a higher priority
                                than ``shape(list|tuple)`` but not ``shape(Variable)``. \
                                This argument ``actual_shape`` will be removed in a future version. \
                                Instructions for updating: ``actual_shape`` will be removed in future versions and replaced by ``shape``.
        act (str, optional): The non-linear activation to be applied to the reshaped input. Default None.
        inplace(bool, optional): If ``inplace`` is True, the input and output of ``layers.reshape``
                       are the same variable. Otherwise, the input and output of
                       ``layers.reshape`` are different variable. Default False. Note that if ``x``
                       is more than one OPs' input, ``inplace`` must be False.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.
                            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``x``. It is a new tensor variable if ``inplace`` is ``False``, otherwise it is ``x``. If ``act`` is None, return the reshaped tensor variable, otherwise return the activated tensor variable.

    Raises:
        TypeError: If actual_shape is neither Variable nor None.
        ValueError: If more than one elements of ``shape`` is -1.
        ValueError: If the element of ``shape`` is 0, the corresponding dimension should be less than or equal to the dimension of ``x``.
        ValueError: If the elements in ``shape`` is negative except -1.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            data_1 = fluid.data(
              name='data_1', shape=[2, 4, 6], dtype='float32')
            reshaped_1 = fluid.layers.reshape(
              x=data_1, shape=[-1, 0, 3, 2], inplace=True)
            # the shape of reshaped_1 is [2,4,3,2].

            # example 2:
            # attr shape is a list which contains tensor Variable.
            data_2 = fluid.layers.fill_constant([2,25], "int32", 3)
            dim = fluid.layers.fill_constant([1], "int32", 5)
            reshaped_2 = fluid.layers.reshape(data_2, shape=[dim, 10])
            # the shape of reshaped_2 is [5,10].
    """
    if not isinstance(x, Variable):
        raise TypeError(
            "The type of 'x' in reshape must be Variable, but received %s." %
            (type(x)))

    if convert_dtype(x.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'x' in reshape only support float16 in GPU now.")

    if convert_dtype(x.dtype) not in [
            'float16', 'float32', 'float64', 'int32', 'int64'
    ]:
        raise TypeError(
            "The data type of 'x' in reshape must be float16, float32, float64, int32 or int64, "
            "but received %s." % (convert_dtype(x.dtype)))

    if not isinstance(shape, (list, tuple, Variable)):
        raise TypeError(
            "The type of 'shape' in reshape must be Variable, list or tuple, but "
            "received %s." % (type(shape)))

    if not isinstance(actual_shape, Variable) and (actual_shape is not None):
        raise TypeError(
            "The type of 'actual_shape' in reshape must be Variable "
            "or None, but received %s." % (type(actual_shape)))

    helper = LayerHelper("reshape2", **locals())
    inputs = {"X": x}
    attrs = {}

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_shape_tensor(list_shape):
        new_shape_tensor = []
        for dim in list_shape:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_shape_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    def get_attr_shape(list_shape):
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, Variable):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one dimension value of 'shape' in reshape can "
                        "be -1. But received shape[%d] is also -1." % dim_idx)
                    unk_dim_idx = dim_idx
                elif dim_size == 0:
                    assert dim_idx < len(x.shape), (
                        "The index of 0 in `shape` must be less than "
                        "the input tensor X's dimensions. "
                        "But received shape[%d] = 0, X's dimensions = %d." %
                        (dim_idx, len(x.shape)))
                else:
                    assert dim_size > 0, (
                        "Each dimension value of 'shape' in reshape must not "
                        "be negtive except one unknown dimension. "
                        "But received shape[%d] = %s." %
                        (dim_idx, str(dim_size)))
        return attrs_shape

    if in_dygraph_mode():
        inputs = {'X': x}
        attrs = {'shape': shape}
    else:
        if isinstance(shape, Variable):
            shape.stop_gradient = True
            inputs["Shape"] = shape
        elif isinstance(shape, (list, tuple)):
            assert len(shape) > 0, (
                "The size of 'shape' in reshape can't be zero, "
                "but received %s." % len(shape))
            attrs["shape"] = get_attr_shape(shape)
            if contain_var(shape):
                inputs['ShapeTensor'] = get_new_shape_tensor(shape)
            elif isinstance(actual_shape, Variable):
                actual_shape.stop_gradient = True
                inputs["Shape"] = actual_shape

    out = x if inplace else helper.create_variable_for_type_inference(
        dtype=x.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="reshape2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": out,
                 "XShape": x_shape})

    return helper.append_activation(out)


def squeeze(input, axes, name=None):
    """
    This OP will squeeze single-dimensional entries of input tensor's shape. If axes is provided, will
    remove the dims by axes, the dims selected by axes should be one. If not provide axes, all dims equal
    to one will be deleted.


    .. code-block:: text 

        Case1:

          Input:
            X.shape = (1, 3, 1, 5)
            axes = [0]
          Output:
            Out.shape = (3, 1, 5)

        Case2:

          Input:
            X.shape = (1, 3, 1, 5)
            axes = []
          Output:
            Out.shape = (3, 5)

        Case3:

          Input:
            X.shape = [1,3,1,5]
            axes = [-2]
          Output:
            Out.shape = [1,3,5]

    Args:
        input (Variable): The input Tensor. Support data type: float32, float64, int8, int32, int64.
                          axes (list): One integer or List of integers, indicating the dimensions to be squeezed.
                          Axes range is :math:`[-rank(input), rank(input))`.
                          If axes is negative, :math:`axes=axes+rank(input)`.
        name (str, optional): Please refer to :ref:`api_guide_Name`, Default None.

    Returns:
        Variable: Output squeezed Tensor. Data type is same as input Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            # set batch size=None
            x = fluid.data(name='x', shape=[None, 5, 1, 10])
            y = layers.squeeze(input=x, axes=[2]) # y.shape=[None, 5, 10]

    """
    assert not in_dygraph_mode(), (
        "squeeze layer is not supported in dygraph mode yet.")
    helper = LayerHelper("squeeze", **locals())

    if not isinstance(input, Variable):
        raise TypeError(
            "The type of 'input' in squeeze must be Variable, but received %s" %
            (type(input)))

    if convert_dtype(input.dtype
                     ) not in ['float32', 'float64', 'int8', 'int32', 'int64']:
        raise TypeError(
            "The data type of 'input' in squeeze must be float32, float64, int8, int32,"
            "int64, but received %s." % (convert_dtype(input.dtype)))

    if not isinstance(axes, list):
        raise TypeError(
            "The type of 'axes' in squeeze must be list, but received %s" %
            (type(axes)))

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="squeeze2",
        inputs={"X": input},
        attrs={"axes": axes},
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def unsqueeze(input, axes, name=None):
    """
    Insert single-dimensional entries to the shape of a Tensor. Takes one
    required argument axes, a list of dimensions that will be inserted.
    Dimension indices in axes are as seen in the output tensor.

    For example:

    .. code-block:: text

      Given a tensor such that tensor with shape [3, 4, 5],
      then Unsqueezed tensor with axes=[0, 4] has shape [1, 3, 4, 5, 1].

    Args:
        input (Variable): The input Tensor to be unsqueezed. It is a N-D Tensor of data types float32, float64, int32.
        axes (list): List of integers, indicating the dimensions to be inserted.
        name (str|None): Name for this layer.

    Returns:
        Variable: Output unsqueezed Tensor, with data type being float32, float64, int32, int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[5, 10])
            y = fluid.layers.unsqueeze(input=x, axes=[1])

    """
    helper = LayerHelper("unsqueeze", **locals())
    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    x_shape = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type="unsqueeze2",
        inputs={"X": input},
        attrs={"axes": axes},
        outputs={"Out": out,
                 "XShape": x_shape})

    return out


def lod_reset(x, y=None, target_lod=None):
    """
    Set LoD of :attr:`x` to a new one specified by :attr:`y` or
    :attr:`target_lod`. When :attr:`y` provided, :attr:`y.lod` would be
    considered as target LoD first, otherwise :attr:`y.data` would be
    considered as target LoD. If :attr:`y` is not provided, target LoD should
    be specified by :attr:`target_lod`. If target LoD is specified by
    :attr:`y.data` or :attr:`target_lod`, only one level LoD is supported.

    .. code-block:: text

        * Example 1:

            Given a 1-level LoDTensor x:
                x.lod =  [[ 2,           3,                   1 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            target_lod: [4, 2]

            then we get a 1-level LoDTensor:
                out.lod =  [[4,                          2]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 2:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a Tensor:
                y.data = [[2, 4]]
                y.dims = [1, 3]

            target_lod:
                This parameter does not work when y is not none.

            then we get a 1-level LoDTensor:
                out.lod =  [[2,            4]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

        * Example 3:

            Given a 1-level LoDTensor x:
                x.lod =  [[2,            3,                   1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            y is a 2-level LoDTensor:
                y.lod =  [[2, 2], [2, 2, 1, 1]]
                y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
                y.dims = [6, 1]

            target_lod:
                This parameter does not work when y is not none.

            then we get a 2-level LoDTensor:
                out.lod =  [[2, 2], [2, 2, 1, 1]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a Tensor or LoDTensor.
        y (Variable|optional): If provided, output's LoD would be derived
                           from :attr:`y`.
        target_lod (list|tuple|optional): One level LoD which should be considered
                                      as target LoD when :attr:`y` not provided.

    Returns:
        Variable: Output variable with LoD specified by this layer.

    Raises:
        ValueError: If :attr:`y` and :attr:`target_lod` are both None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # Graph Organizing
            x = fluid.data(name='x', shape=[6])
            y = fluid.data(name='y', shape=[6], lod_level=1)
            output = fluid.layers.lod_reset(x=x, y=y)

            # Create an executor using CPU as an example
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            # Execute
            x_tensor = fluid.core.LoDTensor()
            x_tensor.set(numpy.ones([6]).astype(numpy.float32), place)
            y_ndarray = numpy.ones([6]).astype(numpy.float32)
            y_lod = [[2, 2], [2, 2, 1, 1]]
            y_tensor = fluid.create_lod_tensor(y_ndarray, y_lod, place)

            res, = exe.run(fluid.default_main_program(),
                           feed={'x':x_tensor, 'y':y_tensor},
                           fetch_list=[output],
                           return_numpy=False)
            print(res)
            # Output Value:
            # lod: [[0, 2, 4], [0, 2, 4, 5, 6]]
            # dim: 6
            # layout: NCHW
            # dtype: float
            # data: [1 1 1 1 1 1]
    """
    helper = LayerHelper("lod_reset", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    if y is not None:
        helper.append_op(
            type="lod_reset", inputs={'X': x,
                                      'Y': y}, outputs={'Out': out})
    elif target_lod is not None:
        helper.append_op(
            type="lod_reset",
            inputs={'X': x},
            attrs={'target_lod': target_lod},
            outputs={'Out': out})
    else:
        raise ValueError("y and target_lod should not be both none.")
    return out


def lod_append(x, level):
    """
    Append level to LoD of :attr:`x`.

    .. code-block:: text

        * Example 1:

            given a 1-level LoDTensor x:
                x.lod =  [[ 2,           3,                   1 ]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

            level: [1, 1, 1, 1, 1, 1, 1]

            then we get a 2-level LoDTensor:
                x.lod =  [[ 2, 3, 1 ], [1, 1, 1, 1, 1, 1]]
                x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                x.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a tensor or LoDTensor.
        level (list|tuple|Variable): The LoD level to be appended into LoD of x.

    Returns:
        Variable: Output variable with new LoD level.

    Raises:
        ValueError: If :attr:`y` is None or and :attr:`level` is not Iterator.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[6, 10], lod_level=1)
            out = fluid.layers.lod_append(x, [1,1,1,1,1,1])
    """
    from collections import Iterable
    if x is None:
        raise ValueError("Input(x) can't be None.")
    if (not isinstance(level, Iterable)) and (not isinstance(level, Variable)):
        raise ValueError("Input(level) must be list, tuple or Variable.")

    helper = LayerHelper("lod_append", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {'X': x}
    attrs = {'append': True}

    if isinstance(level, Variable):
        inputs['Y'] = level
    else:
        attrs['target_lod'] = level
    helper.append_op(
        type="lod_reset", inputs=inputs, attrs=attrs, outputs={'Out': out})
    return out


def lrn(input, n=5, k=1.0, alpha=1e-4, beta=0.75, name=None):
    """
    This operator implements the Local Response Normalization Layer.
    This layer performs a type of "lateral inhibition" by normalizing over local input regions.
    For more information, please refer to `ImageNet Classification with Deep Convolutional Neural Networks <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    The formula is as follows:

    .. math::

        Output(i, x, y) = Input(i, x, y) / \\left(k + \\alpha \\sum\\limits^{\\min(C-1, i + n/2)}_{j = \\max(0, i - n/2)}(Input(j, x, y))^2\\right)^{\\beta}

    In the above equation:

    - :math:`n` : The number of channels to sum over.
    - :math:`k` : The offset (avoid being divided by 0).
    - :math:`\\alpha` : The scaling parameter.
    - :math:`\\beta` : The exponent parameter.


    Args:
        input (Variable): Input feature, 4D-Tensor with the shape of [N,C,H,W], where N is the batch size, C is the input channel, H is Height, W is weight. The data type is float32. The rank of this tensor must be 4, otherwise it will raise ValueError.
        n (int, optional): The number of channels to sum over. Default: 5
        k (float, optional): An offset, positive. Default: 1.0
        alpha (float, optional): The scaling parameter, positive. Default:1e-4
        beta (float, optional): The exponent, positive. Default:0.75
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` 

    Returns:
        Variable: A tensor variable storing the transformation result with the same shape and data type as input.


    Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        data = fluid.data(
            name="data", shape=[None, 3, 112, 112], dtype="float32")
        lrn = fluid.layers.lrn(input=data)
        print(lrn.shape)  # [-1, 3, 112, 112]
        print(lrn.dtype)  # float32
    """
    helper = LayerHelper('lrn', **locals())
    dtype = helper.input_dtype()
    input_shape = input.shape
    dims = len(input_shape)

    if dims != 4:
        raise ValueError(
            "dims of input must be 4(not %d), and it's order must be NCHW" %
            (dims))

    mid_out = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    lrn_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="lrn",
        inputs={"X": input},
        outputs={
            "Out": lrn_out,
            "MidOut": mid_out,
        },
        attrs={"n": n,
               "k": k,
               "alpha": alpha,
               "beta": beta})

    return lrn_out


def pad(x, paddings, pad_value=0., name=None):
    """
    This op will pad a tensor with a constant value given by :attr:`pad_value`, and the
    padded shape is specified by :attr:`paddings`.

    Specifically, the number of values padded before the elements of :attr:`x`
    in dimension :attr:`i` is indicated by :attr:`paddings[2*i]`, and the number
    of values padded after the elements of :attr:`x` in dimension :attr:`i` is
    indicated by :attr:`paddings[2*i+1]`.

    See below for an example.

    .. code-block:: text

        Given:
            x = [[1, 2], [3, 4]]

            paddings = [0, 1, 1, 2]

            pad_value = 0

        Return:

            out = [[0, 1, 2, 0, 0]
                   [0, 3, 4, 0, 0]
                   [0, 0, 0, 0, 0]]

    Args:
        x (Variable): Tensor, data type is float32.
        paddings (list): A list of integers. Its elements specify the padded
                         width before and after each dimension in turn.
                         The length of :attr:`paddings` must be equal to 
                         :math:`rank(x) \\times 2`.
        pad_value (float): The constant value used to pad.
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        The padded tensor, with the same data type and rank as :attr:`x`

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            # x is a rank 2 tensor variable with shape [100, 224].
            # out will be a tensor of shape [101, 227] 
            import paddle.fluid as fluid
            x = fluid.data(name='data', shape=[100, 224], dtype='float32')
            out = fluid.layers.pad(
                x=x, paddings=[0, 1, 1, 2], pad_value=0.)
    """
    helper = LayerHelper('pad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='pad',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'paddings': paddings,
               'pad_value': float(pad_value)})
    return out


def pad_constant_like(x, y, pad_value=0., name=None):
    """
    Pad :attr:`y` with :attr:`pad_value`, the number of values padded to
    the edges of each axis is specified by the difference of the shape
    of :attr:`x` and :attr:`y` . ((0, shape_x_0 - shape_y_0), ... (0, shape_x_n - shape_y_n))
    specify padding widths for each axis. The input should be a k-D tensor(k > 0 and k < 7).

    See below for an example.

    .. code-block:: text

        Given:
            X = [[[[ 0,  1,  2],
                   [ 3,  4,  5]],
                  [[ 6,  7,  8],
                   [ 9, 10, 11]],
                  [[12, 13, 14],
                   [15, 16, 17]]],
                 [[[18, 19, 20],
                   [21, 22, 23]],
                  [[24, 25, 26],
                   [27, 28, 29]],
                  [[30, 31, 32],
                   [33, 34, 35]]]]
            X.shape = (2, 3, 2, 3)

            Y = [[[[35, 36, 37]],
                  [[38, 39, 40]],
                  [[41, 42, 43]]]]
            Y.shape = (1, 3, 1, 3)
		And
            pad_value = -1,

        Return:
            Out = [[[[35, 36, 37],
                     [-1, -1, -1]],
                    [[38, 39, 40],
                     [-1, -1, -1]],
                    [[41, 42, 43],
                     [-1, -1, -1]]],
                  [[[-1, -1, -1],
                    [-1, -1, -1]],
                   [[-1, -1, -1],
                    [-1, -1, -1]],
                   [[-1, -1, -1],
                    [-1, -1, -1]]]]
            Out.shape = (2, 3, 2, 3)

    Args:
        x (Variable): Tensor, its shape spicifies the shape of output.
        y (Variable): Tensor, its rank is the same with :attr:`x`, and for each dimension :math:`i` , 
                      :math:`y\_shape[i] <= x\_shape[i]` . The data type can be float32 or float64.
        pad_value (float): The constant value used to pad.
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        The padded tensor, with the same shape as :attr:`x` and the same data type as :attr:`y`

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            # x is a rank 4 tensor variable, x.shape = (2, 3, 2, 3)
            # y is a rank 4 tensor variable, y.shape = (1, 3, 1, 3)
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[2,3,2,3], dtype='float32')
            y = fluid.data(name='y', shape=[1,3,1,3], dtype='float32')
            out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
            # out is a rank 4 tensor variable, and out.shape = [2, 3 ,2 , 3]
    """
    helper = LayerHelper('pad_constant_like', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='pad_constant_like',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'pad_value': float(pad_value)})
    return out


def label_smooth(label,
                 prior_dist=None,
                 epsilon=0.1,
                 dtype="float32",
                 name=None):
    """
    Label smoothing is a mechanism to regularize the classifier layer and is called 
    label-smoothing regularization (LSR). 

    Label smoothing is proposed to encourage the model to be less confident,
    since optimizing the log-likelihood of the correct label directly may
    cause overfitting and reduce the ability of the model to adapt. Label
    smoothing replaces the ground-truth label :math:`y` with the weighted sum
    of itself and some fixed distribution :math:`\mu`. For class :math:`k`,
    i.e.

    .. math::

        \\tilde{y_k} = (1 - \epsilon) * y_k + \epsilon * \mu_k,

    where :math:`1 - \epsilon` and :math:`\epsilon` are the weights
    respectively, and :math:`\\tilde{y}_k` is the smoothed label. Usually
    uniform distribution is used for :math:`\mu`.

    See more details about label smoothing in https://arxiv.org/abs/1512.00567.

    Parameters:
        label(Variable): The input variable containing the label data. The
                        label data should use one-hot representation. It's 
                        a multidimensional tensor with a shape of 
                        :math:`[N_1, ..., Depth]`, where Depth is class number.
        prior_dist(Variable, optional): The prior distribution to be used to smooth
                        labels. If not provided, an uniform distribution
                        is used. It's a multidimensional tensor with a shape of
                        :math:`[1, class\_num]` . The default value is None.
        epsilon(float, optional): The weight used to mix up the original ground-truth
                        distribution and the fixed distribution. The default value is 
                        0.1.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type can be set
                        as 'float32', 'float64'. The default value is 'float32'.
        name(str, optional): The default value is None. Normally there is no need for user 
                        to set this property. For more information, please refer to 
                        :ref:`api_guide_Name`.

    Returns:
        Variable: The tensor variable containing the smoothed labels.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            label = layers.data(name="label", shape=[1], dtype="float32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="float32")
    """
    if epsilon > 1. or epsilon < 0.:
        raise ValueError("The value of epsilon must be between 0 and 1.")
    helper = LayerHelper("label_smooth", **locals())
    label.stop_gradient = True
    smooth_label = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="label_smooth",
        inputs={"X": label,
                "PriorDist": prior_dist} if prior_dist else {"X": label},
        outputs={"Out": smooth_label},
        attrs={"epsilon": float(epsilon)})
    return smooth_label


@templatedoc()
def roi_pool(input, rois, pooled_height=1, pooled_width=1, spatial_scale=1.0):
    """
    This operator implements the roi_pooling layer. 
    Region of interest pooling (also known as RoI pooling) is to perform max pooling on inputs of nonuniform sizes to obtain fixed-size feature maps (e.g. 7*7).
    
    The operator has three steps:
    
        1. Dividing each region proposal into equal-sized sections with the pooled_width and pooled_height;
        2. Finding the largest value in each section;
        3. Copying these max values to the output buffer.
    
    For more information, please refer to https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn
    
    Args:
        input (Variable): Input feature, 4D-Tensor with the shape of [N,C,H,W], where N is the batch size, C is the input channel, H is Height, W is weight. The data type is float32 or float64.
        rois (Variable): ROIs (Regions of Interest) to pool over. 2D-LoDTensor with the shape of [num_rois,4], the lod level is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is the top left coordinates, and (x2, y2) is the bottom right coordinates.
        pooled_height (int, optional): The pooled output height, data type is int32. Default: 1
        pooled_width (int, optional): The pooled output height, data type is int32. Default: 1
        spatial_scale (float, optional): Multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0
    
    Returns:
        Variable: The pooled feature, 4D-Tensor with the shape of [num_rois, C, pooled_height, pooled_width].
    
    
    Examples:
    
    ..  code-block:: python
    
        import paddle.fluid as fluid
        import numpy as np
    
        DATATYPE='float32'
    
        place = fluid.CPUPlace()
        #place = fluid.CUDAPlace(0)
    
        input_data = np.array([i for i in range(1,17)]).reshape(1,1,4,4).astype(DATATYPE)
        roi_data =fluid.create_lod_tensor(np.array([[1., 1., 2., 2.], [1.5, 1.5, 3., 3.]]).astype(DATATYPE),[[2]], place)
    
        x = fluid.data(name='input', shape=[None,1,4,4], dtype=DATATYPE)
        rois = fluid.data(name='roi', shape=[None,4], dtype=DATATYPE)
    
        pool_out = fluid.layers.roi_pool(
                input=x,
                rois=rois,
                pooled_height=1,
                pooled_width=1,
                spatial_scale=1.0)
    
        exe = fluid.Executor(place)
        out, = exe.run(feed={'input':input_data ,'roi':roi_data}, fetch_list=[pool_out.name])
        print(out)   #array([[[[11.]]], [[[16.]]]], dtype=float32)
        print(np.array(out).shape)  # (2, 1, 1, 1)
    """
    helper = LayerHelper('roi_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    argmaxes = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="roi_pool",
        inputs={"X": input,
                "ROIs": rois},
        outputs={"Out": pool_out,
                 "Argmax": argmaxes},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale
        })
    return pool_out


@templatedoc()
def roi_align(input,
              rois,
              pooled_height=1,
              pooled_width=1,
              spatial_scale=1.0,
              sampling_ratio=-1,
              name=None):
    """
    ${comment}

    Args:
        input (Variable): ${x_comment}
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
            a 2-D LoDTensor of shape (num_rois, 4), the lod level is 1. The 
            data type is float32 or float64. Given as [[x1, y1, x2, y2], ...], 
            (x1, y1) is the top left coordinates, and (x2, y2) is the bottom
            right coordinates. 
        pooled_height (int32, optional): ${pooled_height_comment} Default: 1
        pooled_width (int32, optional): ${pooled_width_comment} Default: 1
        spatial_scale (float32, optional): ${spatial_scale_comment} Default: 1.0
        sampling_ratio(int32, optional): ${sampling_ratio_comment} Default: -1
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Variable:

        Output: ${out_comment}.


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(
                name='data', shape=[None, 256, 32, 32], dtype='float32')
            rois = fluid.data(
                name='rois', shape=[None, 4], dtype='float32')
            align_out = fluid.layers.roi_align(input=x,
                                               rois=rois,
                                               pooled_height=7,
                                               pooled_width=7,
                                               spatial_scale=0.5,
                                               sampling_ratio=-1)
    """
    helper = LayerHelper('roi_align', **locals())
    dtype = helper.input_dtype()
    align_out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="roi_align",
        inputs={"X": input,
                "ROIs": rois},
        outputs={"Out": align_out},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale,
            "sampling_ratio": sampling_ratio
        })
    return align_out


def dice_loss(input, label, epsilon=0.00001, name=None):
    """
    Dice loss for comparing the similarity between the input predictions and the label.
    This implementation is for binary classification, where the input is sigmoid
    predictions of each pixel, usually used for segmentation task. The dice loss can
    be defined as the following equation:

    .. math::

        dice\_loss &= 1 - \\frac{2 * intersection\_area}{total\_area} \\\\
                  &= \\frac{(total\_area - intersection\_area) - intersection\_area}{total\_area} \\\\
                  &= \\frac{(union\_area - intersection\_area)}{total\_area}


    Parameters:
        input (Variable): Tensor, rank>=2, shape is :math:`[N_1, N_2, ..., N_D]`, where :math:`N_1` is
                          the batch_size, :math:`N_D` is 1. It is usually the output predictions of sigmoid activation.
                          The data type can be float32 or float64.
        label (Variable): Tensor, the groud truth with the same rank as input, shape is :math:`[N_1, N_2, ..., N_D]`. 
                          where :math:`N_1` is the batch_size, :math:`N_D` is 1. The data type can be float32 or float64.
        epsilon (float): The epsilon will be added to the numerator and denominator.
                         If both input and label are empty, it makes sure dice is 1.
                         Default: 0.00001
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        The dice loss with shape [1], data type is the same as `input` .
    Return Type:
        Varaible

    Example:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='data', shape = [3, 224, 224, 1], dtype='float32')
            label = fluid.data(name='label', shape=[3, 224, 224, 1], dtype='float32')
            predictions = fluid.layers.sigmoid(x)
            loss = fluid.layers.dice_loss(input=predictions, label=label)
    """
    label = one_hot(label, depth=input.shape[-1])
    reduce_dim = list(range(1, len(input.shape)))
    inse = reduce_sum(input * label, dim=reduce_dim)
    dice_denominator = reduce_sum(
        input, dim=reduce_dim) + reduce_sum(
            label, dim=reduce_dim)
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    return reduce_mean(dice_score)


def image_resize(input,
                 out_shape=None,
                 scale=None,
                 name=None,
                 resample='BILINEAR',
                 actual_shape=None,
                 align_corners=True,
                 align_mode=1,
                 data_format='NCHW'):
    """
    This op resizes a batch of images.

    The input must be a 4-D Tensor of the shape (num_batches, channels, in_h, in_w) 
    or (num_batches, in_h, in_w, channels), or a 5-D Tensor of the shape 
    (num_batches, channels, in_d, in_h, in_w) or (num_batches, in_d, in_h, in_w, channels), 
    and the resizing only applies on the three dimensions(depth, hight and width).

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the
    future and only use :attr:`out_shape` instead.

    Supporting resample methods:

        'BILINEAR' : Bilinear interpolation

        'TRILINEAR' : Trilinear interpolation

        'NEAREST' : Nearest neighbor interpolation

    Nearest neighbor interpolation is to perform nearest neighbor interpolation
    in both the 3rd dimention(in height direction) and the 4th dimention(in width 
    direction) on input tensor.
            
    Bilinear interpolation is an extension of linear interpolation for 
    interpolating functions of two variables (e.g. H-direction and 
    W-direction in this op) on a rectilinear 2D grid. The key idea is 
    to perform linear interpolation first in one direction, and then 
    again in the other direction.

    Trilinear interpolation is an extension of linear interpolation for 
    interpolating functions of three variables (e.g. D-direction, 
    H-direction and W-direction in this op) on a rectilinear 3D grid. 
    The linear interpolation is performed on three directions.

    Align_corners and align_mode are optinal parameters,the calculation method 
    of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)
            
          
        Nearest neighbor interpolation:
          
          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = floor (H_{in} * scale_{factor})
              W_out = floor (W_{in} * scale_{factor})

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

        Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:
           
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

        Trilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5


          else:
           
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}
          
    For details of nearest neighbor interpolation, please refer to Wikipedia: 
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation.

    For details of bilinear interpolation, please refer to Wikipedia: 
    https://en.wikipedia.org/wiki/Bilinear_interpolation.

    For details of trilinear interpolation, please refer to Wikipedia: 
    https://en.wikipedia.org/wiki/Trilinear_interpolation.



    Parameters:
        input (Variable): 4-D or 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of image resize
             layer, the shape is (out_h, out_w) when input is a 4-D Tensor and is
             (out_d, out_h, out_w) when input is a 5-D Tensor. Default: None. If 
             a list, each element can be an integer or a Tensor Variable of shape: [1].
             If a Tensor Variable, its dimensions size should be a 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set.
             And :attr:`out_shape` has a higher priority than :attr:`scale`.
             Default: None.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
        resample(str): The resample method. It supports 'BILINEAR', 'TRILINEAR'
                       and 'NEAREST' currently. Default: 'BILINEAR'
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool) :  An optional bool, If True, the centers of the 4 corner pixels of the 
                               input and output tensors are aligned, preserving the values at the 
                               corner pixels.
                               Default: True
        align_mode(int)  :  An optional for bilinear interpolation. can be \'0\' 
                            for src_idx = scale*(dst_indx+0.5)-0.5 , can be \'1\' for 
                            src_idx = scale*dst_index.
        data_format(str, optional): NCHW(num_batches, channels, height, width) or 
                                    NHWC(num_batches, height, width, channels) for 4-D Tensor,
                                    NCDHW(num_batches, channels, depth, height, width) or 
                                    NDHWC(num_batches, depth, height, width, channels) for 5-D Tensor.
                                    Default: 'NCHW'.

    Returns:
        A 4-D Tensor of the shape (num_batches, channels, out_h, out_w) or (num_batches, out_h, out_w, channels),
        or 5-D Tensor of the shape (num_batches, channels, out_d, out_h, out_w) or (num_batches, out_d, out_h, out_w, channels).

    Raises:
        TypeError: out_shape should be a list or tuple or Variable.
        TypeError: actual_shape should either be Variable or None.
        ValueError: The 'resample' of image_resize can only be 'BILINEAR',
                    'TRILINEAR' or 'NEAREST' currently.
        ValueError: 'BILINEAR' and 'NEAREST' only support 4-D tensor.
        ValueError: 'TRILINEAR' only support 5-D tensor.
        ValueError: One of out_shape and scale must not be None.
        ValueError: out_shape length should be 2 for input 4-D tensor.
        ValueError: out_shape length should be 3 for input 5-D tensor.
        ValueError: scale should be greater than zero.
        TypeError: align_corners shoule be a bool value
        ValueError: align_mode can only be '0' or '1'
        ValueError: data_format can only be 'NCHW', 'NHWC', 'NCDHW' or 'NDHWC'.

    Examples:
        .. code-block:: python
	
	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[None,3,6,10])

	    #1
	    output = fluid.layers.image_resize(input=input,out_shape=[12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.image_resize(input=input,out_shape=[12,dim1])

	    #3
	    #x = np.array([3,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.image_resize(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.image_resize(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.random.rand(2,3,6,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12)
	    #2
	    # (2, 3, 12, 2)
	    #3
	    # (2, 3, 3, 12)
	    #4
	    # (2, 3, 3, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.image_resize(input=input, out_shape=[12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L]

    """
    resample_methods = {
        'BILINEAR': 'bilinear',
        'TRILINEAR': 'trilinear',
        'NEAREST': 'nearest',
    }
    if resample not in resample_methods:
        raise ValueError(
            "The 'resample' of image_resize can only be 'BILINEAR', 'TRILINEAR' "
            "or 'NEAREST' currently.")
    resample_type = resample_methods[resample]

    if resample in ['BILINEAR', 'NEAREST'] and len(input.shape) != 4:
        raise ValueError("'BILINEAR' and 'NEAREST' only support 4-D tensor.")
    if resample == 'TRILINEAR' and len(input.shape) != 5:
        raise ValueError("'TRILINEAR'only support 5-D tensor.")

    if not isinstance(align_corners, bool):
        raise TypeError("Attr align_corners should be a bool value")
    if align_mode != 0 and align_mode != 1:
        raise ValueError("align_mode can only be 0 or 1")

    if out_shape is None and scale is None:
        raise ValueError("One of out_shape and scale must not be None.")
    helper = LayerHelper('{}_interp'.format(resample_type), **locals())
    dtype = helper.input_dtype()

    if len(input.shape) == 4 and data_format not in ['NCHW', 'NHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCHW` or `NHWC` supported for 4-D input.")
    elif len(input.shape) == 5 and data_format not in ['NCDHW', 'NDHWC']:
        raise ValueError(
            "Got wrong value for param `data_format`: " + data_format +
            " received but only `NCDHW` or `NDHWC` supported for 5-D input.")

    def _is_list_or_turple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if data_format == 'NCHW' or data_format == 'NCDHW':
        data_layout = 'NCHW'
    if data_format == 'NHWC' or data_format == 'NDHWC':
        data_layout = 'NHWC'

    inputs = {"X": input}
    attrs = {
        "out_d": -1,
        "out_h": -1,
        "out_w": -1,
        "interp_method": resample_type,
        "align_corners": align_corners,
        "align_mode": align_mode,
        "data_layout": data_layout
    }

    if out_shape is not None:
        if isinstance(out_shape, Variable):
            out_shape.stop_gradient = True
            inputs['OutSize'] = out_shape
        else:
            if not (_is_list_or_turple_(out_shape)):
                raise TypeError(
                    "out_shape should be a list or tuple or Variable.")
            # Validate the shape
            contain_var = False
            for dim_idx, dim_size in enumerate(out_shape):
                if isinstance(dim_size, Variable):
                    contain_var = True
                    continue
                assert dim_size > 0, (
                    "Each dimension size given in out_shape must be greater than 0."
                )

            if contain_var:
                new_size_tensor = []
                size_list = []
                for dim in out_shape:
                    if isinstance(dim, Variable):
                        dim.stop_gradient = True
                        new_size_tensor.append(dim)
                        size_list.append(-1)
                    else:
                        assert (isinstance(dim, int))
                        temp_out = helper.create_variable_for_type_inference(
                            'int32')
                        fill_constant(
                            [1], 'int32', dim, force_cpu=True, out=temp_out)
                        new_size_tensor.append(temp_out)
                        size_list.append(dim)
                inputs['SizeTensor'] = new_size_tensor

            if len(input.shape) == 4:
                if len(out_shape) != 2:
                    raise ValueError("out_shape length should be 2 for "
                                     "input 4-D tensor.")
                if contain_var:
                    attrs['out_h'] = size_list[0]
                    attrs['out_w'] = size_list[1]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_h'] = out_shape[0]
                    attrs['out_w'] = out_shape[1]
            if len(input.shape) == 5:
                if len(out_shape) != 3:
                    raise ValueError("out_shape length should be 3 for "
                                     "input 5-D tensor.")
                if contain_var:
                    attrs['out_d'] = size_list[0]
                    attrs['out_h'] = size_list[1]
                    attrs['out_w'] = size_list[2]
                else:
                    out_shape = list(map(int, out_shape))
                    attrs['out_d'] = out_shape[0]
                    attrs['out_h'] = out_shape[1]
                    attrs['out_w'] = out_shape[2]

    else:
        if isinstance(scale, Variable):
            scale.stop_gradient = True
            inputs["Scale"] = scale
        elif isinstance(scale, float) or isinstance(scale, int):
            if scale <= 0:
                raise ValueError("Attr(scale) should be greater than zero.")
            attrs['scale'] = float(scale)
        else:
            raise TypeError(
                "Attr(scale)'s type should be float, int or Variable.")

    if isinstance(actual_shape, Variable):
        warnings.warn(
            "actual_shape will be deprecated, it is recommended to use "
            "out_shape instead of actual_shape to specify output shape dynamically."
        )
        actual_shape.stop_gradient = True
        inputs["OutSize"] = actual_shape
    elif actual_shape is not None:
        raise TypeError("actual_shape should either be Variable or None.")

    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='{}_interp'.format(resample_type),
        inputs=inputs,
        outputs={"Out": out},
        attrs=attrs)
    return out


@templatedoc(op_type="bilinear_interp")
def resize_bilinear(input,
                    out_shape=None,
                    scale=None,
                    name=None,
                    actual_shape=None,
                    align_corners=True,
                    align_mode=1,
                    data_format='NCHW'):
    """
    This op resizes the input by performing bilinear interpolation based on given
    output shape which specified by actual_shape, out_shape and scale
    in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in 
    the future and only use :attr:`out_shape` instead.

    Bilinear interpolation is an extension of linear interpolation for
    interpolating functions of two variables (e.g. H-direction and
    W-direction in this op) on a rectilinear 2D grid. The key idea is
    to perform linear interpolation first in one direction, and then
    again in the other direction.

    For details of bilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Bilinear_interpolation

    Align_corners and align_mode are optinal parameters,the calculation 
    method of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)

        Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    Parameters:
        input(Variable): 4-D Tensor(NCHW), its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): Output shape of resize bilinear
            layer, the shape is (out_h, out_w).Default: None. If a list, each 
            element can be an integer or a Tensor Variable with shape: [1]. If a 
            Tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set. 
             And :attr:`out_shape` has a higher priority than :attr:`scale`. 
             Default: None.
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        align_mode(bool): ${align_mode_comment}
        data_format(str, optional): NCHW(num_batches, channels, height, width) or 
                                    NHWC(num_batches, height, width, channels). Default: 'NCHW'.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`

    Returns:
	Variable: 4-D tensor(NCHW or NHWC).
    
    Examples:
        .. code-block:: python
	
	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[None,3,6,10])

	    #1
	    output = fluid.layers.resize_bilinear(input=input,out_shape=[12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.resize_bilinear(input=input,out_shape=[12,dim1])

	    #3
	    #x = np.array([3,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.resize_bilinear(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.resize_bilinear(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.random.rand(2,3,6,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12)
	    #2
	    # (2, 3, 12, 2)
	    #3
	    # (2, 3, 3, 12)
	    #4
	    # (2, 3, 3, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.resize_bilinear(input=input, out_shape=[12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L]

    """

    return image_resize(input, out_shape, scale, name, 'BILINEAR', actual_shape,
                        align_corners, align_mode, data_format)


@templatedoc(op_type="trilinear_interp")
def resize_trilinear(input,
                     out_shape=None,
                     scale=None,
                     name=None,
                     actual_shape=None,
                     align_corners=True,
                     align_mode=1,
                     data_format='NCDHW'):
    """
    This op resizes the input by performing trilinear interpolation based on given
    output shape which specified by actual_shape, out_shape and scale
    in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated 
    in the future and only use :attr:`out_shape` instead.

    Trilinear interpolation is an extension of linear interpolation for 
    interpolating functions of three variables (e.g. D-direction, 
    H-direction and W-direction in this op) on a rectilinear 3D grid. 
    The linear interpolation is performed on three directions.

    For details of trilinear interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Trilinear_interpolation

    Align_corners and align_mode are optinal parameters,the calculation 
    method of interpolation can be selected by them.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :

              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)     

        Bilinear interpolation:

          if:

              align_corners = False , align_mode = 0
              
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5

          else:

              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

    Parameters:
        input(${x_type}): 5-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): The output shape of resized tensor, the shape is (out_d, out_h, out_w). Default: None. Every element should be an integer or a Tensor Variable with shape: [1] if it is a list. If it is a Tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input depth, height or width.
             At least one of :attr:`out_shape` or :attr:`scale` must be set. 
             And :attr:`out_shape` has a higher priority than :attr:`scale`. 
             Default: None.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
        actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        align_mode(bool): ${align_mode_comment}
        data_format(str, optional): NCDHW(num_batches, channels, depth, height, width) or 
                                    NDHWC(num_batches, depth, height, width, channels).
                                    Default: 'NCDHW'.

    Returns:
        Variable: A 5-D Tensor(NCDHW or NDHWC) 

    Examples:
        .. code-block:: python
	
	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[None,3,6,8,10])

	    #1
	    output = fluid.layers.resize_trilinear(input=input,out_shape=[12,12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.resize_trilinear(input=input,out_shape=[12,dim1,4])

	    #3
	    #x = np.array([3,12,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[3], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.resize_trilinear(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.resize_trilinear(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.random.rand(2,3,6,8,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12, 12)
	    #2
	    # (2, 3, 12, 2, 4)
	    #3
	    # (2, 3, 3, 12, 12)
	    #4
	    # (2, 3, 3, 4, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.resize_trilinear(input=input, out_shape=[12,12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L, 12L]



    """

    return image_resize(input, out_shape, scale, name, 'TRILINEAR',
                        actual_shape, align_corners, align_mode, data_format)


@templatedoc(op_type="nearest_interp")
def resize_nearest(input,
                   out_shape=None,
                   scale=None,
                   name=None,
                   actual_shape=None,
                   align_corners=True,
                   data_format='NCHW'):
    """
    This op resizes the input by performing nearest neighbor interpolation in both the
    height direction and the width direction based on given output shape 
    which is specified by actual_shape, out_shape and scale in priority order.

    **Warning:** the parameter :attr:`actual_shape` will be deprecated in the 
    future and only use :attr:`out_shape` instead.

    Example:

    .. code-block:: text

        For scale:
          
            if align_corners = True && out_size > 1 :
              scale_factor = (in_size-1.0)/(out_size-1.0)
            
            else:
              
              scale_factor = float(in_size/out_size)
          
        Nearest neighbor interpolation:
          
          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = floor(H_{in} * scale_{factor})
              W_out = floor(W_{in} * scale_{factor})

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})


    For details of nearest neighbor interpolation, please refer to Wikipedia:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation

    Parameters:
        input(${x_type}): 4-D Tensor, its data type is float32, float64, or uint8,
                          its data format is specified by :attr:`data_format`.
        out_shape(list|tuple|Variable|None): The output shape of resized tensor, the shape is (out_h, out_w). Default: None. Every element should be an integer or a tensor Variable with shape: [1] if it is a list. If it is a tensor Variable, its dimension size should be 1.
        scale(float|Variable|None): The multiplier for the input height or width. At
             least one of :attr:`out_shape` or :attr:`scale` must be set. 
             And :attr:`out_shape` has a higher priority than :attr:`scale`. 
             Default: None. 
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
	actual_shape(Variable): An optional input to specify output shape
                                dynamically. If provided, image resize
                                according to this given shape rather than
                                :attr:`out_shape` and :attr:`scale` specifying
                                shape. That is to say actual_shape has the
                                highest priority. It is recommended to use
                                :attr:`out_shape` if you want to specify output 
                                shape dynamically, because :attr:`actual_shape` 
                                will be deprecated. When using actual_shape to 
                                specify output shape, one of :attr:`out_shape` 
                                and :attr:`scale` should also be set, otherwise 
                                errors would be occured in graph constructing stage.
                                Default: None
        align_corners(bool): ${align_corners_comment}
        data_format(str, optional): NCHW(num_batches, channels, height, width) or 
                                    NHWC(num_batches, height, width, channels).
                                    Default: 'NCHW'.

    Returns:
	Variable: 4-D tensor(NCHW or NHWC).

    Examples:
        .. code-block:: python
	
	    #declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[None,3,6,10])

	    #1
	    output = fluid.layers.resize_nearest(input=input,out_shape=[12,12])

	    #2
	    #x = np.array([2]).astype("int32")
	    #dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
	    #fluid.layers.assign(input=x, output=dim1)
	    #output = fluid.layers.resize_nearest(input=input,out_shape=[12,dim1])

	    #3
	    #x = np.array([3,12]).astype("int32")
	    #shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
	    #fluid.layers.assign(input=x, output=shape_tensor)
	    #output = fluid.layers.resize_nearest(input=input,out_shape=shape_tensor)

	    #4
	    #x = np.array([0.5]).astype("float32")
	    #scale_tensor = fluid.data(name="scale", shape=[1], dtype="float32")
	    #fluid.layers.assign(x,scale_tensor)
	    #output = fluid.layers.resize_nearest(input=input,scale=scale_tensor)

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.random.rand(2,3,6,10).astype("float32")

	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data[0].shape)

	    #1
	    # (2, 3, 12, 12)
	    #2
	    # (2, 3, 12, 2)
	    #3
	    # (2, 3, 3, 12)
	    #4
	    # (2, 3, 3, 5)

	    #imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		output = fluid.layers.resize_nearest(input=input, out_shape=[12,12])
    		print(output.shape)

		# [2L, 3L, 12L, 12L]



    """

    return image_resize(
        input,
        out_shape,
        scale,
        name,
        'NEAREST',
        actual_shape,
        align_corners,
        align_mode=1,
        data_format=data_format)


def image_resize_short(input, out_short_len, resample='BILINEAR'):
    """
    This op resizes a batch of images. The short edge of input images will be
    resized to the given 'out_short_len'. The long edge of input images
    will be resized proportionately to make images' length-width ratio
    constant.

    Parameters:
        input (Variable): 4-D tensor(NCHW), The input tensor of image resize layer.
        out_short_len(int): The length of output images' short edge.
        resample (str): resample method, default: BILINEAR.

    Returns:
        Variable: 4-D tensor(NCHW).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(name="input", shape=[None,3,6,9], dtype="float32")
            out = fluid.layers.image_resize_short(input, out_short_len=3)
    """
    in_shape = input.shape
    if len(in_shape) != 4:
        raise ValueError(
            "The rank of input must be 4 (num_batches, channels, in_h, in_w).")
    hw = in_shape[2:4]
    short_idx = hw.index(min(hw))
    long_idx = 1 - short_idx
    out_shape = list(hw)
    out_shape[short_idx] = out_short_len
    out_shape[long_idx] = int(
        float(out_shape[long_idx]) * (float(out_short_len) / float(hw[
            short_idx])) + 0.5)
    return image_resize(input=input, out_shape=out_shape, resample=resample)


def gather(input, index, overwrite=True):
    """
    **Gather Layer**

    Output is obtained by gathering entries of the outer-most dimension
    of X indexed by `index` and concatenate them together.

    .. math::

        Out = X[Index]


    .. code-block:: text


                Given:

                X = [[1, 2],
                     [3, 4],
                     [5, 6]]

                Index = [1, 2]

                Then:

                Out = [[3, 4],
                       [5, 6]]

    Args:
        input (Variable): The source input tensor with rank>=1. Supported data type is 
            int32, int64, float32, float64 and uint8 (only for CPU), 
            float16 (only for GPU).
        index (Variable): The index input tensor with rank=1. Data type is int32 or int64.
        overwrite (bool, optional): The mode that updating the grad when has same index.
            If True, use the overwrite mode to update the grad of the same index,
	    if False, use the accumulate mode to update the grad of the same index. 
	    Default value is True.
	    


    Returns:
        output (Variable): The output is a tensor with the same rank as input.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[-1, 5], dtype='float32')
            index = fluid.data(name='index', shape=[-1, 1], dtype='int32')
            output = fluid.layers.gather(x, index)
    """
    helper = LayerHelper('gather', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="gather",
        inputs={"X": input,
                "Index": index},
        outputs={"Out": out},
        attrs={'overwrite': overwrite})
    return out


def gather_nd(input, index, name=None):
    """
    **Gather Nd Layer**

    This function is actually a high-dimensional extension of :code:`gather` 
    and supports for simultaneous indexing by multiple axes. :attr:`index` is a 
    K-dimensional integer tensor, which is regarded as a (K-1)-dimensional 
    tensor of :attr:`index` into :attr:`input`, where each element defines 
    a slice of params:

    .. math::

        output[(i_0, ..., i_{K-2})] = input[index[(i_0, ..., i_{K-2})]]

    Obviously, :code:`index.shape[-1] <= input.rank` . And, the output tensor has
    shape :code:`index.shape[:-1] + input.shape[index.shape[-1]:]` .

    .. code-block:: text

            Given:
                input = [[[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]],
                         [[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]]]
                input.shape = (2, 3, 4)

            * Case 1:
                index = [[1]]
                
                gather_nd(input, index)  
                         = [input[1, :, :]] 
                         = [[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]]

            * Case 2:
                index = [[0,2]]

                gather_nd(input, index)
                         = [input[0, 2, :]]
                         = [8, 9, 10, 11]

            * Case 3:
                index = [[1, 2, 3]]

                gather_nd(input, index)
                         = [input[1, 2, 3]]
                         = [23]

    Args:
        input (Variable): The source input. Its dtype should be int32, int64, float32, float64.
        index (Variable): The index input with rank > 1, index.shape[-1] <= input.rank.
                          Its dtype should be int32, int64.
        name (str|None): A name for this layer(optional). If set None, the
                         layer will be named automatically.

    Returns:
        output (Variable): A tensor with the shape index.shape[:-1] + input.shape[index.shape[-1]:]

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[3, 4, 5], dtype='float32')
            index = fluid.data(name='index', shape=[2, 2], dtype='int32')
            output = fluid.layers.gather_nd(x, index)

    """
    helper = LayerHelper('gather_nd', **locals())
    dtype = helper.input_dtype()
    if name is None:
        output = helper.create_variable_for_type_inference(dtype)
    else:
        output = helper.create_variable(
            name=name, dtype=dtype, persistable=False)
    helper.append_op(
        type="gather_nd",
        inputs={"X": input,
                "Index": index},
        outputs={"Out": output})
    return output


def scatter(input, index, updates, name=None, overwrite=True):
    """
    **Scatter Layer**

    Output is obtained by updating the input on selected indices based on updates.

    .. code-block:: python
        import numpy as np
                
        #input:
        input = np.array([[1, 1], [2, 2], [3, 3]])
        index = np.array([2, 1, 0, 1])
        # shape of updates should be the same as input
        # shape of updates with dim > 1 should be the same as input
        updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        overwrite = False

        # calculation:
        if not overwrite:
            for i in range(len(index)):
                input[index[i]] = np.zeros((2))

        for i in range(len(index)):
            if (overwrite):
                input[index[i]] = updates[i]
            else:
                input[index[i]] += updates[i]
        # output:
        out = np.array([[3, 3], [6, 6], [1, 1]])
        out.shape # [3, 2]

    Args:
        input (Variable): The input N-D Tensor with rank>=1. Data type can be float32.
        index (Variable): The index 1-D Tensor. Data type can be int32, int64. The length of index cannot exceed updates's length, and the value in index cannot exceed input's length.
        updates (Variable): update input with updates parameter based on index. shape should be the same as input, and dim value with dim > 1 shoule be the same as input.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .
        overwrite (bool): The mode that updating the output when there are same indices.
            If True, use the overwrite mode to update the output of the same index,
	    if False, use the accumulate mode to update the output of the same index. 
	    Default value is True.

    Returns:
        Variable(Tensor|LoDTensor): The output is a Tensor with the same shape as input.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            input = fluid.layers.data(name='data', shape=[3, 2], dtype='float32', append_batch_size=False)
            index = fluid.layers.data(name='index', shape=[4], dtype='int64', append_batch_size=False)
            updates = fluid.layers.data(name='update', shape=[4, 2], dtype='float32', append_batch_size=False)

            output = fluid.layers.scatter(input, index, updates, overwrite=False)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            in_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            update_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'data':in_data, "index":index_data, "update":update_data}, fetch_list=[output])
            print(res)
            # [array([[3., 3.],
            #   [6., 6.],
            #   [1., 1.]], dtype=float32)]
    """
    helper = LayerHelper('scatter', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="scatter",
        inputs={"X": input,
                "Ids": index,
                "Updates": updates},
        attrs={'overwrite': overwrite},
        outputs={"Out": out})
    return out


def scatter_nd_add(ref, index, updates, name=None):
    """
    **Scatter_nd_add Layer**

    Output is obtained by applying sparse addition to a single value
    or slice in a Variable. 

    :attr:`ref` is a Tensor with rank :math:`R` 
    and :attr:`index` is a Tensor with rank :math:`K` . Thus, :attr:`index` 
    has shape :math:`[i_0, i_1, ..., i_{K-2}, Q]` where :math:`Q \leq R` . :attr:`updates` 
    is a Tensor with rank :math:`K - 1 + R - Q` and its
    shape is :math:`index.shape[:-1] + ref.shape[index.shape[-1]:]` .

    According to the :math:`[i_0, i_1, ..., i_{K-2}]` of :attr:`index` ,
    add the corresponding :attr:`updates` slice to the :attr:`ref` slice
    which is obtained by the last one dimension of :attr:`index` .

    .. code-block:: text
        
        Given:

        * Case 1:
            ref = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          we get:
             
            output = [0, 22, 12, 14, 4, 5]

        * Case 2:
            ref = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            ref.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          we get:
             
            output = [[67, 19], [-16, -27]]

    Args:
        ref (Variable): The ref input. Its dtype should be int32, int64, float32, float64.
        index (Variable): The index input with rank > 1 and index.shape[-1] <= ref.rank.
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Variable): The updated value of scatter_nd_add op, and it must have the same dtype
                            as ref. It must have the shape index.shape[:-1] + ref.shape[index.shape[-1]:].
        name (str|None): The output variable name. If set None, the layer will be named automatically.

    Returns:
        output (Variable): The output is a tensor with the same shape and dtype as ref.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            ref = fluid.data(name='ref', shape=[3, 5, 9, 10], dtype='float32')
            index = fluid.data(name='index', shape=[3, 2], dtype='int32')
            updates = fluid.data(name='update', shape=[3, 9, 10], dtype='float32')

            output = fluid.layers.scatter_nd_add(ref, index, updates)
    """
    if ref.dtype != updates.dtype:
        raise ValueError("ref and updates must have same data type.")

    helper = LayerHelper('scatter_nd_add', **locals())
    dtype = helper.input_dtype()
    if name is None:
        output = helper.create_variable_for_type_inference(dtype)
    else:
        output = helper.create_variable(
            name=name, dtype=dtype, persistable=False)
    helper.append_op(
        type="scatter_nd_add",
        inputs={"X": ref,
                "Index": index,
                "Updates": updates},
        outputs={"Out": output})
    return output


def scatter_nd(index, updates, shape, name=None):
    """
    **Scatter_nd Layer**

    Output is obtained by scattering the :attr:`updates` in a new tensor according 
    to :attr:`index` . This op is similar to :code:`scatter_nd_add`, except the 
    tensor of :attr:`shape` is zero-initialized. Correspondingly, :code:`scatter_nd(index, updates, shape)` 
    is equal to :code:`scatter_nd_add(fluid.layers.zeros(shape, updates.dtype), index, updates)` . 
    If :attr:`index` has repeated elements, then the corresponding updates are accumulated. 
    Because of the numerical approximation issues, the different order of repeated elements 
    in :attr:`index` may cause different results. The specific calculation method can be 
    seen :code:`scatter_nd_add` . This op is the inverse of the :code:`gather_nd` op.

    Args:
        index (Variable): The index input with rank > 1 and index.shape[-1] <= len(shape).
                          Its dtype should be int32 or int64 as it is used as indexes.
        updates (Variable): The updated value of scatter_nd op. Its dtype should be int32, int64, float32, float64.
                            It must have the shape index.shape[:-1] + shape[index.shape[-1]:]
        shape(tuple|list): Shape of output tensor.
        name (str|None): The output variable name. If set None, the layer will be named automatically.

    Returns:
        output (Variable): The output is a tensor with the same type as :attr:`updates` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            index = fluid.data(name='index', shape=[3, 2], dtype='int64')
            updates = fluid.data(name='update', shape=[3, 9, 10], dtype='float32')
            shape = [3, 5, 9, 10]

            output = fluid.layers.scatter_nd(index, updates, shape)
    """
    return scatter_nd_add(zeros(shape, updates.dtype), index, updates, name)


def sequence_scatter(input, index, updates, name=None):
    """
    **Note**:
    
    **The index and updates parameters of the OP must be LoDTensor.**
     
    Plus the updates data to the correspoding input according to the index.
 
    The updated algorithm is as follows: output[instance_index][index [pos]] = input[instance_index][index [pos]] +  updates[pos], 
    where instance_idx is the K sample corresponding to pos in batch.

    The value of output[i][j] depends on whether j can be found in the i+1th interval of the index. If found, 
    out[i][j] = input[i][j] + update[m] [n], otherwise, out[i][j] = input[i][j].

    For example, in the following example, the lod information for index is divided into three sequences. Among 
    them, because the element 0 can be found in the first interval of the index, it is updated with the value of 
    the corresponding position of the updates, out[0][0] = input[0][0]+updates[0][0] . Because element 1 cannot 
    be found in the third interval of index, out[2][1] = input[2][1].

    .. code-block:: text
        
        *Case 1:

            Given:
                input.data = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
                              input.dims = [3, 6]

                index.data = [[0], [1], [2], [5], [4], [3], [2], [1], [3], [2], [5], [4]]
                index.lod =  [[0,        3,                       8,                 12]]

                updates.data = [[0.3], [0.3], [0.4], [0.1], [0.2], [0.3], [0.4], [0.0], [0.2], [0.3], [0.1], [0.4]]
                updates.lod =  [[  0,            3,                                 8,                         12]]

            Then:
                out.data = [[1.3, 1.3, 1.4, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.4, 1.3, 1.2, 1.1],
                            [1.0, 1.0, 1.3, 1.2, 1.4, 1.1]]
                out.dims = X.dims = [3, 6]

    Args:
        input (Variable): A Tensor with shape of  :math:`[N, k_1... k_n]`. Supported data types: float32, float64, int32, int64.
        index (Variable):  A LoDTensor contains index information. Its LoD level must be 1 and its data type must be int64.
        updates (Variable): A LodTensor contains updates information. It has the same  LoD level with the index and has the 
                            same data type  with the input. Supported data types: float32, float64, int32, int64.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, 
                              please refer to :ref:`api_guide_Name`

    Returns:
        Variable: A Tensor which has been updated. It has the same shape and data type with input.

    Examples:

        .. code-block:: python
	
            import paddle.fluid as fluid

            input = fluid.data( name="x", shape=[None, 3, 6], dtype='float32' )
            index = fluid.data( name='index', shape=[12, 1],  dtype='int64', lod_level=1)
            updates = fluid.data( name='updates', shape=[12, 1], dtype='float32', lod_level=1)
            output = fluid.layers.sequence_scatter(input, index, updates)

    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_scatter', **locals())
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="sequence_scatter",
        inputs={"X": input,
                "Ids": index,
                "Updates": updates},
        outputs={"Out": out})
    return out


@templatedoc()
def random_crop(x, shape, seed=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        shape(${shape_type}): ${shape_comment}
        seed(int|${seed_type}|None): ${seed_comment} By default, the seed will
            get from `random.randint(-65536, 65535)`.

    Returns:
        ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            img = fluid.data("img", [None, 3, 256, 256])
            # cropped_img is [-1, 3, 224, 224]
            cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])

            # cropped_img2 shape: [-1, 2, 224, 224]
            # cropped_img2 = fluid.layers.random_crop(img, shape=[2, 224, 224])

            # cropped_img3 shape: [-1, 3, 128, 224]
            # cropped_img3 = fluid.layers.random_crop(img, shape=[128, 224])

    """
    helper = LayerHelper("random_crop", **locals())
    dtype = x.dtype
    out = helper.create_variable_for_type_inference(dtype)
    if seed is None:
        seed = np.random.randint(-65536, 65536)
    op_attrs = {"shape": shape}
    if isinstance(seed, int):
        op_attrs["startup_seed"] = seed
        seed = helper.create_variable(
            name=unique_name.generate("random_crop_seed"),
            dtype="int64",
            persistable=True)
    elif not isinstance(seed, Variable):
        raise ValueError("'seed' must be a Variable or an int.")
    helper.append_op(
        type="random_crop",
        inputs={"X": x,
                "Seed": seed},
        outputs={"Out": out,
                 "SeedOut": seed},
        attrs=op_attrs)
    return out


def log(x, name=None):
    """
    Calculates the natural log of the given input tensor, element-wise.

    .. math::

        Out = \\ln(x)

    Args:
        x (Variable): Input LoDTensor or Tensor. Must be one of the following types: float32, float64.
        name (str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`
    

    Returns:
        Variable: The natural log of the input LoDTensor or Tensor computed element-wise.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Graph Organizing
            x = fluid.layers.data(name="x", shape=[1], dtype="float32")
            res = fluid.layers.log(x)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            x_i = np.array([[1], [2]]).astype(np.float32)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
            print(res_val) # [[0.], [0.6931472]]
    """
    helper = LayerHelper('log', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(type="log", inputs={"X": x}, outputs={"Out": out})
    return out


@templatedoc()
def relu(x, name=None):
    """
    ${comment}

    Args:
        x(Variable): ${x_comment}
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        Variable: ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[-1,0],[1,2.6]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.relu(x1)
                print(out1.numpy())
                # [[0.  0. ]
                #  [1.  2.6]]
"""
    helper = LayerHelper('relu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="relu", inputs={"X": helper.input('x')}, outputs={"Out": out})
    return out


def selu(x, scale=None, alpha=None, name=None):
    """
    Selu Operator.

    The equation is:
    
    .. math::
        selu= \\lambda*
        \\begin{cases}
            x                      &\\quad \\text{ if } x>0 \n
            \\alpha * e^x - \\alpha  &\\quad \\text{ if } x<=0
        \\end{cases}
    

    The input `X` can carry the LoD (Level of Details) information,
    or not. And the output shares the LoD information with input `X`.

    Args:
        x (Variable): The input N-D Tensor.
        scale(float, optional): lambda in selu activation function,
            the default value is 1.0507009873554804934193349852946.
            For more information about this value, please refer
            to: https://arxiv.org/abs/1706.02515.
        alpha(float, optional): alpha in selu activation function,
            the default value is 1.6732632423543772848170429916717.
            For more information about this value, please refer
            to: https://arxiv.org/abs/1706.02515.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .


    Returns:
        Variable(Tensor|LoDTensor): The output Tensor or LoDTensor with the same shape and LoD information as input.

    Examples:

        .. code-block:: python
             
            import paddle.fluid as fluid
            import numpy as np

            inputs = fluid.layers.data(name="x", shape=[2, 2], dtype="float32")
            output = fluid.layers.selu(inputs)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img = np.array([[0, 1],[2, 3]]).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
            print(res) # [array([[0.      , 1.050701],[2.101402, 3.152103]], dtype=float32)]
    """
    helper = LayerHelper('selu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    attrs = {}
    if scale is not None:
        attrs["scale"] = scale
    if alpha is not None:
        attrs["alpha"] = alpha

    helper.append_op(
        type="selu", inputs={"X": x}, outputs={"Out": out}, attrs=attrs)
    return out


def mean_iou(input, label, num_classes):
    """
    Mean Intersection-Over-Union is a common evaluation metric for
    semantic image segmentation, which first computes the IOU for each
    semantic class and then computes the average over classes.
    IOU is defined as follows:

    .. math::

        IOU = \\frac{true\_positive}{(true\_positive + false\_positive + false\_negative)}.

    The predictions are accumulated in a confusion matrix and mean-IOU
    is then calculated from it.


    Parameters:
        input (Variable): A n-D Tensor of prediction results for semantic labels with type int32 or int64.
        label (Variable): A Tensor of ground truth labels with type int32 or int64.
                           Its shape should be the same as input.
        num_classes (int32): The possible number of labels.

    Returns: 
	Three Variables.

        - mean_iou(Variable) : A 1-D Tensor representing the mean intersection-over-union with shape [1]. \
			    Data type is float32.
        - out_wrong(Variable) : A 1-D Tensor with shape [num_classes]. Data type is int32. \
			     The wrong numbers of each class.
        - out_correct(Variable): A 1-D  Tensor with shape [num_classes]. Data type is int32. The correct numbers of each class.
 
   
    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            iou_shape = [None, 32, 32]
            num_classes = 5
            predict = fluid.data(name='predict', shape=iou_shape, dtype='int64')
            label = fluid.data(name='label', shape=iou_shape, dtype='int64')
            mean_iou, out_wrong, out_correct = fluid.layers.mean_iou(predict, label,
                                                          num_classes)
    """
    helper = LayerHelper('mean_iou', **locals())
    dtype = helper.input_dtype()
    out_mean_iou = helper.create_variable_for_type_inference(dtype='float32')
    out_wrong = helper.create_variable_for_type_inference(dtype='int32')
    out_correct = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="mean_iou",
        inputs={"Predictions": input,
                "Labels": label},
        outputs={
            "OutMeanIou": out_mean_iou,
            "OutWrong": out_wrong,
            "OutCorrect": out_correct
        },
        attrs={"num_classes": num_classes})
    return out_mean_iou, out_wrong, out_correct


def crop(x, shape=None, offsets=None, name=None):
    """
    Crop input into output, as specified by offsets and shape.

    **Warning:** THIS OP IS DEPRECATED. It will be removed in the future version.
    Instructions for updating: Use :ref:`api_fluid_layers_crop_tensor` instead.

    .. code-block:: text

        * Case 1:
            Given
                X = [[0, 1, 2, 0, 0]
                     [0, 3, 4, 0, 0]
                     [0, 0, 0, 0, 0]],
            and
                shape = [2, 2],
                offsets = [0, 1],
            output is:
                Out = [[1, 2],
                       [3, 4]].
        * Case 2:
            Given
                X = [[0, 1, 2, 5, 0]
                     [0, 3, 4, 6, 0]
                     [0, 0, 0, 0, 0]],
            and shape is tensor
                shape = [[0, 0, 0]
                         [0, 0, 0]]
            and
                offsets = [0, 1],

            output is:
                Out = [[1, 2, 5],
                       [3, 4, 6]].

    Parameters:
        x (Variable): Tensor, data type can be float32 or float64.
        shape (Variable|list/tuple of integers): The output shape is specified
            by `shape`, which can be a Tensor or a list/tuple of integers.
            If it is a Tensor, it's rank must be the same as `x` , only 
            it's shape will be used, and the value of it will be ignored. This way
            is suitable for the case that the output shape may be changed each
            iteration. If it is a list/tuple of integers, it's length must be the same
            as the rank of `x`
        offsets (Variable|list/tuple of integers|None): Specifies the cropping
            offsets at each dimension. It can be a Tensor or a list/tuple
            of integers. If it is a Tensor, it's rank must be the same as `x`.
            This way is suitable for the case that the offsets may be changed
            each iteration. If it is a list/tuple of integers, it's length must be the
            same as the rank of `x`. If None, the offsets are 0 at each dimension.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name` . Usually name is no need to set and 
            None by default. 

    Returns:
        The cropped Tensor, which has the same rank and data type with `x`

    Return Type:
        Variable

    Raises:
        ValueError: If shape is not a list, tuple or Variable.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name="x", shape=[3, 3, 5], dtype="float32")
            y = fluid.data(name="y", shape=[2, 2, 3], dtype="float32")
            crop = fluid.layers.crop(x, shape=y)

            # or
            z = fluid.data(name="z", shape=[3, 3, 5], dtype="float32")
            crop = fluid.layers.crop(z, shape=[2, 2, 3])

    """
    helper = LayerHelper('crop', **locals())

    if not (isinstance(shape, list) or isinstance(shape, tuple) or \
            isinstance(shape, Variable)):
        raise ValueError("The shape should be a list, tuple or Variable.")

    if offsets is None:
        offsets = [0] * len(x.shape)

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x}
    attrs = {}
    if isinstance(shape, Variable):
        ipts['Y'] = shape
    else:
        attrs['shape'] = shape
    if isinstance(offsets, Variable):
        ipts['Offsets'] = offsets
    else:
        attrs['offsets'] = offsets

    helper.append_op(
        type='crop',
        inputs=ipts,
        outputs={'Out': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def crop_tensor(x, shape=None, offsets=None, name=None):
    """
    Crop input into output, as specified by offsets and shape.

    .. code-block:: text

        * Case 1 (input is a 2-D Tensor):
            Input:
                X.shape = [3. 5]
                X.data = [[0, 1, 2, 0, 0],
                          [0, 3, 4, 0, 0],
                          [0, 0, 0, 0, 0]]
            Parameters:
                shape = [2, 2]
                offsets = [0, 1]
            Output:
                Out = [[1, 2],
                       [3, 4]]
        * Case 2 (input is a 3-D Tensor):
            Input:
                X.shape = [2, 3, 4]
                X.data =  [[[0, 1, 2, 3],
                            [0, 5, 6, 7],
                            [0, 0, 0, 0]],
                           [[0, 3, 4, 5],
                            [0, 6, 7, 8],
                            [0, 0, 0, 0]]]
            Parameters:
                shape = [2, 2, 3]
                offsets = [0, 0, 1]
            Output:
                Out = [[[1, 2, 3],
                        [5, 6, 7]],
                       [[3, 4, 5],
                        [6, 7, 8]]]

    Parameters:
        x (Variable): 1-D to 6-D Tensor, the data type is float32 or float64.
        shape (list|tuple|Variable): The output shape is specified
            by `shape`. Its data type is int32. If a list/tuple, it's length must be
            the same as the dimension size of `x`. If a Variable, it shoule be a 1-D Tensor.
            When it is a list, each element can be an integer or a Tensor of shape: [1].
            If Variable contained, it is suitable for the case that the shape may 
            be changed each iteration. Only the first element of list/tuple can be 
            set to -1, it means that the first dimension's size of the output is the same 
            as the input.
        offsets (list|tuple|Variable, optional): Specifies the cropping
            offsets at each dimension. Its data type is int32. If a list/tuple, it's length
            must be the same as the dimension size of `x`. If a Variable, it shoule be a 1-D
            Tensor. When it is a list, each element can be an integer or a Tensor of shape: [1].
            If Variable contained, it is suitable for the case that the offsets may be changed
            each iteration. Default: None, the offsets are 0 at each dimension.
        name(str, optional): The default value is None. Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: The cropped Tensor has same data type with `x`.

    Raises:
        ValueError: If shape is not a list, tuple or Variable.
        ValueError: If offsets is not None and not a list, tuple or Variable.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name="x", shape=[None, 3, 5], dtype="float32")
            # x.shape = [-1, 3, 5], where -1 indicates batch size, and it will get the exact value in runtime.

            # shape is a 1-D Tensor
            crop_shape = fluid.data(name="crop_shape", shape=[3], dtype="int32")
            crop0 = fluid.layers.crop_tensor(x, shape=crop_shape)
            # crop0.shape = [-1, -1, -1], it means crop0.shape[0] = x.shape[0] in runtime.

            # or shape is a list in which each element is a constant
            crop1 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3])
            # crop1.shape = [-1, 2, 3]

            # or shape is a list in which each element is a constant or Variable
            y = fluid.data(name="y", shape=[3, 8, 8], dtype="float32")
            dim1 = fluid.data(name="dim1", shape=[1], dtype="int32")
            crop2 = fluid.layers.crop_tensor(y, shape=[3, dim1, 4])
            # crop2.shape = [3, -1, 4]

            # offsets is a 1-D Tensor
            crop_offsets = fluid.data(name="crop_offsets", shape=[3], dtype="int32")
            crop3 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3], offsets=crop_offsets)
            # crop3.shape = [-1, 2, 3]

            # offsets is a list in which each element is a constant or Variable
            offsets_var =  fluid.data(name="dim1", shape=[1], dtype="int32")
            crop4 = fluid.layers.crop_tensor(x, shape=[-1, 2, 3], offsets=[0, 1, offsets_var])
            # crop4.shape = [-1, 2, 3]

    """
    helper = LayerHelper('crop_tensor', **locals())

    if not (isinstance(shape, list) or isinstance(shape, tuple) or \
            isinstance(shape, Variable)):
        raise ValueError("The shape should be a list, tuple or Variable.")

    if offsets is None:
        offsets = [0] * len(x.shape)

    if not (isinstance(offsets, list) or isinstance(offsets, tuple) or \
            isinstance(offsets, Variable)):
        raise ValueError("The offsets should be a list, tuple or Variable.")

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x}
    attrs = {}

    def contain_var(input_list):
        for ele in input_list:
            if isinstance(ele, Variable):
                return True
        return False

    if isinstance(offsets, Variable):
        offsets.stop_gradient = True
        ipts['Offsets'] = offsets
    elif contain_var(offsets):
        new_offsets_tensor = []
        for dim in offsets:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_offsets_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                assert dim >= 0, ("offsets should be greater or equal to zero.")
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_offsets_tensor.append(temp_out)
        ipts['OffsetsTensor'] = new_offsets_tensor
    else:
        attrs['offsets'] = offsets

    unk_dim_idx = -1
    if isinstance(shape, Variable):
        shape.stop_gradient = True
        ipts['Shape'] = shape
    elif contain_var(shape):
        new_shape_tensor = []
        shape_attr = []
        for dim_idx, dim_size in enumerate(shape):
            if isinstance(dim_size, Variable):
                dim_size.stop_gradient = True
                new_shape_tensor.append(dim_size)
                shape_attr.append(-1)
            else:
                assert (isinstance(dim_size, int))
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one element in shape can be unknown.")
                    assert dim_idx == 0, (
                        "Only the first element in shape can be -1.")
                    unk_dim_idx = dim_idx
                else:
                    assert dim_size > 0, (
                        "Each dimension size given in shape must be greater than zero."
                    )
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant(
                    [1], 'int32', dim_size, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
                shape_attr.append(dim_size)
        ipts['ShapeTensor'] = new_shape_tensor
        attrs['shape'] = shape_attr
    else:
        attrs['shape'] = shape

    helper.append_op(
        type='crop_tensor',
        inputs=ipts,
        outputs={'Out': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def affine_grid(theta, out_shape, name=None):
    """
    It generates a grid of (x,y) coordinates using the parameters of
    the affine transformation that correspond to a set of points where
    the input feature map should be sampled to produce the transformed
    output feature map.

    Args:
        theta (Variable) - A Tensor with shape [N, 2, 3]. It contains a batch of affine transform parameters.
                           The data type can be float32 or float64.
        out_shape (Variable | list | tuple): The shape of target output with format [batch_size, channel, height, width].
                                             ``out_shape`` can be a Tensor or a list or tuple. The data
                                             type must be int32.
        name(str|None): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Variable: A Tensor with shape [batch_size, H, W, 2] while 'H' and 'W' are the height and width of feature map in affine transformation. The data type is the same as `theta`. 

    Raises:
        ValueError: If the type of arguments is not supported.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            place = fluid.CPUPlace()
            theta = fluid.data(name="x", shape=[None, 2, 3], dtype="float32")
            out_shape = fluid.data(name="y", shape=[4], dtype="int32")
            grid_0 = fluid.layers.affine_grid(theta, out_shape)
            grid_1 = fluid.layers.affine_grid(theta, [5, 3, 28, 28])
            batch_size=2
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            output= exe.run(feed={"x": np.random.rand(batch_size,2,3).astype("float32"),
                                  "y": np.array([5, 3, 28, 28]).astype("int32")},
                                  fetch_list=[grid_0.name, grid_1.name])
            print(output[0])
            print(output[1])
    """
    helper = LayerHelper('affine_grid')

    if not (isinstance(out_shape, list) or isinstance(out_shape, tuple) or \
            isinstance(out_shape, Variable)):
        raise ValueError("The out_shape should be a list, tuple or Variable.")

    if not isinstance(theta, Variable):
        raise ValueError("The theta should be a Variable.")

    out = helper.create_variable_for_type_inference(theta.dtype)
    ipts = {'Theta': theta}
    attrs = {}
    if isinstance(out_shape, Variable):
        ipts['OutputShape'] = out_shape
    else:
        attrs['output_shape'] = out_shape

    helper.append_op(
        type='affine_grid',
        inputs=ipts,
        outputs={'Output': out},
        attrs=None if len(attrs) == 0 else attrs)
    return out


def rank_loss(label, left, right, name=None):
    """
    This operator implements the sort loss layer in the RankNet model. RankNet is a pairwise ranking model 
    with a training sample consisting of a pair of documents (A and B), The label (P) 
    indicates whether A is ranked higher than B or not. Please refer to more details: 
    `RankNet <http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_

    Rank loss layer takes three inputs: left ( :math:`o_i` ), right ( :math:`o_j` ) and
    label ( :math:`P_{i,j}` ). The inputs respectively represent RankNet's output scores
    for documents A and B and the value of label P. Rank loss layer takes batch inputs 
    with size batch_size (batch_size >= 1), P = {0, 1} or {0, 0.5, 1}, 
    where 0.5 means that there is no information about the rank of the input pair.
    The following equation computes rank loss C_{i,j} from the inputs:

    .. math::
      C_{i,j} &= -\\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\\\
    .. math::
      o_{i,j} &=  o_i - o_j  \\\\
    .. math::
      \\tilde{P_{i,j}} &= \\left \{0, 0.5, 1 \\right \} \ or \ \\left \{0, 1 \\right \}

    Parameters:
        label (Variable): 2-D ``Tensor`` with the shape of :math:`[batch,1]`, the data type is float32, batch indicates the size of the data. Indicats whether A ranked higher than B or not.
        left (Variable): 2-D ``Tensor`` with the shape of :math:`[batch,1]`, the data type is float32. RankNet's output score for doc A.
        right (Variable): 2-D ``Tensor`` with the shape of :math:`[batch,1]`, the data type is float32. RankNet's output score for doc B.
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: ``Tensor`` indicating the output value of the sort loss layer, the data type is float32, and the return value's shape is :math:`[batch,1]` .

    Raises:
        ValueError: Any of label, left, and right is not a ``Variable`` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            label = fluid.data(name="label", shape=[-1, 1], dtype="float32")
            left = fluid.data(name="left", shape=[-1, 1], dtype="float32")
            right = fluid.data(name="right", shape=[-1, 1], dtype="float32")
            out = fluid.layers.rank_loss(label, left, right)

    """
    helper = LayerHelper('rank_loss', **locals())

    if not (isinstance(label, Variable)):
        raise ValueError("The label should be a Variable")

    if not (isinstance(left, Variable)):
        raise ValueError("The left should be a Variable")

    if not (isinstance(right, Variable)):
        raise ValueError("The right should be a Variable")

    out = helper.create_variable_for_type_inference("float32")

    helper.append_op(
        type='rank_loss',
        inputs={"Label": label,
                "Left": left,
                "Right": right},
        outputs={'Out': out})
    return out


def margin_rank_loss(label, left, right, margin=0.1, name=None):
    """
    Margin Ranking Loss Layer for ranking problem,
    which compares left score and right score passed in.
    The ranking loss can be defined as following equation:

    .. math::

        rank\_loss = max(0, -label * (left - right) + margin)

    Args:
       label (Variable): Indicates whether the left is ranked higher than the right or not.
           Data type is float32.
       left (Variable): Ranking score for left. Data type float32.
       right (Variable): Ranking score for right. Data type float32.
       margin (float): Indicates the given margin.
       name(str|None): For detailed information, please refer to 
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
       Variable: The ranking loss.

    Raises:
       ValueError: Any of label, left, and right is not a Variable.

    Examples:

        .. code-block:: python

           import paddle.fluid as fluid
           label = fluid.data(name="label", shape=[-1, 1], dtype="float32")
           left = fluid.data(name="left", shape=[-1, 1], dtype="float32")
           right = fluid.data(name="right", shape=[-1, 1], dtype="float32")
           out = fluid.layers.margin_rank_loss(label, left, right)
    """
    helper = LayerHelper('margin_rank_loss', **locals())
    if not isinstance(label, Variable):
        raise ValueError("The label should be a Variable.")
    if not isinstance(left, Variable):
        raise ValueError("The left should be a Variable.")
    if not isinstance(right, Variable):
        raise ValueError("The right should be a Variable.")
    out = helper.create_variable_for_type_inference(left.dtype)
    act = helper.create_variable_for_type_inference(left.dtype)
    helper.append_op(
        type='margin_rank_loss',
        inputs={"Label": label,
                "X1": left,
                "X2": right},
        outputs={'Out': out,
                 'Activated': act},
        attrs={'margin': margin})
    return out


def pad2d(input,
          paddings=[0, 0, 0, 0],
          mode='constant',
          pad_value=0.0,
          data_format="NCHW",
          name=None):
    """
    Pad 2-d images accordding to 'paddings' and 'mode'.
    If mode is 'reflect', paddings[0] and paddings[1] must be no greater
    than height-1. And the width dimension has the same condition.

    Parameters:
        input (Variable): The input image with [N, C, H, W] format or [N, H, W, C] format, which is a 4-D Tensor with data type float32.
        paddings (Variable | List[int32]): The padding size. If padding is a List, it must
            contain four integers, (padding_top, padding_bottom, padding_left, padding_right).
            Otherwise, it is a 1-D Tensor with shape [4]. Data type is int32.
            Default is [0, 0, 0, 0].
        mode (str): Three modes: 'constant' (default), 'reflect', 'edge' .
        	When in 'constant' mode, this op uses a constant value to pad the input tensor.
        	When in 'reflect' mode, uses reflection of the input boundaries to pad the input tensor.
        	When in 'edge' mode, uses input boundaries to pad the input tensor.
        	Default is 'constant'
        pad_value (float32): The value to fill the padded areas in 'constant' mode . Default is 0.0
        data_format (str): An string from: "NHWC", "NCHW". Specify the data format of
                           the input data.
                           Default is  "NCHW"
        name (str, optional) : The default value is None.  Normally there is no need for
                    user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns: a 4-D Tensor padded accordding to paddings and mode and data type is same as input.

    Return Type: Variable


    Examples:
        .. code-block:: text

	      Given that X is a channel of image from input:

	      X = [[1, 2, 3],
		   [4, 5, 6]]

	      Case 0:

		paddings = [0, 1, 2, 3],
		mode = 'constant'
		pad_value = 0

		Out = [[0, 0, 1, 2, 3, 0, 0, 0]
		       [0, 0, 4, 5, 6, 0, 0, 0]
		       [0, 0, 0, 0, 0, 0, 0, 0]]

	      Case 1:

		paddings = [0, 1, 2, 1],
		mode = 'reflect'

		Out = [[3, 2, 1, 2, 3, 2]
		       [6, 5, 4, 5, 6, 5]
		       [3, 2, 1, 2, 3, 2]]

	      Case 2:

		paddings = [0, 1, 2, 1],
		mode = 'edge'

		Out = [[1, 1, 1, 2, 3, 3]
		       [4, 4, 4, 5, 6, 6]
		       [4, 4, 4, 5, 6, 6]]

    Code Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          data = fluid.data(name='data', shape=[None, 3, 32, 32],
                                   dtype='float32')
          result = fluid.layers.pad2d(input=data, paddings=[1, 2, 3, 4],
                                      mode='reflect')
    """

    helper = LayerHelper('pad2d', **locals())

    assert mode in ['reflect', 'edge', 'constant'
                    ], "mode should be one of constant, reflect, edge."

    dtype = helper.input_dtype(input_param_name='input')
    out = helper.create_variable_for_type_inference(dtype)
    inputs = {'X': input}
    attrs = {'mode': mode, 'pad_value': pad_value, 'data_format': data_format}

    if isinstance(paddings, Variable):
        inputs['Paddings'] = paddings
        attrs['paddings'] = []
    else:
        attrs['paddings'] = paddings

    helper.append_op(
        type='pad2d', inputs=inputs, outputs={"Out": out}, attrs=attrs)

    return out


@templatedoc()
def elu(x, alpha=1.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        alpha(${alpha_type}|1.0): ${alpha_comment}
        name(str|None): The default value is None. Normally there is no need for user to set this property. 
                        For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        ${out_type}: ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
         
            input_elu = np.array([[-1,6],[1,15.6]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(input_elu)
                y = fluid.layers.elu(x, alpha=0.2)
                print(y.numpy())
                # [[-0.12642411  6.        ]
                # [ 1.          15.6       ]]
    """
    helper = LayerHelper('elu', **locals())
    if not isinstance(x, Variable):
        raise TypeError(
            "The type of 'x' in elu must be Variable, but received %s" %
            (type(x)))
    if convert_dtype(x.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'x' in elu only support float16 in GPU now.")
    if convert_dtype(x.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'x' in elu must be float16 (only support on GPU), float32 or float64, but received %s."
            % (convert_dtype(x.dtype)))
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='elu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'alpha': alpha})
    return out


@templatedoc()
def relu6(x, threshold=6.0, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        threshold(float, optional): ${threshold_comment}
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[-1,0],[2.5,7.8]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.relu6(x=x1, threshold=6.0)
                print(out1.numpy())
                # [[0.  0. ]
                #  [2.5 6. ]]
    """
    helper = LayerHelper('relu6', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='relu6',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


@templatedoc()
def pow(x, factor=1.0, name=None):
    """
    This is Pow Activation Operator.

    :math:`out = x^{factor}`

    Args:
        x(Variable): A ``Tensor`` or ``LoDTensor`` . The data type is ``float32`` or ``float64``.
        factor(float32|Variable, optional): A scalar with type ``float32`` or a ``Tensor`` with shape [1] and type ``float32``.  The exponential factor of Pow. Default 1.0.
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``x``.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data(name="x", shape=[32,32], dtype="float32")

            # example 1: argument factor is float
            y_1 = fluid.layers.pow(x, factor=2.0)
            # y_1 is x^{2.0}

            # example 2: argument factor is Variable
            factor_tensor = fluid.layers.fill_constant([1], "float32", 3.0)
            y_2 = fluid.layers.pow(x, factor=factor_tensor)
            # y_2 is x^{3.0}
    """
    helper = LayerHelper('pow', **locals())
    inputs = {'X': x}
    attrs = {}
    if isinstance(factor, Variable):
        factor.stop_gradient = True
        inputs['FactorTensor'] = factor
    else:
        attrs['factor'] = factor

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='pow', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


@templatedoc()
def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        scale_a(${scale_a_type}|2.0 / 3.0): ${scale_a_comment}
        scale_b(${scale_b_type}|1.7159): ${scale_b_comment}
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        output(${out_type}): ${out_comment}. 

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            data = fluid.data(name="input", shape=[-1, 3])
            result = fluid.layers.stanh(data,scale_a=0.67, scale_b=1.72)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            x = np.random.random(size=(3, 3)).astype('float32')
            output= exe.run(feed={"input": x},
                         fetch_list=[result])
            print(output)

            #[array([[0.626466  , 0.89842904, 0.7501062 ],
            #       [0.25147712, 0.7484996 , 0.22902708],
            #       [0.62705994, 0.23110689, 0.56902856]], dtype=float32)]

    """
    helper = LayerHelper('stanh', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='stanh',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'scale_a': scale_a,
               'scale_b': scale_b})
    return out


@templatedoc()
def hard_sigmoid(x, slope=0.2, offset=0.5, name=None):
    """
    ${comment}
    Parameters:
        x (${x_type}): ${x_comment}
        slope (float, optional): ${slope_comment}
        offset (float, optional): ${offset_comment}
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`

    Returns:
        ${out_type}: ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.layers.fill_constant(shape=[3, 2], value=0.5, dtype='float32') # [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
            result = fluid.layers.hard_sigmoid(data) # [[0.6, 0.6], [0.6, 0.6], [0.6, 0.6]]
    """
    helper = LayerHelper('hard_sigmoid', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='hard_sigmoid',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'slope': slope,
               'offset': offset})
    return out


@templatedoc()
def swish(x, beta=1.0, name=None):
    """
    Elementwise swish activation function. See `Searching for Activation Functions <https://arxiv.org/abs/1710.05941>`_ for more details.
    
    Equation:

    .. math::
        out = \\frac{x}{1 + e^{- beta * x}}
    
    Args:
        x(Variable): Tensor or LoDTensor, dtype: float32 or float64, the input of swish activation.
        
        beta(float): Constant beta of swish operator, default 1.0.
        
        name(str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:

        Variable: Output of the swish activation, Tensor or LoDTensor, with the same dtype and shape with the input x.

    Examples:

        .. code-block:: python
            
            # declarative mode
            import numpy as np
            from paddle import fluid
            
            x = fluid.data(name="x", shape=(-1, 3), dtype="float32")
            y = fluid.layers.swish(x, beta=2.0)
            
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            start = fluid.default_startup_program()
            main = fluid.default_main_program()
            
            data = np.random.randn(2, 3).astype("float32")
            exe.run(start)
            y_np, = exe.run(main, feed={"x": data}, fetch_list=[y])
            
            data
            # array([[-1.1239197 ,  1.3391294 ,  0.03921051],
            #        [ 1.1970421 ,  0.02440812,  1.2055548 ]], dtype=float32)
            y_np
            # array([[-0.2756806 ,  1.0610548 ,  0.01998957],
            #        [ 0.9193261 ,  0.01235299,  0.9276883 ]], dtype=float32)


        .. code-block:: python

            # imperative mode
            import numpy as np
            from paddle import fluid
            import paddle.fluid.dygraph as dg
            
            data = np.random.randn(2, 3).astype("float32")
            place = fluid.CPUPlace()
            with dg.guard(place) as g:
                x = dg.to_variable(data)
                y = fluid.layers.swish(x)
                y_np = y.numpy()
            data
            # array([[-0.0816701 ,  1.1603649 , -0.88325626],
            #        [ 0.7522361 ,  1.0978601 ,  0.12987892]], dtype=float32)
            y_np
            # array([[-0.03916847,  0.8835007 , -0.25835553],
            #        [ 0.51126915,  0.82324016,  0.06915068]], dtype=float32)
    """
    helper = LayerHelper('swish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='swish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'slope': beta})
    return out


def prelu(x, mode, param_attr=None, name=None):
    """
    Equation:

    .. math::
        y = \max(0, x) + \\alpha * \min(0, x)

    There are three modes for the activation:

    .. code-block:: text

        all: All elements share same alpha.
        channel: Elements in same channel share same alpha.
        element: All elements do not share alpha. Each element has its own alpha.

    Args:
        x (Variable): The input Tensor or LoDTensor with data type float32.
        mode (str): The mode for weight sharing. 
        param_attr(ParamAttr|None): The parameter attribute for the learnable
          weight (alpha), it can be create by ParamAttr. None by default.
          For detailed information, please refer to :ref:`api_fluid_ParamAttr`.
        name(str|None): For detailed information, please refer 
          to :ref:`api_guide_Name`. Usually name is no need to set and 
          None by default. 

    Returns:
        Variable:

        output(Variable): The tensor or LoDTensor with the same shape as input.
        The data type is float32.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            from paddle.fluid.param_attr import ParamAttr
            x = fluid.data(name="x", shape=[None,5,10,10], dtype="float32")
            mode = 'channel'
            output = fluid.layers.prelu(
                     x,mode,param_attr=ParamAttr(name='alpha'))

    """
    helper = LayerHelper('prelu', **locals())
    if mode not in ['all', 'channel', 'element']:
        raise ValueError('mode should be one of all, channel, element.')
    alpha_shape = [1]
    if mode == 'channel':
        alpha_shape = [1, x.shape[1], 1, 1]
    elif mode == 'element':
        alpha_shape = x.shape
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype='float32',
        is_bias=False,
        default_initializer=Constant(1.0))
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x,
                'Alpha': alpha},
        attrs={"mode": mode},
        outputs={"Out": out})
    return out


@templatedoc()
def brelu(x, t_min=0.0, t_max=24.0, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        t_min(${t_min_type}|0.0): ${t_min_comment}
        t_max(${t_max_type}|24.0): ${t_max_comment}
        name(str|None): The default value is None. Normally there is no need for user to set this property. 
                        For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        ${out_type}: ${out_comment}

    Examples:

    .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            
            input_brelu = np.array([[-1,6],[1,15.6]])
            with fluid.dygraph.guard():
                x = fluid.dygraph.to_variable(input_brelu)
                y = fluid.layers.brelu(x, t_min=1.0, t_max=10.0)
                print(y.numpy())
                #[[ 1.  6.]
                #[ 1. 10.]] 
    """
    helper = LayerHelper('brelu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='brelu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'t_min': t_min,
               't_max': t_max})
    return out


@templatedoc()
def leaky_relu(x, alpha=0.02, name=None):
    """
    ${comment}
    Args:
        x(${x_type}): ${x_comment}
        alpha(${alpha_type}|0.02): ${alpha_comment}
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        output(${out_type}): ${out_comment}

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Graph Organizing
            x = fluid.layers.data(name="x", shape=[2], dtype="float32")
            res = fluid.layers.leaky_relu(x, alpha=0.1)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            x_i = np.array([[-1, 2], [3, -4]]).astype(np.float32)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
            print(res_val) # [[-0.1, 2], [3, -0.4]]
    """
    helper = LayerHelper('leaky_relu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='leaky_relu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'alpha': alpha})
    return out


def soft_relu(x, threshold=40.0, name=None):
    """
    SoftRelu Activation Operator.

    $out = \ln(1 + \exp(\max(\min(x, threshold), -threshold)))$

    Args:
        x(Variable): Input of soft_relu operator. Data type can be float32, float64.
        threshold(float, optional): The threshold value of soft_relu, default value being 40.0.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable(Tensor|LoDTensor)): Output of soft_relu operator, shape and LoD same as input.

    Examples:

        .. code-block:: python 
 
            import paddle.fluid as fluid
            import numpy as np

            inputs = fluid.layers.data(name="x", shape=[2, 2], dtype="float32")
            output = fluid.layers.soft_relu(inputs, threshold=20.0)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img = np.array([[0, 1],[2, 3]]).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
            print(res) # [array([[0.6931472, 1.3132616], [2.126928 , 3.0485873]], dtype=float32)]
    """
    helper = LayerHelper('soft_relu', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='soft_relu',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold})
    return out


def flatten(x, axis=1, name=None):
    """
    **Flatten op**

    Flatten the input tensor into a 2D matrix.

    For Example:

    .. code-block:: text

        Case 1:

          Given
            X.shape = (3, 100, 100, 4)

          and
            axis = 2

          We get:
            Out.shape = (3 * 100, 4 * 100)

        Case 2:

          Given
            X.shape = (3, 100, 100, 4)

          and
            axis = 0

          We get:
            Out.shape = (1, 3 * 100 * 100 * 4)

    Args:
        x (Variable): A tensor of rank >= axis. A tensor with type float32,
                      float64, int8, int32, int64.
        axis (int): Indicate up to which input dimensions (exclusive) should
                    be flattened to the outer dimension of the output.
                    The value for axis must be in the range [0, R], where R
                    is the rank of the input tensor. Default: 1.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.

    Returns:
        Variable: A 2D tensor with the contents of the input tensor, with input \
                  dimensions up to axis flattened to the outer dimension of \
                  the output and remaining input dimensions flattened into the \
                  inner dimension of the output. A Tensor with type same as input x.

    Raises:
        ValueError: If x is not a variable.
        ValueError: If axis is not in range [0, rank(x)].

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name="x", shape=[4, 4, 3], dtype="float32")
            # x shape is [4, 4, 3]
            out = fluid.layers.flatten(x=x, axis=2)
            # out shape is [16, 3]
    """
    helper = LayerHelper('flatten', **locals())

    if not (isinstance(x, Variable)):
        raise ValueError("The input x should be a Variable")

    if not (isinstance(axis, int)) or axis > len(x.shape) or axis < 0:
        raise ValueError("The axis should be a int, and in range [0, rank(x)]")

    out = helper.create_variable_for_type_inference(x.dtype)
    x_shape = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(
        type='flatten2',
        inputs={"X": x},
        outputs={'Out': out,
                 'XShape': x_shape},
        attrs={"axis": axis})
    return out


def sequence_enumerate(input, win_size, pad_value=0, name=None):
    """
    Generate a new sequence for the input index sequence with \
        shape ``[d_1, win_size]``, which enumerates all the \
        sub-sequences with length ``win_size`` of the input with \
        shape ``[d_1, 1]``, and padded by ``pad_value`` if necessary in generation.

    Please note that the `input` must be LodTensor.

    .. code-block:: text

        Input x:
            x.lod = [[0, 3, 5]]
            x.data = [[1], [2], [3], [4], [5]]
            x.dims = [5, 1]

        Attrs:
            win_size = 2
            pad_value = 0

        Output:
            out.lod = [[0, 3, 5]]
            out.data = [[1, 2], [2, 3], [3, 0], [4, 5], [5, 0]]
            out.dims = [5, 2]


    Args:
        input (Variable): The input variable which is a index sequence, \
            which should be a LodTensor with shape ``[d_1, 1]`` and 1-level lod info. \
            The data type should be float32, float64, int8, int32 or int64.
        win_size (int): The window size for enumerating all sub-sequences.
        pad_value (int, optional): The padding value, default 0.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The enumerate sequence variable which is a LoDTensor with \
            shape ``[d_1, win_size]`` and 1-level lod info. \
            The data type is same as ``input``.

    Return Type: Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data(name='x', shape=[-1, 1], dtype='int32', lod_level=1)
            out = fluid.layers.sequence_enumerate(input=x, win_size=3, pad_value=0)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper('sequence_enumerate', **locals())
    out = helper.create_variable_for_type_inference(
        helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='sequence_enumerate',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'win_size': win_size,
               'pad_value': pad_value})
    return out


def sequence_mask(x, maxlen=None, dtype='int64', name=None):
    """
    **SequenceMask Layer**

    This layer outputs a mask according to the input :code:`x` and
    :code:`maxlen` with data type of :code:`dtype`.

    Supposing :code:`x` is a Tensor with shape [d_1, d_2, ..., d_n], the
    :code:`y` is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

    .. math::

        y(i_1, i_2,..., i_n, j) = (j < x(i_1, i_2,..., i_n))

    .. code-block:: text

        Case:

        Consider input:
            x = [3, 1, 1, 0]    max_len = 4

        then we get out:
            mask = [[1, 1, 1, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0]]

    Args:
        x (Variable): Input tensor of sequence_mask layer, \
            whose elements are integers less than :code:`maxlen`. \
            Tensor or LodTensor with shape [d_1, d_2, ..., d_n].
        maxlen (int, optional): Maximum length of the sequence. If :code:`maxlen` \
                           is None, it would be replace with :math:`max(x)`.
        dtype (np.dtype|core.VarDesc.VarType|str, optional): Data type of the output, \
             ``int64`` by default.
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The output sequence mask. Tensor or LodTensor with shape [d_1, d_2, ..., d_n, maxlen] \
            and data type of :code:`dtype`. The data type should be float32, float64, int8, \
            int32 or int64.

    Return Type: Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            x = fluid.data(name='x', shape=[10], dtype='float32', lod_level=1)
            mask = layers.sequence_mask(x=x)

    """
    helper = LayerHelper('sequence_mask', **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=dtype, name=name)

    inputs = {'X': [x]}
    attrs = {'out_dtype': out.dtype}
    if maxlen is not None:
        if isinstance(maxlen, Variable):
            inputs['MaxLenTensor'] = maxlen
        else:
            attrs['maxlen'] = maxlen

    helper.append_op(
        type='sequence_mask', inputs=inputs, outputs={'Y': out}, attrs=attrs)

    out.stop_gradient = True
    return out


def stack(x, axis=0):
    """

    This OP stacks all the inputs :code:`x` along axis.

    .. code-block:: text

        Case 1:

          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]

          Attrs:
            axis = 0

          Output:
            Out.dims = [3, 1, 2]
            Out.data =[ [ [1.0, 2.0] ],
                        [ [3.0, 4.0] ],
                        [ [5.0, 6.0] ] ]


        Case 2:


          Input:
            x[0].shape = [1, 2]
            x[0].data = [ [1.0 , 2.0 ] ]
            x[1].shape = [1, 2]
            x[1].data = [ [3.0 , 4.0 ] ]
            x[2].shape = [1, 2]
            x[2].data = [ [5.0 , 6.0 ] ]


          Attrs:
            axis = 1 or axis = -2

          Output:
            Out.shape = [1, 3, 2]
            Out.data =[ [ [1.0, 2.0]
                          [3.0, 4.0]
                          [5.0, 6.0] ] ]


    Args:
        x (Variable|list(Variable)): Input :code:`x` can be a single Tensor, a :code:`list` of Tensors.
                                     If :code:`x` is a :code:`list`, the shapes of all these Tensors
                                     must be the same. Supposing input is N dims
                                     Tensors :math:`[d_0, d_1, ..., d_{n-1}]`, the output is N+1 dims
                                     Tensor :math:`[d_0, d_1, d_{axis-1}, len(x), d_{axis}, ..., d_{n-1}]`.
                                     Support data types: float32, float64, int32, int64.
        axis (int, optional): The axis along which all inputs are stacked. ``axis`` range is :math:`[-(R+1), R+1)`.
                              R is the first tensor of inputs. If ``axis`` < 0, :math:`axis=axis+rank(x[0])+1`.
                              The default value of axis is 0.

    Returns:
        Variable: The stacked Tensor, has same data type with input Tensors. Output dim is :math:`rank(x[0])+1`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            # set batch size=None
            x1 = fluid.data(name='x1', shape=[None, 1, 2], dtype='int32')
            x2 = fluid.data(name='x2', shape=[None, 1, 2], dtype='int32')
            # stack Tensor list
            data = layers.stack([x1,x2]) # stack according to axis 0, data.shape=[2, None, 1, 2]

            data = layers.stack([x1,x2], axis=1) # stack according to axis 1, data.shape=[None, 2, 1, 2]

            # stack single Tensor
            data = layers.stack(x1)  # stack according to axis 0, data.shape=[1, None, 1, 2]

    """

    helper = LayerHelper('stack', **locals())
    axis = 0 if axis is None else axis

    if not isinstance(x, list) and not isinstance(x, tuple):
        x = [x]

    out = helper.create_variable_for_type_inference(x[0].dtype)
    helper.append_op(
        type='stack', inputs={'X': x}, outputs={'Y': out},
        attrs={'axis': axis})

    return out


@templatedoc(op_type="filter_by_instag")
def filter_by_instag(ins, ins_tag, filter_tag, is_lod):
    """
    **Filter By Instag Layer**
   
    This function filter a batch of ins by instag, 
    There are multiple ins, and every ins belongs to some tags. 
    We can specify some tags we want. So the ins which belongs to that tags
    remains in the output, and others removed.
 
    For example, one batch has 4 ins. Every ins has its tag list. 
     
       | Ins   |   Ins_Tag |
       |:-----:|:------:|
       |  0    |   0, 1 |
       |  1    |   1, 3 |
       |  2    |   0, 3 |
       |  3    |   2, 6 |

    And Lod is [1,1,1,1]

    And the filter tags [1]

    From the definition above, ins which has tag 1 can pass the filter
    So Ins 0 and Ins 1 can pass and be seen in the output,
    Ins 2 and 3 cannot pass because they do not has tag 1.

    Actually, if is_lod is false, it is normal tensor that equals to 
    lod_tensor with all 1, similar to the example above.

    Args:
        ins (Variable): Input Variable (LoDTensor), usually it is 2D tensor
                        And first dimension can have lod info or not.
        ins_tag (Variable): Input Variable (LoDTensor), usually it is 1D list
                        And split them by lod info
        filter_tag (Variable): Input Variable (1D Tensor/List), usually it is 
                        list that holds the tags.
        is_lod (Bool): Boolean value to indicate ins is lod tensor or not.

    Returns:
        Variable: filtered ins (LoDTensor) and loss weight (Tensor)

    Examples:
        .. code-block:: python

          import paddle.fluid.layers as layers
          ins = layers.data(name='Ins', shape=[-1,32], lod_level=0, dtype='float64')
          ins_tag = layers.data(name='Ins_tag', shape=[-1,16], lod_level=0, dtype='int64')
          filter_tag = layers.data(name='Filter_tag', shape=[-1,16], dtype='int64')
          out, loss_weight = layers.filter_by_instag(ins,  ins_tag,  filter_tag, True)
        		
    """
    helper = LayerHelper('filter_by_instag', **locals())

    out = helper.create_variable_for_type_inference(dtype=ins.dtype)
    loss_weight = helper.create_variable_for_type_inference(dtype=np.float64)
    mmap = helper.create_variable_for_type_inference(dtype=ins_tag.dtype)
    helper.append_op(
        type='filter_by_instag',
        inputs={'Ins': ins,
                'Ins_tag': ins_tag,
                'Filter_tag': filter_tag},
        outputs={'Out': out,
                 'LossWeight': loss_weight,
                 'IndexMap': mmap},
        attrs={'is_lod': is_lod})

    return [out, loss_weight]


def unstack(x, axis=0, num=None):
    """
    **UnStack Layer**

    This layer unstacks input Tensor :code:`x` into several Tensors along :code:`axis`.

    If :code:`axis` < 0, it would be replaced with :code:`axis+rank(x)`.
    If :code:`num` is None, it would be inferred from :code:`x.shape[axis]`,
    and if :code:`x.shape[axis]` <= 0 or is unknown, :code:`ValueError` is
    raised.

    Args:
        x (Variable): Input Tensor. It is a N-D Tensors of data types float32, float64, int32, int64.
        axis (int): The axis along which the input is unstacked.
        num (int|None): The number of output variables.

    Returns:
        list(Variable): The unstacked Tensors list. The list elements are N-D Tensors of data types float32, float64, int32, int64.

    Raises:
        ValueError: If x.shape[axis] <= 0 or axis is not in range [-D, D).

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[2, 3, 5], dtype='float32')  # create a tensor with shape=[2, 3, 5]
            y = fluid.layers.unstack(x, axis=1)  # unstack with second axis, which results 3 tensors with shape=[2, 5]

    """
    helper = LayerHelper('unstack', **locals())
    if num is None:
        if axis is None or x.shape[axis] <= 0:
            raise ValueError('unknown unstack number')
        else:
            num = x.shape[axis]

    outs = []
    for _ in range(num):
        outs.append(helper.create_variable_for_type_inference(x.dtype))

    helper.append_op(
        type='unstack',
        inputs={'X': [x]},
        outputs={'Y': outs},
        attrs={'axis': axis,
               'num': num})
    return outs


def expand(x, expand_times, name=None):
    """
    This operation tiles ``x`` multiple times according to the parameter ``expand_times``.
    The times number for each dimension of ``x`` is set by the parameter ``expand_times``.
    The rank of ``x`` should be less than or equal to 6. Please note that size of ``expand_times`` must be the same
    with X's rank. Following is a using case:


    .. code-block:: text

        Input(X) is a 3-D tensor with shape [2, 3, 1]:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        Attr(expand_times):  [1, 2, 2]

        Output(Out) is a 3-D tensor with shape [2, 6, 2]:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]

    Args:
        x (Variable): A ``Tensor`` or ``LoDTensor`` with dimension in [1, 6]. The data type is ``bool``, ``float32``, ``float64`` or ``int32`` .
        expand_times (list|tuple|Variable): The data type is ``int32`` . If ``expand_times`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``expand_times`` is an Variable, it should be an 1-D Tensor.
                Expand times number for each dimension of ``x`` .
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``x``. After expanding, size of each dimension of output is equal to the size of the corresponding dimension of ``x`` multiplying the corresponding value given by ``expand_times`` .

    Raises:
        TypeError: The type of ``expand_times`` must be list, tuple or Variable.
        ValueError: The elements of ``expand_times`` cannot be negative.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # example 1:
            data_1 = fluid.layers.fill_constant(shape=[2, 3, 1], dtype='int32', value=0)
            expanded_1 = fluid.layers.expand(data_1, expand_times=[1, 2, 2])
            # the shape of expanded_1 is [2, 6, 2].

            # example 2:
            data_2 = fluid.layers.fill_constant(shape=[12, 14], dtype="int32", value=3)
            expand_times = fluid.layers.fill_constant(shape=[2], dtype="int32", value=4)
            expanded_2 = fluid.layers.expand(data_2, expand_times=expand_times)
            # the shape of expanded_2 is [48, 56].
    """
    if not isinstance(x, Variable):
        raise TypeError(
            "The type of 'input' in reduce_sum must be Variable, but received %s"
            % (type(x)))
    if not isinstance(expand_times, (list, tuple, Variable)):
        raise ValueError(
            "Input expand_times must be an Variable, python list or tuple.")
    if convert_dtype(
            x.dtype) not in ['bool', 'float32', 'float64', 'int32', 'int64']:
        raise TypeError(
            "The data type of input  in expand  must be one of bool float32, float64, int32 or int64, but received %s."
            % (convert_dtype(x.dtype)))
    if convert_dtype(x.dtype) == 'bool' and x.stop_gradient == True:
        raise ValueError(
            "expand op bool date type must set the stop_gradient to be False")

    helper = LayerHelper('expand', input=x, **locals())
    inputs = {"X": x}
    attrs = {}

    def contain_var(expand_times):
        for ele in expand_times:
            if isinstance(ele, Variable):
                return True
        return False

    def get_attr_expand_times(list_expand_times):
        attrs_expand_times = []
        for idx, times in enumerate(list_expand_times):
            if isinstance(times, Variable):
                attrs_expand_times.append(-1)
            else:
                attrs_expand_times.append(times)
                assert times > 0, (
                    "Each element given in expand_times must not be negtive.")
        return attrs_expand_times

    def get_new_expand_times_tensor(list_expand_times):
        new_expand_times_tensor = []
        for ele in list_expand_times:
            if isinstance(ele, Variable):
                ele.stop_gradient = True
                new_expand_times_tensor.append(ele)
            else:
                assert (isinstance(ele, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', ele, force_cpu=True, out=temp_out)
                new_expand_times_tensor.append(temp_out)
        return new_expand_times_tensor

    if in_dygraph_mode():
        inputs = {'X': x}
        attrs = {'expand_times': expand_times}
    else:
        if isinstance(expand_times, Variable):
            expand_times.stop_gradient = True
            inputs['ExpandTimes'] = expand_times
        elif isinstance(expand_times, (list, tuple)):
            attrs['expand_times'] = get_attr_expand_times(expand_times)
            if contain_var(expand_times):
                inputs['expand_times_tensor'] = get_new_expand_times_tensor(
                    expand_times)

    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='expand', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return out


def expand_as(x, target_tensor, name=None):
    """
    expand_as operator tiles to the input by given expand tensor. You should set expand tensor
    for each dimension by providing tensor 'target_tensor'. The rank of X
    should be in [1, 6]. Please note that size of 'target_tensor' must be the same
    with X's rank. Following is a using case:


    .. code-block:: text

        Input(X) is a 3-D tensor with shape [2, 3, 1]:

                [
                   [[1], [2], [3]],
                   [[4], [5], [6]]
                ]

        target_tensor's shape:  [2, 6, 2] 

        Output(Out) is a 3-D tensor with shape [2, 6, 2]:

                [
                    [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
                    [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
                ]
                

    Args:
        x (Variable): A Tensor with dtype float64, float32, int32.
        A tensor with rank in [1, 6].
        target_tensor (Variable): A Tensor with dtype float64, float32, int32.
        target_tensor for expanding to Input(X). Only use target_tensor'shape.

    Returns:
        Variable: A Tensor with dtype float64, float32, int32. 
        After expanding, size of each dimension of Output(Out) is equal to the size 
        of the corresponding dimension of target_tensor multiplying the corresponding
        value given by target_tensor.


    Examples:
        .. code-block:: python
          
        import paddle.fluid as fluid
        import numpy as np

        data = fluid.layers.data(name="data", shape=[-1,10], dtype='float64')
        target_tensor = fluid.layers.data(
          name="target_tensor", shape=[-1,20], dtype='float64')
        result = fluid.layers.expand_as(x=data, target_tensor=target_tensor) 
        use_cuda = False
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        x = np.random.rand(3,10)
        y = np.random.rand(3,20)
        output= exe.run(feed={"data":x,"target_tensor":y},fetch_list=[result.name])
        print(output[0].shape)
        #(3,20)

    """

    helper = LayerHelper('expand_as', input=x, **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    inputs = {'X': x, 'target_tensor': target_tensor}
    helper.append_op(type='expand_as', inputs=inputs, outputs={'Out': out})
    return out


from paddle.fluid.framework import convert_np_dtype_to_dtype_


@templatedoc()
def uniform_random_batch_size_like(input,
                                   shape,
                                   dtype='float32',
                                   input_dim_idx=0,
                                   output_dim_idx=0,
                                   min=-1.0,
                                   max=1.0,
                                   seed=0):
    """
    This OP initializes a variable with random values sampled from a
    uniform distribution in the range [min, max). The input_dim_idx used to get the input dimension value which will be used to resize the output dimension.

    .. code-block:: text

        *Case 1:

            Given:
                input =[[0.946741  , 0.1357001 , 0.38086128]]    # input.shape=[1,3]
                shape=[2,4]

            result.shape[output_dim_idx] = input.shape[input_dim_idx],
            output_dim_idx = 0, 
            input_dim_idx = 0,
            result.shape[0] = input.shape[0], 
            then:
                result=[[ 0.3443427 , -0.23056602,  0.3477049 ,  0.06139076]]    # result.shape=[1,4]
            
       *Case 2:
           
           Given:
               input =[[0.946741  , 0.1357001 , 0.38086128]]     # input.shape=[1,3]
               shape=[2,4]
               input_dim_idx=1
               output_dim_idx=1
         
           result.shape[output_dim_idx] = input.shape[input_dim_idx],
           output_dim_idx = 1, 
           input_dim_idx = 1,
           result.shape[1] = input.shape[1], 
           then:
               result=[[-0.23133647, -0.84195036,  0.21441269],
                       [-0.08774924,  0.25605237, -0.09403259]]    # result.shape=[2,3]
    Args:
        input (Variable): A Tensor. Supported data types: float32, float64.
        shape (tuple|list): A python list or python tuple. The shape of the output Tensor, the data type is int.
        input_dim_idx (int, optional): An index used to get the input dimension value which will be used to resize the output dimension. Default  0. 
        output_dim_idx (int, optional): An index used to indicate the specific dimension that will be replaced by corresponding input dimension value. Default 0.
        min (float, optional): The lower bound on the range of random values to generate, the min is included in the range. Default -1.0.
        max (float, optional): The upper bound on the range of random values to generate, the max is excluded in the range. Default 1.0.
        seed (int, optional):  Random seed used for generating samples. 0 means use a seed generated by the system.Note that if seed is not 0, this operator will always generate the same random numbers every time.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The data type of output Tensor. Supported data types: float32, float64. Default float32.
    Returns:
        Variable: A Tensor of the specified shape filled with uniform_random values. The shape of the Tensor is determined by the shape parameter and the specified dimension of the input Tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            
            # example 1: 
            input = fluid.data(name="input", shape=[1, 3], dtype='float32')
            out_1 = fluid.layers.uniform_random_batch_size_like(input, [2, 4]) # out_1.shape=[1, 4]

            # example 2: 
            out_2 = fluid.layers.uniform_random_batch_size_like(input, [2, 4], input_dim_idx=1, output_dim_idx=1) # out_2.shape=[2, 3]

            
    """

    helper = LayerHelper('uniform_random_batch_size_like', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='uniform_random_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'min': min,
            'max': max,
            'seed': seed,
            'dtype': c_dtype
        })

    return out


@templatedoc()
def gaussian_random(shape, mean=0.0, std=1.0, seed=0, dtype='float32'):
    """
    Generate a random tensor whose data is drawn from a Gaussian distribution.

    Args:
        shape (Tuple[int] | List[int]): Shape of the generated random tensor.
        
        mean (float): Mean of the random tensor, defaults to 0.0.
            
        std (float): Standard deviation of the random tensor, defaults to 1.0.
        
        seed (int): ${seed_comment}
        
        dtype(np.dtype | core.VarDesc.VarType | str): Output data type, float32 or float64.

    Returns:
        Variable: Random tensor whose data is drawn from a Gaussian distribution, dtype: flaot32 or float64 as specified.

    Examples:
       .. code-block:: python
       
           # declarative mode 
           import numpy as np
           from paddle import fluid
   
           x = fluid.layers.gaussian_random((2, 3), std=2., seed=10)
   
           place = fluid.CPUPlace()
           exe = fluid.Executor(place)
           start = fluid.default_startup_program()
           main = fluid.default_main_program()
   
           exe.run(start)
           x_np, = exe.run(main, feed={}, fetch_list=[x])

           x_np
           # array([[2.3060477, 2.676496 , 3.9911983],
           #        [0.9990833, 2.8675377, 2.2279181]], dtype=float32)

       .. code-block:: python

           # imperative mode
           import numpy as np
           from paddle import fluid
           import paddle.fluid.dygraph as dg
    
           place = fluid.CPUPlace()
           with dg.guard(place) as g:
               x = fluid.layers.gaussian_random((2, 4), mean=2., dtype="float32", seed=10)
               x_np = x.numpy()       
           x_np
           # array([[2.3060477 , 2.676496  , 3.9911983 , 0.9990833 ],
           #        [2.8675377 , 2.2279181 , 0.79029655, 2.8447366 ]], dtype=float32)
    """

    helper = LayerHelper('gaussian_random', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='gaussian_random',
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'mean': mean,
            'std': std,
            'seed': seed,
            'dtype': c_dtype,
            'use_mkldnn': False
        })

    return out


@templatedoc()
def sampling_id(x, min=0.0, max=1.0, seed=0, dtype='float32'):
    """
    This op is used for sampling id from multinomial distribution from the input, sampling one id for one sample.

    Parameters:
        x (Variable): 2-D tensor, [batch_size, input_feature_dimensions]
        min (Float): minimum , default 0.0.
        max (Float): maximum, default 1.0.
        seed (Float): Random seed, default 0. if seed is not 0, will generate same number every time. 
        dtype(np.dtype|core.VarDesc.VarType|str): The type of output data : float32, float_16, int etc

    Returns:
        Variable: sampling tensor.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(
                name="X",
                shape=[13, 11],
                dtype='float32')

            out = fluid.layers.sampling_id(x)
    """

    helper = LayerHelper('sampling_id', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='sampling_id',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'min': min,
               'max': max,
               'seed': seed})

    return out


@templatedoc()
def gaussian_random_batch_size_like(input,
                                    shape,
                                    input_dim_idx=0,
                                    output_dim_idx=0,
                                    mean=0.0,
                                    std=1.0,
                                    seed=0,
                                    dtype='float32'):
    """
    ${comment}

    Args:
        input (Variable): ${input_comment}
        shape (tuple|list): ${shape_comment}
        input_dim_idx (int): ${input_dim_idx_comment}
        output_dim_idx (int): ${output_dim_idx_comment}
        mean (float): ${mean_comment}
        std (float): ${std_comment}
        seed (int): ${seed_comment}
        dtype(np.dtype|core.VarDesc.VarType|str): The type of output data, float32 or float_64.

    Returns:
        out (Variable): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(name="input", shape=[13, 11], dtype='float32')

            out = fluid.layers.gaussian_random_batch_size_like(
                input, shape=[-1, 11], mean=1.0, std=2.0)
    """

    helper = LayerHelper('gaussian_random_batch_size_like', **locals())
    out = helper.create_variable_for_type_inference(dtype)
    c_dtype = convert_np_dtype_to_dtype_(dtype)
    helper.append_op(
        type='gaussian_random_batch_size_like',
        inputs={'Input': input},
        outputs={'Out': out},
        attrs={
            'shape': shape,
            'input_dim_idx': input_dim_idx,
            'output_dim_idx': output_dim_idx,
            'mean': mean,
            'std': std,
            'seed': seed,
            'dtype': c_dtype
        })

    return out


@templatedoc()
def sum(x):
    """
    ${comment}
    
    Case 1:
    ::
        Input:
            Input. Shape = [2, 3]
            Input = [[1, 2, 3],
                     [4, 5, 6]]

        Output:
            The output. Shape = [2, 3]
            Output = [[1, 2, 3],
                      [4, 5, 6]]

    Case 2:
    ::
        Input:
            First input:
            Input1. Shape = [2, 3]
            Input1 = [[1, 2, 3],
                      [4, 5, 6]]

        The second input:
            Input2. Shape = [2, 3]
            Input2 = [[7, 8, 9],
                      [10, 11, 12]]

        Output:
            The output. Shape = [2, 3]
            Output = [[8, 10, 12],
                      [14, 16, 18]]

    Args:
        x (Variable|list(Variable)): ${x_comment}

    Returns:
        Variable: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input0 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=5)
            input1 = fluid.layers.fill_constant(shape=[2, 3], dtype='int64', value=3)
            sum = fluid.layers.sum([input0, input1])

            # You can print out 'sum' via executor.
            out = fluid.layers.Print(sum, message="the sum of input0 and input1: ")
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_main_program())

            # The printed result is:
            # 1570701754	the sum of input0 and input1: 	The place is:CPUPlace
            # Tensor[sum_0.tmp_0]
            #    shape: [2,3,]
            #    dtype: l
            #    data: 8,8,8,8,8,8,

            # the sum of input0 and input1 is 2-D Tensor with shape [2,3].
            # dtype is the corresponding C++ data type, which may vary in different environments.
            # Eg: if the data type of tensor is int64, then the corresponding C++ data type is int64_t, 
            #       so the dtype value is typeid(int64_t).Name(), which is 'x' on MacOS, 'l' on Linux, 
            #       and '__int64' on Windows. They both represent 64-bit integer variables.
    """

    helper = LayerHelper('sum', **locals())
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('x'))
    helper.append_op(
        type='sum',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'use_mkldnn': False})

    return out


@templatedoc()
def slice(input, axes, starts, ends):
    """
    This operator produces a slice of ``input`` along multiple axes. Similar to numpy:
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
    end dimension for each axis in the list of axes and Slice uses this information
    to slice the input data tensor. If a negative value is passed to
    ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
    axis :math:`i-1` (here 0 is the initial position).
    If the value passed to ``starts`` or ``ends`` is greater than n
    (the number of elements in this dimension), it represents n.
    For slicing to the end of a dimension with unknown size, it is recommended
    to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` and ``ends``.
    Following examples will explain how slice works:

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
            Then:
                result = [ [5, 6, 7], ]

        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [-1, 1000]       # -1 denotes the reverse 0th position of dimension 0.
            Then:
                result = [ [2, 3, 4], ] # result = data[0:1, 1:4]
    Args:
        input (Variable): A ``Tensor`` or ``LoDTensor`` . The data type is ``float16``, ``float32``, ``float64``, ``int32`` or ``int64``.
        axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to.
                            It's optional. If it is not provides, it will be treated as :math:`[0,1,...,len(starts)-1]`.
        starts (list|tuple|Variable): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``starts`` is an Variable, it should be an 1-D Tensor.
                It represents starting indices of corresponding axis in ``axes``.
        ends (list|tuple|Variable): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``ends`` is an Variable, it should be an 1-D Tensor .
                It represents ending indices of corresponding axis in ``axes``.

    Returns:
        Variable:  A ``Tensor`` or ``LoDTensor``. The data type is same as ``input``.

    Raises:
        TypeError: The type of ``starts`` must be list, tuple or Variable.
        TypeError: The type of ``ends`` must be list, tuple or Variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.data(
                name="input", shape=[4, 5, 6], dtype='float32')

            # example 1:
            # attr starts is a list which doesn't contain tensor Variable.
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            sliced_1 = fluid.layers.slice(input, axes=axes, starts=starts, ends=ends)
            # sliced_1 is input[0:3, 0:2, 2:4].

            # example 2:
            # attr starts is a list which contain tensor Variable.
            minus_3 = fluid.layers.fill_constant([1], "int32", -3)
            sliced_2 = fluid.layers.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)
            # sliced_2 is input[0:3, 0:2, 2:4].
    """

    if not isinstance(starts, (list, tuple, Variable)):
        raise ValueError(
            "Input starts must be an Variable, python list or tuple.")
    if not isinstance(ends, (list, tuple, Variable)):
        raise ValueError(
            "Input ends must be an Variable, python list or tuple.")

    helper = LayerHelper('slice', **locals())

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_list_tensor(old_list):
        new_list_tensor = []
        for dim in old_list:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_list_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_list_tensor.append(temp_out)
        return new_list_tensor

    inputs = {'Input': input}
    attrs = {'axes': axes}
    infer_flags = list(1 for i in range(len(axes)))

    if in_dygraph_mode():
        inputs = {'Input': input}
        attrs = {
            'axes': axes,
            'starts': starts,
            'ends': ends,
            'infer_flags': infer_flags
        }
    else:
        # starts
        if isinstance(starts, Variable):
            starts.stop_gradient = True
            inputs['StartsTensor'] = starts
            infer_flags = list(-1 for i in range(len(axes)))
        elif isinstance(starts, (list, tuple)):
            attrs['starts'] = []
            if not contain_var(starts):
                attrs['starts'] = starts
            else:
                inputs['StartsTensorList'] = get_new_list_tensor(starts)
                for i, dim in enumerate(starts):
                    if isinstance(dim, Variable):
                        attrs['starts'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['starts'].append(dim)

        # ends
        if isinstance(ends, Variable):
            ends.stop_gradient = True
            inputs['EndsTensor'] = ends
            infer_flags = list(-1 for i in range(len(axes)))
        elif isinstance(ends, (list, tuple)):
            attrs['ends'] = []
            if not contain_var(ends):
                attrs['ends'] = ends
            else:
                inputs['EndsTensorList'] = get_new_list_tensor(ends)
                for i, dim in enumerate(ends):
                    if isinstance(dim, Variable):
                        attrs['ends'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['ends'].append(dim)
        # infer_flags
        attrs['infer_flags'] = infer_flags
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('input'))
    helper.append_op(
        type='slice', inputs=inputs, attrs=attrs, outputs={'Out': out})

    return out


@templatedoc()
def strided_slice(input, axes, starts, ends, strides):
    """
    This operator produces a slice of ``input`` along multiple axes. Similar to numpy:
    https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    Slice uses ``axes``, ``starts`` and ``ends`` attributes to specify the start and
    end dimension for each axis in the list of axes and Slice uses this information
    to slice the input data tensor. If a negative value is passed to
    ``starts`` or ``ends`` such as :math:`-i`,  it represents the reverse position of the
    axis :math:`i-1` th(here 0 is the initial position). The ``strides`` represents steps of
    slicing and if the ``strides`` is negative, slice operation is in the opposite direction.
    If the value passed to ``starts`` or ``ends`` is greater than n
    (the number of elements in this dimension), it represents n.
    For slicing to the end of a dimension with unknown size, it is recommended
    to pass in INT_MAX. The size of ``axes`` must be equal to ``starts`` , ``ends`` and ``strides``.
    Following examples will explain how strided_slice works:

    .. code-block:: text

        Case1:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [1, 0]
                ends = [2, 3]
                strides = [1, 1]
            Then:
                result = [ [5, 6, 7], ]
        
        Case2:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [0, 1]
                ends = [2, 0]
                strides = [1, -1]
            Then:
                result = [ [8, 7, 6], ]
        
        Case3:
            Given:
                data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
                axes = [0, 1]
                starts = [-1, 1000]
                ends = [-1, 1000]
                strides = [1, 3]
            Then:
                result = [ [2], ]
    Args:
        input (Variable): An N-D ``Tensor`` or ``LoDTensor`` . The data type is ``float32``, ``float64``, ``int32`` or ``int64``.
        axes (list|tuple): The data type is ``int32`` . Axes that `starts` and `ends` apply to.
                            It's optional. If it is not provides, it will be treated as :math:`[0,1,...,len(starts)-1]`.
        starts (list|tuple|Variable): The data type is ``int32`` . If ``starts`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``starts`` is an Variable, it should be an 1-D Tensor.
                It represents starting indices of corresponding axis in ``axes``.
        ends (list|tuple|Variable): The data type is ``int32`` . If ``ends`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``ends`` is an Variable, it should be an 1-D Tensor .
                It represents ending indices of corresponding axis in ``axes``.
        strides (list|tuple|Variable): The data type is ``int32`` . If ``strides`` is a list or tuple, the elements of
                it should be integers or Tensors with shape [1]. If ``strides`` is an Variable, it should be an 1-D Tensor .
                It represents slice step of corresponding axis in ``axes``.

    Returns:
        Variable:  A ``Tensor`` or ``LoDTensor`` with the same dimension as ``input``. The data type is same as ``input``.

    Raises:
        TypeError: The type of ``starts`` must be list, tuple or Variable.
        TypeError: The type of ``ends`` must be list, tuple or Variable.
        TypeError: The type of ``strides`` must be list, tuple or Variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.data(
                name="input", shape=[3, 4, 5, 6], dtype='float32')

            # example 1:
            # attr starts is a list which doesn't contain tensor Variable.
            axes = [0, 1, 2]
            starts = [-3, 0, 2]
            ends = [3, 2, 4]
            strides_1 = [1, 1, 1]
            strides_2 = [1, 1, 2]
            sliced_1 = fluid.layers.strided_slice(input, axes=axes, starts=starts, ends=ends, strides=strides_1)
            # sliced_1 is input[:, 0:3:1, 0:2:1, 2:4:1].


            # example 2:
            # attr starts is a list which contain tensor Variable.
            minus_3 = fluid.layers.fill_constant([1], "int32", -3)
            sliced_2 = fluid.layers.strided_slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)
            # sliced_2 is input[:, 0:3:1, 0:2:1, 2:4:2].
    """
    if not isinstance(starts, (list, tuple, Variable)):
        raise ValueError(
            "Input starts must be an Variable, python list or tuple.")
    if not isinstance(ends, (list, tuple, Variable)):
        raise ValueError(
            "Input ends must be an Variable, python list or tuple.")
    if not isinstance(strides, (list, tuple, Variable)):
        raise ValueError(
            "Input strides must be an Variable, python list or tuple.")

    helper = LayerHelper('strided_slice', **locals())

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_list_tensor(old_list):
        new_list_tensor = []
        for dim in old_list:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_list_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_list_tensor.append(temp_out)
        return new_list_tensor

    inputs = {'Input': input}
    attrs = {'axes': axes}
    infer_flags = list(1 for i in range(len(axes)))

    if in_dygraph_mode():
        inputs = {'Input': input}
        attrs = {
            'axes': axes,
            'starts': starts,
            'ends': ends,
            'strides': strides,
            'infer_flags': infer_flags
        }
    else:
        # starts
        if isinstance(starts, Variable):
            starts.stop_gradient = True
            inputs['StartsTensor'] = starts
        elif isinstance(starts, (list, tuple)):
            attrs['starts'] = []
            if not contain_var(starts):
                attrs['starts'] = starts
            else:
                inputs['StartsTensorList'] = get_new_list_tensor(starts)
                for i, dim in enumerate(starts):
                    if isinstance(dim, Variable):
                        attrs['starts'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['starts'].append(dim)

        # ends
        if isinstance(ends, Variable):
            ends.stop_gradient = True
            inputs['EndsTensor'] = ends
        elif isinstance(ends, (list, tuple)):
            attrs['ends'] = []
            if not contain_var(ends):
                attrs['ends'] = ends
            else:
                inputs['EndsTensorList'] = get_new_list_tensor(ends)
                for i, dim in enumerate(ends):
                    if isinstance(dim, Variable):
                        attrs['ends'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['ends'].append(dim)
        # strides
        if isinstance(strides, Variable):
            strides.stop_gradient = True
            inputs['StridesTensor'] = strides
        elif isinstance(strides, (list, tuple)):
            attrs['strides'] = []
            if not contain_var(strides):
                attrs['strides'] = strides
            else:
                inputs['StridesTensorList'] = get_new_list_tensor(strides)
                for i, dim in enumerate(strides):
                    if isinstance(dim, Variable):
                        attrs['strides'].append(-1)
                        infer_flags[i] = -1
                    else:
                        attrs['strides'].append(dim)
        attrs['infer_flags'] = infer_flags
    out = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype('input'))
    helper.append_op(
        type='strided_slice', inputs=inputs, attrs=attrs, outputs={'Out': out})

    return out


def shape(input):
    """
    **Shape Layer**

    Get the shape of the input.

    Args:
        input (Variable): The input N-D Tensor. Datatype can be float32, float64, int32, int64.

    Returns:
        Variable (Tensor): The shape of the input variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            inputs = fluid.layers.data(name="x", shape=[3, 100, 100], dtype="float32")
            output = fluid.layers.shape(inputs)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img = np.ones((3, 100, 100)).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
            print(res) # [array([  3, 100, 100], dtype=int32)]
    """

    helper = LayerHelper('shape', **locals())
    out = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type='shape', inputs={'Input': input}, outputs={'Out': out})

    return out


def rank(input):
    """
    The OP returns the number of dimensions for a tensor, which is a 0-D int32 Tensor.

    Args:
        input (Variable): The input N-D tensor with shape of :math:`[N_1, N_2, ..., N_k]`, the data type is arbitrary.

    Returns:
        Variable, the output data type is int32.: The 0-D tensor with the dimensions of the input variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            input = fluid.data(name="input", shape=[3, 100, 100], dtype="float32")
            rank = fluid.layers.rank(input) # rank=(3,)
    """

    ndims = len(input.shape)
    out = assign(np.array(ndims, 'int32'))

    return out


def size(input):
    """
    **Size Layer**

    Returns the number of elements for a tensor, which is a int64 Tensor with shape [1].

    Args:
        input (Variable): The input variable.

    Returns:
        Variable: The number of elements for the input variable.

    Examples:
        .. code-block:: python

            import paddle.fluid.layers as layers

            input = layers.data(
                name="input", shape=[3, 100], dtype="float32", append_batch_size=False)
            rank = layers.size(input) # 300
    """

    helper = LayerHelper('size', **locals())
    out = helper.create_variable_for_type_inference(dtype='int64')
    helper.append_op(type='size', inputs={'Input': input}, outputs={'Out': out})

    return out


def _elementwise_op(helper):
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)
    if in_dygraph_mode():
        x = base.to_variable(x)
        y = base.to_variable(y)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)
    if not isinstance(x, Variable):
        raise TypeError(
            "The type of 'x' in %s must be Variable, but received %s" %
            (op_type, type(x)))
    if not isinstance(y, Variable):
        raise TypeError(
            "The type of 'y' in %s must be Variable, but received %s" %
            (op_type, type(y)))
    if convert_dtype(x.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'x' in %s only support float16 on GPU now." %
            (op_type))
    if convert_dtype(y.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'y' in %s only support float16 on GPU now." %
            (op_type))
    if convert_dtype(x.dtype) not in [
            'float16', 'float32', 'float64', 'int32', 'int64'
    ]:
        raise TypeError(
            "The data type of 'x' in %s must be float16 or float32 or float64 or int32 or int64, "
            "but received %s." % (op_type, convert_dtype(x.dtype)))
    if convert_dtype(y.dtype) not in [
            'float16', 'float32', 'float64', 'int32', 'int64'
    ]:
        raise TypeError(
            "The data type of 'y' in %s must be float16 or float32 or float64 or int32 or int64, "
            "but received %s." % (op_type, convert_dtype(y.dtype)))

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type=op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_activation(out)


def scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Scale operator.

    Putting scale and bias to the input Tensor as following:

    ``bias_after_scale`` is True:

    .. math::
                            Out=scale*X+bias

    ``bias_after_scale`` is False:

    .. math::
                            Out=scale*(X+bias)

    Args:
        x(Variable): Input N-D Tensor of scale operator. Data type can be float32, float64, int8, int16, int32, int64, uint8.
        scale(float): The scale factor of the input.
        bias(float): The bias to be put on the input.
        bias_after_scale(bool): Apply bias addition after or before scaling. It is useful for numeric stability in some circumstances.
        act(str, optional): Activation applied to the output such as tanh, softmax, sigmoid, relu.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name` 

    Returns:
        Variable(Tensor|LoDTensor): Output tensor of scale operator, with shape and data type same as input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            inputs = fluid.layers.data(name="x", shape=[2, 3], dtype='float32')
            output = fluid.layers.scale(inputs, scale = 2.0, bias = 1.0)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)

            res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
            print(res) # [array([[ 3.,  5.,  7.], [ 9., 11., 13.]], dtype=float32)]
    """

    helper = LayerHelper('scale', **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type='scale',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={
            'scale': float(scale),
            'bias': float(bias),
            'bias_after_scale': bias_after_scale
        })
    return helper.append_activation(out)


def elementwise_add(x, y, axis=-1, act=None, name=None):
    """
Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_add(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[3., 8., 6.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_add(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_add(x, y, axis=3)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]

    """
    return _elementwise_op(LayerHelper('elementwise_add', **locals()))


def elementwise_div(x, y, axis=-1, act=None, name=None):
    """
Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_div(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[2., 0.6, 2.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_div(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_div(x, y, axis=3)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]

    """
    return _elementwise_op(LayerHelper('elementwise_div', **locals()))


def elementwise_sub(x, y, axis=-1, act=None, name=None):
    """
Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_sub(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[1., -2., 2.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_sub(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_sub(x, y, axis=3)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]

    """
    return _elementwise_op(LayerHelper('elementwise_sub', **locals()))


def elementwise_mul(x, y, axis=-1, act=None, name=None):
    """
Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_mul(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[2., 15., 8.]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_mul(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) # z.shape=[2,3,4,5]


    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.random.randint(1, 5, size=[2, 3, 4, 5]).astype('float32'),
                "y": np.random.randint(1, 5, size=[5]).astype('float32')
            }
        
        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[5], dtype='float32')
        z = fluid.layers.elementwise_mul(x, y, axis=3)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) # z.shape=[2,3,4,5]
 
    """
    return _elementwise_op(LayerHelper('elementwise_mul', **locals()))


def elementwise_max(x, y, axis=-1, act=None, name=None):
    """
Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_max(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[2, 5, 4]


    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_max(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value)#[[[[1., 1., 1., 1., 1.] .... [1., 1., 1., 1., 1.]]]]

    """
    return _elementwise_op(LayerHelper('elementwise_max', **locals()))


def elementwise_min(x, y, axis=-1, act=None, name=None):
    """
Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_max(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[1, 3, 2]

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }

        x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_max(x, y, axis=1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value)#[[[[0., 0., 0., 0., 0.] .... [0., 0., 0., 0., 0.]]]]
    """

    return _elementwise_op(LayerHelper('elementwise_min', **locals()))


def elementwise_pow(x, y, axis=-1, act=None, name=None):
    """
Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([2, 3, 4]).astype('float32'),
                "y": np.array([1, 5, 2]).astype('float32')
            }

        x = fluid.data(name="x", shape=[3], dtype='float32')
        y = fluid.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_pow(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[2, 243, 16]
    """

    return _elementwise_op(LayerHelper('elementwise_pow', **locals()))


def elementwise_mod(x, y, axis=-1, act=None, name=None):
    """
Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([10, 15, 8]).astype('int32'),
                "y": np.array([3, 6, 5]).astype('int32')
            }

        x = fluid.data(name="x", shape=[3], dtype='int32')
        y = fluid.data(name="y", shape=[3], dtype='int32')
        z = fluid.layers.elementwise_mod(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[1, 3, 3]
    """
    return _elementwise_op(LayerHelper('elementwise_mod', **locals()))


def elementwise_floordiv(x, y, axis=-1, act=None, name=None):
    """
Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        def gen_data():
            return {
                "x": np.array([10, 15, 8]).astype('int32'),
                "y": np.array([3, 7, 5]).astype('int32')
            }

        x = fluid.data(name="x", shape=[3], dtype='int32')
        y = fluid.data(name="y", shape=[3], dtype='int32')
        z = fluid.layers.elementwise_floordiv(x, y)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])

        print(z_value) #[3, 2, 1]
    """
    return _elementwise_op(LayerHelper('elementwise_floordiv', **locals()))


for func in [
        elementwise_add,
        elementwise_div,
        elementwise_sub,
        elementwise_mul,
        elementwise_max,
        elementwise_pow,
        elementwise_min,
        elementwise_mod,
        elementwise_floordiv,
]:
    op_proto = OpProtoHolder.instance().get_op_proto(func.__name__)
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "axis (int32, optional): If X.dimension != Y.dimension, \
            Y.dimension must be a subsequence of x.dimension. \
            And axis is the start dimension index for broadcasting Y onto X. ",
            "act (string, optional): Activation applied to the output. \
            Default is None. Details: :ref:`api_guide_activations_en` ",
            "name (string, optional): Name of the output. \
            Default is None. It's used to print debug info for developers. Details: \
            :ref:`api_guide_Name` "
        ],
        skip_attrs_set={"x_data_format", "y_data_format", "axis"
                        }) + """\n""" + str(func.__doc__)

for func in []:
    op_proto = OpProtoHolder.instance().get_op_proto(func.__name__)
    func.__doc__ = _generate_doc_string_(
        op_proto,
        additional_args_lines=[
            "act (basestring|None): Activation applied to the output.",
            "name (basestring|None): Name of the output."
        ])
    func.__doc__ = func.__doc__ + """

Examples:
  .. code-block:: python
    
    import paddle.fluid as fluid
    # example 1: shape(x) = (2, 3, 4, 5), shape(y) = (2, 3, 4, 5)
    x0 = fluid.layers.data(name="x0", shape=[2, 3, 4, 5], dtype='float32')
    y0 = fluid.layers.data(name="y0", shape=[2, 3, 4, 5], dtype='float32')
    z0 = fluid.layers.%s(x0, y0)

    # example 2: shape(X) = (2, 3, 4, 5), shape(Y) = (5)
    x1 = fluid.layers.data(name="x1", shape=[2, 3, 4, 5], dtype='float32')
    y1 = fluid.layers.data(name="y1", shape=[5], dtype='float32')
    z1 = fluid.layers.%s(x1, y1)

    # example 3: shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    x2 = fluid.layers.data(name="x2", shape=[2, 3, 4, 5], dtype='float32')
    y2 = fluid.layers.data(name="y2", shape=[4, 5], dtype='float32')
    z2 = fluid.layers.%s(x2, y2, axis=2)

    # example 4: shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    x3 = fluid.layers.data(name="x3", shape=[2, 3, 4, 5], dtype='float32')
    y3 = fluid.layers.data(name="y3", shape=[3, 4], dtype='float32')
    z3 = fluid.layers.%s(x3, y3, axis=1)

    # example 5: shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    x4 = fluid.layers.data(name="x4", shape=[2, 3, 4, 5], dtype='float32')
    y4 = fluid.layers.data(name="y4", shape=[2], dtype='float32')
    z4 = fluid.layers.%s(x4, y4, axis=0)

    # example 6: shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0
    x5 = fluid.layers.data(name="x5", shape=[2, 3, 4, 5], dtype='float32')
    y5 = fluid.layers.data(name="y5", shape=[2], dtype='float32')
    z5 = fluid.layers.%s(x5, y5, axis=0)
    """ % (func.__name__, func.__name__, func.__name__, func.__name__,
           func.__name__, func.__name__)


def _logical_op(op_name, x, y, out=None, name=None, binary_op=True):
    helper = LayerHelper(op_name, **locals())

    if binary_op:
        assert x.dtype == y.dtype

    if out is None:
        if name is None:
            out = helper.create_variable_for_type_inference(dtype=x.dtype)
        else:
            out = helper.create_variable(
                name=name, dtype=x.dtype, persistable=False)

    if binary_op:
        helper.append_op(
            type=op_name, inputs={"X": x,
                                  "Y": y}, outputs={"Out": out})
    else:
        helper.append_op(type=op_name, inputs={"X": x}, outputs={"Out": out})

    return out


@templatedoc()
def logical_and(x, y, out=None, name=None):
    """
    logical_and Operator

    It operates element-wise on X and Y, and returns the Out. X, Y and Out are N-dim boolean LoDTensor or Tensor.
    Each element of Out is calculated by
    
    .. math::

        Out = X \land Y

    Args:
        x(${x_type}): ${x_comment}
        y(${y_type}): ${y_comment}
        out(LoDTensor or Tensor): The LoDTensor or Tensor that specifies the output of the operator, which can be any Variable that has been created in the program. The default value is None, and a new Variable will be created to save the output.
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_type}: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Graph organizing
            x = fluid.layers.data(name='x', shape=[2], dtype='bool')
            y = fluid.layers.data(name='y', shape=[2], dtype='bool')
            res = fluid.layers.logical_and(x=x, y=y)
            # The comment lists another available method.
            # res = fluid.layers.fill_constant(shape=[2], dtype='bool', value=0)
            # fluid.layers.logical_and(x=x, y=y, out=res)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            x_i = np.array([[1, 0], [0, 1]]).astype(np.bool)
            y_i = np.array([[1, 1], [0, 0]]).astype(np.bool)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i, 'y':y_i}, fetch_list=[res])
            print(res_val) # [[True, False], [False, False]]
    """

    return _logical_op(
        op_name="logical_and", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def logical_or(x, y, out=None, name=None):
    """
    logical_or Operator

    It operates element-wise on X and Y, and returns the Out. X, Y and Out are N-dim boolean LoDTensor or Tensor.
    Each element of Out is calculated by
    
    .. math::

        Out = X \lor Y

    Args:
        x(${x_type}): ${x_comment}
        y(${y_type}): ${y_comment}
        out(LoDTensor or Tensor): The LoDTensor or Tensor that specifies the output of the operator, which can be any Variable that has been created in the program. The default value is None, and a new Variable will be created to save the output.
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_type}: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Graph organizing
            x = fluid.layers.data(name='x', shape=[2], dtype='bool')
            y = fluid.layers.data(name='y', shape=[2], dtype='bool')
            res = fluid.layers.logical_or(x=x, y=y)
            # The comment lists another available method.
            # res = fluid.layers.fill_constant(shape=[2], dtype='bool', value=0)
            # fluid.layers.logical_or(x=x, y=y, out=res)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            x_i = np.array([[1, 0], [0, 1]]).astype(np.bool)
            y_i = np.array([[1, 1], [0, 0]]).astype(np.bool)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i, 'y':y_i}, fetch_list=[res])
            print(res_val) # [[True, True], [False, True]]
    """

    return _logical_op(
        op_name="logical_or", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def logical_xor(x, y, out=None, name=None):
    """
    logical_xor Operator

    It operates element-wise on X and Y, and returns the Out. X, Y and Out are N-dim boolean LoDTensor or Tensor.
    Each element of Out is calculated by
    
    .. math::

        Out = (X \lor Y) \land \lnot (X \land Y)

    Args:
        x(${x_type}): ${x_comment}
        y(${y_type}): ${y_comment}
        out(LoDTensor or Tensor): The LoDTensor or Tensor that specifies the output of the operator, which can be any Variable that has been created in the program. The default value is None, and a new Variable will be created to save the output.
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_type}: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Graph organizing
            x = fluid.layers.data(name='x', shape=[2], dtype='bool')
            y = fluid.layers.data(name='y', shape=[2], dtype='bool')
            res = fluid.layers.logical_xor(x=x, y=y)
            # The comment lists another available method.
            # res = fluid.layers.fill_constant(shape=[2], dtype='bool', value=0)
            # fluid.layers.logical_xor(x=x, y=y, out=res)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            x_i = np.array([[1, 0], [0, 1]]).astype(np.bool)
            y_i = np.array([[1, 1], [0, 0]]).astype(np.bool)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i, 'y':y_i}, fetch_list=[res])
            print(res_val) # [[False, True], [False, True]]
    """

    return _logical_op(
        op_name="logical_xor", x=x, y=y, name=name, out=out, binary_op=True)


@templatedoc()
def logical_not(x, out=None, name=None):
    """
    logical_not Operator

    It operates element-wise on X, and returns the Out. X and Out are N-dim boolean LoDTensor or Tensor.
    Each element of Out is calculated by
    
    .. math::

        Out = \lnot X

    Args:
        x(${x_type}): ${x_comment}
        out(LoDTensor/Tensor): The LoDTensor/Tensor that specifies the output of the operator, which can be any Variable that has been created in the program. The default value is None, and a new Variable will be created to save the output.
        name(str|None): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_type}: ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            # Graph organizing
            x = fluid.layers.data(name='x', shape=[2], dtype='bool')
            res = fluid.layers.logical_not(x)
            # The comment lists another availble method.
            # res = fluid.layers.fill_constant(shape=[2], dtype='bool', value=0)
            # fluid.layers.logical_not(x, out=res)

            # Create an executor using CPU as an example
            exe = fluid.Executor(fluid.CPUPlace())

            # Execute
            x_i = np.array([[1, 0]]).astype(np.bool)
            res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
            print(res_val) # [[False, True]]
    """

    return _logical_op(
        op_name="logical_not", x=x, y=None, name=name, out=out, binary_op=False)


@templatedoc()
def clip(x, min, max, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        min(float): ${min_comment}
        max(float): ${max_comment}
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_comment}

    Return Type:
        ${out_type}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(
                name='data', shape=[1], dtype='float32')
            reward = fluid.layers.clip(x=input, min=-1.0, max=1.0)
    """

    helper = LayerHelper("clip", **locals())

    if name is None:
        name = unique_name.generate_with_ignorable_key(".".join(
            [helper.name, 'tmp']))

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="clip",
        inputs={"X": x},
        attrs={"min": min,
               "max": max},
        outputs={"Out": out})

    return out


@templatedoc()
def clip_by_norm(x, max_norm, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        max_norm(${max_norm_type}): ${max_norm_comment}
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Variable:

        out(${out_type}): ${out_comment}


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(
                name='data', shape=[None, 1], dtype='float32')
            reward = fluid.layers.clip_by_norm(x=input, max_norm=1.0)
    """

    helper = LayerHelper("clip_by_norm", **locals())

    if name is None:
        name = unique_name.generate_with_ignorable_key(".".join(
            [helper.name, 'tmp']))

    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="clip_by_norm",
        inputs={"X": x},
        attrs={"max_norm": max_norm},
        outputs={"Out": out})

    return out


@templatedoc()
def mean(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # Graph Organizing
            input = fluid.data(
                name='data', shape=[2, 3], dtype='float32')
            output = fluid.layers.mean(input)

            # Create an executor using CPU as an example
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            # Execute
            x_ndarray = numpy.ones([2, 3]).astype(numpy.float32)
            res, = exe.run(fluid.default_main_program(),
                           feed={'data':x_ndarray},
                           fetch_list=[output])
            print(res)
            '''
            Output Value:
            [1.]
            '''
    """

    helper = LayerHelper("mean", **locals())

    if not isinstance(x, Variable):
        raise TypeError(
            "The type of 'x' in mean must be Variable, but received %s.\n" %
            (type(x)))

    if convert_dtype(x.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'x' in mean only support float16 in GPU now.")

    if convert_dtype(x.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'x' in mean must be float16 or float32 or float64, but received %s.\n"
            % (convert_dtype(x.dtype)))

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mean", inputs={"X": x}, attrs={}, outputs={"Out": out})

    return out


@templatedoc()
def merge_selected_rows(x, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        name(basestring|None): Name of the output.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            place = fluid.CPUPlace()
            block = fluid.default_main_program().global_block()

            var = block.create_var(name="X2",
                                   dtype="float32",
                                   persistable=True,
                                   type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            y = fluid.layers.merge_selected_rows(var)
            z = fluid.layers.get_tensor_from_selected_rows(y)

            x_rows = [0, 2, 2, 4, 19]
            row_numel = 2
            np_array = numpy.ones((len(x_rows), row_numel)).astype("float32")

            x = fluid.global_scope().var("X2").get_selected_rows()
            x.set_rows(x_rows)
            x.set_height(20)
            x_tensor = x.get_tensor()
            x_tensor.set(np_array, place)

            exe = fluid.Executor(place=place)
            result = exe.run(fluid.default_main_program(), fetch_list=[z])

            print("x_rows: ", x_rows)
            print("np_array: ", np_array)
            print("result: ", result)
            '''
            Output Values:
            ('x_rows: ', [0, 2, 2, 4, 19])
            ('np_array: ', array([[1., 1.],
                   [1., 1.],
                   [1., 1.],
                   [1., 1.],
                   [1., 1.]], dtype=float32))
            ('result: ', [array([[1., 1.],
                   [2., 2.],
                   [1., 1.],
                   [1., 1.]], dtype=float32)])
            '''
    """

    helper = LayerHelper("merge_selected_rows", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="merge_selected_rows",
        inputs={"X": x},
        attrs={},
        outputs={"Out": out})
    return out


def mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None):
    """
    Mul Operator.
    This operator is used to perform matrix multiplication for input $x$ and $y$.
    The equation is:

    ..  math::
        Out = x * y

    Both the input $x$ and $y$ can carry the LoD (Level of Details) information, or not. But the output only shares the LoD information with input $x$.

    Args:
        x (Variable): The first input Tensor/LoDTensor of mul_op.
        y (Variable): The second input Tensor/LoDTensor of mul_op.
        x_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $x$ is a tensor with more than two dimensions, $x$ will be flattened into a two-dimensional matrix first. The flattening rule is: the first `num_col_dims` will be flattened to form the first dimension of the final matrix (the height of the matrix), and the rest `rank(x) - num_col_dims` dimensions are flattened to form the second dimension of the final matrix (the width of the matrix). As a result, height of the flattened matrix is equal to the product of $x$'s first `x_num_col_dims` dimensions' sizes, and width of the flattened matrix is equal to the product of $x$'s last `rank(x) - num_col_dims` dimensions' size. For example, suppose $x$ is a 6-dimensional tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3. Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default is 1. 
        y_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs. If the input $y$ is a tensor with more than two dimensions, $y$ will be flattened into a two-dimensional matrix first. The attribute `y_num_col_dims` determines how $y$ is flattened. See comments of `x_num_col_dims` for more details. Default is 1. 
        name (str, optional): Name of the output. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name`. Default is None. 

    Returns:
        Variable(Tensor/LoDTensor): The output Tensor/LoDTensor of mul op.

    Examples:
        ..  code-block:: python
            
            import paddle.fluid as fluid
            dataX = fluid.layers.data(name="dataX", append_batch_size = False, shape=[2, 5], dtype="float32")
            dataY = fluid.layers.data(name="dataY", append_batch_size = False, shape=[5, 3], dtype="float32")
            output = fluid.layers.mul(dataX, dataY,
                                      x_num_col_dims = 1,
                                      y_num_col_dims = 1)
            

    """

    helper = LayerHelper("mul", **locals())

    if not isinstance(x, Variable):
        raise TypeError(
            "The type of 'x' in mul must be Variable, but received %s" %
            (type(x)))
    if not isinstance(y, Variable):
        raise TypeError(
            "The type of 'y' in mul must be Variable, but received %s" %
            (type(y)))
    if convert_dtype(x.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'x' in mul only support float16 in GPU now.")
    if convert_dtype(y.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'y' in mul only support float16 in GPU now.")
    if convert_dtype(x.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'x' in mul must be float16, float32 or float64, but received %s."
            % (convert_dtype(x.dtype)))
    if convert_dtype(y.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'y' in mul must be float16, float32 or float64, but received %s."
            % (convert_dtype(y.dtype)))

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mul",
        inputs={"X": x,
                "Y": y},
        attrs={
            "x_num_col_dims": x_num_col_dims,
            "y_num_col_dims": y_num_col_dims
        },
        outputs={"Out": out})
    return out


@templatedoc()
def sigmoid_cross_entropy_with_logits(x,
                                      label,
                                      ignore_index=kIgnoreIndex,
                                      name=None,
                                      normalize=False):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        label(${label_type}): ${label_comment}
        ignore_index(int): ${ignore_index_comment}
        name(str|None): The default value is None.  Normally there is
            no need for user to set this property.  For more information,
            please refer to :ref:`api_guide_Name`
        normalize(bool): If true, divide the output by the number of
            targets != ignore_index.

    Returns:
        out(${out_type}): ${out_comment}

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(
                name='data', shape=[10], dtype='float32')
            label = fluid.data(
                name='data', shape=[10], dtype='float32')
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=input,
                label=label,
                ignore_index=-1,
                normalize=True) # or False
            # loss = fluid.layers.reduce_sum(loss) # summation of loss
    """

    helper = LayerHelper("sigmoid_cross_entropy_with_logits", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="sigmoid_cross_entropy_with_logits",
        inputs={"X": x,
                "Label": label},
        attrs={"ignore_index": ignore_index,
               'normalize': normalize},
        outputs={"Out": out})
    return out


@templatedoc()
def maxout(x, groups, name=None):
    """
    ${comment}

    Args:
        x(${x_type}): ${x_comment}
        groups(${groups_type}): ${groups_comment}
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.

    Returns:
        Variable:

        out(${out_type}): ${out_comment}


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(
                name='data', 
                shape=[None, 256, 32, 32], 
                dtype='float32')
            out = fluid.layers.maxout(input, groups=2)
    """
    helper = LayerHelper("maxout", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="maxout",
        inputs={"X": x},
        attrs={"groups": groups},
        outputs={"Out": out})
    return out


def space_to_depth(x, blocksize, name=None):
    """
    Gives a blocksize to space_to_depth the input LoDtensor with Layout: [batch, channel, height, width]

    This op rearranges blocks of spatial data, into depth. More specifically, this op outputs a copy of \
        theinput LoDtensor where values from the height and width dimensions are moved to the channel \
        dimension.
    The attr blocksize indicates the input block size.

    space_to_depth will reorgnize the elements of input with shape[batch, channel, height, width] \
        according to blocksize to construct output with shape \
        [batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]:

    - Non-overlapping blocks of size block_size x block size are rearranged into depth at each location.
    - The Y, X coordinates within each block of the input become the high order component of the output channel index
    - channel should be divisible by square of blocksize
    - height, width should be divsible by blocksize

    This OP is useful for resizing the activations between convolutions \
        (but keeping all data)

    .. code-block:: text

        Given the input x with the shape [1, 1, 4, 4]:
        x.data = [[[[1,   2,  5,  6],
                    [3,   4,  7,  8],
                    [9,  10, 13, 14],
                    [11, 12, 15, 16]]]]
        blocksize = 2

        then get the output with the shape [1, 4, 2, 2]:
        out.data = [[[[1,   2],  [3,  4]],
                     [[5,   6],  [7,  8]],
                     [[9,  10], [11, 12]],
                     [[13, 14], [15, 16]]]]

    Args:
        x (Variable): The input, which should be 4 dims Tensor or LodTensor, with the shape \
            [batch, channel, height, width]
        blocksize (int): The blocksize to select the element on each feature map should be > 2
        name(str, optional): For detailed information, please refer \
            to :ref:`api_guide_Name`. Usually name is no need to set and \
            None by default.

    Returns: The output, which should be 4 dims Tensor or LodTensor, with the shape \
            [batch, channel * blocksize * blocksize, height/blocksize, width/blocksize]

    Return Type: Variable

    Raises:
        TypeError: blocksize type must be int64.

    Examples:
        .. code-block:: python
    
            import paddle.fluid as fluid
            import numpy as np

            data = fluid.data(
                name='data', shape=[1, 4, 2, 2], dtype='float32')
            space_to_depthed = fluid.layers.space_to_depth(
                x=data, blocksize=2)

            exe = fluid.Executor(fluid.CPUPlace())
            data_np = np.arange(0,16).reshape((1,4,2,2)).astype('float32')

            print(data_np)
            #array([[[[ 0.,  1.], [ 2.,  3.]],
            #        [[ 4.,  5.], [ 6.,  7.]],
            #        [[ 8.,  9.], [10., 11.]],
            #        [[12., 13.], [14., 15.]]]], dtype=float32)

            out_main = exe.run(fluid.default_main_program(),
                        feed={'data': data_np},
                        fetch_list=[space_to_depthed])

            print(out_main)
            #[array([[[[ 0.]], [[ 4.]], [[ 1.]], [[ 5.]],
            #         [[ 8.]], [[12.]], [[ 9.]], [[13.]],
            #         [[ 2.]], [[ 6.]], [[ 3.]], [[ 7.]],
            #         [[10.]], [[14.]], [[11.]], [[15.]]]], dtype=float32)]

    """

    helper = LayerHelper("space_to_depth", **locals())

    if not (isinstance(blocksize, int)):
        raise ValueError("blocksize must be a python Int")

    if name is None:
        out = helper.create_variable_for_type_inference(
            dtype=x.dtype)  #fix create
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="space_to_depth",
        inputs={"X": x},
        attrs={"blocksize": blocksize},
        outputs={"Out": out})
    return out


@templatedoc()
def sequence_reverse(x, name=None):
    """
    **Notes: The Op only receives LoDTensor as input. If your input is Tensor, please use reverse Op.(fluid.layers.** :ref:`api_fluid_layers_reverse` ).

    This operator only supports LoDTensor as input. It will reverse each sequence for input LoDTensor.
    Currently it only supports 1-level LoDTensor. This operator is very useful when building a
    reverse :ref:`api_fluid_layers_DynamicRNN` network.

    .. code-block:: text

        input(x) is a LoDTensor:
            x.lod  = [[0, 2, 5]]
            x.data = [[1,  2,  3,  4],
                      [5,  6,  7,  8],
                      [9, 10, 11, 12],
                      [13,14, 15, 16],
                      [17,18, 19, 20]]
            x.shape = [5, 4]

        output LoDTensor with same shape and LoD info:
            out.lod  = [[0, 2, 5]]
            out.data = [[5,  6,  7,  8],
                        [1,  2,  3,  4],
                        [17,18, 19, 20],
                        [13,14, 15, 16],
                        [9, 10, 11, 12]]
            out.shape = [5, 4]

    Args:
        x(Variable): LoDTensor with 1-level LoD info. Currently it only supports 1-level LoDTensor.
            The data type should be float32, float64, int8, int32 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: LoDTensor reversed from input. The data type is same with input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 10], dtype='float32', lod_level=1)
            x_reversed = fluid.layers.sequence_reverse(x)
    """
    assert not in_dygraph_mode(), (
        "sequence layer is not supported in dygraph mode yet.")
    helper = LayerHelper("sequence_reverse", **locals())
    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="sequence_reverse",
        inputs={"X": x},
        outputs={"Y": out},
        attrs=dict())
    return out


def affine_channel(x,
                   scale=None,
                   bias=None,
                   data_layout='NCHW',
                   name=None,
                   act=None):
    """
    Applies a separate affine transformation to each channel of the input.
    Useful for replacing spatial batch norm with its equivalent fixed
    transformation. The input also can be 2D tensor and applies a affine
    transformation in second dimension.

    Args:
        x (Variable): Feature map input can be a 4D tensor with order NCHW
            or NHWC. It also can be a 2D tensor and the affine transformation
            is applied in the second dimension.The data type is float32 or float64.
        scale (Variable): 1D input of shape (C), the c-th element is the scale
            factor of the affine transformation for the c-th channel of
            the input.The data type is float32 or float64.
        bias (Variable): 1D input of shape (C), the c-th element is the bias
            of the affine transformation for the c-th channel of the input.
            The data type is float32 or float64.
        data_layout (str, default NCHW): NCHW or NHWC. If input is 2D
            tensor, you can ignore data_layout.
        name (str, default None): The name of this layer. For more information,
            please refer to :ref:`api_guide_Name` .
        act (str, default None): Activation to be applied to the output of this layer.

    Returns:
        Variable: A tensor which has the same shape, data layout and data type with x.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            use_gpu = False
            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)

            data = fluid.data(name='data', shape=[None, 1, 2, 2], dtype='float32')
            input_scale = fluid.layers.create_parameter(shape=[1], dtype="float32",
                                    default_initializer=fluid.initializer.Constant(2.0))
            input_bias = fluid.layers.create_parameter(shape=[1],dtype="float32",
                                    default_initializer=fluid.initializer.Constant(0.5))
            out = fluid.layers.affine_channel(data,scale=input_scale,
                                    bias=input_bias)

            exe.run(fluid.default_startup_program())
            test_program = fluid.default_main_program().clone(for_test=True)

            [out_array] = exe.run(test_program,
                                  fetch_list=out,
                                  feed={'data': np.ones([1,1,2,2]).astype('float32')})
            # out_array is [[[[2.5, 2.5],
            #                [2.5, 2.5]]]] with shape: [1, 1, 2, 2]

    """
    helper = LayerHelper("affine_channel", **locals())

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="affine_channel",
        inputs={"X": x,
                'Scale': scale,
                'Bias': bias},
        attrs={"data_layout": data_layout},
        outputs={"Out": out})
    return helper.append_activation(out)


def similarity_focus(input, axis, indexes, name=None):
    """
    SimilarityFocus Operator

    Generate a similarity focus mask with the same shape of input using the following method:

    1. Extract the 3-D tensor(here the first dimension is BatchSize) corresponding
       to the axis according to the indexes. For example, if axis=1 and indexes=[a],
       it will get the matrix T=X[:, a, :, :]. In this case, if the shape of input X
       is (BatchSize, A, B, C), the shape of tensor T is (BatchSize, B, C).
    2. For each index, find the largest numbers in the tensor T, so that the same
       row and same column has at most one number(what it means is that if the
       largest number has been found in the i-th row and the j-th column, then
       the numbers in the i-th row or j-th column will be skipped. And then the
       next largest number will be selected from the remaining numbers. Obviously
       there will be min(B, C) numbers), and mark the corresponding position of the
       3-D similarity focus mask as 1, otherwise as 0. Do elementwise-or for
       each index.
    3. Broadcast the 3-D similarity focus mask to the same shape of input X.

    Refer to `Similarity Focus Layer <http://www.aclweb.org/anthology/N16-1108>`_

    .. code-block:: text

        * Example :

            Given a 4-D tensor x with the shape (BatchSize, C, A, B), where C is
            the number of channels and the shape of feature map is (A, B):
                x.shape = (2, 3, 2, 2)
                x.data = [[[[0.8, 0.1],
                            [0.4, 0.5]],

                           [[0.9, 0.7],
                            [0.9, 0.9]],

                           [[0.8, 0.9],
                            [0.1, 0.2]]],


                          [[[0.2, 0.5],
                            [0.3, 0.4]],

                           [[0.9, 0.7],
                            [0.8, 0.4]],

                           [[0.0, 0.2],
                            [0.4, 0.7]]]]

            Given axis: 1 (the axis of the channel)
            Given indexes: [0]

            then we get a 4-D tensor out with the same shape of input x:
                out.shape = (2, 3, 2, 2)
                out.data = [[[[1.0, 0.0],
                              [0.0, 1.0]],

                             [[1.0, 0.0],
                              [0.0, 1.0]],

                             [[1.0, 0.0],
                              [0.0, 1.0]]],

                            [[[0.0, 1.0],
                              [1.0, 0.0]],

                             [[0.0, 1.0],
                              [1.0, 0.0]],

                             [[0.0, 1.0],
                              [1.0, 0.0]]]]

    Args:
        input(Variable): The input tensor variable(default float). It should
            be a 4-D tensor with shape [BatchSize, A, B, C]. Data type is 
            float32 or float64.
        axis(int): Indicating the dimension to be selected. It can only be
            1, 2 or 3.
        indexes(list): Indicating the indexes of the selected dimension.

    Returns:
        Variable: A tensor variable with the same shape and same type \
                  as the input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.data(
                name='data', shape=[-1, 3, 2, 2], dtype='float32')
            fluid.layers.similarity_focus(input=data, axis=1, indexes=[0])
    """
    helper = LayerHelper('similarity_focus', **locals())
    # check attrs
    if isinstance(axis, int) is False:
        raise TypeError("axis must be int type.")
    if isinstance(indexes, list) is False:
        raise TypeError("indexes must be list type.")
    if axis != 1 and axis != 2 and axis != 3:
        raise ValueError("axis must be 1, 2 or 3.")
    if len(indexes) == 0:
        raise ValueError("indexes can not be empty.")

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=input.dtype)
    else:
        out = helper.create_variable(
            name=name, dtype=input.dtype, persistable=False)
    helper.append_op(
        type='similarity_focus',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={"axis": axis,
               "indexes": indexes})
    return out


def hash(input, hash_size, num_hash=1, name=None):
    """
    This OP hash the input to an integer less than the hash_size.
    The hash algorithm we used was xxHash - Extremely fast hash algorithm
    (https://github.com/Cyan4973/xxHash/tree/v0.6.5)

    Args:
        input(Variable): A **Two-Dimensional** LoDTensor with type int32, int64.
             **Only support LoDTensor**.
        num_hash(int, optional): The times of hash, default is 1.
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
       Variable: A LoDTensor with the same data type as input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            place = fluid.core.CPUPlace()

            x = fluid.data(name="x", shape=[1], dtype="int32", lod_level=1)
            res = fluid.layers.hash(name="res",input=x, hash_size=1000, num_hash=4)

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            in1 = np.array([[1,2],[3,4]]).astype("int32")
            print(in1)
            x_i = fluid.core.LoDTensor()
            x_i.set(in1,place)
            x_i.set_recursive_sequence_lengths([[0,2]])
            res = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res], return_numpy=False)
            print(np.array(res[0]))
            # [[[722]
            #   [407]
            #   [337]
            #   [395]]
            #  [[603]
            #   [590]
            #   [386]
            #   [901]]]
    """
    helper = LayerHelper('hash', **locals())
    out = helper.create_variable_for_type_inference(
        helper.input_dtype(), stop_gradient=True)
    helper.append_op(
        type='hash',
        inputs={'X': input},
        outputs={'Out': out},
        attrs={'num_hash': num_hash,
               'mod_by': hash_size})
    return out


@templatedoc()
def grid_sampler(x, grid, name=None):
    """
    This operation samples input X by using bilinear interpolation based on
    flow field grid, which is usually gennerated by :code:`affine_grid` . The grid of
    shape [N, H, W, 2] is the concatenation of (x, y) coordinates
    with shape [N, H, W] each, where x is indexing the 4th dimension
    (in width dimension) of input data x and y is indexng the 3rd
    dimention (in height dimension), finally results is the bilinear
    interpolation value of 4 nearest corner points. The output tensor 
    shape will be [N, C, H, W].

    .. code-block:: text

        Step 1:
        Get (x, y) grid coordinates and scale to [0, H-1/W-1].

        .. code-block:: text

            grid_x = 0.5 * (grid[:, :, :, 0] + 1) * (W - 1)
            grid_y = 0.5 * (grid[:, :, :, 1] + 1) * (H - 1)

        Step 2:
        Indices input data X with grid (x, y) in each [H, W] area, and bilinear
        interpolate point value by 4 nearest points.

          wn ------- y_n ------- en
          |           |           |
          |          d_n          |
          |           |           |
         x_w --d_w-- grid--d_e-- x_e
          |           |           |
          |          d_s          |
          |           |           |
          ws ------- y_s ------- wn

        x_w = floor(x)              // west side x coord
        x_e = x_w + 1               // east side x coord
        y_n = floor(y)              // north side y coord
        y_s = y_s + 1               // south side y coord

        d_w = grid_x - x_w          // distance to west side
        d_e = x_e - grid_x          // distance to east side
        d_n = grid_y - y_n          // distance to north side
        d_s = y_s - grid_y          // distance to south side

        wn = X[:, :, y_n, x_w]      // north-west point value
        en = X[:, :, y_n, x_e]      // north-east point value
        ws = X[:, :, y_s, x_w]      // south-east point value
        es = X[:, :, y_s, x_w]      // north-east point value

        output = wn * d_e * d_s + en * d_w * d_s
               + ws * d_e * d_n + es * d_w * d_n

    Args:
        x(Variable): The input tensor, which is a 4-D tensor with shape
                     [N, C, H, W], N is the batch size, C is the channel
                     number, H and W is the feature height and width.
                     The data type is float32 or float64.
        grid(Variable): Input grid tensor of shape [N, H, W, 2]. The
                        data type is float32 or float64.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Variable: Output of shape [N, C, H, W] data samples input X
                  using bilnear interpolation based on input grid.
                  The data type is same as input tensor.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            # use with affine_grid
            x = fluid.data(name='x', shape=[None, 10, 32, 32], dtype='float32')
            theta = fluid.layers.data(name='theta', shape=[2, 3], dtype='float32')
            grid = fluid.layers.affine_grid(theta=theta, out_shape=[3, 10, 32, 32])
            out = fluid.layers.grid_sampler(x=x, grid=grid)

    """
    helper = LayerHelper("grid_sampler", **locals())

    if not isinstance(x, Variable):
        return ValueError("The x should be a Variable")

    if not isinstance(grid, Variable):
        return ValueError("The grid should be a Variable")

    out = helper.create_variable_for_type_inference(x.dtype)
    ipts = {'X': x, 'Grid': grid}

    helper.append_op(type='grid_sampler', inputs=ipts, outputs={'Output': out})
    return out


def log_loss(input, label, epsilon=1e-4, name=None):
    """
    **Negative Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    negative log loss.

    .. math::

        Out = -label * \\log{(input + \\epsilon)}
              - (1 - label) * \\log{(1 - input + \\epsilon)}

    Args:
        input (Variable|list):  A 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator. Data type float32.
        label (Variable|list):  The ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size. 
                                Data type float32.
        epsilon (float, optional): A small number for numerical stability. Default 1e-4.
        name(str|None): For detailed information, please refer to 
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Returns:
        Variable: A 2-D tensor with shape [N x 1], the negative log loss.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
          prob = fluid.data(name='prob', shape=[-1, 10], dtype='float32')
          cost = fluid.layers.log_loss(input=prob, label=label)
    """
    helper = LayerHelper('log_loss', **locals())

    if name is None:
        loss = helper.create_variable_for_type_inference(dtype=input.dtype)
    else:
        loss = helper.create_variable(
            name=name, dtype=input.dtype, persistable=False)

    helper.append_op(
        type='log_loss',
        inputs={'Predicted': [input],
                'Labels': [label]},
        outputs={'Loss': [loss]},
        attrs={'epsilon': epsilon})
    return loss


def teacher_student_sigmoid_loss(input,
                                 label,
                                 soft_max_up_bound=15.0,
                                 soft_max_lower_bound=-15.0):
    """
    **Teacher Student Log Loss Layer**

    This layer accepts input predictions and target label and returns the
    teacher_student loss. Z is click or not, z' is value of teacher loss, label = {-2, -1, [0, 2]}
    when z' is not exist, clk = 0 : label = -2; when z' is not exist, clk = 1 : label = -1;
    when z' is exist    , clk = 0 : label = 0 + z'; when z' is exist    , clk = 1 : label = 1 + z'

    .. math::
        loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x 1], where N is the
                                batch size. This input is a probability computed
                                by the previous operator.
        label (Variable|list):  the ground truth which is a 2-D tensor with
                                shape [N x 1], where N is the batch size.
        soft_max_up_bound  (float):  if input > soft_max_up_bound, will be bound
        soft_max_lower_bound (float): if input < soft_max_lower_bound, will be bound

    Returns:
        Variable: A 2-D tensor with shape [N x 1], the teacher_student_sigmoid_loss.

    Examples:
        .. code-block:: python
          
          import paddle.fluid as fluid

          batch_size = 64
          label = fluid.data(
                    name="label", shape=[batch_size, 1], dtype="int64")
          similarity = fluid.data(
                    name="similarity", shape=[batch_size, 1], dtype="float32")
          cost = fluid.layers.teacher_student_sigmoid_loss(input=similarity, label=label)

    """
    helper = LayerHelper('teacher_student_sigmoid_loss', **locals())
    out = helper.create_variable(dtype=input.dtype)
    helper.append_op(
        type='teacher_student_sigmoid_loss',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs={"soft_max_lower_bound": float(soft_max_lower_bound), \
                "soft_max_up_bound": float(soft_max_up_bound)})
    return out


def add_position_encoding(input, alpha, beta, name=None):
    """
    This operator performs weighted sum of input feature at each position
    (position in the sequence) and the corresponding position encoding.

    For more details of position encoding, please refer to `Attention Is All You 
    Need <http://arxiv.org/pdf/1706.03762.pdf>`_ .

    The formula is as follows:

    .. math::
        PE(pos, 2i) &= \\sin{(pos / 10000^{2i / P})}   \\\\
        PE(pos, 2i + 1) &= \\cos{(pos / 10000^{2i / P})}  \\\\
        Out(:, pos, i) &= \\alpha * input(:, pos, i) + \\beta * PE(pos, i)

    Where:
      - :math:`PE(pos, 2i)` : the value at even index `2i` for encoding of position `pos`.
      - :math:`PE(pos, 2i + 1)` : the value at odd index `2i+1` for encoding of position `pos`

    Args:
        input(Variable): A Tensor or LoDTensor (lod level is 1). If it is a
            Tensor, the shape should be `[N, M, P]`, where `N` stands for
            batch size, `M` for sequence length, `P` for the size of feature
            dimension. If it is a LoDTensor, the shape should be `[N, P]`,
            where `N` stands for the total sequence lengths in this mini-batch,
            `P` for the size of feature. The data type should be float32 or float64.
        alpha(float): Indicate the weight coefficient for `input` when performing
            weighted sum.
        beta(float): Indicate the weight coefficient for position encoding when
            performing weighted sum.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.

    Returns:
        Variable: A Tensor or LoDTensor. It has the same shape, data type and lod as `input`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid

          tensor = fluid.data(
              name='tensor',
              shape=[None, 64, 512],
              dtype='float32')
          position_tensor = fluid.layers.add_position_encoding(
              input=tensor, alpha=1.0, beta=1.0)

    """
    helper = LayerHelper('add_position_encoding', **locals())
    dtype = helper.input_dtype()

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable(name=name, dtype=dtype, persistable=False)

    helper.append_op(
        type="add_position_encoding",
        inputs={"X": input},
        outputs={"Out": out},
        attrs={"alpha": alpha,
               "beta": beta})
    return out


def bilinear_tensor_product(x,
                            y,
                            size,
                            act=None,
                            name=None,
                            param_attr=None,
                            bias_attr=None):
    """
    **Bilinear Tensor Product Layer**

    This layer performs bilinear tensor product on two inputs.
    For example:

    .. math::
       out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

    In this formula:
      - :math:`x`: the first input contains M elements, shape is [batch_size, M].
      - :math:`y`: the second input contains N elements, shape is [batch_size, N].
      - :math:`W_{i}`: the i-th learned weight, shape is [M, N].
      - :math:`out_{i}`: the i-th element of out, shape is [batch_size, size].
      - :math:`y^\mathrm{T}`: the transpose of :math:`y_{2}`.

    Args:
        x (Variable): 2-D input tensor with shape [batch_size, M]. Data type 
            is float32 or float64.
        y (Variable): 2-D input tensor with shape [batch_size, N]. Data type 
            should be same as **x**.
        size (int): The dimension of this layer.
        act (str|None): Activation to be applied to the output of this layer. Default None.
        name(str|None): For detailed information, please refer to 
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        param_attr (ParamAttr|None): To specify the weight parameter attribute. 
            Default: None, which means the default weight parameter property is 
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr|None): To specify the bias parameter attribute. 
            Default: None, which means the default bias parameter property is 
            used. See usage for details in :ref:`api_fluid_ParamAttr` .
    Returns:
        Variable: A 2-D Tensor of shape [batch_size, size]. Data type is the same as input **x**.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          layer1 = fluid.data("t1", shape=[-1, 5], dtype="float32")
          layer2 = fluid.data("t2", shape=[-1, 4], dtype="float32")
          tensor = fluid.layers.bilinear_tensor_product(x=layer1, y=layer2, size=1000)
    """
    helper = LayerHelper('bilinear_tensor_product', **locals())
    dtype = helper.input_dtype('x')

    param_shape = [size, x.shape[1], y.shape[1]]

    w = helper.create_parameter(
        attr=helper.param_attr, shape=param_shape, dtype=dtype, is_bias=False)

    if name is None:
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable(name=name, dtype=dtype, persistable=False)

    inputs = {"X": x, "Y": y, "Weight": w}
    if helper.bias_attr:
        bias_size = [1, size]
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)
        inputs["Bias"] = bias
    helper.append_op(
        type="bilinear_tensor_product", inputs=inputs, outputs={"Out": out})

    # add activation
    return helper.append_activation(out)


@templatedoc()
def get_tensor_from_selected_rows(x, name=None):
    """
    This operator gets tensor data from input with SelectedRows type, and outputs a LoDTensor.

    .. code-block:: text

        input x is SelectedRows:
           x.rows = [0, 5, 5, 4, 19]
           x.height = 20
           x.value = [[1, 1] [2, 2] [2, 2] [3, 3] [6, 6]]

        Ouput is LoDTensor:
           out.shape = [5, 2]
           out.data = [[1, 1],
                       [2, 2],
                       [2, 2],
                       [3, 3],
                       [6, 6]]

    Args:
        x(SelectedRows): Input with SelectedRows type. The data type is float32, float64, int32 or int64.
        name(str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: LoDTensor transformed from SelectedRows. The data type is same with input.

    Examples:
        .. code-block:: python
	    
            import paddle.fluid as fluid
            b = fluid.default_main_program().global_block()
            input = b.create_var(name="X", dtype="float32", persistable=True, type=fluid.core.VarDesc.VarType.SELECTED_ROWS)
            out = fluid.layers.get_tensor_from_selected_rows(input)
    """

    helper = LayerHelper('get_tensor_from_selected_rows', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='get_tensor_from_selected_rows',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={})
    return out


def shuffle_channel(x, group, name=None):
    """
    This operator shuffles the channels of input x.
    It divide the input channels in each group into :attr:`group` subgroups,
    and obtain a new order by selecting element from every subgroup one by one.

    Please refer to the paper
    https://arxiv.org/pdf/1707.01083.pdf
    
    .. code-block:: text

        Given a 4-D tensor input with the shape (N, C, H, W):
            input.shape = (1, 4, 2, 2)
            input.data =[[[[0.1, 0.2],
                           [0.2, 0.3]],

                          [[0.3, 0.4],
                           [0.4, 0.5]],

                          [[0.5, 0.6],
                           [0.6, 0.7]],

                          [[0.7, 0.8],
                           [0.8, 0.9]]]]
            Given group: 2
            then we get a 4-D tensor out whth the same shape of input:
            out.shape = (1, 4, 2, 2)
            out.data = [[[[0.1, 0.2],
                          [0.2, 0.3]],
                          
                         [[0.5, 0.6],
                          [0.6, 0.7]],
                          
                         [[0.3, 0.4],
                          [0.4, 0.5]],
                          
                         [[0.7, 0.8],
                          [0.8, 0.9]]]]
                        
    Args: 
        x(Variable): The input tensor variable. It should be a 4-D tensor with shape [N, C, H, W]
        group(int): Indicating the conuts of subgroups, It should divide the number of channels.

    Returns:
        out(Variable): the channels shuffling result is a tensor variable with the 
        same shape and same type as the input.

    Raises:
        ValueError: If group is not an int type variable.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(name='input', shape=[None,4,2,2], dtype='float32')
            out = fluid.layers.shuffle_channel(x=input, group=2)
    """
    helper = LayerHelper("shuffle_channel", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(group, int):
        raise TypeError("group must be int type")

    helper.append_op(
        type="shuffle_channel",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"group": group})
    return out


@templatedoc()
def temporal_shift(x, seg_num, shift_ratio=0.25, name=None):
    """
    **Temporal Shift Operator**
    
    ${comment}
                        
    Args: 
        x(Variable): ${x_comment}
        seg_num(int): ${seg_num_comment}
        shift_ratio(float): ${shift_ratio_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        out(Variable): The temporal shifting result is a tensor variable with the 
        same shape and same data type as the input.

    Raises:
        TypeError: seg_num must be int type.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            input = fluid.data(name='input', shape=[None,4,2,2], dtype='float32')
            out = fluid.layers.temporal_shift(x=input, seg_num=2, shift_ratio=0.2)
    """
    helper = LayerHelper("temporal_shift", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(seg_num, int):
        raise TypeError("seg_num must be int type.")

    helper.append_op(
        type="temporal_shift",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"seg_num": seg_num,
               "shift_ratio": shift_ratio})
    return out


class PyFuncRegistry(object):
    _register_funcs = []

    def __init__(self, func):
        if func is None or not callable(func):
            raise TypeError('func must be a Python function')

        self._func = func
        # find named args using reflection
        args = inspect.getargspec(self._func)
        if len(args[0]) == 0 and args[1] is None and args[2] is None:
            # Function with no inputs
            self._named_args = None
        else:
            self._named_args = args[0]
        self._id = core._append_python_callable_object_and_return_id(self)
        '''
        Why record self here?

        1. For debug usage. Users can call
           :code:`py_func.registered_func(idx)` method
           to find the registered function corresponding
           to :code:`idx`.

        2. For increasing reference count of self.
           It seems that to release Python object
           whose reference count is 1 would cause
           segmentation fault error in C++ side.
           May be lack of Python GC in C++ side?
        '''
        PyFuncRegistry._register_funcs.append(self)

    @classmethod
    def registered_func(cls, idx):
        return cls._register_funcs[idx]._func

    @classmethod
    def registered_func_num(cls):
        return len(cls._register_funcs)

    @property
    def id(self):
        return self._id

    def __call__(self, *args):
        if self._named_args is None:
            func_ret = self._func()
        else:
            kwargs = dict()
            idx = 0
            for arg in self._named_args:
                kwargs[arg] = args[idx]
                idx += 1
            func_ret = self._func(*args[idx:], **kwargs)

        if not isinstance(func_ret, (list, tuple)):
            func_ret = (func_ret, )

        ret = []
        for each_ret in func_ret:
            if each_ret is None or isinstance(each_ret, core.LoDTensor):
                ret.append(each_ret)
                continue

            if not isinstance(each_ret, np.ndarray):
                each_ret = np.array(each_ret)

            tensor = core.LoDTensor()
            tensor.set(each_ret, core.CPUPlace())
            ret.append(tensor)

        return tuple(ret)


@templatedoc()
def py_func(func, x, out, backward_func=None, skip_vars_in_backward_input=None):
    """
    This API is used to register customized OP to Fluid. The forward  function 
    of the registered OP is ``func`` and the backward function of that is 
    ``backward_func``. Paddle will call ``func`` at forward runtime  and call 
    ``backward_func`` at backward runtime(if ``backward_func`` is not  None). 
    ``x`` is the input of ``func``, whose type must be LoDTensor; ``out`` is 
    the output of ``func``, whose type can be either LoDTensor or NumPy array.

    The input of the backward function ``backward_func`` is ``x``, ``out`` and 
    the gradient of ``out``. If some variables of ``out`` have no gradient, the 
    relevant input variable of ``backward_func`` is None. If some variables of 
    ``x`` do not have a gradient, the user should return None in ``backward_func``.

    The data type and shape of ``out`` should also be set correctly before this 
    API is called, and the data type and shape of the gradient of ``out`` and 
    ``x`` will be inferred automatically.

    This API can also be used to debug the neural network by setting the ``func``
    as a function that only print variables.

    Args:
        func (callable): The forward function of the registered OP. When the network
            is running, the forward output ``out`` will be calculated according to this 
            function and the forward input ``x``.
        x (Variable): The input of the forward function ``func``, its type can be 
            Variable | tuple[Variable] | list[Variale], in which Variable is LoDTensor.
        out (Variable): The output of the forward function ``func``, its type can be
            Variable | tuple[Variable] | list[Variale], in which Variable can be either 
            LoDTensor or NumPy array. Since Paddle cannot automatically infer the shape
            and data type of ``out``, ``out`` must be created in advance.
        backward_func (callable, optional): The backward function of the registered OP. 
            Its default value is None, which means there is no reverse calculation. If 
            it is not None, ``backward_func`` is called to calculate the gradient of 
            ``x`` when the network is at backward runtime.
        skip_vars_in_backward_input (Variable, optional): It's used to limit the input 
            variable list of ``backward_func``, and it can be single Variable, tuple[Variable]
            or list[Variable]. It must belong to either ``x`` or ``out``. The default 
            value is None, which means that no variables need to be removed from ``x`` 
            and ``out``. If it is not None, these variables will not be the input of 
            ``backward_func``. This parameter is only useful when ``backward_func`` is 
            not None.
    
    Returns: 
        Variable: The output ``out`` of the forward function ``func``.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import six

            def create_tmp_var(name, dtype, shape):
            return fluid.default_main_program().current_block().create_var(
            name=name, dtype=dtype, shape=shape)

            # Tanh activation function provided by Paddle C++ op
            # Here, tanh is used as an example to show how to use py_func
            def tanh(x):
                return np.tanh(x)

            # Skip forward input x
            def tanh_grad(y, dy):
                return np.array(dy) * (1 - np.square(np.array(y)))

            def debug_func(x):
                print(x)

            def simple_net(img, label):
                hidden = img
                for idx in six.moves.range(4):
                    hidden = fluid.layers.fc(hidden, size=200)
                    new_hidden = create_tmp_var(name='hidden_{}'.format(idx),
                        dtype=hidden.dtype, shape=hidden.shape)

                    # User-defined forward and backward 
                    hidden = fluid.layers.py_func(func=tanh, x=hidden,
                        out=new_hidden, backward_func=tanh_grad,
                        skip_vars_in_backward_input=hidden)

                    # User-defined debugging layer, which can print out variable details
                    fluid.layers.py_func(func=debug_func, x=hidden, out=None)

                prediction = fluid.layers.fc(hidden, size=10, act='softmax')
                loss = fluid.layers.cross_entropy(input=prediction, label=label)
                return fluid.layers.mean(loss)
    """
    helper = LayerHelper('py_func', **locals())
    if x is None:
        x = []
    elif isinstance(x, Variable):
        x = [x]
    elif not isinstance(x, (list, tuple)):
        raise TypeError('Input must be Variable/list(Variable)/tuple(Variable)')

    if out is None:
        out_list = []
    elif isinstance(out, Variable):
        out_list = [out]
    elif isinstance(out, (list, tuple)):
        out_list = out
    else:
        raise TypeError(
            'Output must be Variable/list(Variable)/tuple(Variable)')

    fwd_func_id = PyFuncRegistry(func).id
    bwd_func_id = PyFuncRegistry(
        backward_func).id if backward_func is not None else -1

    for each_out in out_list:
        if len(each_out.shape) == 0:
            raise ValueError(
                'Output shapes of py_func op should be provided by users manually'
            )

    backward_skip_vars = set()
    if backward_func is not None and skip_vars_in_backward_input is not None:
        if isinstance(skip_vars_in_backward_input, Variable):
            skip_vars_in_backward_input = [skip_vars_in_backward_input]

        fwd_in_out = [v.name for v in x]
        fwd_in_out.extend([v.name for v in out_list])
        fwd_in_out = set(fwd_in_out)
        backward_skip_vars = set()
        for v in skip_vars_in_backward_input:
            if not v.name in fwd_in_out:
                raise ValueError(
                    'Variable {} is not found in forward inputs and outputs'
                    .format(v.name))
            backward_skip_vars.add(v.name)

    helper.append_op(
        type='py_func',
        inputs={'X': x},
        outputs={'Out': out_list},
        attrs={
            'forward_callable_id': fwd_func_id,
            'backward_callable_id': bwd_func_id,
            'backward_skip_vars': list(backward_skip_vars)
        })
    return out


# For debug usage
py_func.registered_func = PyFuncRegistry.registered_func
py_func.registered_func_num = PyFuncRegistry.registered_func_num


@templatedoc()
def psroi_pool(input,
               rois,
               output_channels,
               spatial_scale,
               pooled_height,
               pooled_width,
               name=None):
    """
    ${comment}

    Parameters:
        input (Variable): ${x_comment}
        rois (Variable): LoDTensor, ROIs (Regions of Interest) to pool over.It should be
                         a 2-D LoDTensor of shape (num_rois, 4), the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates. The data type is the same as `input`
        output_channels (int): ${output_channels_comment}
        spatial_scale (float): ${spatial_scale_comment} Default: 1.0
        pooled_height (int): ${pooled_height_comment} Default: 1
        pooled_width (int): ${pooled_width_comment} Default: 1
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name`

    Returns:
        ${out_comment}.

    Return Type:
        Variable

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[100, 490, 28, 28], dtype='float32')
            rois = fluid.data(name='rois', shape=[None, 4], lod_level=1, dtype='float32')
            pool_out = fluid.layers.psroi_pool(x, rois, 10, 1.0, 7, 7)
    """
    helper = LayerHelper('psroi_pool', **locals())
    # check attrs
    if not isinstance(output_channels, int):
        raise TypeError("output_channels must be int type")
    if not isinstance(spatial_scale, float):
        raise TypeError("spatial_scale must be float type")
    if not isinstance(pooled_height, int):
        raise TypeError("pooled_height must be int type")
    if not isinstance(pooled_width, int):
        raise TypeError("pooled_width must be int type")
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='psroi_pool',
        inputs={'X': input,
                'ROIs': rois},
        outputs={'Out': out},
        attrs={
            'output_channels': output_channels,
            'spatial_scale': spatial_scale,
            'pooled_height': pooled_height,
            'pooled_width': pooled_width
        })
    return out


@templatedoc()
def prroi_pool(input,
               rois,
               spatial_scale=1.0,
               pooled_height=1,
               pooled_width=1,
               name=None):
    """
    The precise roi pooling implementation for paddle?https://arxiv.org/pdf/1807.11590.pdf

    Args:
        input (Variable):The input of Deformable PSROIPooling.The shape of input tensor is
                        [N,C,H,W]. Where N is batch size,C is number of input channels,H
                        is height of the feature, and W is the width of the feature.
        rois (Variable): ROIs (Regions of Interest) to pool over.It should be
                        a 2-D LoDTensor of shape (num_rois, 4), the lod level
                        is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                        the top left coordinates, and (x2, y2) is the bottom
                        right coordinates.
        spatial_scale (float): Ratio of input feature map height (or width) to raw image height (or width).
                             Equals the reciprocal of total stride in convolutional layers, Default: 1.0.
        pooled_height (integer): The pooled output height. Default: 1.
        pooled_width (integer): The pooled output width. Default: 1.
        name (str, default None): The name of this operation.

    Returns:
        Variable(Tensor): The shape of the returned Tensor is (num_rois, output_channels, pooled_h, pooled_w), with value type float32,float16..

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.layers.data(name='x', shape=[490, 28, 28], dtype='float32')
            rois = fluid.layers.data(name='rois', shape=[4], lod_level=1, dtype='float32')
            pool_out = fluid.layers.prroi_pool(x, rois, 1.0, 7, 7)
    """
    helper = LayerHelper('prroi_pool', **locals())
    # check attrs
    if not isinstance(spatial_scale, float):
        raise TypeError("spatial_scale must be float type")
    if not isinstance(pooled_height, int):
        raise TypeError("pooled_height must be int type")
    if not isinstance(pooled_width, int):
        raise TypeError("pooled_width must be int type")
    dtype = helper.input_dtype()
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type='prroi_pool',
        inputs={'X': input,
                'ROIs': rois},
        outputs={'Out': out},
        attrs={
            'spatial_scale': spatial_scale,
            'pooled_height': pooled_height,
            'pooled_width': pooled_width
        })
    return out


def huber_loss(input, label, delta):
    """
    This operator computes the Huber loss between input and label.
    Huber loss is commonly used in regression tasks. Compared to square_error_cost, Huber loss is more robust and less sensitivity to outliers.

    When the absolute difference between input and label is greater than delta, the linear error is calculated:

    .. math::
            huber\_loss = delta * (label - input) - 0.5 * delta * delta

    When the absolute difference between input and label is greater than delta, the square error is calculated:

    .. math::
            huber\_loss = 0.5 * (label - input) * (label - input)


    Args:
        input (Variable): Predicted data, 2D-Tensor with the shape of [batch_size, 1]. The data type should be float32 or float64.
        label (Variable): Ground truth label, 2D-Tensor with the shape of [batch_size, 1]. The data type should be float32 or float64.
        delta (float): The threshold for Huber loss, which is used to control the balance between the linear error and square error. The data type should be float32.

    Returns:
        Variable: The huber loss, a tensor with the same shape and data type as input.


    Examples:

    ..  code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        DATATYPE='float32'
        input_data = np.array([[1.],[2.],[3.],[4.]]).astype(DATATYPE)
        label_data = np.array([[3.],[3.],[4.],[4.]]).astype(DATATYPE)

        x = fluid.data(name='input', shape=[None, 1], dtype=DATATYPE)
        y = fluid.data(name='label', shape=[None, 1], dtype=DATATYPE)
        loss = fluid.layers.huber_loss(input=x, label=y, delta=1.0)

        place = fluid.CPUPlace()
        #place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        HuberLoss, = exe.run(feed={'input':input_data ,'label':label_data}, fetch_list=[loss.name])
        print(HuberLoss)  #[[1.5], [0.5], [0.5], [0. ]], dtype=float32
    """
    helper = LayerHelper('huber_loss', **locals())
    residual = helper.create_variable_for_type_inference(
        dtype=helper.input_dtype())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='huber_loss',
        inputs={'X': input,
                'Y': label},
        outputs={'Out': out,
                 'Residual': residual},
        attrs={'delta': delta})
    return out


@templatedoc()
def kldiv_loss(x, target, reduction='mean', name=None):
    """
    ${comment}

    Args:
        x (Variable): ${x_comment}
        target (Variable): ${target_comment}
        reduction (Variable): ${reduction_comment}
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.

    Returns:
        Variable(Tensor): The KL divergence loss. The data type is same as input tensor

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None,4,2,2], dtype='float32')
            target = fluid.layers.data(name='target', shape=[4,2,2], dtype='float32')
            loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='batchmean')
    """
    helper = LayerHelper('kldiv_loss', **locals())
    loss = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='kldiv_loss',
        inputs={'X': x,
                'Target': target},
        outputs={'Loss': loss},
        attrs={'reduction': reduction})
    return loss


from .ops import square
from .control_flow import equal


def npair_loss(anchor, positive, labels, l2_reg=0.002):
    '''
  **Npair Loss Layer**

  Read `Improved Deep Metric Learning with Multi class N pair Loss Objective\
       <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/\
       papers/nips16_npairmetriclearning.pdf>`_ .

  Npair loss requires paired data. Npair loss has two parts: the first part is L2
  regularizer on the embedding vector; the second part is cross entropy loss which
  takes the similarity matrix of anchor and positive as logits.

  Args:
    anchor(Variable): embedding vector for the anchor image. shape=[batch_size, embedding_dims], 
                      the data type is float32 or float64.
    positive(Variable): embedding vector for the positive image. shape=[batch_size, embedding_dims], 
                      the data type is float32 or float64.
    labels(Variable): 1-D tensor. shape=[batch_size], the data type is float32 or float64 or int64.
    l2_reg(float32): L2 regularization term on embedding vector, default: 0.002.

  Returns:
    A Variable holding Tensor representing the npair loss, the data type is the same as 
    anchor, the shape is [1].

  Examples:
    .. code-block:: python

       import paddle.fluid as fluid
       anchor = fluid.data(
                     name = 'anchor', shape = [18, 6], dtype = 'float32')
       positive = fluid.data(
                     name = 'positive', shape = [18, 6], dtype = 'float32')
       labels = fluid.data(
                     name = 'labels', shape = [18], dtype = 'float32')

       npair_loss = fluid.layers.npair_loss(anchor, positive, labels, l2_reg = 0.002)
  '''
    Beta = 0.25
    batch_size = labels.shape[0]

    labels = reshape(labels, shape=[batch_size, 1], inplace=True)
    labels = expand(labels, expand_times=[1, batch_size])

    labels = equal(labels, transpose(labels, perm=[1, 0])).astype('float32')
    labels = labels / reduce_sum(labels, dim=1, keep_dim=True)

    l2loss = reduce_mean(reduce_sum(square(anchor), 1)) \
             + reduce_mean(reduce_sum(square(positive), 1))
    l2loss = l2loss * Beta * l2_reg

    similarity_matrix = matmul(
        anchor, positive, transpose_x=False, transpose_y=True)
    softmax_ce = softmax_with_cross_entropy(
        logits=similarity_matrix, label=labels, soft_label=True)
    cross_entropy = reduce_sum(labels * softmax_ce, 0)
    celoss = reduce_mean(cross_entropy)

    return l2loss + celoss


def pixel_shuffle(x, upscale_factor):
    """

    This op rearranges elements in a tensor of shape [N, C, H, W]
    to a tensor of shape [N, C/r**2, H*r, W*r].
    This is useful for implementing efficient sub-pixel convolution
    with a stride of 1/r.
    Please refer to the paper: `Real-Time Single Image and Video Super-Resolution 
    Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158v2>`_ .
    by Shi et. al (2016) for more details.

    Parameters:

        x(Variable): 4-D tensor, the data type should be float32 or float64.
        upscale_factor(int): factor to increase spatial resolution.

    Returns:
        Out(Variable): Reshaped tensor according to the new dimension.

    Raises:
        ValueError: If the square of upscale_factor cannot divide the channels of input.

    Examples:
        .. code-block:: python

	    # declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[2,9,4,4])
	    output = fluid.layers.pixel_shuffle(x=input, upscale_factor=3)
	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.random.rand(2,9,4,4).astype("float32")
	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data},
                fetch_list=[output],
                return_numpy=True)
 
 	    # print(output.shape)
	    # (2L, 1L, 12L, 12L)

    """

    helper = LayerHelper("pixel_shuffle", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    if not isinstance(upscale_factor, int):
        raise TypeError("upscale factor must be int type")

    helper.append_op(
        type="pixel_shuffle",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={"upscale_factor": upscale_factor})
    return out


def fsp_matrix(x, y):
    """

    **FSP matrix op**

    This op is used to calculate the flow of solution procedure (FSP) matrix of two 4-D Tensor feature maps.
    Given feature map x with shape [x_channel, h, w] and feature map y with shape
    [y_channel, h, w], we can get the fsp matrix of x and y in two steps:

    1. reshape x into matrix with shape [x_channel, h * w] and reshape and
       transpose y into matrix with shape [h * w, y_channel].
    2. multiply x and y to get fsp matrix with shape [x_channel, y_channel].

    The output is a batch of fsp matrices.

    Args:

        x (Variable): A 4-D Tensor feature map with shape [batch_size, x_channel, height, width].
                      A Tensor with type float32, float64.
        y (Variable): A 4-D Tensor feature map with shape [batch_size, y_channel, height, width].
                      The y_channel can be different with the x_channel of Input(X)
                      while the other dimensions must be the same with Input(X)'s. A Tensor with
                      type float32, float64.

    Returns:

        fsp matrix (Variable): The output of FSP op with shape [batch_size, x_channel, y_channel].
        The x_channel is the channel of x and the y_channel is the channel of y. A Tensor with
        type float32, float64.

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            data = fluid.data(name='data', shape=[None, 3, 32, 32])
            feature_map_0 = fluid.layers.conv2d(data, num_filters=2,
                                                filter_size=3)
            feature_map_1 = fluid.layers.conv2d(feature_map_0, num_filters=2,
                                                filter_size=1)
            loss = fluid.layers.fsp_matrix(feature_map_0, feature_map_1)

    """
    helper = LayerHelper('fsp_matrix', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype(
        input_param_name='x'))
    helper.append_op(type='fsp', inputs={'X': x, 'Y': y}, outputs={'Out': out})
    return out


def continuous_value_model(input, cvm, use_cvm=True):
    """

    **continuous_value_model layers**

    Now, this OP is used in CTR project to remove or dispose show and click value in :attr:`input`.

    :attr:`input` is an embedding vector including show and click value, whose shape is :math:`[N, D]` (N is batch size. D is `2 + embedding dim` ).
    Show and click at first two dims of embedding vector D.
    If :attr:`use_cvm` is True, it will caculate :math:`log(show)` and :math:`log(click)` , and output shape is :math:`[N, D]` .
    If :attr:`use_cvm` is False, it will remove show and click from :attr:`input` , and output shape is :math:`[N, D - 2]` .
    :attr:`cvm` is show_click info, whose shape is :math:`[N, 2]` .

    Args:
        input (Variable): The input variable. A 2-D LoDTensor with shape :math:`[N, D]` , where N is the batch size, D is `2 + the embedding dim` . `lod level = 1` .
        A Tensor with type float32, float64.
        cvm (Variable): Show and click variable. A 2-D Tensor with shape :math:`[N, 2]` , where N is the batch size, 2 is show and click.
        A Tensor with type float32, float64.
        use_cvm  (bool):  Use show_click or not. if use, the output dim is the same as input.
                          if not use, the output dim is `input dim - 2` (remove show and click)

    Returns:

        Variable: A 2-D LodTensor with shape :math:`[N, M]` . if :attr:`use_cvm` = True, M is equal to input dim D. if False, M is equal to `D - 2`. \
        A Tensor with same type as input.

    Examples:

        .. code-block:: python

          import paddle.fluid as fluid
          input = fluid.data(name="input", shape=[64, 1], dtype="int64")
          label = fluid.data(name="label", shape=[64, 1], dtype="int64")
          embed = fluid.layers.embedding(
                            input=input,
                            size=[100, 11],
                            dtype='float32')
          ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=[-1, 1], dtype="int64", value=1)
          show_clk = fluid.layers.cast(fluid.layers.concat([ones, label], axis=1), dtype='float32')
          show_clk.stop_gradient = True
          input_with_cvm = fluid.layers.continuous_value_model(embed, show_clk, True)

    """
    helper = LayerHelper('cvm', **locals())
    out = helper.create_variable(dtype=input.dtype)
    helper.append_op(
        type='cvm',
        inputs={'X': [input],
                'CVM': [cvm]},
        outputs={'Y': [out]},
        attrs={"use_cvm": use_cvm})
    return out


def where(condition):
    """
    Return an int64 tensor with rank 2, specifying the coordinate of true element in `condition`.

    Args:
        condition(Variable): A bool tensor with rank at least 1, the data type is bool.

    Returns:
        Variable, the output data type is int64. : The tensor variable storing a 2-D tensor, which involves all coordinate. 

    Examples:
        .. code-block:: python

             import paddle.fluid as fluid
             import paddle.fluid.layers as layers
             import numpy as np

             # condition is a tensor [True, False, True]
             condition = layers.assign(np.array([1, 0, 1], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[0], [2]]

             # condition is a tensor [[True, False], [False, True]]
             condition = layers.assign(np.array([[1, 0], [0, 1]], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[0, 0], [1, 1]]

             # condition is a tensor [False, False, False]
             condition = layers.assign(np.array([0, 0, 0], dtype='int32'))
             condition = layers.cast(condition, 'bool')
             out = layers.where(condition) # [[]]

    """
    helper = LayerHelper("where", **locals())

    out = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.INT64)

    helper.append_op(
        type='where', inputs={'Condition': condition}, outputs={'Out': [out]})
    return out


def sign(x):
    """
    This OP returns sign of every element in `x`: 1 for positive, -1 for negative and 0 for zero.

    Args:
        x(Variable|numpy.ndarray): The input variable could be N-D tensor or N-D numpy array, \
            the input data type is float32 or float64.

    Returns:
        Variable, the output data type is the same as input data type. : The output sign tensor with identical shape to input :attr:`x`.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          # [1.0, 0.0, -1.0]
          data = fluid.layers.sign(np.array([3.0, 0.0, -2.0], dtype='float32')) 
    """

    helper = LayerHelper("sign", **locals())

    if not isinstance(x, Variable):
        if isinstance(x, np.ndarray):
            x = assign(x)
        else:
            raise TypeError(
                "The type of 'x' in sign_op must be Variable or numpy.ndarray, but received %s."
                % (type(x)))

    if convert_dtype(x.dtype) in ['float16']:
        warnings.warn(
            "The data type of 'x' in sign_op only support float16 in GPU now.")
    if convert_dtype(x.dtype) not in ['float16', 'float32', 'float64']:
        raise TypeError(
            "The data type of 'x' in sign_op must be float16, float32 or float64, but received %s."
            % (convert_dtype(x.dtype)))

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(type='sign', inputs={'X': [x]}, outputs={'Out': [out]})

    return out


def unique(x, dtype='int32'):
    """
    **unique** 

    Return a unique tensor for `x` and an index tensor pointing to this unique tensor.

    Args:
        x(Variable): A 1-D input tensor.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of index tensor: int32, int64.

    Returns:
        tuple: (out, index). `out` is the unique tensor for `x`, with identical dtype to `x`, and \
            `index` is an index tensor pointing to `out`, by which user can recover the original `x` tensor.

    Examples:
        .. code-block:: python

             import numpy as np
             import paddle.fluid as fluid
             x = fluid.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
             out, index = fluid.layers.unique(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]
    """

    helper = LayerHelper("unique", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    index = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='unique',
        inputs={'X': x},
        attrs={'dtype': convert_np_dtype_to_dtype_(dtype)},
        outputs={'Out': [out],
                 'Index': [index]})

    return out, index


def unique_with_counts(x, dtype='int32'):
    """
    This OP return a unique tensor for `x` , and count tensor that the count of unqiue result in raw input, \
    and an index tensor pointing to this unique tensor. 

    **NOTICE**: This op just be supported in device of CPU, and support the variable type of Tensor only.

    Args:
        x(Variable): A 1-D input tensor with input shape of :math:`[N]` , the input data type is float32, float64, int32, int64.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of count and index tensor, it could be int32, int64. Defalut value is int32.

    Returns: 
        tuple, the variable type in tuple is Tensor, the output :attr:`out` data type is the same as input :attr:`x`, \
        and data type of output :attr:`index` and :attr:`count` will be int32 or int64.: The :attr:`out` is unique tensor for input :attr:`x`,\
        the data shape is :math:`[K]`, the `K` may be different to the `N` in shape of :attr:`x`. :attr:`index` is an index tensor pointing\
        to :attr:`out`, the data shape is :math:`[N]` , the data shape is the same as input :attr:`x`. :attr:`count` is count of unqiue element in\
        the :attr:`x`, the data shape is :math:`[K]`, the data shape is the same as output :attr:`out`.

    Examples:
        .. code-block:: python

             import numpy as np
             import paddle.fluid as fluid
             x = fluid.layers.assign(np.array([2, 3, 3, 1, 5, 3], dtype='int32'))
             out, index, count = fluid.layers.unique_with_counts(x) # out is [2, 3, 1, 5]; index is [0, 1, 1, 2, 3, 1]
                                                        # count is [1, 3, 1, 1]
            # x.shape=(6,) out.shape=(4,), index.shape=(6,), count.shape=(4,)
    """
    if not (dtype == 'int32' or dtype == 'int64'):
        raise TypeError(
            "Op unique_with_counts, index dtype must be int32 or int64")

    if x is None or len(x.shape) != 1:
        raise ValueError(
            "Op unique_with_counts, x must not be null and size of dim must be 1"
        )

    helper = LayerHelper("unique_with_counts", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    index = helper.create_variable_for_type_inference(dtype)

    count = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type='unique_with_counts',
        inputs={'X': x},
        attrs={'dtype': convert_np_dtype_to_dtype_(dtype)},
        outputs={'Out': [out],
                 'Index': [index],
                 'Count': [count]})

    return out, index, count


def deformable_conv(input,
                    offset,
                    mask,
                    num_filters,
                    filter_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=None,
                    deformable_groups=None,
                    im2col_step=None,
                    param_attr=None,
                    bias_attr=None,
                    modulated=True,
                    name=None):
    """
    **Deformable Convolution op**

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

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`

          Offset shape: :math:`(N, 2 * deformable\_groups * H_f * H_w, H_{in}, W_{in})`

          Mask shape: :math:`(N, deformable\_groups * H_f * H_w, H_{in}, W_{in})`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Args:
        input (Variable): The input image with [N, C, H, W] format. A Tensor with type
            float32, float64.
        offset (Variable): The input coordinate offset of deformable convolution layer.
            A Tensor with type float32, float64.
        Mask (Variable, Optional): The input mask of deformable covolution layer.
            A Tensor with type float32, float64.It should be None when you use
            deformable_conv_v2.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int): The groups number of the deformable conv layer. According to
            grouped convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1.
        deformable_groups (int): The number of deformable group partitions.
            Default: deformable_groups = 1.
        im2col_step (int): Maximum number of images per im2col computation; 
            The total batch size should be divisable by this value or smaller
            than this value; if you face out of memory problem, you can try
            to use a smaller value here.
            Default: im2col_step = 64.
        param_attr (ParamAttr, Optional): The parameter attribute for learnable parameters/weights
            of deformable conv. If it is set to None or one attribute of ParamAttr,
            deformable conv will create ParamAttr as param_attr.
            If the Initializer of the param_attr is not set, the parameter is
            initialized with :math:`Normal(0.0, std)`, and the 
            :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr|bool, Optional): The parameter attribute for the bias of
            deformable conv layer. If it is set to False, no bias will be added
            to the output units. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        modulated (bool): Make sure which version should be used between v1 and v2, where v2 is \
            used while True. Default: True.
        name(str, Optional): For details, please refer to :ref:`api_guide_Name`.
                        Generally, no setting is required. Default: None.
    Returns:
        Variable: The tensor variable storing the deformable convolution \
                  result. A Tensor with type float32, float64.
    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.
    Examples:
        .. code-block:: python

          #deformable conv v2:
         
          import paddle.fluid as fluid
          C_in, H_in, W_in = 3, 32, 32
          filter_size, deformable_groups = 3, 1
          data = fluid.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
          offset = fluid.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          mask = fluid.data(name='mask', shape=[None, deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          out = fluid.layers.deformable_conv(input=data, offset=offset, mask=mask,
                                             num_filters=2, filter_size=filter_size, padding=1, modulated=True)

          #deformable conv v1:

          import paddle.fluid as fluid
          C_in, H_in, W_in = 3, 32, 32
          filter_size, deformable_groups = 3, 1
          data = fluid.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
          offset = fluid.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
          out = fluid.layers.deformable_conv(input=data, offset=offset, mask=None,
                                             num_filters=2, filter_size=filter_size, padding=1, modulated=False)
    """

    num_channels = input.shape[1]
    assert param_attr is not False, "param_attr should not be False here."

    helper = LayerHelper('deformable_conv', **locals())
    dtype = helper.input_dtype()

    if not isinstance(input, Variable):
        raise TypeError("Input of deformable_conv must be Variable")
    if not isinstance(offset, Variable):
        raise TypeError("Input Offset of deformable_conv must be Variable")

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels // groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    input_shape = input.shape
    filter_shape = [num_filters, int(num_filter_channels)] + filter_size

    def _get_default_param_initializer():
        filter_elem_num = filter_size[0] * filter_size[1] * num_channels
        std = (2.0 / filter_elem_num)**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_variable_for_type_inference(dtype)

    if modulated:
        helper.append_op(
            type='deformable_conv',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
                'Mask': mask,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    else:
        helper.append_op(
            type='deformable_conv_v1',
            inputs={
                'Input': input,
                'Filter': filter_param,
                'Offset': offset,
            },
            outputs={"Output": pre_bias},
            attrs={
                'strides': stride,
                'paddings': padding,
                'dilations': dilation,
                'groups': groups,
                'deformable_groups': deformable_groups,
                'im2col_step': im2col_step,
            })

    output = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    return output


def unfold(x, kernel_sizes, strides=1, paddings=0, dilations=1, name=None):
    """

    This op returns a col buffer of sliding local blocks of input x, also known
    as im2col for batched 2D image tensors. For each block under the convolution filter,
    all element will be rearranged as a column. While the convolution filter silding over
    the input feature map, a series of such columns will be formed.

    For each input :math:`x` with shape [N, C, H, W], the output shape [N, Cout, Lout]
    can be calculated as following.

    .. math::

        dkernel[0] &= dilations[0] \\times (kernel\_sizes[0] - 1) + 1

        dkernel[1] &= dilations[1] \\times (kernel\_sizes[1] - 1) + 1

        hout &= \\frac{H + paddings[0] + paddings[2] - dkernel[0]}{strides[0]} + 1

        wout &= \\frac{W + paddings[1] + paddings[3] - dkernel[1]}{strides[1]} + 1

        Cout &= C \\times kernel\_sizes[0] \\times kernel\_sizes[1]

        Lout &= hout \\times wout


    Parameters:
        x(Varaible):              4-D Tensor, input tensor of format [N, C, H, W], 
                                  data type can be float32 or float64
        kernel_sizes(int|list):   The size of convolution kernel, should be [k_h, k_w]
                                  or an integer k treated as [k, k].
        strides(int|list):        The strides, should be [stride_h, stride_w]
                                  or an integer stride treated as [sride, stride].
                                  For default, strides will be [1, 1].
        paddings(int|list):       The paddings of each dimension, should be
                                  [padding_top, padding_left, padding_bottom, padding_right]
                                  or [padding_h, padding_w] or an integer padding.
                                  If [padding_h, padding_w] was given, it will expanded to
                                  [padding_h, padding_w, padding_h, padding_w]. If an integer
                                  padding was given, [padding, padding, padding, padding] will
                                  be used. For default, paddings will be [0, 0, 0, 0]
        dilations(int|list):      the dilations of convolution kernel, shold be
                                  [dilation_h, dilation_w], or an integer dialtion treated as
                                  [dilation, dilation]. For default, it will be [1, 1].
        name(str, optional): The default value is None.  
                             Normally there is no need for user to set this property.  
                             For more information, please refer to :ref:`api_guide_Name`

    
    Returns:
        The tensor variable corresponding to the sliding local blocks. 
        The output shape is [N, Cout, Lout] as decribled above. 
        Cout is the  total number of values within each block, 
        and Lout is the total number of such blocks. 
        The data type of output is the same as the input :math:`x`

    Return Type:
        Variable

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name = 'data', shape = [100, 3, 224, 224], dtype = 'float32')
            y = fluid.layers.unfold(x, [3, 3], 1, 1, 1)
    """

    helper = LayerHelper("unfold", **locals())

    assert len(x.shape) == 4, \
            "input should be the format of [N, C, H, W]"

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes, kernel_sizes]
    else:
        assert isinstance(kernel_sizes, list) and (len(kernel_sizes) == 2), \
            "kernel_sizes should either be an integer or a list of two integers"

    if isinstance(strides, int):
        strides = [strides, strides]
    else:
        assert isinstance(strides, list) and (len(strides) == 2), \
            "strides should either be an integer or a list of two integers"

    if isinstance(dilations, int):
        dilations = [dilations, dilations]
    else:
        assert isinstance(dilations, list) and (len(dilations) == 2), \
            "dilations should either be an integer or a list of two integers"

    if isinstance(paddings, int):
        paddings = [paddings] * 4
    elif isinstance(paddings, list):
        if len(paddings) == 2:
            paddings = paddings * 2
        elif len(paddings) == 4:
            pass
        else:
            raise ValueError(
                "paddings should either be an integer or a list of 2 or 4 integers"
            )
    else:
        raise ValueError(
            "Unexpected type of paddings, it should be either an integer or a list"
            "of 2 or 4 integers")

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="unfold",
        inputs={"X": x},
        outputs={"Y": out},
        attrs={
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "paddings": paddings,
            "dilations": dilations
        })
    return out


def deformable_roi_pooling(input,
                           rois,
                           trans,
                           no_trans=False,
                           spatial_scale=1.0,
                           group_size=[1, 1],
                           pooled_height=1,
                           pooled_width=1,
                           part_size=None,
                           sample_per_part=1,
                           trans_std=0.1,
                           position_sensitive=False,
                           name=None):
    """
    Deformable ROI Pooling Layer
  
    Performs deformable region-of-interest pooling on inputs. As described
    in `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_, it will get offset for each bin after 
    roi pooling so that pooling at correct region. Batch_size will change to the number of region bounding boxes after deformable_roi_pooling.
  
    The operation has three steps:
    
    1. Dividing each region proposal into equal-sized sections with the pooled_width and pooled_height.
  
    2. Add offset to pixel in ROI to get new location and the new value which are computed directly through
       bilinear interpolation with four nearest pixel.
     
    3. Sample several points in each bin to get average values as output.
  
  
    Args:
        input (Variable):The input of deformable roi pooling and it is tensor which value type is float32. The shape of input is
                         [N, C, H, W]. Where N is batch size, C is number of input channels,
                         H is height of the feature, and W is the width of the feature.
        rois (Variable): ROIs (Regions of Interest) with type float32 to pool over. It should be
                         a 2-D LoDTensor of shape (num_rois, 4), and the lod level
                         is 1. Given as [[x1, y1, x2, y2], ...], (x1, y1) is
                         the top left coordinates, and (x2, y2) is the bottom
                         right coordinates, which value type is float32.
        trans (Variable): Offset of features on ROIs while pooling which value type is float32. The format is [N, C, H, W], where 
                          N is number of ROIs, C is number of channels, which indicate the offset distance 
                          in the x and y directions, H is pooled height, and W is pooled width. 
        no_trans (bool): Whether to add offset to get new value or not while roi pooling, which value with type bool is True or False.
                         If value is True, no offset will be added in operation. Default: False.
        spatial_scale (float): Ratio of input feature map height (or width) to raw image height (or width), which value type is float32.
                         Equals the reciprocal of total stride in convolutional layers, Default: 1.0.
        group_size (list|tuple): The number of groups which input channels are divided and the input is list or tuple, which value type is int32. (eg.number of input channels 
                          is k1 * k2 * (C + 1), which k1 and k2 are group width and height and C+1 is number of output
                          chanels.) eg.(4, 6), which 4 is height of group and 6 is width of group. Default: [1, 1].
        pooled_height (int): The pooled output height which value type is int32. Default: 1.
        pooled_width (int): The pooled output width which value type is int32. Default: 1.
        part_size (list|tuple): The height and width of offset which values in list or tuple is int32, eg.(4, 6), which height is 4 and width is 6, and values always equal to pooled_height \
                         and pooled_width. Default: if None, default value is [pooled_height, pooled_width].
        sample_per_part (int): The number of samples in each bin which value type is int32. If value is bigger, it will consume more performance. Default: 1.
        trans_std (float): Coefficient of offset which value type is float32. It controls weight of offset. Default: 0.1.
        position_sensitive (bool): Whether to choose deformable psroi pooling mode or not, and value type is bool(True or False). If value is False, input dimension equals to output dimension. \
                                   If value is True, input dimension shoule be output dimension * pooled_height * pooled_width. Default: False.
        name (str|None): Name of layer. Default: None.
    Returns:
        Variable: Output of deformable roi pooling is that, if position sensitive is False, input dimension equals to output dimension. If position sensitive is True,\
                  input dimension should be the result of output dimension divided by pooled height and pooled width.

    Examples:
      .. code-block:: python

        # position_sensitive=True
        import paddle.fluid as fluid
        input = fluid.data(name="input",
                           shape=[2, 192, 64, 64], 
                           dtype='float32')                   
        rois = fluid.data(name="rois",
                          shape=[-1, 4],
                          dtype='float32', 
                          lod_level=1)
        trans = fluid.data(name="trans",
                           shape=[2, 384, 64, 64], 
                           dtype='float32') 
        x = fluid.layers.deformable_roi_pooling(input=input, 
                                                rois=rois, 
                                                trans=trans, 
                                                no_trans=False,
                                                spatial_scale=1.0, 
                                                group_size=(1, 1),
                                                pooled_height=8,
                                                pooled_width=8,
                                                part_size=(8, 8),
                                                sample_per_part=4, 
                                                trans_std=0.1,
                                                position_sensitive=True)
  
        # position_sensitive=False
        import paddle.fluid as fluid
        input = fluid.data(name="input",
                           shape=[2, 192, 64, 64], 
                           dtype='float32')                   
        rois = fluid.data(name="rois",
                          shape=[-1, 4],
                          dtype='float32', 
                          lod_level=1)
        trans = fluid.data(name="trans",
                           shape=[2, 384, 64, 64], 
                           dtype='float32') 
        x = fluid.layers.deformable_roi_pooling(input=input, 
                                                rois=rois, 
                                                trans=trans, 
                                                no_trans=False,
                                                spatial_scale=1.0, 
                                                group_size=(1, 1),
                                                pooled_height=8,
                                                pooled_width=8,
                                                part_size=(8, 8),
                                                sample_per_part=4, 
                                                trans_std=0.1,
                                                position_sensitive=False)
    """

    input_channels = input.shape[1]
    if position_sensitive == False:
        output_channels = input_channels
    else:
        output_channels = input_channels / pooled_height / pooled_width

    if part_size is None:
        part_height = pooled_height
        part_width = pooled_width
        part_size = [part_height, part_width]
    part_size = utils.convert_to_list(part_size, 2, 'part_size')
    group_size = utils.convert_to_list(group_size, 2, 'group_size')
    helper = LayerHelper('deformable_psroi_pooling', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    top_count = helper.create_variable_for_type_inference(dtype='int32')
    helper.append_op(
        type="deformable_psroi_pooling",
        inputs={"Input": input,
                "ROIs": rois,
                "Trans": trans},
        outputs={"Output": output,
                 "TopCount": top_count},
        attrs={
            "no_trans": no_trans,
            "spatial_scale": spatial_scale,
            "output_dim": output_channels,
            "group_size": group_size,
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "part_size": part_size,
            "sample_per_part": sample_per_part,
            "trans_std": trans_std
        })
    return output


def shard_index(input, index_num, nshards, shard_id, ignore_value=-1):
    """
    This operator recomputes the `input` indices according to the offset of the
    shard. The length of the indices is evenly divided into N shards, and if
    the `shard_id` matches the shard with the input index inside, the index is
    recomputed on the basis of the shard offset, elsewise it is set to
    `ignore_value`. The detail is as follows:
    :: 
        
        shard_size = (index_num + nshards - 1) // nshards
        y = x % shard_size if x // shard_size == shard_id else ignore_value

    NOTE: If the length of indices cannot be evely divided by the shard number,
    the size of the last shard will be less than the calculated `shard_size`

    Examples:
    ::
    
        Input:
          X.shape = [4, 1]
          X.data = [[1], [6], [12], [19]]
          index_num = 20
          nshards = 2
          ignore_value = -1
        
        if shard_id == 0, we get:
          Out.shape = [4, 1]
          Out.data = [[1], [6], [-1], [-1]]
        
        if shard_id == 1, we get:
          Out.shape = [4, 1]
          Out.data = [[-1], [-1], [2], [9]]
    
    Args:
        - **input** (Variable): Input indices, last dimension must be 1.
        - **index_num** (scalar): An interger defining the range of the index.
        - **nshards** (scalar): The number of shards
        - **shard_id** (scalar): The index of the current shard
        - **ignore_value** (scalar): An ingeter value out of sharded index range

    Returns:
        Variable: The sharded index of input.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            batch_size = 32
            label = fluid.data(name="label", shape=[batch_size, 1], dtype="int64")
            shard_label = fluid.layers.shard_index(input=label,
                                                   index_num=20,
                                                   nshards=2,
                                                   shard_id=0)
    """
    op_type = 'shard_index'
    helper = LayerHelper(op_type, **locals())
    if index_num % nshards != 0:
        raise ValueError(
            'The index_num(%d) cannot be evenly divided by nshards(%d)' %
            (index_num, nshards))
    if shard_id < 0 or shard_id >= nshards:
        raise ValueError('The shard_id(%d) should be in [0, %d)' %
                         (shard_id, nshards))

    out = helper.create_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'X': [input]},
        outputs={'Out': out},
        attrs={
            'index_num': index_num,
            'nshards': nshards,
            'shard_id': shard_id,
            'ignore_value': ignore_value
        },
        stop_gradient=True)
    return out


@templatedoc()
def hard_swish(x, threshold=6.0, scale=6.0, offset=3.0, name=None):
    """
    This operator implements the hard_swish activation function.
    Hard_swish is proposed in MobileNetV3, and performs better in computational stability and efficiency compared to swish function.
    For more details please refer to: https://arxiv.org/pdf/1905.02244.pdf

    The formula is as follows:

    .. math::

        out = \\frac{x * (min(max(0, x+offset), threshold))}{scale}

    In the above equation:

    ``threshold`` and ``scale`` should be positive, ``offset`` can be positive or negative. It is recommended to use default parameters.

    Args:
        x (Variable): Input feature, multi-dimensional Tensor. The data type should be float32 or float64.
        threshold (float, optional): The threshold in Relu function. Default: 6.0
        scale (float, optional): The scale factor. Default: 6.0
        offset (float, optional): The offset factor. Default: 3.0
        name (str, optional): The default value is None. Normally there is no need for user to set this property. For more information, please refer to :ref:`api_guide_Name` 
        
    Returns:
        Variable: The output tensor with the same shape and data type as input.
    
    
    Examples:
    
    .. code-block:: python
    
        import paddle.fluid as fluid
        import numpy as np
    
        DATATYPE='float32'
    
        x_data = np.array([i for i in range(1,5)]).reshape([1,1,4]).astype(DATATYPE)
    
        x = fluid.data(name="x", shape=[None,1,4], dtype=DATATYPE)
        y = fluid.layers.hard_swish(x)
    
        place = fluid.CPUPlace()
        #place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        out, = exe.run(feed={'x':x_data}, fetch_list=[y.name])
        print(out)  # [[0.66666667, 1.66666667,3., 4.]]
    """
    helper = LayerHelper('hard_swish', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='hard_swish',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'threshold': threshold,
               'scale': scale,
               'offset': offset})
    return out


def gather_tree(ids, parents):
    """
    To be used after beam search. After beam search, we get selected ids at
    each time step and the corresponding parents in the search tree. Both ids
    and parents have the layout :attr:`[max_time, batch_size, beam_size]`. Then
    :attr:`gather_tree` is used to backtrace from the last time step and
    generate the full sequences by collecting selected ids.

    Here is an example:

    .. code-block:: text

            Given:
                ids = [[[2 2]
                        [6 1]]
                       [[3 9]
                        [6 1]]
                       [[0 1]
                        [9 0]]]
                parents = [[[0 0]
                            [1 1]]
                           [[1 0]
                            [1 0]]
                           [[0 0]
                            [0 1]]]

            Then:                
                gather_tree(ids, parents)  
                         = [[[2 2]
                             [1 6]]
                            [[3 3]
                             [6 1]]
                            [[0 1]
                             [9 0]]]

    Args:
        ids(Variable): A Tensor with shape :attr:`[length, batch_size, beam_size]`
            and data type :attr:`int32` or :attr:`int64`. It contains the selected
            ids of all time steps.
        parents(Variable): A Tensor with the same shape and data type as :attr:`ids`,
            It contains the parents corresponding to selected ids when searching
            among beams.

    Returns:
        Variable: A Tensor with the same shape and data type as :attr:`ids`. \
            It contains the full sequences. The sequences are collected from \
            :attr:`ids` by backtracing according to :attr:`parents`.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            ids = fluid.layers.data(name='ids',
                                    shape=[5, 2, 2],
                                    dtype='int64',
                                    append_batch_size=False)
            parents = fluid.layers.data(name='parents',
                                        shape=[5, 2, 2],
                                        dtype='int64',
                                        append_batch_size=False)
            final_sequences = fluid.layers.gather_tree(ids, parents)
    """
    helper = LayerHelper('gather_tree', **locals())
    out = helper.create_variable_for_type_inference(dtype=ids.dtype)

    helper.append_op(
        type="gather_tree",
        inputs={"Ids": ids,
                "Parents": parents},
        outputs={"Out": out})

    return out


def mse_loss(input, label):
    """
    This op accepts input predications and target label and returns the mean square error.

    The loss can be described as:

    .. math::
        
        Out = MEAN((input - label)^2)

    Parameters: 
        input (Variable): Input tensor, the data type should be float32.
        label (Variable): Label tensor, the data type shoulf be float32.

    Returns:
        Variable: The tensor variable storing the mean square error difference of input and label.

    Return type: Variable.
    
    Examples:
        .. code-block:: python
	    # declarative mode
	    import paddle.fluid as fluid
	    import numpy as np
	    input = fluid.data(name="input", shape=[1])
	    label = fluid.data(name="label", shape=[1])
	    output = fluid.layers.mse_loss(input,label)
	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())
 
	    input_data = np.array([1.5]).astype("float32")
	    label_data = np.array([1.7]).astype("float32")
	    output_data = exe.run(fluid.default_main_program(),
                feed={"input":input_data, "label":label_data},
                fetch_list=[output],
                return_numpy=True)
 
	    print(output_data)
	    # [array([0.04000002], dtype=float32)]
	    
	    # imperative mode
	    import paddle.fluid.dygraph as dg

	    with dg.guard(place) as g:
    		input = dg.to_variable(input_data)
    		label = dg.to_variable(label_data)
    		output = fluid.layers.mse_loss(input, label)
    		print(output.numpy())
	        
	        # [0.04000002]

    """
    return reduce_mean(square_error_cost(input, label))


@templatedoc()
def uniform_random(shape, dtype='float32', min=-1.0, max=1.0, seed=0):
    """
    This OP initializes a variable with random values sampled from a
    uniform distribution in the range [min, max).

    Examples:
    ::
    
        Input:
          shape = [1, 2]
        
        Output:
          result=[[0.8505902, 0.8397286]]

    Args:
        shape (list|tuple|Variable): The shape of the output Tensor,  if the shape is a list or tuple, 
                                     its elements can be an integer
                                     or a Tensor with the shape [1], and the type of the Tensor is int64. 
                                     If the shape is a Variable, it is a 1-D Tensor, and the type of the Tensor is int64.
        dtype(np.dtype|core.VarDesc.VarType|str, optional): The type of the output Tensor. Supported data types: float32, float64.
                                                  Default: float32.
        min (float, optional): The lower bound on the range of random values to generate, the min is included in the range. Default -1.0.
        max (float, optional): The upper bound on the range of random values to generate, the max is excluded in the range. Default 1.0.
        seed (int, optional): Random seed used for generating samples. 0 means use a
            seed generated by the system. Note that if seed is not 0, this
            operator will always generate the same random numbers every time.
            Default 0.

    Returns: 
        Variable: A Tensor of the specified shape filled with uniform_random values.

    Raises:
        TypeError: The shape type should be list or tupple or variable.
    
    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = fluid.layers.uniform_random(shape=[3, 4])

            # example 2:
            # attr shape is a list which contains tensor Variable.
            dim_1 = fluid.layers.fill_constant([1],"int64",3)
            result_2 = fluid.layers.uniform_random(shape=[dim_1, 5])

            # example 3:
            # attr shape is a Variable, the data type must be int64
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            result_3 = fluid.layers.uniform_random(var_shape)

    """
    if not (isinstance(shape, (list, tuple, Variable))):
        raise TypeError(
            "Input shape must be a python list,Variable or tuple. But received %s"
            % (type(shape)))

    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    if convert_dtype(dtype) not in ['float32', 'float64']:
        raise TypeError(
            "The attribute dtype in uniform_random op must be float32 or float64, but received %s."
            % (convert_dtype(dtype)))

    def contain_var(one_list):
        for ele in one_list:
            if isinstance(ele, Variable):
                return True
        return False

    def get_new_shape_tensor(list_shape):
        new_shape_tensor = []
        for dim in list_shape:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_shape_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = helper.create_variable_for_type_inference('int64')
                fill_constant([1], 'int64', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    def get_attr_shape(list_shape):
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, Variable):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
                assert dim_size > 0, (
                    "Each dimension size given in shape must not be negtive "
                    "except one unknown dimension.")
        return attrs_shape

    helper = LayerHelper("uniform_random", **locals())
    inputs = dict()
    attrs = {'seed': seed, 'min': min, 'max': max}
    if in_dygraph_mode():
        attrs = {'shape': shape}
    else:
        if isinstance(shape, Variable):
            shape.stop_gradient = True
            inputs["ShapeTensor"] = shape
        elif isinstance(shape, (list, tuple)):
            assert len(shape) > 0, (
                "The size of argument(shape) can't be zero.")
            attrs["shape"] = get_attr_shape(shape)
            if contain_var(shape):
                inputs['ShapeTensorList'] = get_new_shape_tensor(shape)

    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="uniform_random", inputs=inputs, attrs=attrs,
        outputs={"Out": out})

    return helper.append_activation(out)
