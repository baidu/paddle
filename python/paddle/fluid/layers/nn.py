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

#   Copyright (c ) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from ..layer_helper import LayerHelper
from ..initializer import Normal, Constant
from ..framework import Variable
from ..param_attr import ParamAttr
from .layer_function_generator import autodoc, templatedoc
from .tensor import concat
from . import utils
import random
from .. import unique_name
from functools import reduce

__all__ = [
    'fc',
    'embedding',
    'dynamic_lstm',
    'dynamic_lstmp',
    'dynamic_gru',
    'gru_unit',
    'linear_chain_crf',
    'crf_decoding',
    'cos_sim',
    'cross_entropy',
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
    'batch_norm',
    'beam_search_decode',
    'conv2d_transpose',
    'conv3d_transpose',
    'sequence_expand',
    'lstm_unit',
    'reduce_sum',
    'reduce_mean',
    'reduce_max',
    'reduce_min',
    'reduce_prod',
    'sequence_first_step',
    'sequence_last_step',
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
    'hsigmoid',
    'beam_search',
    'row_conv',
    'multiplex',
    'layer_norm',
    'softmax_with_cross_entropy',
    'smooth_l1',
    'one_hot',
    'autoincreased_step_counter',
    'reshape',
    'lod_reset',
    'lrn',
    'pad',
    'label_smooth',
    'roi_pool',
    'dice_loss',
    'image_resize',
    'image_resize_short',
    'resize_bilinear',
    'gather',
    'random_crop',
    'mean_iou',
    'relu',
    'log',
    'crop',
    'rank_loss',
]


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       use_mkldnn=False,
       act=None,
       is_test=False,
       name=None):
    """
    **Fully Connected Layer**

    This function creates a fully connected layer in the network. It can take
    multiple tensors as its inputs. It creates a variable called weights for
    each input tensor, which represents a fully connected weight matrix from
    each input unit to each output unit. The fully connected layer multiplies
    each input tensor with its coresponding weight to produce an output Tensor.
    If multiple input tensors are given, the results of multiple multiplications
    will be sumed up. If bias_attr is not None, a bias variable will be created
    and added to the output. Finally, if activation is not None, it will be applied
    to the output as well.

    This process can be formulated as follows:

    .. math::

        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})

    In the above equation:

    * :math:`N`: Number of the input.
    * :math:`X_i`: The input tensor.
    * :math:`W`: The weights created by this layer.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output tensor.

    Args:
        input (Variable|list of Variable): The input tensor(s) of this layer, and the dimension of
            the input tensor(s) is at least 2.
        size(int): The number of output units in this layer.
        num_flatten_dims (int, default 1): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input
            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, suppose
            `X` is a 6-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30].
        param_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for learnable
            parameters/weights of this layer.
        bias_attr (ParamAttr|list of ParamAttr, default None): The parameter attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act (str, default None): Activation to be applied to the output of this layer.
        is_test(bool): A flag indicating whether execution is in test phase.
        use_mkldnn(bool): Use mkldnn kernel or not, it is valid only when the mkldnn
            library is installed. Default: False
        name (str, default None): The name of this layer.

    Returns:
        Variable: The transformation result.

    Raises:
        ValueError: If rank of the input tensor is less than 2.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
          fc = fluid.layers.fc(input=data, size=1000, act="tanh")
    """

    helper = LayerHelper("fc", **locals())

    dtype = helper.input_dtype()

    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[num_flatten_dims:], 1)
        ] + [size]

        w = helper.create_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_tmp_variable(dtype)
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
        pre_bias = helper.create_tmp_variable(dtype)
        helper.append_op(
            type="sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": use_mkldnn})
    # add bias
    pre_activation = helper.append_bias_op(pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_activation(pre_activation)


def embedding(input,
              size,
              is_sparse=False,
              is_distributed=False,
              padding_idx=None,
              param_attr=None,
              dtype='float32'):
    """
    **Embedding Layer**

    This layer is used to lookup embeddings of IDs, provided by :attr:`input`, in
    a lookup table. The result of this lookup is the embedding of each ID in the
    :attr:`input`.

    All the input variables are passed in as local variables to the LayerHelper
    constructor.

    Args:
        input(Variable): The tensor variable containing the IDs.
        size(tuple|list): The shape of the look up table parameter. It should
            have two elements which indicate the size of the dictionary of
            embeddings and the size of each embedding vector respectively.
        is_sparse(bool): The flag indicating whether to use sparse update.
        is_distributed(bool): Whether to run lookup table from remote parameter server.
        padding_idx(int|long|None): If :attr:`None`, it makes no effect to lookup.
            Otherwise the given :attr:`padding_idx` indicates padding the output
            with zeros whenever lookup encounters it in :attr:`input`. If
            :math:`padding_idx < 0`, the :attr:`padding_idx` to use in lookup is
            :math:`size[0] + dim`.
        param_attr(ParamAttr): Parameters for this layer
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32, float_16, int etc

    Returns:
        Variable: The tensor variable storing the embeddings of the \
                  supplied inputs.

    Examples:
        .. code-block:: python

          dict_size = len(dataset.ids)
          data = fluid.layers.data(name='ids', shape=[32, 32], dtype='float32')
          fc = fluid.layers.embedding(input=data, size=[dict_size, 16])
    """

    helper = LayerHelper('embedding', **locals())
    w = helper.create_parameter(
        attr=helper.param_attr, shape=size, dtype=dtype, is_bias=False)
    tmp = helper.create_tmp_variable(dtype)
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
            'padding_idx': padding_idx
        })
    return tmp


@templatedoc(op_type="lstm")
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
    ${comment}

    Args:
        input (Variable): ${input_comment}
        size (int): 4 * hidden size.
        h_0(Variable): The initial hidden state is an optional input, default is zero.
                       This is a tensor with shape (N x D), where N is the
                       batch size and D is the hidden size.
        c_0(Variable): The initial cell state is an optional input, default is zero.
                       This is a tensor with shape (N x D), where N is the
                       batch size. `h_0` and `c_0` can be NULL but only at the same time.

        param_attr(ParamAttr|None): The parameter attribute for the learnable
                               hidden-hidden weights.

                               - Weights = {:math:`W_{ch}, W_{ih}, \
                                                W_{fh}, W_{oh}`}
                               - The shape is (D x 4D), where D is the hidden
                                 size.
        bias_attr (ParamAttr|None): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.

                              1. `use_peepholes = False`
                                 - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                 - The shape is (1 x 4D).
                              2. `use_peepholes = True`
                                 - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                 - The shape is (1 x 7D).
        use_peepholes (bool): ${use_peepholes_comment}
        is_reverse (bool): ${is_reverse_comment}
        gate_activation (str): ${gate_activation_comment}
        cell_activation (str): ${cell_activation_comment}
        candidate_activation (str): ${candidate_activation_comment}
        dtype (str): Data type. Choices = ["float32", "float64"], default "float32".
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.

    Returns:
        tuple: The hidden state, and cell state of LSTM. The shape of both \
        is (T x D), and lod is the same with the `input`.

    Examples:
        .. code-block:: python

            hidden_dim = 512
            forward_proj = fluid.layers.fc(input=input_seq, size=hidden_dim * 4,
                                           act=None, bias_attr=None)
            forward, _ = fluid.layers.dynamic_lstm(
                input=forward_proj, size=hidden_dim * 4, use_peepholes=False)
    """

    helper = LayerHelper('lstm', **locals())
    size = size / 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 4 * size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    hidden = helper.create_tmp_variable(dtype)
    cell = helper.create_tmp_variable(dtype)
    batch_gate = helper.create_tmp_variable(dtype)
    batch_cell_pre_act = helper.create_tmp_variable(dtype)
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
                  name=None):
    """
    **Dynamic LSTMP Layer**

    LSTMP (LSTM with recurrent projection) layer has a separate projection
    layer after the LSTM layer, projecting the original hidden state to a
    lower-dimensional one, which is proposed to reduce the number of total
    parameters and furthermore computational complexity for the LSTM,
    espeacially for the case that the size of output units is relative
    large (https://research.google.com/pubs/archive/43905.pdf).

    The formula is as follows:

    .. math::

        i_t & = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i)

        f_t & = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f)

        \\tilde{c_t} & = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c)

        o_t & = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_t + b_o)

        c_t & = f_t \odot c_{t-1} + i_t \odot \\tilde{c_t}

        h_t & = o_t \odot act_h(c_t)

        r_t & = \overline{act_h}(W_{rh}h_t)

    In the above formula:

    * :math:`W`: Denotes weight matrices (e.g. :math:`W_{xi}` is \
          the matrix of weights from the input gate to the input).
    * :math:`W_{ic}`, :math:`W_{fc}`, :math:`W_{oc}`: Diagonal weight \
          matrices for peephole connections. In our implementation, \
          we use vectors to reprenset these diagonal weight matrices.
    * :math:`b`: Denotes bias vectors (e.g. :math:`b_i` is the input gate \
          bias vector).
    * :math:`\sigma`: The activation, such as logistic sigmoid function.
    * :math:`i, f, o` and :math:`c`: The input gate, forget gate, output \
          gate, and cell activation vectors, respectively, all of which have \
          the same size as the cell output activation vector :math:`h`.
    * :math:`h`: The hidden state.
    * :math:`r`: The recurrent projection of the hidden state.
    * :math:`\\tilde{c_t}`: The candidate hidden state, whose \
          computation is based on the current input and previous hidden state.
    * :math:`\odot`: The element-wise product of the vectors.
    * :math:`act_g` and :math:`act_h`: The cell input and cell output \
          activation functions and `tanh` is usually used for them.
    * :math:`\overline{act_h}`: The activation function for the projection \
          output, usually using `identity` or same as :math:`act_h`.

    Set `use_peepholes` to `False` to disable peephole connection. The formula
    is omitted here, please refer to the paper
    http://www.bioinf.jku.at/publications/older/2604.pdf for details.

    Note that these :math:`W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}`
    operations on the input :math:`x_{t}` are NOT included in this operator.
    Users can choose to use fully-connected layer before LSTMP layer.

    Args:
        input(Variable): The input of dynamic_lstmp layer, which supports
                         variable-time length input sequence. The underlying
                         tensor in this Variable is a matrix with shape
                         (T X 4D), where T is the total time steps in this
                         mini-batch, D is the hidden size.
        size(int): 4 * hidden size.
        proj_size(int): The size of projection output.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
                               hidden-hidden weight and projection weight.

                               - Hidden-hidden weight = {:math:`W_{ch}, W_{ih}, \
                                                W_{fh}, W_{oh}`}.
                               - The shape of hidden-hidden weight is (P x 4D),
                                 where P is the projection size and D the hidden
                                 size.
                               - Projection weight = {:math:`W_{rh}`}.
                               - The shape of projection weight is (D x P).
        bias_attr(ParamAttr|None): The bias attribute for the learnable bias
                              weights, which contains two parts, input-hidden
                              bias weights and peephole connections weights if
                              setting `use_peepholes` to `True`.

                              1. `use_peepholes = False`
                                - Biases = {:math:`b_c, b_i, b_f, b_o`}.
                                - The shape is (1 x 4D).
                              2. `use_peepholes = True`
                                - Biases = { :math:`b_c, b_i, b_f, b_o, W_{ic}, \
                                                 W_{fc}, W_{oc}`}.
                                - The shape is (1 x 7D).
        use_peepholes(bool): Whether to enable diagonal/peephole connections,
                             default `True`.
        is_reverse(bool): Whether to compute reversed LSTM, default `False`.
        gate_activation(str): The activation for input gate, forget gate and
                              output gate. Choices = ["sigmoid", "tanh", "relu",
                              "identity"], default "sigmoid".
        cell_activation(str): The activation for cell output. Choices = ["sigmoid",
                              "tanh", "relu", "identity"], default "tanh".
        candidate_activation(str): The activation for candidate hidden state.
                              Choices = ["sigmoid", "tanh", "relu", "identity"],
                              default "tanh".
        proj_activation(str): The activation for projection output.
                              Choices = ["sigmoid", "tanh", "relu", "identity"],
                              default "tanh".
        dtype(str): Data type. Choices = ["float32", "float64"], default "float32".
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        tuple: A tuple of two output variable: the projection of hidden state, \
               and cell state of LSTMP. The shape of projection is (T x P), \
               for the cell state which is (T x D), and both LoD is the same \
               with the `input`.

    Examples:

        .. code-block:: python

            dict_dim, emb_dim = 128, 64
            data = fluid.layers.data(name='sequence', shape=[1],
                                     dtype='int32', lod_level=1)
            emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim, proj_dim = 512, 256
            fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                                     act=None, bias_attr=None)
            proj_out, _ = fluid.layers.dynamic_lstmp(input=fc_out,
                                                     size=hidden_dim * 4,
                                                     proj_size=proj_dim,
                                                     use_peepholes=False,
                                                     is_reverse=True,
                                                     cell_activation="tanh",
                                                     proj_activation="tanh")
    """

    helper = LayerHelper('lstmp', **locals())
    size = size / 4
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[proj_size, 4 * size], dtype=dtype)
    proj_weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, proj_size], dtype=dtype)
    bias_size = [1, 7 * size]
    if not use_peepholes:
        bias_size[1] = 4 * size
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=bias_size, dtype=dtype, is_bias=True)

    projection = helper.create_tmp_variable(dtype)
    cell = helper.create_tmp_variable(dtype)
    ordered_proj0 = helper.create_tmp_variable(dtype)
    batch_hidden = helper.create_tmp_variable(dtype)
    batch_gate = helper.create_tmp_variable(dtype)
    batch_cell_pre_act = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='lstmp',
        inputs={
            'Input': input,
            'Weight': weight,
            'ProjWeight': proj_weight,
            'Bias': bias
        },
        outputs={
            'Projection': projection,
            'Cell': cell,
            'OrderedP0': ordered_proj0,
            'BatchHidden': batch_hidden,
            'BatchGate': batch_gate,
            'BatchCellPreAct': batch_cell_pre_act
        },
        attrs={
            'use_peepholes': use_peepholes,
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
                h_0=None):
    """
    **Gated Recurrent Unit (GRU) Layer**

    Refer to `Empirical Evaluation of Gated Recurrent Neural Networks on
    Sequence Modeling <https://arxiv.org/abs/1412.3555>`_ .

    The formula is as follows:

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \\tilde{h_t}

    The :math:`\odot` is the element-wise product of the vectors. :math:`act_g`
    is the update gate and reset gate activation function and :math:`sigmoid`
    is usually used for it. :math:`act_c` is the activation function for
    candidate hidden state and :math:`tanh` is usually used for it.

    Note that these :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` operations on
    the input :math:`x_{t}` are NOT included in this operator. Users can choose
    to use fully-connect layer before GRU layer.

    Args:
        input(Variable): The input of dynamic_gru layer, which supports
            variable-time length input sequence. The underlying tensor in this
            Variable is a matrix with shape :math:`(T \\times 3D)`, where
            :math:`T` is the total time steps in this mini-batch, :math:`D`
            is the hidden size.
        size(int): The dimension of the gru cell.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            hidden-hidden weight matrix. Note:

            - The shape of the weight matrix is :math:`(T \\times 3D)`, where
              :math:`D` is the hidden size.
            - All elements in the weight matrix can be divided into two parts.
              The first part are weights of the update gate and reset gate with
              shape :math:`(D \\times 2D)`, and the second part are weights for
              candidate hidden state with shape :math:`(D \\times D)`.
        bias_attr(ParamAttr): The parameter attribute for learnable the
            hidden-hidden bias.
        is_reverse(bool): Whether to compute reversed GRU, default
            :attr:`False`.
        gate_activation(str): The activation for update gate and reset gate.
            Choices = ["sigmoid", "tanh", "relu", "identity"], default "sigmoid".
        candidate_activation(str): The activation for candidate hidden state.
            Choices = ["sigmoid", "tanh", "relu", "identity"], default "tanh".
        h_0 (Variable): This is initial hidden state. If not set, default is
            zero. This is a tensor with shape (N x D), where N is the number of
            total time steps of input mini-batch feature and D is the hidden
            size.

    Returns:
        Variable: The hidden state of GRU. The shape is :math:`(T \\times D)`, \
            and sequence length is the same with the input.

    Examples:

        .. code-block:: python

            dict_dim, emb_dim = 128, 64
            data = fluid.layers.data(name='sequence', shape=[1],
                                     dtype='int32', lod_level=1)
            emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            hidden = fluid.layers.dynamic_gru(input=x, dim=hidden_dim)
    """

    helper = LayerHelper('gru', **locals())
    dtype = helper.input_dtype()

    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)
    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=[1, 3 * size], dtype=dtype, is_bias=True)
    batch_size = input.shape[0]
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    if h_0 != None:
        assert h_0.shape == (
            batch_size, size
        ), 'The shape of h0 should be(batch_size, %d)' % size
        inputs['H0'] = h_0

    hidden = helper.create_tmp_variable(dtype)
    batch_gate = helper.create_tmp_variable(dtype)
    batch_reset_hidden_prev = helper.create_tmp_variable(dtype)
    batch_hidden = helper.create_tmp_variable(dtype)

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
            'activation': candidate_activation
        })
    return hidden


def gru_unit(input,
             hidden,
             size,
             param_attr=None,
             bias_attr=None,
             activation='tanh',
             gate_activation='sigmoid'):
    """
    GRU unit layer. The equation of a gru step is:

        .. math::
            u_t & = actGate(xu_{t} + W_u h_{t-1} + b_u)

            r_t & = actGate(xr_{t} + W_r h_{t-1} + b_r)

            m_t & = actNode(xm_t + W_c dot(r_t, h_{t-1}) + b_m)

            h_t & = dot((1-u_t), m_t) + dot(u_t, h_{t-1})

    The inputs of gru unit includes :math:`z_t`, :math:`h_{t-1}`. In terms
    of the equation above, the :math:`z_t` is split into 3 parts -
    :math:`xu_t`, :math:`xr_t` and :math:`xm_t`. This means that in order to
    implement a full GRU unit operator for an input, a fully
    connected layer has to be applied, such that :math:`z_t = W_{fc}x_t`.

    The terms :math:`u_t` and :math:`r_t` represent the update and reset gates
    of the GRU cell. Unlike LSTM, GRU has one lesser gate. However, there is
    an intermediate candidate hidden output, which is denoted by :math:`m_t`.
    This layer has three outputs :math:`h_t`, :math:`dot(r_t, h_{t-1})`
    and concatenation of :math:`u_t`, :math:`r_t` and :math:`m_t`.

    Args:
        input (Variable): The fc transformed input value of current step.
        hidden (Variable): The hidden value of lstm unit from previous step.
        size (integer): The input dimension value.
        param_attr (ParamAttr): The weight parameters for gru unit. Default: None
        bias_attr (ParamAttr): The bias parameters for gru unit. Default: None
        activation (string): The activation type for cell (actNode).
                             Default: 'tanh'
        gate_activation (string): The activation type for gates (actGate).
                                  Default: 'sigmoid'

    Returns:
        tuple: The hidden value, reset-hidden value and gate values.

    Examples:

        .. code-block:: python

             # assuming we have x_t_data and prev_hidden of size=10
             x_t = fluid.layers.fc(input=x_t_data, size=30)
             hidden_val, r_h_val, gate_val = fluid.layers.gru_unit(input=x_t,
                                                    hidden = prev_hidden)

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
    size = size / 3

    # create weight
    weight = helper.create_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)

    gate = helper.create_tmp_variable(dtype)
    reset_hidden_pre = helper.create_tmp_variable(dtype)
    updated_hidden = helper.create_tmp_variable(dtype)
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
        })

    return updated_hidden, reset_hidden_pre, gate


@templatedoc()
def linear_chain_crf(input, label, param_attr=None):
    """
    Linear Chain CRF.

    ${comment}

    Args:
        input(${emission_type}): ${emission_comment}
        input(${transition_type}): ${transition_comment}
        label(${label_type}): ${label_comment}
        param_attr(ParamAttr): The attribute of the learnable parameter.

    Returns:
        output(${emission_exps_type}): ${emission_exps_comment} \n
        output(${transition_exps_type}): ${transition_exps_comment} \n
        output(${log_likelihood_type}): ${log_likelihood_comment}

    """
    helper = LayerHelper('linear_chain_crf', **locals())
    size = input.shape[1]
    transition = helper.create_parameter(
        attr=helper.param_attr,
        shape=[size + 2, size],
        dtype=helper.input_dtype())
    alpha = helper.create_tmp_variable(dtype=helper.input_dtype())
    emission_exps = helper.create_tmp_variable(dtype=helper.input_dtype())
    transition_exps = helper.create_tmp_variable(dtype=helper.input_dtype())
    log_likelihood = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='linear_chain_crf',
        inputs={"Emission": [input],
                "Transition": transition,
                "Label": label},
        outputs={
            "Alpha": [alpha],
            "EmissionExps": [emission_exps],
            "TransitionExps": transition_exps,
            "LogLikelihood": log_likelihood
        })

    return log_likelihood


@templatedoc()
def crf_decoding(input, param_attr, label=None):
    """
    ${comment}

    Args:
        input(${emission_type}): ${emission_comment}

        param_attr(ParamAttr): The parameter attribute for training.

        label(${label_type}): ${label_comment}

    Returns:
        Variable: ${viterbi_path_comment}

    Examples:
        .. code-block:: python

           crf_decode = layers.crf_decoding(
                input=hidden, param_attr=ParamAttr(name="crfw"))
    """
    helper = LayerHelper('crf_decoding', **locals())
    transition = helper.get_parameter(param_attr.name)
    viterbi_path = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='crf_decoding',
        inputs={"Emission": [input],
                "Transition": transition,
                "Label": label},
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
        Variable: the output of cosine(X, Y).
    """
    helper = LayerHelper('cos_sim', **locals())
    out = helper.create_tmp_variable(dtype=X.dtype)
    xnorm = helper.create_tmp_variable(dtype=X.dtype)
    ynorm = helper.create_tmp_variable(dtype=X.dtype)
    helper.append_op(
        type='cos_sim',
        inputs={'X': [X],
                'Y': [Y]},
        outputs={'Out': [out],
                 'XNorm': [xnorm],
                 'YNorm': [ynorm]})
    return out


def dropout(x, dropout_prob, is_test=False, seed=None, name=None):
    """
    Computes dropout.

    Drop or keep each element of `x` independently. Dropout is a regularization
    technique for reducing overfitting by preventing neuron co-adaption during
    training. The dropout operator randomly sets (according to the given dropout
    probability) the outputs of some units to zero, while others are remain
    unchanged.

    Args:
        x (Variable): The input tensor variable.
        dropout_prob (float): Probability of setting units to zero.
        is_test (bool): A flag indicating whether it is in test phrase or not.
        seed (int): A Python integer used to create random seeds. If this
                    parameter is set to None, a random seed is used.
                    NOTE: If an integer seed is given, always the same output
                    units will be dropped. DO NOT use a fixed seed in training.
        name (str|None): A name for this layer(optional). If set None, the layer
                         will be named automatically.

    Returns:
        Variable: A tensor variable is the shape with `x`.

    Examples:

        .. code-block:: python

            x = fluid.layers.data(name="data", shape=[32, 32], dtype="float32")
            droped = fluid.layers.dropout(x, dropout_prob=0.5)
    """

    helper = LayerHelper('dropout', **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    mask = helper.create_tmp_variable(dtype=x.dtype, stop_gradient=True)
    helper.append_op(
        type='dropout',
        inputs={'X': [x]},
        outputs={'Out': [out],
                 'Mask': [mask]},
        attrs={
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0
        })
    return out


def cross_entropy(input, label, soft_label=False):
    """
    **Cross Entropy Layer**

    This layer computes the cross entropy between `input` and `label`. It
    supports both standard cross-entropy and soft-label cross-entropy loss
    computation.

    1) One-hot cross-entropy:
        `soft_label = False`, `Label[i, 0]` indicates the class index for sample i:

        .. math::

            Y[i] = -\log(X[i, Label[i]])

    2) Soft-label cross-entropy:
        `soft_label = True`, `Label[i, j]` indicates the soft label of class j
        for sample i:

        .. math::

            Y[i] = \sum_j{-Label[i, j] * log(X[i, j])}

       Please make sure that in this case the summation of each row of `label`
       equals one.

    3) One-hot cross-entropy with vecterized `label`:
         As a special case of 2), when each row of 'label' has only one
         non-zero element which is equal to 1, soft-label cross-entropy degenerates
         to a one-hot cross-entropy with one-hot label representation.

    Args:
        input (Variable|list):  a 2-D tensor with shape [N x D], where N is the
                                batch size and D is the number of classes. This
                                input is a probability computed by the previous
                                operator, which is almost always the result of
                                a softmax operator.
        label (Variable|list): the ground truth which is a 2-D tensor. When
                               `soft_label` is set to `False`, `label` is a
                               tensor<int64> with shape [N x 1]. When
                               `soft_label` is set to `True`, `label` is a
                               tensor<float/double> with shape [N x D].
        soft_label (bool): a flag indicating whether to
                                           interpretate the given labels as soft
                                           labels, default `False`.

    Returns:
         A 2-D tensor with shape [N x 1], the cross entropy loss.

    Raises:
        `ValueError`: 1) the 1st dimension of `input` and `label` are not equal.
                      2) when `soft_label == True`, and the 2nd dimension of
                         `input` and `label` are not equal.
                      3) when `soft_label == False`, and the 2nd dimension of
                         `label` is not 1.

    Examples:
        .. code-block:: python

          predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
          cost = fluid.layers.cross_entropy(input=predict, label=label)
    """
    helper = LayerHelper('cross_entropy', **locals())
    out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='cross_entropy',
        inputs={'X': [input],
                'Label': [label]},
        outputs={'Y': [out]},
        attrs={"soft_label": soft_label})
    return out


def square_error_cost(input, label):
    """
    **Square error cost layer**

    This layer accepts input predictions and target label and returns the
    squared error cost.

    For predictions, :math:`X`, and target labels, :math:`Y`, the equation is:

    .. math::

        Out = (X - Y)^2

    In the above equation:

        * :math:`X`: Input predictions, a tensor.
        * :math:`Y`: Input labels, a tensor.
        * :math:`Out`: Output value, same shape with :math:`X`.

    Args:
        input (Variable): Input tensor, has predictions.
        label (Variable): Label tensor, has target labels.

    Returns:
        Variable: The tensor variable storing the element-wise squared error \
                  difference of input and label.

    Examples:
        .. code-block:: python

          y = layers.data(name='y', shape=[1], dtype='float32')
          y_predict = layers.data(name='y_predict', shape=[1], dtype='float32')
          cost = layers.square_error_cost(input=y_predict, label=y)

    """
    helper = LayerHelper('square_error_cost', **locals())
    minus_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='square', inputs={'X': [minus_out]},
        outputs={'Out': [square_out]})
    return square_out


@templatedoc()
def chunk_eval(input,
               label,
               chunk_scheme,
               num_chunk_types,
               excluded_chunk_types=None):
    """
    **Chunk Evaluator**

    This function computes and outputs the precision, recall and
    F1-score of chunk detection.

    For some basics of chunking, please refer to
    'Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>'.

    ChunkEvalOp computes the precision, recall, and F1-score of chunk detection,
    and supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.
    Here is a NER example of labeling for these tagging schemes:

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
    and LOC(LOCATION), and we can see that the labels have the form <tag type>-<chunk type>.

    Since the calculations actually use label ids rather than labels, extra attention
    should be paid when mapping labels to ids to make CheckEvalOp work. The key point
    is that the listed equations are satisfied by ids.

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

    Still use NER as example, assuming the tagging scheme is IOB while chunk types are ORG,
    PER and LOC. To satisfy the above equations, the label map can be like this:

    .. code-block:: python

       B-ORG  0
       I-ORG  1
       B-PER  2
       I-PER  3
       B-LOC  4
       I-LOC  5
       O      6

    It's not hard to verify the equations noting that the num of chunk types
    is 3 and the num of tag types in IOB scheme is 2. For example, the label
    id of I-LOC is 5, the tag type id of I-LOC is 1, and the chunk type id of
    I-LOC is 2, which consistent with the results from the equations.

    Args:
        input (Variable): prediction output of the network.
        label (Variable): label of the test data set.
        chunk_scheme (str): ${chunk_scheme_comment}
        num_chunk_types (int): ${num_chunk_types_comment}
        excluded_chunk_types (list): ${excluded_chunk_types_comment}

    Returns:
        tuple: tuple containing: precision, recall, f1_score,
        num_infer_chunks, num_label_chunks,
        num_correct_chunks

    Examples:
        .. code-block:: python

            crf = fluid.layers.linear_chain_crf(
                input=hidden, label=label, param_attr=ParamAttr(name="crfw"))
            crf_decode = fluid.layers.crf_decoding(
                input=hidden, param_attr=ParamAttr(name="crfw"))
            fluid.layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) / 2)
    """
    helper = LayerHelper("chunk_eval", **locals())

    # prepare output
    precision = helper.create_tmp_variable(dtype="float32")
    recall = helper.create_tmp_variable(dtype="float32")
    f1_score = helper.create_tmp_variable(dtype="float32")
    num_infer_chunks = helper.create_tmp_variable(dtype="int64")
    num_label_chunks = helper.create_tmp_variable(dtype="int64")
    num_correct_chunks = helper.create_tmp_variable(dtype="int64")

    helper.append_op(
        type="chunk_eval",
        inputs={"Inference": [input],
                "Label": [label]},
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
                  padding=None,
                  bias_attr=None,
                  param_attr=None,
                  act=None):
    """
    This function creates the op for sequence_conv, using the inputs and
    other convolutional configurations for the filters and stride as given
    in the input parameters to the function.

    Args:
        input (Variable): ${x_comment}
        num_filters (int): number of filters.
        filter_size (int): the filter size (H and W).
        filter_stride (int): stride of the filter.
        padding (bool): if True, add paddings.
        bias_attr (ParamAttr|None): attributes for bias
        param_attr (ParamAttr|None): attributes for parameter
        act (str): the activation type

    Returns:
        Variable: output of sequence_conv
    """

    helper = LayerHelper('sequence_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [filter_size * input.shape[1], num_filters]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    pre_bias = helper.create_tmp_variable(dtype)

    helper.append_op(
        type='sequence_conv',
        inputs={
            'X': [input],
            'Filter': [filter_param],
        },
        outputs={"Out": pre_bias},
        attrs={
            'contextStride': filter_stride,
            'contextStart': -int(filter_size / 2),
            'contextLength': filter_size
        })
    pre_act = helper.append_bias_op(pre_bias)
    return helper.append_activation(pre_act)


def sequence_softmax(input, param_attr=None, bias_attr=None, use_cudnn=True):
    """
    This function computes the softmax activation among all time-steps for each
    sequence. The dimension of each time-step should be 1. Thus, the shape of
    input Tensor can be either :math:`[N, 1]` or :math:`[N]`, where :math:`N`
    is the sum of the length of all sequences.

    For i-th sequence in a mini-batch:

    .. math::

        Out(X[lod[i]:lod[i+1]], :) = \\frac{\exp(X[lod[i]:lod[i+1], :])}{\sum(\exp(X[lod[i]:lod[i+1], :]))}

    For example, for a mini-batch of 3 sequences with variable-length,
    each containing 2, 3, 2 time-steps, the lod of which is [0, 2, 5, 7],
    then softmax will be computed among :math:`X[0:2, :]`, :math:`X[2:5, :]`,
    :math:`X[5:7, :]`, and :math:`N` turns out to be 7.

    Args:
        input (Variable): The input variable which is a LoDTensor.
        bias_attr (ParamAttr|None): attributes for bias
        param_attr (ParamAttr|None): attributes for parameter
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn \
        library is installed. Default: True

    Returns:
        Variable: output of sequence_softmax

    Examples:

        .. code-block:: python

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_sequence_softmax = fluid.layers.sequence_softmax(input=x)
    """
    helper = LayerHelper('sequence_softmax', **locals())
    dtype = helper.input_dtype()
    softmax_out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="sequence_softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn})
    return softmax_out


def softmax(input, param_attr=None, bias_attr=None, use_cudnn=True, name=None):
    """
    The input of the softmax layer is a 2-D tensor with shape N x K (N is the
    batch_size, K is the dimension of input feature). The output tensor has the
    same shape as the input tensor.

    For each row of the input tensor, the softmax operator squashes the
    K-dimensional vector of arbitrary real values to a K-dimensional vector of real
    values in the range [0, 1] that add up to 1.

    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.

    For each row :math:`i` and each column :math:`j` in Input(X), we have:

    .. math::

        Out[i, j] = \\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])}

    Args:
        input (Variable): The input variable.
        bias_attr (ParamAttr): attributes for bias
        param_attr (ParamAttr): attributes for parameter
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn \
        library is installed.

    Returns:
        Variable: output of softmax

    Examples:

        .. code-block:: python

             fc = fluid.layers.fc(input=x, size=10)
             softmax = fluid.layers.softmax(input=fc)

    """
    helper = LayerHelper('softmax', **locals())
    dtype = helper.input_dtype()
    softmax_out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="softmax",
        inputs={"X": input},
        outputs={"Out": softmax_out},
        attrs={"use_cudnn": use_cudnn})
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
           use_mkldnn=False,
           act=None,
           name=None):
    """
    The convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    channels, H is the height of the feature, and W is the width of the feature.
    Filter is in MCHW format, where M is the number of output image channels,
    C is the number of input image channels, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input image channels divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more detials.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW format.
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
        input (Variable): The input image with [N, C, H, W] format.
        num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple|None): The filter size. If filter_size is a tuple,
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
        groups (int): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        param_attr (ParamAttr): The parameters to the Conv2d Layer. Default: None
        bias_attr (ParamAttr): Bias parameter for the Conv2d layer. Default: None
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        use_mkldnn (bool): Use mkldnn kernels or not, it is valid only when compiled
            with mkldnn library. Default: False
        act (str): Activation type. Default: None
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
          conv2d = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu")
    """

    num_channels = input.shape[1]

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
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels / groups

    filter_size = utils.convert_to_list(filter_size, 2, 'filter_size')
    stride = utils.convert_to_list(stride, 2, 'stride')
    padding = utils.convert_to_list(padding, 2, 'padding')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        std = (2.0 / (filter_size[0]**2 * num_channels))**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_tmp_variable(dtype)

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
            'use_mkldnn': use_mkldnn
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
           use_mkldnn=False,
           act=None,
           name=None):
    """
    **Convlution3D Layer**

    The convolution3D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input(Input) and
    Output(Output) are in NCDHW format. Where N is batch size C is the number of
    channels, D is the depth of the feature, H is the height of the feature,
    and W is the width of the feature. Convlution3D is similar with Convlution2D
    but adds one dimension(depth). If bias attribution and activation type are
    provided, bias is added to the output of the convolution, and the
    corresponding activation function is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW format.
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
        input (Variable): The input image with [N, C, D, H, W] format.
            num_filters(int): The number of filter. It is as same as the output
            image channel.
        filter_size (int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_D, filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        stride (int|tuple): The stride size. If stride is a tuple, it must
            contain three integers, (stride_D, stride_H, stride_W). Otherwise, the
            stride_D = stride_H = stride_W = stride. Default: stride = 1.
        padding (int|tuple): The padding size. If padding is a tuple, it must
            contain three integers, (padding_D, padding_H, padding_W). Otherwise, the
            padding_D = padding_H = padding_W = padding. Default: padding = 0.
        dilation (int|tuple): The dilation size. If dilation is a tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups (int): The groups number of the Conv3d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: groups=1
        param_attr (ParamAttr): The parameters to the Conv3d Layer. Default: None
        bias_attr (ParamAttr): Bias parameter for the Conv3d layer. Default: None
        use_cudnn (bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        use_mkldnn (bool): Use mkldnn kernels or not.
        act (str): Activation type. Default: None
        name (str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The tensor variable storing the convolution and \
                  non-linearity activation result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
          conv3d = fluid.layers.conv3d(input=data, num_filters=2, filter_size=3, act="relu")
    """

    l_type = 'conv3d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()

    num_channels = input.shape[1]

    if groups is None:
        num_filter_channels = num_channels
    else:
        if num_channels % groups != 0:
            raise ValueError("num_channels must be divisible by groups.")
        num_filter_channels = num_channels / groups

    filter_size = utils.convert_to_list(filter_size, 3, 'filter_size')
    stride = utils.convert_to_list(stride, 3, 'stride')
    padding = utils.convert_to_list(padding, 3, 'padding')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    input_shape = input.shape
    filter_shape = [num_filters, num_filter_channels] + filter_size

    def _get_default_param_initializer():
        std = (2.0 / (filter_size[0]**3 * num_channels))**0.5
        return Normal(0.0, std, 0)

    filter_param = helper.create_parameter(
        attr=helper.param_attr,
        shape=filter_shape,
        dtype=dtype,
        default_initializer=_get_default_param_initializer())

    pre_bias = helper.create_tmp_variable(dtype)

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
            'use_mkldnn': use_mkldnn
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)

    return helper.append_activation(pre_act)


def sequence_pool(input, pool_type):
    """
    This function add the operator for sequence pooling.
    It pools features of all time-steps of each instance, and is applied
    on top of the input using pool_type mentioned in the parameters.

    It supports four pool_type:

    - average: :math:`Out[i] = \\frac{\sum_i X_i}{N}`
    - sum:     :math:`Out[i] = \sum_jX_{ij}`
    - sqrt:    :math:`Out[i] = \\frac{\sum_jX_{ij}}{\sqrt{len(X_i)}}`
    - max:     :math:`Out[i] = max(X_i)`

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[2, 3, 2]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) == out.dims[0]

       for different pool_type:
         average: out.data = [2, 4, 3], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
         sum    : out.data = [4, 12, 6], where 4=1+3, 12=2+4+6, 6=5+1
         sqrt   : out.data = [2.82, 6.93, 4.24], where 2.82=(1+3)/sqrt(2),
                    6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
         max    : out.data = [3, 6, 5], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
         last   : out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
         first  : out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.
        pool_type (string): The pooling type of sequence_pool.
            It supports average, sum, sqrt and max.

    Returns:
        The sequence pooling variable which is a Tensor.

    Examples:

        .. code-block:: python

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
             sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
             sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
             max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
             last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
             first_x = fluid.layers.sequence_pool(input=x, pool_type='first')
    """
    helper = LayerHelper('sequence_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)
    max_index = helper.create_tmp_variable(dtype)

    helper.append_op(
        type="sequence_pool",
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "MaxIndex": max_index},
        attrs={"pooltype": pool_type.upper()})

    # when pool_type is max, variable max_index is initialized,
    # so we stop the gradient explicitly here
    if pool_type == 'max':
        max_index.stop_gradient = True

    return pool_out


def sequence_first_step(input):
    """
    This function gets the first step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[2, 3, 2]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) == out.dims[0]
         out.data = [1, 2, 5], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's first step variable which is a Tensor.

    Examples:

        .. code-block:: python

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_first_step = fluid.layers.sequence_first_step(input=x)
    """
    return sequence_pool(input=input, pool_type="first")


def sequence_last_step(input):
    """
    This function gets the last step of sequence.

    .. code-block:: text

       x is a 1-level LoDTensor:
         x.lod = [[2, 3, 2]]
         x.data = [1, 3, 2, 4, 6, 5, 1]
         x.dims = [7, 1]

       then output is a Tensor:
         out.dim = [3, 1]
         with condition len(x.lod[-1]) == out.dims[0]
         out.data = [3, 6, 1], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)

    Args:
        input(variable): The input variable which is a LoDTensor.

    Returns:
        The sequence's last step variable which is a Tensor.

    Examples:

        .. code-block:: python

             x = fluid.layers.data(name='x', shape=[7, 1],
                              dtype='float32', lod_level=1)
             x_last_step = fluid.layers.sequence_last_step(input=x)
    """
    return sequence_pool(input=input, pool_type="last")


@templatedoc()
def pool2d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           use_mkldnn=False,
           name=None):
    """
    ${comment}

    Args:
        input (Variable): The input tensor of pooling operator. The format of
                          input tensor is NCHW, where N is batch size, C is
                          the number of channels, H is the height of the
                          feature, and W is the width of the feature.
        pool_size (int): The side length of pooling windows. All pooling
                         windows are squares with pool_size on a side.
        pool_type: ${pooling_type_comment}
        pool_stride (int): stride of the pooling layer.
        pool_padding (int): padding size.
        global_pooling: ${global_pooling_comment}
        use_cudnn: ${use_cudnn_comment}
        ceil_mode: ${ceil_mode_comment}
        use_mkldnn: ${use_mkldnn_comment}
        name (str|None): A name for this layer(optional). If set None, the
                        layer will be named automatically.

    Returns:
        Variable: The pooling result.

    Raises:
        ValueError: If 'pool_type' is not "max" nor "avg"
        ValueError: If 'global_pooling' is False and 'pool_size' is -1
        ValueError: If 'use_cudnn' is not a bool value.

    Examples:

        .. code-block:: python

          data = fluid.layers.data(
              name='data', shape=[3, 32, 32], dtype='float32')
          conv2d = fluid.layers.pool2d(
                            input=data,
                            pool_size=2,
                            pool_type='max',
                            pool_stride=1,
                            global_pooling=False)
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When the global_pooling is False, pool_size must be passed "
            "and be a valid value. Received pool_size: " + str(pool_size))

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
    pool_padding = utils.convert_to_list(pool_padding, 2, 'pool_padding')
    pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    l_type = 'pool2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": use_mkldnn
        })

    return pool_out


def pool3d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           use_cudnn=True,
           ceil_mode=False,
           use_mkldnn=False,
           name=None):
    """
    This function adds the operator for pooling in 3-dimensions, using the
    pooling configurations mentioned in input parameters.

    Args:
        input (Variable): ${input_comment}
        pool_size (int): ${ksize_comment}
        pool_type (str): ${pooling_type_comment}
        pool_stride (int): stride of the pooling layer.
        pool_padding (int): padding size.
        global_pooling (bool): ${global_pooling_comment}
        use_cudnn (bool): ${use_cudnn_comment}
        ceil_mode (bool): ${ceil_mode_comment}
        use_mkldnn (bool): ${use_mkldnn_comment}
        name (str): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: output of pool3d layer.
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown pool_type: '%s'. It can only be 'max' or 'avg'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When the global_pooling is False, pool_size must be passed "
            "and be a valid value. Received pool_size: " + str(pool_size))

    pool_size = utils.convert_to_list(pool_size, 3, 'pool_size')
    pool_padding = utils.convert_to_list(pool_padding, 3, 'pool_padding')
    pool_stride = utils.convert_to_list(pool_stride, 3, 'pool_stride')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    l_type = "pool3d"
    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)

    helper.append_op(
        type=l_type,
        inputs={"X": input},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "use_cudnn": use_cudnn,
            "ceil_mode": ceil_mode,
            "use_mkldnn": use_mkldnn
        })

    return pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               in_place=False,
               use_mkldnn=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=False,
               fuse_with_relu=False):
    """
    **Batch Normalization Layer**

    Can be used as a normalizer function for conv2d and fully_connected operations.
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

    Args:
        input(variable): The input variable which is a LoDTensor.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test(bool, Default False): Used for training or training.
        momentum(float, Default 0.9):
        epsilon(float, Default 1e-05):
        param_attr(ParamAttr): The parameter attribute for Parameter `scale`.
        bias_attr(ParamAttr): The parameter attribute for Parameter `bias`.
        data_layout(string, default NCHW): NCHW|NHWC
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        use_mkldnn(bool, Default false): ${use_mkldnn_comment}
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.
        moving_mean_name(string, Default None): The name of moving_mean which store the global Mean.
        moving_variance_name(string, Default None): The name of the moving_variance which store the global Variance.
        do_model_average_for_mean_and_var(bool, Default False): Do model average for mean and variance or not.
        fuse_with_relu (bool): if True, this OP performs relu after batch norm.

    Returns:
        Variable: A tensor variable which is the result after applying batch normalization on the input.

    Examples:

        .. code-block:: python

            hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            hidden2 = fluid.layers.batch_norm(input=hidden1)
    """
    helper = LayerHelper('batch_norm', **locals())
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
        dtype=input.dtype)
    mean.stop_gradient = True

    variance = helper.create_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(1.0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=input.dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)

    batch_norm_out = input if in_place else helper.create_tmp_variable(dtype)

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
            "use_mkldnn": use_mkldnn,
            "fuse_with_relu": fuse_with_relu
        })

    return helper.append_activation(batch_norm_out)


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
    ${comment}

    The formula is as follows:

    ..  math::

        \\mu & = \\frac{1}{H}\\sum_{i=1}^{H} a_i

        \\sigma & = \\sqrt{\\frac{1}{H}\sum_{i=1}^{H}(a_i - \\mu)^2}

        h & = f(\\frac{g}{\\sigma}(a - \\mu) + b)

    * :math:`a`: the vector representation of the summed inputs to the neurons
    in that layer.

    * :math:`H`: the number of hidden units in a layers

    * :math:`g`: the trainable scale parameter.

    * :math:`b`: the trainable bias parameter.

    Args:
        input(Variable): The input tensor variable.
        scale(bool): Whether to learn the adaptive gain :math:`g` after
            normalization.
        shift(bool): Whether to learn the adaptive bias :math:`b` after
            normalization.
        begin_norm_axis(bool): The normalization will be performed along
            dimensions from :attr:`begin_norm_axis` to :attr:`rank(input)`.
        epsilon(float): The small value added to the variance to prevent
            division by zero.
        param_attr(ParamAttr|None): The parameter attribute for the learnable
            gain :math:`g`.
        bias_attr(ParamAttr|None): The parameter attribute for the learnable
            bias :math:`b`.
        act(str): Activation to be applied to the output of layer normalizaiton.
        name (str): The name of this layer. It is optional.

    Returns:
        ${y_comment}

    Examples:

        >>> data = fluid.layers.data(name='data', shape=[3, 32, 32],
        >>>                          dtype='float32')
        >>> x = fluid.layers.layer_norm(input=data, begin_norm_axis=1)
    """
    helper = LayerHelper('layer_norm', **locals())
    dtype = helper.input_dtype()

    # create intput and parameters
    inputs = {'X': input}
    input_shape = input.shape
    param_shape = [reduce(lambda x, y: x * y, input_shape[begin_norm_axis:])]
    if scale:
        scale = helper.create_parameter(
            attr=helper.param_attr,
            shape=param_shape,
            dtype=dtype,
            default_initializer=Constant(1.0))
        inputs['Scale'] = scale
    if shift:
        assert bias_attr is not False
        bias = helper.create_parameter(
            attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)
        inputs['Bias'] = bias

    # create output
    mean_out = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    variance_out = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    layer_norm_out = helper.create_tmp_variable(dtype)

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
                     name=None):
    """
    **Convlution2D transpose layer**

    The convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCHW format. Where N is batch size, C is the number of channels,
    H is the height of the feature, and W is the width of the feature.
    Parameters(dilations, strides, paddings) are two elements. These two elements
    represent height and width, respectively. The details of convolution transpose
    layer, please refer to the following explanation and references
    `therein <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    Where:

    * :math:`X`: Input value, a tensor with NCHW format.
    * :math:`W`: Filter value, a tensor with MCHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:

        - Input:

          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`

          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`

        - Output:

          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`

        Where

        .. math::

           H_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1

    Args:
        input(Variable): The input image with [N, C, H, W] format.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple|None): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). This
            parameter only works when filter_size is None.
        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square. None if use output size to
            calculate filter_size.
        padding(int|tuple): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: padding = 0.
        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.
        dilation(int|tuple): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups(int): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups=1
        param_attr(ParamAttr): The parameters to the Conv2d_transpose Layer.
                               Default: None
        bias_attr(ParamAttr): Bias parameter for the Conv2d layer. Default: None
        use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act(str): Activation type. Default: None
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The tensor variable storing the convolution transpose result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
          conv2d_transpose = fluid.layers.conv2d_transpose(input=data, num_filters=2, filter_size=3)
    """

    input_channel = input.shape[1]

    op_type = 'conv2d_transpose'
    if (input_channel == groups and num_filters == input_channel and
            not use_cudnn):
        op_type = 'depthwise_conv2d_transpose'

    helper = LayerHelper(op_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv2d_transpose must be Variable")

    padding = utils.convert_to_list(padding, 2, 'padding')
    stride = utils.convert_to_list(stride, 2, 'stride')
    dilation = utils.convert_to_list(dilation, 2, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        h_in = input.shape[2]
        w_in = input.shape[3]

        filter_size_h = (output_size[0] - (h_in - 1) * stride[0] + 2 *
                         padding[0] - 1) / dilation[0] + 1
        filter_size_w = (output_size[1] - (w_in - 1) * stride[1] + 2 *
                         padding[1] - 1) / dilation[1] + 1
        filter_size = [filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 2,
                                            'conv2d_transpose.filter_size')

    groups = 1 if groups is None else groups
    filter_shape = [input_channel, num_filters / groups] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    pre_bias = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type=op_type,
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn
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
                     name=None):
    """
    **Convlution3D transpose layer**

    The convolution3D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input(Input) and output(Output)
    are in NCDHW format. Where N is batch size, C is the number of channels,
    D is the depth of the feature, H is the height of the feature, and W
    is the width of the feature. Parameters(dilations, strides, paddings) are
    two elements. These two elements represent height and width, respectively.
    The details of convolution transpose layer, please refer to the following
    explanation and references `therein <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.

    For each input :math:`X`, the equation is:

    .. math::

        Out = \sigma (W \\ast X + b)

    In the above equation:

    * :math:`X`: Input value, a tensor with NCDHW format.
    * :math:`W`: Filter value, a tensor with MCDHW format.
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D tensor with shape [M, 1].
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

           D_{out} &= (D_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (D_f - 1) + 1 \\\\
           H_{out} &= (H_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (H_f - 1) + 1 \\\\
           W_{out} &= (W_{in} - 1) * strides[2] - 2 * paddings[2] + dilations[2] * (W_f - 1) + 1

    Args:
        input(Variable): The input image with [N, C, D, H, W] format.
        num_filters(int): The number of the filter. It is as same as the output
            image channel.
        output_size(int|tuple|None): The output image size. If output size is a
            tuple, it must contain three integers, (image_D, image_H, image_W). This
            parameter only works when filter_size is None.
        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain three integers, (filter_size_D, filter_size_H, filter_size_W).
            Otherwise, the filter will be a square. None if use output size to
            calculate filter_size.
        padding(int|tuple): The padding size. If padding is a tuple, it must
            contain three integers, (padding_D, padding_H, padding_W). Otherwise, the
            padding_D = padding_H = padding_W = padding. Default: padding = 0.
        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain three integers, (stride_D, stride_H, stride_W). Otherwise, the
            stride_D = stride_H = stride_W = stride. Default: stride = 1.
        dilation(int|tuple): The dilation size. If dilation is a tuple, it must
            contain three integers, (dilation_D, dilation_H, dilation_W). Otherwise, the
            dilation_D = dilation_H = dilation_W = dilation. Default: dilation = 1.
        groups(int): The groups number of the Conv3d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: groups=1
        param_attr(ParamAttr): The parameters to the Conv3d_transpose Layer.
            Default: None
        bias_attr(ParamAttr): Bias parameter for the Conv3d layer. Default: None
        use_cudnn(bool): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True
        act(str): Activation type. Default: None
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The tensor variable storing the convolution transpose result.

    Raises:
        ValueError: If the shapes of input, filter_size, stride, padding and
                    groups mismatch.

    Examples:
       .. code-block:: python

          data = fluid.layers.data(name='data', shape=[3, 12, 32, 32], dtype='float32')
          conv3d_transpose = fluid.layers.conv3d_transpose(input=data, num_filters=2, filter_size=3)
    """
    l_type = "conv3d_transpose"
    helper = LayerHelper(l_type, **locals())
    if not isinstance(input, Variable):
        raise TypeError("Input of conv3d_transpose must be Variable")
    input_channel = input.shape[1]

    padding = utils.convert_to_list(padding, 3, 'padding')
    stride = utils.convert_to_list(stride, 3, 'stride')
    dilation = utils.convert_to_list(dilation, 3, 'dilation')

    if not isinstance(use_cudnn, bool):
        raise ValueError("use_cudnn should be True or False")

    if filter_size is None:
        if output_size is None:
            raise ValueError("output_size must be set when filter_size is None")
        if isinstance(output_size, int):
            output_size = [output_size, output_size]

        d_in = input.shape[2]
        h_in = input.shape[3]
        w_in = input.shape[4]

        filter_size_d = (output_size[0] - (d_in - 1) * stride[0] + 2 *
                         padding[0] - 1) / dilation[0] + 1
        filter_size_h = (output_size[1] - (h_in - 1) * stride[1] + 2 *
                         padding[1] - 1) / dilation[1] + 1
        filter_size_w = (output_size[2] - (w_in - 1) * stride[2] + 2 *
                         padding[2] - 1) / dilation[2] + 1
        filter_size = [filter_size_d, filter_size_h, filter_size_w]
    else:
        filter_size = utils.convert_to_list(filter_size, 3,
                                            'conv3d_transpose.filter_size')

    groups = 1 if groups is None else groups
    filter_shape = [input_channel, num_filters / groups] + filter_size
    img_filter = helper.create_parameter(
        dtype=input.dtype, shape=filter_shape, attr=helper.param_attr)

    pre_bias = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type=l_type,
        inputs={'Input': [input],
                'Filter': [img_filter]},
        outputs={'Output': pre_bias},
        attrs={
            'strides': stride,
            'paddings': padding,
            'dilations': dilation,
            'groups': groups,
            'use_cudnn': use_cudnn
        })

    pre_act = helper.append_bias_op(pre_bias, dim_start=1, dim_end=2)
    out = helper.append_activation(pre_act)
    return out


def sequence_expand(x, y, ref_level=-1, name=None):
    """Sequence Expand Layer. This layer will expand the input variable **x**
    according to specified level lod of **y**. Please note that lod level of
    **x** is at most 1 and rank of **x** is at least 2. When rank of **x**
    is greater than 2, then it would be viewed as a 2-D tensor.
    Following examples will explain how sequence_expand works:

    .. code-block:: text

        * Case 1
            x is a LoDTensor:
                x.lod  = [[2,        2]]
                x.data = [[a], [b], [c], [d]]
                x.dims = [4, 1]

            y is a LoDTensor:
                y.lod = [[2,    2],
                         [3, 3, 1, 1]]

            ref_level: 0

            then output is a 1-level LoDTensor:
                out.lod =  [[2,        2,        2,        2]]
                out.data = [[a], [b], [a], [b], [c], [d], [c], [d]]
                out.dims = [8, 1]

        * Case 2
            x is a Tensor:
                x.data = [[a], [b], [c]]
                x.dims = [3, 1]

            y is a LoDTensor:
                y.lod = [[2, 0, 3]]

            ref_level: -1

            then output is a Tensor:
                out.data = [[a], [a], [c], [c], [c]]
                out.dims = [5, 1]
    Args:
        x (Variable): The input variable which is a Tensor or LoDTensor.
        y (Variable): The input variable which is a LoDTensor.
        ref_level (int): Lod level of `y` to be referred by `x`. If set to -1,
                         refer the last level of lod.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The expanded variable which is a LoDTensor.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[10], dtype='float32')
            y = fluid.layers.data(name='y', shape=[10, 20],
                             dtype='float32', lod_level=1)
            out = layers.sequence_expand(x=x, y=y, ref_level=0)
    """
    helper = LayerHelper('sequence_expand', input=x, **locals())
    dtype = helper.input_dtype()
    tmp = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='sequence_expand',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': tmp},
        attrs={'ref_level': ref_level})
    return tmp


def beam_search(pre_ids,
                pre_scores,
                ids,
                scores,
                beam_size,
                end_id,
                level=0,
                name=None):
    """
    Beam search is a classical algorithm for selecting candidate words in a
    machine translation task.

    Refer to `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_
    for more details.
    
    This layer does the search in beams for one time step. Specifically, it 
    selects the top-K candidate word ids of current step from :attr:`ids`
    according to their :attr:`scores` for all source sentences, where K is
    :attr:`beam_size` and :attr:`ids, scores` are predicted results from the
    computation cell. Additionally, :attr:`pre_ids` and :attr:`pre_scores` are
    the output of beam_search at previous step, they are needed for special use
    to handle ended candidate translations.
 
    Note that the :attr:`scores` passed in should be accumulated scores, and
    length penalty should be done with extra operators before calculating the
    accumulated scores if needed, also suggest finding top-K before it and
    using the top-K candidates following.

    Please see the following demo for a fully beam search usage example:

        fluid/tests/book/test_machine_translation.py

    Args:
        pre_ids(Variable): The LodTensor variable which is the output of
            beam_search at previous step. It should be a LodTensor with shape
            :math:`(batch_size, 1)` and lod
            :math:`[[0, 1, ... , batch_size], [0, 1, ..., batch_size]]` at the
            first step.
        pre_scores(Variable): The LodTensor variable which is the output of
            beam_search at previous step.
        ids(Variable): The LodTensor variable containing the candidates ids.
            Its shape should be :math:`(batch_size \\times beam_size, K)`,
            where :math:`K` supposed to be :attr:`beam_size`.
        scores(Variable): The LodTensor variable containing the accumulated
            scores corresponding to :attr:`ids` and its shape is the same as
            the shape of :attr:`ids`.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        level(int, default 0): It can be ignored and mustn't change currently.
            It means the source level of lod, which is explained as following.
            The lod level of :attr:`ids` should be 2. The first level is source
            level which describes how many prefixes (branchs) for each source
            sentece (beam), and the second level is sentence level which
            describes how these candidates belong to the prefix. The paths
            linking prefixes and selected candidates are organized and reserved
            in lod.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The LodTensor pair containing the selected ids and the \
            corresponding scores.

    Examples:
        .. code-block:: python

            # Suppose `probs` contains predicted results from the computation
            # cell and `pre_ids` and `pre_scores` is the output of beam_search
            # at previous step.
            topk_scores, topk_indices = layers.topk(probs, k=beam_size)
            accu_scores = layers.elementwise_add(
                x=layers.log(x=topk_scores)),
                y=layers.reshape(
                    pre_scores, shape=[-1]),
                axis=0)
            selected_ids, selected_scores = layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=end_id)
    """
    helper = LayerHelper('beam_search', **locals())
    score_type = scores.dtype
    id_type = ids.dtype

    selected_scores = helper.create_tmp_variable(dtype=score_type)
    selected_ids = helper.create_tmp_variable(dtype=id_type)

    helper.append_op(
        type='beam_search',
        inputs={
            'pre_ids': pre_ids,
            'pre_scores': pre_scores,
            'ids': ids,
            'scores': scores,
        },
        outputs={
            'selected_ids': selected_ids,
            'selected_scores': selected_scores,
        },
        attrs={
            # TODO(ChunweiYan) to assure other value support
            'level': level,
            'beam_size': beam_size,
            'end_id': end_id,
        })

    return selected_ids, selected_scores


def beam_search_decode(ids, scores, beam_size, end_id, name=None):
    """
    Beam Search Decode Layer. This layer constructs the full hypotheses for
    each source sentence by walking back along the LoDTensorArray :attr:`ids`
    whose lods can be used to restore the path in the beam search tree.
    Please see the following demo for a fully beam search usage example:
        fluid/tests/book/test_machine_translation.py

    Args:
        ids(Variable): The LodTensorArray variable containing the selected ids
            of all steps.
        scores(Variable): The LodTensorArray variable containing the selected
            scores of all steps.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The LodTensor pair containing the generated id sequences \
            and the corresponding scores. The shapes and lods of the two \
            LodTensor are same. The lod level is 2 and the two levels \
            separately indicate how many hypotheses each source sentence has \
            and how many ids each hypothesis has.

    Examples:
        .. code-block:: python
            # Suppose `ids` and `scores` are LodTensorArray variables reserving
            # the selected ids and scores of all steps
            finished_ids, finished_scores = layers.beam_search_decode(
                ids, scores, beam_size=5, end_id=0)
    """
    helper = LayerHelper('beam_search_decode', **locals())
    sentence_ids = helper.create_tmp_variable(dtype=ids.dtype)
    sentence_scores = helper.create_tmp_variable(dtype=ids.dtype)

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
    """Lstm unit layer. The equation of a lstm step is:

        .. math::

            i_t & = \sigma(W_{x_i}x_{t} + W_{h_i}h_{t-1} + b_i)

            f_t & = \sigma(W_{x_f}x_{t} + W_{h_f}h_{t-1} + b_f)

            c_t & = f_tc_{t-1} + i_t tanh (W_{x_c}x_t + W_{h_c}h_{t-1} + b_c)

            o_t & = \sigma(W_{x_o}x_{t} + W_{h_o}h_{t-1} + b_o)

            h_t & = o_t tanh(c_t)

    The inputs of lstm unit include :math:`x_t`, :math:`h_{t-1}` and
    :math:`c_{t-1}`. The 2nd dimensions of :math:`h_{t-1}` and :math:`c_{t-1}`
    should be same. The implementation separates the linear transformation and
    non-linear transformation apart. Here, we take :math:`i_t` as an example.
    The linear transformation is applied by calling a `fc` layer and the
    equation is:

        .. math::

            L_{i_t} = W_{x_i}x_{t} + W_{h_i}h_{t-1} + b_i

    The non-linear transformation is applied by calling `lstm_unit_op` and the
    equation is:

        .. math::

            i_t = \sigma(L_{i_t})

    This layer has two outputs including :math:`h_t` and :math:`o_t`.

    Args:
        x_t (Variable): The input value of current step, a 2-D tensor with shape
            M x N, M for batch size and N for input size.
        hidden_t_prev (Variable): The hidden value of lstm unit, a 2-D tensor
            with shape M x S, M for batch size and S for size of lstm unit.
        cell_t_prev (Variable): The cell value of lstm unit, a 2-D tensor with
            shape M x S, M for batch size and S for size of lstm unit.
        forget_bias (float): The forget bias of lstm unit.
        param_attr (ParamAttr): The attributes of parameter weights, used to set
            initializer, name etc.
        bias_attr (ParamAttr): The attributes of bias weights, if not False,
            bias weights will be created and be set to default value.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        tuple: The hidden value and cell value of lstm unit.

    Raises:
        ValueError: The ranks of **x_t**, **hidden_t_prev** and **cell_t_prev**
                    not be 2 or the 1st dimensions of **x_t**, **hidden_t_prev**
                    and **cell_t_prev** not be the same or the 2nd dimensions of
                    **hidden_t_prev** and **cell_t_prev** not be the same.

    Examples:

        .. code-block:: python

             x_t = fluid.layers.fc(input=x_t_data, size=10)
             prev_hidden = fluid.layers.fc(input=prev_hidden_data, size=30)
             prev_cell = fluid.layers.fc(input=prev_cell_data, size=30)
             hidden_value, cell_value = fluid.layers.lstm_unit(x_t=x_t,
                                                    hidden_t_prev=prev_hidden,
                                                    cell_t_prev=prev_cell)
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
    c = helper.create_tmp_variable(dtype)
    h = helper.create_tmp_variable(dtype)

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
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool|False): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the corresponding output tensor.
            fluid.layers.reduce_sum(x)  # [3.5]
            fluid.layers.reduce_sum(x, dim=0)  # [0.3, 0.5, 1.1, 1.6]
            fluid.layers.reduce_sum(x, dim=-1)  # [1.9, 1.6]
            fluid.layers.reduce_sum(x, dim=1, keep_dim=True)  # [[1.9], [1.6]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1, 2], [3, 4]],
            #      [[5, 6], [7, 8]]]
            # Each example is followed by the corresponding output tensor.
            fluid.layers.reduce_sum(x, dim=[1, 2]) # [10, 26]
            fluid.layers.reduce_sum(x, dim=[0, 1]) # [16, 20]

    """
    helper = LayerHelper('reduce_sum', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
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
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimension along which the mean is computed. If
            `None`, compute the mean over all elements of :attr:`input`
            and return a variable with a single element, otherwise it
            must be in the range :math:`[-rank(input), rank(input))`. If
            :math:`dim[i] < 0`, the dimension to reduce is
            :math:`rank(input) + dim[i]`.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set `None`, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced mean Variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_mean(x)  # [0.4375]
            fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
            fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
            fluid.layers.reduce_mean(
                x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_mean(x, dim=[1, 2]) # [2.5, 6.5]
            fluid.layers.reduce_mean(x, dim=[0, 1]) # [4.0, 5.0]
    """
    helper = LayerHelper('reduce_mean', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
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
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimension along which the maximum is computed.
            If :attr:`None`, compute the maximum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_max(x)  # [0.9]
            fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
            fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
            fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_max(x, dim=[1, 2]) # [4.0, 8.0]
            fluid.layers.reduce_max(x, dim=[0, 1]) # [7.0, 8.0]
    """
    helper = LayerHelper('reduce_max', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
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
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimensions along which the minimum is computed.
            If :attr:`None`, compute the minimum over all elements of
            :attr:`input` and return a Tensor variable with a single element,
            otherwise must be in the range :math:`[-rank(input), rank(input))`.
            If :math:`dim[i] < 0`, the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_min(x)  # [0.1]
            fluid.layers.reduce_min(x, dim=0)  # [0.1, 0.2, 0.5, 0.7]
            fluid.layers.reduce_min(x, dim=-1)  # [0.2, 0.1]
            fluid.layers.reduce_min(x, dim=1, keep_dim=True)  # [[0.2], [0.1]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_min(x, dim=[1, 2]) # [1.0, 5.0]
            fluid.layers.reduce_min(x, dim=[0, 1]) # [1.0, 2.0]
    """
    helper = LayerHelper('reduce_min', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
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
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (list|int|None): The dimensions along which the product is performed. If
            :attr:`None`, multipy all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
        keep_dim (bool|False): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true.
        name(str|None): A name for this layer(optional). If set None, the
            layer will be named automatically.

    Returns:
        Variable: The reduced Tensor variable.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with following elements:
            #    [[0.2, 0.3, 0.5, 0.9]
            #     [0.1, 0.2, 0.6, 0.7]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_prod(x)  # [0.0002268]
            fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
            fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
            fluid.layers.reduce_prod(x, dim=1,
                                     keep_dim=True)  # [[0.027], [0.0084]]

            # x is a Tensor variable with shape [2, 2, 2] and elements as below:
            #      [[[1.0, 2.0], [3.0, 4.0]],
            #      [[5.0, 6.0], [7.0, 8.0]]]
            # Each example is followed by the correspending output tensor.
            fluid.layers.reduce_prod(x, dim=[1, 2]) # [24.0, 1680.0]
            fluid.layers.reduce_prod(x, dim=[0, 1]) # [105.0, 384.0]
    """
    helper = LayerHelper('reduce_prod', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
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


def split(input, num_or_sections, dim=-1, name=None):
    """
    Split the input tensor into multiple sub-tensors.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        num_or_sections (int|list): If :attr:`num_or_sections` is an integer,
            then the integer indicates the number of equal sized sub-tensors
            that the tensor will be divided into. If :attr:`num_or_sections`
            is a list of integers, the length of list indicates the number of
            sub-tensors and the integers indicate the sizes of sub-tensors'
            :attr:`dim` dimension orderly.
        dim (int): The dimension along which to split. If :math:`dim < 0`, the
            dimension to split along is :math:`rank(input) + dim`.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.

    Returns:
        list(Variable): The list of segmented tensor variables.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with shape [3, 9, 5]:
            x0, x1, x2 = fluid.layers.split(x, num_or_sections=3, dim=1)
            x0.shape  # [3, 3, 5]
            x1.shape  # [3, 3, 5]
            x2.shape  # [3, 3, 5]
            x0, x1, x2 = fluid.layers.split(
                x, num_or_sections=[2, 3, 4], dim=1)
            x0.shape  # [3, 2, 5]
            x1.shape  # [3, 3, 5]
            x2.shape  # [3, 4, 5]
    """
    helper = LayerHelper('split', **locals())
    input_shape = input.shape
    dim = (len(input_shape) + dim) if dim < 0 else dim
    if isinstance(num_or_sections, int):
        assert num_or_sections > 1, 'num_or_sections must be more than 1.'
        num = num_or_sections
    else:
        assert len(num_or_sections) < input_shape[
            dim], 'len(num_or_sections) must not be more than input.shape[dim].'
        num = len(num_or_sections)
    outs = [
        helper.create_tmp_variable(dtype=helper.input_dtype())
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
    **L2 normalize Layer**

    The l2 normalize layer normalizes `x` along dimension `axis` using an L2
    norm. For a 1-D tensor (`dim` is fixed to 0), this layer computes

    .. math::

        y = \\frac{x}{ \sqrt{\sum {x^2} + epsion }}

    For `x` with more dimensions, this layer independently normalizes each 1-D
    slice along dimension `axis`.

    Args:
        x(Variable|list): The input tensor to l2_normalize layer.
        axis(int): The axis on which to apply normalization. If `axis < 0`, \
            the dimension to normalization is rank(X) + axis. -1 is the
            last dimension.
        epsilon(float): The epsilon value is used to avoid division by zero, \
            the defalut value is 1e-10.
        name(str|None): A name for this layer(optional). If set None, the layer \
            will be named automatically.

    Returns:
        Variable: The output tensor variable is the same shape with `x`.

    Examples:

        .. code-block:: python

            data = fluid.layers.data(name="data",
                                     shape=(3, 17, 13),
                                     dtype="float32")
            normed = fluid.layers.l2_normalize(x=data, axis=1)
    """

    if len(x.shape) == 1:
        axis = 0
    helper = LayerHelper("l2_normalize", **locals())

    out = helper.create_tmp_variable(dtype=x.dtype)
    norm = helper.create_tmp_variable(dtype=x.dtype)
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


def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
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
        name(str|None): A name for this layer(optional). If set None, the layer
            will be named automatically.

    Returns:
        Variable: The product Tensor variable.

    Examples:
        .. code-block:: python

            # Examples to clarify shapes of the inputs and output
            # x: [B, ..., M, K], y: [B, ..., K, N]
            fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

            # x: [B, M, K], y: [B, K, N]
            fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [B, M, K], y: [K, N]
            fluid.layers.matmul(x, y)  # out: [B, M, N]

            # x: [M, K], y: [K, N]
            fluid.layers.matmul(x, y)  # out: [M, N]

            # x: [B, M, K], y: [K]
            fluid.layers.matmul(x, y)  # out: [B, M]

            # x: [K], y: [K]
            fluid.layers.matmul(x, y)  # out: [1]

            # x: [M], y: [N]
            fluid.layers.matmul(x, y, True, True)  # out: [M, N]
    """

    def __check_input(x, y):
        if len(y.shape) > len(x.shape):
            raise ValueError(
                "Invalid inputs for matmul. "
                "x's rank should be always greater than or equal to y'rank.")

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
            raise ValueError("Invalid inputs for matmul.")

        if len(y_shape) > 2:
            for i, dim_x in enumerate(x_shape[:-2]):
                if dim_x != y_shape[i]:
                    raise ValueError("Invalid inputs for matmul.")

    __check_input(x, y)

    helper = LayerHelper('matmul', **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='matmul',
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'transpose_X': transpose_x,
               'transpose_Y': transpose_y})
    return out


def topk(input, k, name=None):
    """
    This operator is used to find values and indices of the k largest entries
    for the last dimension.

    If the input is a vector (1-D Tensor), finds the k largest entries in the vector
    and outputs their values and indices as vectors. Thus values[j] is the j-th
    largest entry in input, and its index is indices[j].

    If the input is a Tensor with higher rank, this operator computes the top k
    entries along the last dimension.

    For example:

    .. code-block:: text

        If:
            input = [[5, 4, 2, 3],
                     [9, 7, 10, 25],
                     [6, 2, 10, 1]]
            k = 2

        Then:
            The first output:
            values = [[5, 4],
                      [10, 25],
                      [6, 10]]

            The second output:
            indices = [[0, 1],
                       [2, 3],
                       [0, 2]]

    Args:
        input(Variable): The input variable which can be a vector or Tensor with
            higher rank.
        k(int):  The number of top elements to look for along the last dimension
                 of input.
        name(str|None): A name for this layer(optional). If set None, the layer
                       will be named automatically.
                       Default: None

    Returns:
        Tuple[Variable]: A tuple with two elements. Each element is a Variable.
        The first one is k largest elements along each last
        dimensional slice. The second one is indices of values
        within the last dimension of input.

    Raises:
        ValueError: If k < 1 or k is not less than the last dimension of input

    Examples:
        .. code-block:: python

            top5_values, top5_indices = layers.topk(input, k=5)
    """
    shape = input.shape
    if k < 1 or k >= shape[-1]:
        raise ValueError("k must be greater than 0 and less than %d." %
                         (shape[-1]))

    helper = LayerHelper("top_k", **locals())
    values = helper.create_tmp_variable(dtype=input.dtype)
    indices = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="top_k",
        inputs={"X": [input]},
        outputs={"Out": [values],
                 "Indices": [indices]},
        attrs={"k": k})
    values.stop_gradient = True
    indices.stop_gradient = True
    return values, indices


def edit_distance(input, label, normalized=True, ignored_tokens=None):
    """
    EditDistance operator computes the edit distances between a batch of
    hypothesis strings and their references. Edit distance, also called
    Levenshtein distance, measures how dissimilar two strings are by counting
    the minimum number of operations to transform one string into anthor.
    Here the operations include insertion, deletion, and substitution.

    For example, given hypothesis string A = "kitten" and reference
    B = "sitting", the edit distance is 3 for A will be transformed into B
    at least after two substitutions and one insertion:

    "kitten" -> "sitten" -> "sittin" -> "sitting"

    The input is a LoDTensor consisting of all the hypothesis strings with
    the total number denoted by `batch_size`, and the separation is specified
    by the LoD information. And the `batch_size` reference strings are arranged
    in order in the same way in the input LoDTensor.

    The output contains the `batch_size` results and each stands for the edit
    distance for a pair of strings respectively. If Attr(normalized) is true,
    the edit distance will be divided by the length of reference string.

    Args:
        input(Variable): The indices for hypothesis strings.
        label(Variable): The indices for reference strings.
        normalized(bool, default True): Indicated whether to normalize the edit distance by
                          the length of reference string.
        ignored_tokens(list<int>, default None): Tokens that should be removed before
                                     calculating edit distance.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: sequence-to-sequence edit distance in shape [batch_size, 1].

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[8], dtype='float32')
            y = fluid.layers.data(name='y', shape=[7], dtype='float32')
            cost = fluid.layers.edit_distance(input=x,label=y)
    """
    helper = LayerHelper("edit_distance", **locals())

    # remove some tokens from input and labels
    if ignored_tokens is not None and len(ignored_tokens) > 0:
        erased_input = helper.create_tmp_variable(dtype="int64")
        erased_label = helper.create_tmp_variable(dtype="int64")

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

    # edit distance op
    edit_distance_out = helper.create_tmp_variable(dtype="int64")
    sequence_num = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="edit_distance",
        inputs={"Hyps": [input],
                "Refs": [label]},
        outputs={"Out": [edit_distance_out],
                 "SequenceNum": [sequence_num]},
        attrs={"normalized": normalized})

    return edit_distance_out, sequence_num


def ctc_greedy_decoder(input, blank, name=None):
    """
    This op is used to decode sequences by greedy policy by below steps:

    1. Get the indexes of max value for each row in input. a.k.a.
       numpy.argmax(input, axis=0).
    2. For each sequence in result of step1, merge repeated tokens between two
       blanks and delete all blanks.

    A simple example as below:

    .. code-block:: text

        Given:

        input.data = [[0.6, 0.1, 0.3, 0.1],
                      [0.3, 0.2, 0.4, 0.1],
                      [0.1, 0.5, 0.1, 0.3],
                      [0.5, 0.1, 0.3, 0.1],

                      [0.5, 0.1, 0.3, 0.1],
                      [0.2, 0.2, 0.2, 0.4],
                      [0.2, 0.2, 0.1, 0.5],
                      [0.5, 0.1, 0.3, 0.1]]

        input.lod = [[4, 4]]

        Then:

        output.data = [[2],
                       [1],
                       [3]]

        output.lod = [[2, 1]]

    Args:

        input(Variable): (LoDTensor<float>), the probabilities of
                         variable-length sequences, which is a 2-D Tensor with
                         LoD information. It's shape is [Lp, num_classes + 1],
                         where Lp is the sum of all input sequences' length and
                         num_classes is the true number of classes. (not
                         including the blank label).
        blank(int): the blank label index of Connectionist Temporal
                    Classification (CTC) loss, which is in thehalf-opened
                    interval [0, num_classes + 1).
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: CTC greedy decode result. If all the sequences in result were
        empty, the result LoDTensor will be [-1] with LoD [[]] and dims [1, 1].

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[8], dtype='float32')

            cost = fluid.layers.ctc_greedy_decoder(input=x, blank=0)
    """
    helper = LayerHelper("ctc_greedy_decoder", **locals())
    _, topk_indices = topk(input, k=1)

    # ctc align op
    ctc_out = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="ctc_align",
        inputs={"Input": [topk_indices]},
        outputs={"Output": [ctc_out]},
        attrs={"merge_repeated": True,
               "blank": blank})
    return ctc_out


def warpctc(input, label, blank=0, norm_by_times=False):
    """
    An operator integrating the open source Warp-CTC library
    (https://github.com/baidu-research/warp-ctc)
    to compute Connectionist Temporal Classification (CTC) loss.
    It can be aliased as softmax with CTC, since a native softmax activation is
    interated to the Warp-CTC library, to to normlize values for each row of the
    input tensor.

    Args:
       input (Variable): The unscaled probabilities of variable-length sequences,
         which is a 2-D Tensor with LoD information.
         It's shape is [Lp, num_classes + 1], where Lp is the sum of all input
         sequences' length and num_classes is the true number of classes.
         (not including the blank label).
       label (Variable): The ground truth of variable-length sequence,
         which is a 2-D Tensor with LoD information. It is of the shape [Lg, 1],
         where Lg is th sum of all labels' length.
       blank (int, default 0): The blank label index of Connectionist
         Temporal Classification (CTC) loss, which is in the
         half-opened interval [0, num_classes + 1).
       norm_by_times(bool, default false): Whether to normalize the gradients
         by the number of time-step, which is also the sequence's length.
         There is no need to normalize the gradients if warpctc layer was
         follewed by a mean_op.

    Returns:
        Variable: The Connectionist Temporal Classification (CTC) loss,
        which is a 2-D Tensor of the shape [batch_size, 1].

    Examples:

        .. code-block:: python

            label = fluid.layers.data(shape=[11, 8], dtype='float32', lod_level=1)
            predict = fluid.layers.data(shape=[11, 1], dtype='float32')
            cost = fluid.layers.warpctc(input=predict, label=label)

    """
    helper = LayerHelper('warpctc', **locals())
    loss_out = helper.create_tmp_variable(dtype=input.dtype)
    grad_out = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='warpctc',
        inputs={'Logits': [input],
                'Label': [label]},
        outputs={'WarpCTCGrad': [grad_out],
                 'Loss': [loss_out]},
        attrs={'blank': blank,
               'norm_by_times': norm_by_times})
    return loss_out


def sequence_reshape(input, new_dim):
    """
    **Sequence Reshape Layer**

    This layer will rearrange the input sequences. The new dimension is set by
    user. Length of each sequence is computed according to original length,
    original dimension and new dimension. The following example will help to
    illustrate the function of this layer:

    .. code-block:: text

        x is a LoDTensor:
            x.lod  = [[0, 2, 6]]
            x.data = [[1,  2], [3,  4],
                      [5,  6], [7,  8],
                      [9, 10], [11, 12]]
            x.dims = [6, 2]

        set new_dim = 4

        then out is a LoDTensor:

            out.lod  = [[0, 1, 3]]

            out.data = [[1,  2,  3,  4],
                        [5,  6,  7,  8],
                        [9, 10, 11, 12]]
            out.dims = [3, 4]

    Currently, only 1-level LoDTensor is supported and please make sure
    (original length * original dimension) can be divided by new dimension with
    no remainder for each sequence.

    Args:

       input (Variable): A 2-D LoDTensor with shape being [N, M] where M for dimension.
       new_dim (int): New dimension that the input LoDTensor is reshaped to.

    Returns:

        Variable: Reshaped LoDTensor according to new dimension.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(shape=[5, 20], dtype='float32', lod_level=1)
            x_reshaped = fluid.layers.sequence_reshape(input=x, new_dim=10)
    """
    helper = LayerHelper('sequence_reshape', **locals())
    out = helper.create_tmp_variable(helper.input_dtype())
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
        num_neg_samples=None):
    """
    ${comment}

    Args:
        input (Variable): input variable.
        label (Variable): label.
        num_total_classes (int):${num_total_classes_comment}
        sample_weight (Variable|None): A Variable of shape [batch_size, 1]
            storing a weight for each sample. The default weight for each
            sample is 1.0.
        param_attr (ParamAttr|None): attributes for parameter
        bias_attr (ParamAttr|None): attributes for bias
        num_neg_samples (int): ${num_neg_samples_comment}

    Returns:
        Variable: The output nce loss.

    Examples:
        .. code-block:: python

            window_size = 5
            words = []
            for i in xrange(window_size):
                words.append(layers.data(
                    name='word_{0}'.format(i), shape=[1], dtype='int64'))

            dict_size = 10000
            label_word = int(window_size / 2) + 1

            embs = []
            for i in xrange(window_size):
                if i == label_word:
                    continue

                emb = layers.embedding(input=words[i], size=[dict_size, 32],
                                       param_attr='emb.w', is_sparse=True)
                embs.append(emb)

            embs = layers.concat(input=embs, axis=1)
            loss = layers.nce(input=embs, label=words[label_word],
                          num_total_classes=dict_size, param_attr='nce.w',
                          bias_attr='nce.b')
    """
    helper = LayerHelper('nce', **locals())
    assert isinstance(input, Variable)
    dim = input.shape[1]
    assert isinstance(label, Variable)
    num_true_class = label.shape[1]
    w = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_total_classes, dim],
        is_bias=False,
        dtype=input.dtype)
    b = helper.create_parameter(
        attr=helper.bias_attr,
        shape=[num_total_classes, 1],
        is_bias=True,
        dtype=input.dtype)
    cost = helper.create_tmp_variable(dtype=input.dtype)
    sample_logits = helper.create_tmp_variable(dtype=input.dtype)
    sample_labels = helper.create_tmp_variable(dtype=label.dtype)

    if num_neg_samples is None:
        num_neg_samples = 10
    else:
        num_neg_samples = int(num_neg_samples)

    attrs = {
        'num_total_classes': int(num_total_classes),
        'num_neg_samples': num_neg_samples
    }

    helper.append_op(
        type='nce',
        inputs={
            'Input': input,
            'Label': label,
            'Weight': w,
            'Bias': b,
            'SampleWeight': sample_weight if sample_weight is not None else []
        },
        outputs={
            'Cost': cost,
            'SampleLogits': sample_logits,
            'SampleLabels': sample_labels
        },
        attrs=attrs)
    return cost / (num_neg_samples + 1)


def hsigmoid(input, label, num_classes, param_attr=None, bias_attr=None):
    """
    The hierarchical sigmoid operator is used to accelerate the training
    process of language model. This operator organizes the classes into a 
    complete binary tree, each leaf node represents a class(a word) and each
    internal node acts as a binary classifier. For each word there's a unique
    path from root to it's leaf node, hsigmoid calculate the cost for each
    internal node on the path, and sum them to get a total cost. hsigmoid can
    achive a acceleration from :math:`O(N)` to :math:`O(logN)`, where :math:`N`
    represents the size of word dict.

    Refer to `Hierarchical Probabilistic Neural Network Language Model
    <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_
    
    Args:
        input (Variable): The input tensor variable with shape 
            :math:`[N \\times D]`, where :math:`N` is the size of mini-batch,
            and :math:`D` is the feature size.
        label (Variable): The tensor variable contains labels of training data.
            It's a tensor with shape is :math:`[N \\times 1]`.
        num_classes: (int), The number of classes, must not be less than 2.
        param_attr (ParamAttr|list of ParamAttr, default None): The parameter
             attribute for learnable parameters/weights of this layer.
        bias_attr (ParamAttr|list of ParamAttr, default None):  The parameter 
             attribute for the bias of this layer. If it is set to False, no
             bias will be applied.

    Returns:
        Out: (Tensor) The cost of hierarchical sigmoid operator. the shape is [N, 1]

    Examples:

        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[2], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='int64')
            out = fluid.layers.hsigmoid(input=x, label=y, num_classes=6)
    """

    helper = LayerHelper('hierarchical_sigmoid', **locals())
    dtype = helper.input_dtype()
    out = helper.create_tmp_variable(dtype)
    pre_out = helper.create_tmp_variable(dtype)
    dim = input.shape[1]
    if num_classes < 2:
        raise ValueError("num_classes must not be less than 2.")
    weights = helper.create_parameter(
        attr=helper.param_attr,
        shape=[num_classes - 1, dim],
        is_bias=False,
        dtype=input.dtype)
    inputs = {"X": input, "W": weights, "Label": label}
    if helper.bias_attr:
        bias = helper.create_parameter(
            attr=helper.bias_attr,
            shape=[1, num_classes - 1],
            is_bias=True,
            dtype=input.dtype)
        inputs['Bias'] = bias
    helper.append_op(
        type="hierarchical_sigmoid",
        inputs=inputs,
        outputs={"Out": out,
                 "PreOut": pre_out},
        attrs={"num_classes": num_classes})
    return out


def transpose(x, perm, name=None):
    """
    Permute the dimensions of `input` according to `perm`.

    The `i`-th dimension  of the returned tensor will correspond to the
    perm[i]-th dimension of `input`.

    Args:
        x (Variable): The input Tensor.
        perm (list): A permutation of the dimensions of `input`.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: A transposed Tensor.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[5, 10, 15], dtype='float32')
            x_transposed = layers.transpose(x, perm=[1, 0, 2])
    """

    if len(perm) != len(x.shape):
        raise ValueError(
            "Input(perm) is the permutation of dimensions of Input(input). "
            "It's length shoud be equal to Input(input)'s rank.")
    for idx, dim in enumerate(perm):
        if dim >= len(x.shape):
            raise ValueError(
                "Each element in perm should be less than x's rank. "
                "%d-th element in perm is %d which accesses x's rank %d." %
                (idx, perm[idx], len(x.shape)))

    helper = LayerHelper('transpose', **locals())
    out = helper.create_tmp_variable(x.dtype)
    helper.append_op(
        type='transpose',
        inputs={'X': [x]},
        outputs={'Out': [out]},
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
    {input.batch_size * output_height * output_width, filter_size_H *
    filter_size_W * input.channels} which is similar with im2col.
    This op use filter / kernel to scan images and convert these images to
    sequences. After expanding, the number of time step are
    output_height * output_width for an image, in which output_height and
    output_width are calculated by below equation:

    .. math::

        output\_size = 1 + \
            (2 * padding + img\_size - block\_size + stride - 1) / stride

    And the dimension of each time step is block_y * block_x * input.channels.

    Args:
        input (Variable): The input should be a tensor in NCHW format.

        filter_size(int|tuple|None): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.

        stride(int|tuple): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: stride = 1.

        padding(int|tuple): The padding size. If padding is a tuple, it can
            contain two integers like (padding_H, padding_W) which means
            padding_up = padding_down = padding_H and
            padding_left = padding_right = padding_W. Or it can use
            (padding_up, padding_left, padding_down, padding_right) to indicate
            paddings of four direction. Otherwise, a scalar padding means
            padding_up = padding_down = padding_left = padding_right = padding
            Default: padding = 0.

        input_image_size(Variable): the input contains image real size.It's dim
            is [batchsize, 2]. It is dispensable.It is just for batch inference.

        out_stride(int|tuple): The scaling of image through CNN. It is
            dispensable. It is valid only when input_image_size is not null.
            If out_stride is tuple,  it must contain two intergers,
            (out_stride_H, out_stride_W). Otherwise,
            the out_stride_H = out_stride_W = out_stride.

        name (int): The name of this layer. It is optional.

    Returns:
        output: The output is a LoDTensor with shape
        {input.batch_size * output_height * output_width,
        filter_size_H * filter_size_W * input.channels}.
        If we regard output as a matrix, each row of this matrix is
        a step of a sequence.

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

            output = fluid.layers.im2sequence(
                input=layer, stride=[1, 1], filter_size=[2, 2])

    """

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
    attrs = {"kernels": filter_size, "strides": stride, "padding": padding}
    if input_image_size:
        if isinstance(out_stride, int):
            out_stride = [out_stride, out_stride]
        inputs["Y"] = input_image_size
        attrs["out_stride"] = out_stride
    helper = LayerHelper('im2sequence', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
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
        >>> import paddle.fluid as fluid
        >>> x = fluid.layers.data(name='x', shape=[16],
        >>>                        dtype='float32', lod_level=1)
        >>> out = fluid.layers.row_conv(input=x, future_context_size=2)
    """
    helper = LayerHelper('row_conv', **locals())
    dtype = helper.input_dtype()
    filter_shape = [future_context_size + 1, input.shape[1]]
    filter_param = helper.create_parameter(
        attr=helper.param_attr, shape=filter_shape, dtype=dtype)
    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='row_conv',
        inputs={'X': [input],
                'Filter': [filter_param]},
        outputs={'Out': [out]})
    return helper.append_activation(out)


@templatedoc()
def multiplex(inputs, index):
    """
    ${comment}

    >>> import paddle.fluid as fluid
    >>> x1 = fluid.layers.data(name='x1', shape=[4], dtype='float32')
    >>> x2 = fluid.layers.data(name='x2', shape=[4], dtype='float32')
    >>> index = fluid.layers.data(name='index', shape=[1], dtype='int32')
    >>> out = fluid.layers.multiplex(inputs=[x1, x2], index=index)

    Args:
       inputs (list): ${x_comment}.
       index (${ids_type}): ${ids_comment}.

    Returns:
        ${out_comment}.
    """
    helper = LayerHelper('multiplex', **locals())

    if not isinstance(inputs, list) and len(inputs) < 2:
        raise ValueError("inputs should be a list object and contains at least "
                         "2 elements.")

    out = helper.create_tmp_variable(inputs[0].dtype)
    helper.append_op(
        type='multiplex',
        inputs={'X': inputs,
                'Ids': index},
        outputs={'Out': [out]})
    return out


def softmax_with_cross_entropy(logits, label, soft_label=False):
    """
    **Softmax With Cross Entropy Operator.**

    Cross entropy loss with softmax is used as the output layer extensively. This
    operator computes the softmax normalized values for each row of the input
    tensor, after which cross-entropy loss is computed. This provides a more
    numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute soft_label is set false, this operators expects mutually
    exclusive hard labels, each sample in a batch is in exactly one class with a
    probability of 1.0. Each sample in the batch will have a single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::

        loss_j =  -\\text{logit}_{label_j} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{logit}_i)\\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::

        loss_j =  -\\sum_{i=0}^{K}\\text{label}_i
        \\left(\\text{logit}_i - \\log\\left(\\sum_{i=0}^{K}
        \\exp(\\text{logit}_i)\\right)\\right), j = 1,...,K

    Args:
        logits (Variable): The unscaled log probabilities, which is a 2-D tensor
            with shape [N x K]. N is the batch_size, and K is the class number.
        label (Variable): The ground truth which is a 2-D tensor. If soft_label
            is set to false, Label is a Tensor<int64> with shape [N x 1]. If
            soft_label is set to true, Label is a Tensor<float/double> with
        soft_label (bool): A flag to indicate whether to interpretate the given
            labels as soft labels. By default, `soft_label` is set to False.
    Returns:
        Variable: The cross entropy loss is a 2-D tensor with shape [N x 1].

    Examples:
        .. code-block:: python

            data = fluid.layers.data(name='data', shape=[128], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            fc = fluid.layers.fc(input=data, size=100)
            out = fluid.layers.softmax_with_cross_entropy(
                logits=fc, label=label)
    """
    helper = LayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_tmp_variable(dtype=logits.dtype)
    loss = helper.create_tmp_variable(dtype=logits.dtype)
    helper.append_op(
        type='softmax_with_cross_entropy',
        inputs={'Logits': logits,
                'Label': label},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs={'soft_label': soft_label})
    return loss


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
        y (Variable): A tensor with rank at least 2. The target value of smooth
            L1 loss op with same shape as :attr:`x`.
        inside_weight (Variable|None):  A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the result of (:attr:`x` - :attr:`y`) will be multiplied
            by this tensor element by element.
        outside_weight (Variable|None): A tensor with rank at least 2. This
            input is optional and should have same shape with :attr:`x`. If
            provided, the out smooth L1 loss will be multiplied by this tensor
            element by element.
        sigma (float|None): Hyper parameter of smooth L1 loss layer. A float
           scalar with default value 1.0.

    Returns:
        Variable: The output smooth L1 loss with shape [batch_size, 1].

    Examples:
        .. code-block:: python

            data = fluid.layers.data(name='data', shape=[128], dtype='float32')
            label = fluid.layers.data(
                name='label', shape=[100], dtype='float32')
            fc = fluid.layers.fc(input=data, size=100)
            out = fluid.layers.smooth_l1(x=fc, y=label)
    """

    helper = LayerHelper('smooth_l1_loss', **locals())
    diff = helper.create_tmp_variable(dtype=x.dtype)
    loss = helper.create_tmp_variable(dtype=x.dtype)
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
        attrs={'sigma': sigma})
    return loss


def one_hot(input, depth):
    """
    This layer creates the one-hot representations for input indices.

    Args:
        input(Variable): Input indices, last dimension must be 1.
        depth(scalar): An interger defining the depth of the one-hot dimension.

    Returns:
        Variable: The one-hot representations of input.

    Examples:
        .. code-block:: python

            label = layers.data(name="label", shape=[1], dtype="float32")
            one_hot_label = layers.one_hot(input=label, depth=10)
    """
    helper = LayerHelper("one_hot", **locals())
    one_hot_out = helper.create_tmp_variable(dtype='float32')
    helper.append_op(
        type="one_hot",
        inputs={'X': input},
        attrs={'depth': depth},
        outputs={'Out': one_hot_out})
    return one_hot_out


def autoincreased_step_counter(counter_name=None, begin=1, step=1):
    """
    Create an auto-increase variable
    which will be automatically increased by 1 every mini-batch
    Return the run counter of the main program, default is started from 1.

    Args:
        counter_name(str): The counter name, default is '@STEP_COUNTER@'.
        begin(int): The first value of this counter.
        step(int): The increment step between each execution.

    Returns:
        Variable: The global run counter.

    Examples:
        .. code-block:: python

           global_step = fluid.layers.autoincreased_step_counter(
               counter_name='@LR_DECAY_COUNTER@', begin=begin, step=1)
    """
    helper = LayerHelper('global_step_counter')
    if counter_name is None:
        counter_name = '@STEP_COUNTER@'
    counter, is_new_var = helper.create_or_get_global_variable(
        name=counter_name, dtype='int64', shape=[1], persistable=True)
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


def reshape(x, shape, actual_shape=None, act=None, inplace=True, name=None):
    """
    Gives a new shape to the input Tensor without changing its data.

    The target shape can be given by :attr:`shape` or :attr:`actual_shape`.
    :attr:`shape` is a list of integer while :attr:`actual_shape` is a tensor
    variable. :attr:`actual_shape` has a higher priority than :attr:`shape`
    if it is provided, while :attr:`shape` still should be set correctly to
    gurantee shape inference in compile-time.

    Some tricks exist when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The indice of 0s in shape can not exceed
    Rank(X).

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

    Args:
        x(variable): The input tensor.
        shape(list): The new shape. At most one dimension of the new shape can
                     be -1.
        actual_shape(variable): An optional input. If provided, reshape
                                according to this given shape rather than
                                :attr:`shape` specifying shape. That is to
                                say :attr:`actual_shape` has a higher priority
                                than :attr:`shape`.
        act (str): The non-linear activation to be applied to output variable.
        inplace(bool): If this flag is set true, the output
                       shares data with input without copying, otherwise
                       a new output tensor is created
                       whose data is copied from input x.
        name (str): The name of this layer. It is optional.

    Returns:
        Variable: The output tensor.

    Raises:
        TypeError: if actual_shape is neither Variable nor None.

    Examples:
        .. code-block:: python

            data = fluid.layers.data(
                name='data', shape=[2, 4, 6], dtype='float32')
            reshaped = fluid.layers.reshape(
                x=data, shape=[-1, 0, 3, 2], act='tanh', inplace=True)
    """

    if not (isinstance(shape, list) or isinstance(shape, tuple)):
        raise ValueError("Input shape must be a python lsit or tuple.")
    inputs = {"X": x}
    if isinstance(actual_shape, Variable):
        inputs["Shape"] = actual_shape
    elif actual_shape is not None:
        raise TypeError("actual_shape should either be Variable or None")

    # Validate the shape
    unk_dim_idx = -1
    for dim_idx, dim_size in enumerate(shape):
        if dim_size == -1:
            assert unk_dim_idx == -1, (
                "Only one dimension in shape can be unknown.")
            unk_dim_idx = dim_idx
        elif dim_size == 0:
            assert dim_idx < len(x.shape), (
                "The indice of 0s in shape can not exceed Rank(X).")
        else:
            assert dim_size > 0, (
                "Each dimension size given in shape must not be negtive "
                "except one unknown dimension.")

    helper = LayerHelper("reshape", **locals())
    reshaped = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="reshape",
        inputs=inputs,
        attrs={"shape": shape,
               "inplace": inplace},
        outputs={"Out": reshaped})

    return helper.append_activation(reshaped)


def lod_reset(x, y=None, target_lod=None):
    """
    Set LoD of :attr:`x` to a new one specified by :attr:`y` or
    :attr:`target_lod`. When :attr:`y` provided, :attr:`y.lod` would be
    considered as target LoD first, otherwise :attr:`y.data` would be
    considered as target LoD. If :attr:`y` is not provided, target LoD should
    be specified by :attr:`target_lod`. If target LoD is specified by
    :attr:`Y.data` or :attr:`target_lod`, only one level LoD is supported.

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

            then we get a 2-level LoDTensor:
                out.lod =  [[2, 2], [2, 2, 1, 1]]
                out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
                out.dims = [6, 1]

    Args:
        x (Variable): Input variable which could be a Tensor or LodTensor.
        y (Variable|None): If provided, output's LoD would be derived
                           from :attr:`y`.
        target_lod (list|tuple|None): One level LoD which should be considered
                                      as target LoD when :attr:`y` not provided.

    Returns:
        Variable: Output variable with LoD specified by this layer.

    Raises:
        ValueError: If :attr:`y` and :attr:`target_lod` are both None.

    Examples:
        .. code-block:: python

            x = layers.data(name='x', shape=[10])
            y = layers.data(name='y', shape=[10, 20], lod_level=2)
            out = layers.lod_reset(x=x, y=y)
    """
    helper = LayerHelper("lod_reset", **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
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
        raise ValueError("y and target_lod should not be both None.")

    return out


def lrn(input, n=5, k=1.0, alpha=1e-4, beta=0.75, name=None):
    """
    Local Response Normalization Layer. This layer performs a type of
    "lateral inhibition" by normalizing over local input regions.

    The formula is as follows:

    .. math::

      Output(i, x, y) = Input(i, x, y) / \\left(k + \\alpha \\sum\\limits^{\\min(C, c + n/2)}_{j = \\max(0, c - n/2)}(Input(j, x, y))^2\\right)^{\\beta}

    In the above equation:

    * :math:`n`: The number of channels to sum over.
    * :math:`k`: The offset (avoid being divided by 0).
    * :math:`alpha`: The scaling parameter.
    * :math:`beta`: The exponent parameter.

    Refer to `ImageNet Classification with Deep Convolutional Neural Networks
    <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_

    Args:
        input (Variable): The input tensor of this layer, and the dimension of input tensor must be 4.
        n (int, default 5): The number of channels to sum over.
        k (float, default 1.0): An offset (usually positive to avoid dividing by 0).
        alpha (float, default 1e-4): The scaling parameter.
        beta (float, default 0.75): The exponent.
        name (str, default None): A name for this operation.

    Raises:
        ValueError: If rank of the input tensor is not 4.

    Returns:
        A tensor variable storing the transformation result.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(
              name="data", shape=[3, 112, 112], dtype="float32")
          lrn = fluid.layers.lrn(input=data)
    """
    helper = LayerHelper('lrn', **locals())
    dtype = helper.input_dtype()
    input_shape = input.shape
    dims = len(input_shape)

    if dims != 4:
        raise ValueError(
            "dims of input must be 4(not %d), and it's order must be NCHW" %
            (dims))

    mid_out = helper.create_tmp_variable(dtype=dtype, stop_gradient=True)
    lrn_out = helper.create_tmp_variable(dtype)
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
    Pads a tensor with a constant value given by :attr:`pad_value`, and the
    padded width is specified by :attr:`paddings`.

    Specifically, the number of values padded before the contents of :attr:`x`
    in dimension :attr:`i` is indicated by :attr:`paddings[i]`, and the number
    of values padded after the contents of :attr:`x` in dimension :attr:`i` is
    indicated by :attr:`paddings[i+1]`.

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
        x (Variable): The input tensor variable.
        paddings (list): A list of integers. Its elements specify the padded
                         width before and after for each dimension in turn.
                         The length of :attr:paddings must be
                         :math:`rank(x) \\times 2`.
        pad_value (float): The constant value used to pad.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The padded tensor variable.

    Examples:
        .. code-block:: python

            # x is a rank 2 tensor variable.
            out = fluid.layers.pad(
                x=x, paddings=[0, 1, 1, 2], pad_value=0.)
    """
    helper = LayerHelper('pad', input=x, **locals())
    dtype = helper.input_dtype()
    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type='pad',
        inputs={'X': x},
        outputs={'Out': out},
        attrs={'paddings': paddings,
               'pad_value': float(pad_value)})
    return out


def label_smooth(label,
                 prior_dist=None,
                 epsilon=0.1,
                 dtype="float32",
                 name=None):
    """
    Label smoothing is a mechanism to regularize the classifier layer and is
    called label-smoothing regularization (LSR).

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

    Args:
        label(Variable): The input variable containing the label data. The
                          label data should use one-hot representation.
        prior_dist(Variable): The prior distribution to be used to smooth
                              labels. If not provided, an uniform distribution
                              is used. The shape of :attr:`prior_dist` should
                              be :math:`(1, class\_num)`.
        epsilon(float): The weight used to mix up the original ground-truth
                        distribution and the fixed distribution.
        dtype(np.dtype|core.VarDesc.VarType|str): The type of data : float32,
                                                  float_64, int etc.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The tensor variable containing the smoothed labels.

    Examples:
        .. code-block:: python

            label = layers.data(name="label", shape=[1], dtype="float32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="float32")
    """
    if epsilon > 1. or epsilon < 0.:
        raise ValueError("The value of epsilon must be between 0 and 1.")
    helper = LayerHelper("label_smooth", **locals())
    label.stop_gradient = True
    smooth_label = helper.create_tmp_variable(dtype)
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
    ${comment}

    Args:
        input (Variable): ${x_comment}
        rois (Variable): ROIs (Regions of Interest) to pool over.
        pooled_height (integer): ${pooled_height_comment} Default: 1
        pooled_width (integer): ${pooled_width_comment} Default: 1
        spatial_scale (float): ${spatial_scale_comment} Default: 1.0

    Returns:
        Variable: ${out_comment}.

    Examples:
        .. code-block:: python

            pool_out = fluid.layers.roi_pool(input=x, rois=rois, 7, 7, 1.0)
    """
    helper = LayerHelper('roi_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_tmp_variable(dtype)
    argmaxes = helper.create_tmp_variable(dtype='int32')
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


def dice_loss(input, label, epsilon=0.00001):
    """
    Dice loss for comparing the similarity of two batch of data,
    usually is used for binary image segmentation i.e. labels are binary.
    The dice loss can be defined as below equation:

    .. math::

        dice\_loss &= 1 - \\frac{2 * intersection\_area}{total\_area} \\\\
                  &= \\frac{(total\_area - intersection\_area) - intersection\_area}{total\_area} \\\\
                  &= \\frac{(union\_area - intersection\_area)}{total\_area}


    Args:
        input (Variable): The predictions with rank>=2. The first dimension is batch size,
                          and the last dimension is class number.
        label (Variable): The groud truth with the same rank with input. The first dimension
                          is batch size, and the last dimension is 1.
        epsilon (float): The epsilon will be added to the numerator and denominator.
                         If both input and label are empty, it makes sure dice is 1.
                         Default: 0.00001

    Returns:
        dice_loss (Variable): The dice loss with shape [1].

    Examples:
        .. code-block:: python

            predictions = fluid.layers.softmax(x)
            loss = fluid.layers.dice_loss(input=predictions, label=label, 2)
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
                 resample='BILINEAR'):
    """
    **Resize a Batch of Images**

    The input must be a tensor of the shape (num_batches, channels, in_h, in_w),
    and the resizing only applies on the last two dimensions(hight and width).

    Supporting resample methods:

        'BILINEAR' : Bilinear interpolation

    Args:
        input (Variable): The input tensor of image resize layer,
                          This is a 4-D tensor of the shape
                          (num_batches, channels, in_h, in_w).
        out_shape(list|tuple|Variable|None): Output shape of image resize
                                    layer, the shape is (out_h, out_w).
                                    Default: None
        scale(float|None): The multiplier for the input height or width.
                         At least one of out_shape or scale must be set.
                         And out_shape has a higher priority than scale.
                         Default: None
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.
        resample(str): The resample method. It can only be 'BILINEAR' currently.
                       Default: 'BILINEAR'

    Returns:
        Variable: The output is a 4-D tensor of the shape
        (num_batches, channls, out_h, out_w).

    Examples:
        .. code-block:: python

            out = fluid.layers.image_resize(input, out_shape=[12, 12])
    """
    resample_methods = {'BILINEAR': 'bilinear_interp'}
    if resample not in resample_methods:
        raise ValueError(
            "The 'resample' of image_resize can only be 'BILINEAR' currently.")
    if out_shape is None and scale is None:
        raise ValueError("One of out_shape and scale must not be None")
    helper = LayerHelper('bilinear_interp', **locals())
    dtype = helper.input_dtype()

    def _is_list_or_turple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    out_h = 0
    out_w = 0
    inputs = {"X": input}
    if out_shape is not None:
        if not (_is_list_or_turple_(out_shape) and
                len(out_shape) == 2) and not isinstance(out_shape, Variable):
            raise ValueError('out_shape should be a list or tuple or variable')
        if _is_list_or_turple_(out_shape):
            out_shape = list(map(int, out_shape))
            out_h = out_shape[0]
            out_w = out_shape[1]
        else:
            inputs['OutSize'] = out_shape
    else:
        out_h = int(input.shape[2] * scale)
        out_w = int(input.shape[3] * scale)

    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type=resample_methods[resample],
        inputs=inputs,
        outputs={"Out": out},
        attrs={"out_h": out_h,
               "out_w": out_w})
    return out


@templatedoc(op_type="bilinear_interp")
def resize_bilinear(input, out_shape=None, scale=None, name=None):
    """
    ${comment}

    Args:
        input(${x_type}): ${x_comment}.

        out_shape(${out_size_type}): ${out_size_comment}.

        scale(float|None): The multiplier for the input height or width. At
             least one of out_shape or scale must be set. And out_shape has
             a higher priority than scale. Default: None.

        name(str|None): The output variable name.

    Returns:
        ${out_comment}.
    """

    return image_resize(input, out_shape, scale, name, 'BILINEAR')


def image_resize_short(input, out_short_len, resample='BILINEAR'):
    """
    Resize a batch of images. The short edge of input images will be
    resized to the given 'out_short_len'. The long edge of input images
    will be resized proportionately to make images' length-width ratio
    constant.

    Args:
        input (Variable): The input tensor of image resize layer,
                          This is a 4-D tensor of the shape
                          (num_batches, channels, in_h, in_w).
        out_short_len(int): The length of output images' short edge.
        resample (str): resample method, default: BILINEAR.

    Returns:
        Variable: The output is a 4-D tensor of the shape
        (num_batches, channls, out_h, out_w).
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


def gather(input, index):
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
        input (Variable): The source input with rank>=1.
        index (Variable): The index input with rank=1.

    Returns:
        output (Variable): The output is a tensor with the same rank as input.

    Examples:

        .. code-block:: python

            output = fluid.layers.gather(x, index)
    """
    helper = LayerHelper('gather', **locals())
    dtype = helper.input_dtype()
    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="gather",
        inputs={"X": input,
                "Index": index},
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
        >>> img = fluid.layers.data("img", [3, 256, 256])
        >>> cropped_img = fluid.layers.random_crop(img, shape=[3, 224, 224])
    """
    helper = LayerHelper("random_crop", **locals())
    dtype = x.dtype
    out = helper.create_tmp_variable(dtype)
    if seed is None:
        seed = random.randint(-65536, 65535)
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


def log(x):
    """
    Calculates the natural log of the given input tensor, element-wise.

    .. math::

        Out = \\ln(x)

    Args:
        x (Variable): Input tensor.

    Returns:
        Variable: The natural log of the input tensor computed element-wise.

    Examples:

        .. code-block:: python

            output = fluid.layers.log(x)
    """
    helper = LayerHelper('log', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_tmp_variable(dtype)
    helper.append_op(type="log", inputs={"X": x}, outputs={"Out": out})
    return out


def relu(x):
    """
    Relu takes one input data (Tensor) and produces one output data (Tensor)
    where the rectified linear function, y = max(0, x), is applied to
    the tensor elementwise.

    .. math::

        Out = \\max(0, x)

    Args:
        x (Variable): The input tensor.

    Returns:
        Variable: The output tensor with the same shape as input.

    Examples:

        .. code-block:: python

            output = fluid.layers.relu(x)
    """
    helper = LayerHelper('relu', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_tmp_variable(dtype)
    helper.append_op(type="relu", inputs={"X": x}, outputs={"Out": out})
    return out


def mean_iou(input, label, num_classes):
    """
    Mean Intersection-Over-Union is a common evaluation metric for
    semantic image segmentation, which first computes the IOU for each
    semantic class and then computes the average over classes.
    IOU is defined as follows:

    .. math::

        IOU = \\frac{true\_positiv}{(true\_positive + false\_positive + false\_negative)}.

    The predictions are accumulated in a confusion matrix and mean-IOU
    is then calculated from it.


    Args:
        input (Variable): A Tensor of prediction results for semantic labels with type int32 or int64.
        label (Variable): A Tensor of ground truth labels with type int32 or int64.
                           Its shape should be the same as input.
        num_classes (int): The possible number of labels.

    Returns:
        mean_iou (Variable): A Tensor representing the mean intersection-over-union with shape [1].
        out_wrong(Variable): A Tensor with shape [num_classes]. The wrong numbers of each class.
        out_correct(Variable): A Tensor with shape [num_classes]. The correct numbers of each class.

    Examples:

        .. code-block:: python

            iou, wrongs, corrects = fluid.layers.mean_iou(predict, label, num_classes)
    """
    helper = LayerHelper('mean_iou', **locals())
    dtype = helper.input_dtype()
    out_mean_iou = helper.create_tmp_variable(dtype='float32')
    out_wrong = helper.create_tmp_variable(dtype='int32')
    out_correct = helper.create_tmp_variable(dtype='int32')
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

    Args:
        x (Variable): The input tensor variable.
        shape (Variable|list/tuple of integer): The output shape is specified
            by `shape`, which can a Variable or a list/tupe of integer.
            If a tensor Variable, it's rank must be the same as `x`. This way
            is suitable for the case that the output shape may be changed each
            iteration. If a list/tupe of integer, it's length must be the same
            as the rank of `x`
        offsets (Variable|list/tuple of integer|None): Specifies the copping
            offsets at each dimension. It can be a Variable or or a list/tupe
            of integer. If a tensor Variable, it's rank must be the same as `x`.
            This way is suitable for the case that the offsets may be changed
            each iteration. If a list/tupe of integer, it's length must be the
            same as the rank of `x`. If None, the offsets are 0 at each
            dimension.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        Variable: The cropped tensor variable.

    Raises:
        ValueError: If shape is not a list, tuple or Variable.

    Examples:

        .. code-block:: python

            x = fluid.layers.data(name="x", shape=[3, 5], dtype="float32")
            y = fluid.layers.data(name="y", shape=[2, 3], dtype="float32")
            crop = fluid.layers.crop(x, shape=y)

            # or
            z = fluid.layers.data(name="z", shape=[3, 5], dtype="float32")
            crop = fluid.layers.crop(z, shape=[2, 3])

    """
    helper = LayerHelper('crop', **locals())

    if not (isinstance(shape, list) or isinstance(shape, tuple) or \
        isinstance(shape, Variable)):
        raise ValueError("The shape should be a list, tuple or Variable.")

    if offsets is None:
        offsets = [0] * len(x.shape)

    out = helper.create_tmp_variable(x.dtype)
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


def rank_loss(label, left, right, name=None):
    """
    **Rank loss layer for RankNet**

    RankNet(http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)
    is a pairwise ranking model with a training sample consisting of a pair
    of documents, A and B. Label P indicates whether A is ranked higher than B
    or not:
 
    P = {0, 1} or {0, 0.5, 1}, where 0.5 means that there is no information
    about the rank of the input pair.
    
    Rank loss layer takes three inputs: left (o_i), right (o_j) and
    label (P_{i,j}). The inputs respectively represent RankNet's output scores
    for documents A and B and the value of label P. The following equation
    computes rank loss C_{i,j} from the inputs:
    
    $$
      C_{i,j} = -\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\
      o_{i,j} =  o_i - o_j  \\
      \tilde{P_{i,j}} = \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}
    $$
    
    Rank loss layer takes batch inputs with size batch_size (batch_size >= 1).   
 
    Args:
        label (Variable): Indicats whether A ranked higher than B or not.
        left (Variable): RankNet's output score for doc A.
        right (Variable): RankNet's output score for doc B.
        name(str|None): A name for this layer(optional). If set None, the layer
                        will be named automatically.

    Returns:
        list: The value of rank loss.

    Raises:
        ValueError: Any of label, left, and right is not a variable.

    Examples:

        .. code-block:: python

            label = fluid.layers.data(name="label", shape=[4, 1], dtype="float32")
            left = fluid.layers.data(name="left", shape=[4, 1], dtype="float32")
            right = fluid.layers.data(name="right", shape=[4, 1], dtype="float32")
            out = fluid.layers.rank_loss(label, left, right)


    """
    helper = LayerHelper('rank_loss', **locals())

    if not (isinstance(label, Variable)):
        raise ValueError("The label should be a Variable")

    if not (isinstance(left, Variable)):
        raise ValueError("The left should be a Variable")

    if not (isinstance(right, Variable)):
        raise ValueError("The right should be a Variable")

    out = helper.create_tmp_variable("float32")

    helper.append_op(
        type='rank_loss',
        inputs={"Label": label,
                "Left": left,
                "Right": right},
        outputs={'Out': out})
    return out
