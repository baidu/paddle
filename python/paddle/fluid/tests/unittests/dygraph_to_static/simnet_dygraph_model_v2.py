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

from functools import reduce
import paddle
from paddle.static import Variable


class EmbeddingLayer(object):
    """
    Embedding Layer class
    """

    def __init__(self, dict_size, emb_dim, name="emb", padding_idx=None):
        """
        initialize
        """
        self.dict_size = dict_size
        self.emb_dim = emb_dim
        self.name = name
        self.padding_idx = padding_idx

    def ops(self):
        """
        operation
        """
        # TODO(huihuangzheng): The original code set the is_sparse=True, but it
        # causes crush in dy2stat. Set it to True after fixing it.
        emb = paddle.fluid.dygraph.Embedding(
            size=[self.dict_size, self.emb_dim],
            is_sparse=True,
            padding_idx=self.padding_idx,
            param_attr=paddle.ParamAttr(
                name=self.name,
                initializer=paddle.nn.initializer.XavierUniform()))

        return emb


class FCLayer(object):
    """
    Fully Connect Layer class
    """

    def __init__(self, fc_dim, act, name="fc"):
        """
        initialize
        """
        self.fc_dim = fc_dim
        self.act = act
        self.name = name

    def ops(self):
        """
        operation
        """
        fc = FC(size=self.fc_dim,
                param_attr=paddle.ParamAttr(name="%s.w" % self.name),
                bias_attr=paddle.ParamAttr(name="%s.b" % self.name),
                act=self.act)
        return fc


class ConcatLayer(object):
    """
    Connection Layer class
    """

    def __init__(self, axis):
        """
        initialize
        """
        self.axis = axis

    def ops(self, inputs):
        """
        operation
        """
        concat = paddle.concat(x=inputs, axis=self.axis)
        return concat


class ReduceMeanLayer(object):
    """
    Reduce Mean Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        mean = paddle.mean(input)
        return mean


class CosSimLayer(object):
    """
    Cos Similarly Calculate Layer
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        sim = paddle.nn.functional.cosine_similarity(x, y)
        return sim


class ElementwiseMaxLayer(object):
    """
    Elementwise Max Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        max = paddle.maximum(x=x, y=y)
        return max


class ElementwiseAddLayer(object):
    """
    Elementwise Add Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        add = paddle.add(x=x, y=y)
        return add


class ElementwiseSubLayer(object):
    """
    Elementwise Add Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, x, y):
        """
        operation
        """
        sub = paddle.fluid.layers.elementwise_sub(x, y)
        return sub


class ConstantLayer(object):
    """
    Generate A Constant Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input, shape, dtype, value):
        """
        operation
        """
        shape = list(shape)
        input_shape = paddle.shape(input)
        shape[0] = input_shape[0]
        constant = paddle.fluid.layers.fill_constant(shape, dtype, value)
        return constant


class SoftsignLayer(object):
    """
    Softsign Layer class
    """

    def __init__(self):
        """
        initialize
        """
        pass

    def ops(self, input):
        """
        operation
        """
        softsign = paddle.nn.functional.softsign(input)
        return softsign


class FC(paddle.nn.Layer):
    r"""
    This interface is used to construct a callable object of the ``FC`` class.
    For more details, refer to code examples.
    It creates a fully connected layer in the network. It can take
    one or multiple ``Tensor`` as its inputs. It creates a Variable called weights for each input tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input tensor
    with its corresponding weight to produce an output Tensor with shape [N, `size`],
    where N is batch size. If multiple input tensors are given, the results of
    multiple output tensors with shape [N, `size`] will be summed up. If ``bias_attr``
    is not None, a bias variable will be created and added to the output.
    Finally, if ``act`` is not None, it will be applied to the output as well.
    When the input is single ``Tensor`` :
    .. math::
        Out = Act({XW + b})
    When the input are multiple ``Tensor`` :
    .. math::
        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})
    In the above equation:
    * :math:`N`: Number of the input. N equals to len(input) if input is list of ``Tensor`` .
    * :math:`X_i`: The i-th input ``Tensor`` .
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output ``Tensor`` .
    See below for an example.
    .. code-block:: text
        Given:
            data_1.data = [[[0.1, 0.2]]]
            data_1.shape = (1, 1, 2) # 1 is batch_size
            data_2.data = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3) # 1 is batch_size
            fc = FC("fc", 2, num_flatten_dims=2)
            out = fc(input=[data_1, data_2])
        Then:
            out.data = [[[0.182996 -0.474117]]]
            out.shape = (1, 1, 2)
    Parameters:
        
        size(int): The number of output units in this layer.
        num_flatten_dims (int, optional): The fc layer can accept an input tensor with more than
            two dimensions. If this happens, the multi-dimension tensor will first be flattened
            into a 2-dimensional matrix. The parameter `num_flatten_dims` determines how the input
            tensor is flattened: the first `num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest `rank(X) - num_flatten_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, suppose
            `X` is a 5-dimensional tensor with a shape [2, 3, 4, 5, 6], and `num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1
        param_attr (ParamAttr or list of ParamAttr, optional): The parameter attribute for learnable
            weights(Parameter) of this layer. Default: None.
        bias_attr (ParamAttr or list of ParamAttr, optional): The attribute for the bias
            of this layer. If it is set to False, no bias will be added to the output units.
            If it is set to None, the bias is initialized zero. Default: None.
        act (str, optional): Activation to be applied to the output of this layer. Default: None.
        is_test(bool, optional): A flag indicating whether execution is in test phase. Default: False.
        dtype(str, optional): Dtype used for weight, it can be "float32" or "float64". Default: "float32".
    Attribute:
        **weight** (list of Parameter): the learnable weights of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Returns:
        None
    
    """

    def __init__(self,
                 size,
                 num_flatten_dims=1,
                 param_attr=None,
                 bias_attr=None,
                 act=None,
                 is_test=False,
                 dtype="float32"):
        super(FC, self).__init__(dtype)

        self._size = size
        self._num_flatten_dims = num_flatten_dims
        self._dtype = dtype
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._act = act
        self.__w = list()

    def _build_once(self, input):
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            input_shape = inp.shape

            param_shape = [
                reduce(lambda a, b: a * b, input_shape[self._num_flatten_dims:],
                       1)
            ] + [self._size]
            self.__w.append(
                self.add_parameter(
                    '_w%d' % i,
                    self.create_parameter(
                        attr=param,
                        shape=param_shape,
                        dtype=self._dtype,
                        is_bias=False)))
            i += 1

        size = list([self._size])
        self._b = self.create_parameter(
            attr=self._bias_attr, shape=size, dtype=self._dtype, is_bias=True)

    # TODO(songyouwei): We should remove _w property
    @property
    def _w(self, i=0):
        return self.__w[i]

    @_w.setter
    def _w(self, value, i=0):
        assert isinstance(self.__w[i], Variable)
        self.__w[i].set_value(value)

    @property
    def weight(self):
        if len(self.__w) > 1:
            return self.__w
        else:
            return self.__w[0]

    @weight.setter
    def weight(self, value):
        if len(self.__w) == 1:
            self.__w[0] = value

    @property
    def bias(self):
        return self._b

    @bias.setter
    def bias(self, value):
        self._b = value

    def forward(self, input):
        mul_results = list()
        i = 0
        for inp, param in self._helper.iter_inputs_and_params(input,
                                                              self._param_attr):
            tmp = self._helper.create_variable_for_type_inference(self._dtype)
            self._helper.append_op(
                type="mul",
                inputs={"X": inp,
                        "Y": self.__w[i]},
                outputs={"Out": tmp},
                attrs={
                    "x_num_col_dims": self._num_flatten_dims,
                    "y_num_col_dims": 1
                })
            i += 1
            mul_results.append(tmp)

        if len(mul_results) == 1:
            pre_bias = mul_results[0]
        else:
            pre_bias = self._helper.create_variable_for_type_inference(
                self._dtype)
            self._helper.append_op(
                type="sum",
                inputs={"X": mul_results},
                outputs={"Out": pre_bias},
                attrs={"use_mkldnn": False})

        if self._b is not None:
            pre_activation = self._helper.create_variable_for_type_inference(
                dtype=self._dtype)
            self._helper.append_op(
                type='elementwise_add',
                inputs={'X': [pre_bias],
                        'Y': [self._b]},
                outputs={'Out': [pre_activation]},
                attrs={'axis': self._num_flatten_dims})
        else:
            pre_activation = pre_bias
        # Currently, we don't support inplace in dygraph mode
        return self._helper.append_activation(pre_activation, act=self._act)


class HingeLoss(object):
    """
    Hing Loss Calculate class
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        self.margin = conf_dict["loss"]["margin"]

    def compute(self, pos, neg):
        """
        compute loss
        """
        elementwise_max = ElementwiseMaxLayer()
        elementwise_add = ElementwiseAddLayer()
        elementwise_sub = ElementwiseSubLayer()
        constant = ConstantLayer()
        reduce_mean = ReduceMeanLayer()
        loss = reduce_mean.ops(
            elementwise_max.ops(
                constant.ops(neg, neg.shape, "float32", 0.0),
                elementwise_add.ops(
                    elementwise_sub.ops(neg, pos),
                    constant.ops(neg, neg.shape, "float32", self.margin))))
        return loss


class BOW(paddle.nn.Layer):
    """
    BOW
    """

    def __init__(self, conf_dict):
        """
        initialize
        """
        super(BOW, self).__init__()
        self.dict_size = conf_dict["dict_size"]
        self.task_mode = conf_dict["task_mode"]
        self.emb_dim = conf_dict["net"]["emb_dim"]
        self.bow_dim = conf_dict["net"]["bow_dim"]
        self.seq_len = conf_dict["seq_len"]
        self.emb_layer = EmbeddingLayer(self.dict_size, self.emb_dim,
                                        "emb").ops()
        self.bow_layer = paddle.nn.Linear(
            in_features=self.bow_dim, out_features=self.bow_dim)
        self.bow_layer_po = FCLayer(self.bow_dim, None, "fc").ops()
        self.softmax_layer = FCLayer(2, "softmax", "cos_sim").ops()

    @paddle.jit.to_static
    def forward(self, left, right):
        """
        Forward network
        """

        # embedding layer
        left_emb = self.emb_layer(left)
        right_emb = self.emb_layer(right)
        left_emb = paddle.reshape(
            left_emb, shape=[-1, self.seq_len, self.bow_dim])
        right_emb = paddle.reshape(
            right_emb, shape=[-1, self.seq_len, self.bow_dim])

        bow_left = paddle.fluid.layers.reduce_sum(left_emb, dim=1)
        bow_right = paddle.fluid.layers.reduce_sum(right_emb, dim=1)
        softsign_layer = SoftsignLayer()
        left_soft = softsign_layer.ops(bow_left)
        right_soft = softsign_layer.ops(bow_right)

        # matching layer
        if self.task_mode == "pairwise":
            left_bow = self.bow_layer(left_soft)
            right_bow = self.bow_layer(right_soft)
            cos_sim_layer = CosSimLayer()
            pred = cos_sim_layer.ops(left_bow, right_bow)
            return left_bow, pred
        else:
            concat_layer = ConcatLayer(1)
            concat = concat_layer.ops([left_soft, right_soft])
            concat_fc = self.bow_layer_po(concat)
            pred = self.softmax_layer(concat_fc)
            return left_soft, pred
