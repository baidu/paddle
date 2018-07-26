#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import itertools

from .framework import Variable, Parameter, default_main_program, default_startup_program, dtype_is_floating
from . import unique_name
from paddle.fluid.initializer import Constant, Xavier
from .param_attr import ParamAttr, WeightNormParamAttr
from . import core


class LayerHelper(object):
    def __init__(self, layer_type, **kwargs):
        self.kwargs = kwargs
        self.layer_type = layer_type
        name = self.kwargs.get('name', None)
        if name is None:
            self.kwargs['name'] = unique_name.generate(self.layer_type)

    @property
    def name(self):
        return self.kwargs['name']

    @property
    def main_program(self):
        return default_main_program()

    @property
    def startup_program(self):
        return default_startup_program()

    def append_op(self, *args, **kwargs):
        return self.main_program.current_block().append_op(*args, **kwargs)

    def multiple_input(self, input_param_name='input'):
        inputs = self.kwargs.get(input_param_name, [])
        type_error = TypeError(
            "Input of {0} layer should be Variable or sequence of Variable".
            format(self.layer_type))
        if isinstance(inputs, Variable):
            inputs = [inputs]
        elif not isinstance(inputs, list) and not isinstance(inputs, tuple):
            raise type_error
        else:
            for each in inputs:
                if not isinstance(each, Variable):
                    raise type_error
        return inputs

    def input(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        if len(inputs) != 1:
            raise "{0} layer only takes one input".format(self.layer_type)
        return inputs[0]

    @property
    def param_attr(self):
        return ParamAttr._to_attr(self.kwargs.get('param_attr', None))

    @property
    def bias_attr(self):
        return ParamAttr._to_attr(self.kwargs.get('bias_attr', None))

    def multiple_param_attr(self, length):
        param_attr = self.param_attr
        if isinstance(param_attr, ParamAttr):
            param_attr = [param_attr]

        if len(param_attr) != 1 and len(param_attr) != length:
            raise ValueError("parameter number mismatch")
        elif len(param_attr) == 1 and length != 1:
            tmp = [None] * length
            for i in range(length):
                tmp[i] = copy.deepcopy(param_attr[0])
            param_attr = tmp
        return param_attr

    def iter_inputs_and_params(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        param_attrs = self.multiple_param_attr(len(inputs))
        for ipt, param_attr in zip(inputs, param_attrs):
            yield ipt, param_attr

    def input_dtype(self, input_param_name='input'):
        inputs = self.multiple_input(input_param_name)
        dtype = None
        for each in inputs:
            if dtype is None:
                dtype = each.dtype
            elif dtype != each.dtype:
                raise ValueError("Data Type mismatch: %d to %d" %
                                 (dtype, each.dtype))
        return dtype

    def _create_weight_normalize(self, attr, shape, dtype):
        from .layers import elementwise_mul, elementwise_div, reshape

        # Remove these ops when LayerHelper and layers support indicating
        # program and block.
        def __norm_op(x,
                      out=None,
                      p=2,
                      dim=None,
                      keep_dim=False,
                      block=self.startup_program.global_block()):
            if out is None:
                out = block.create_var(
                    name=unique_name.generate(".".join(
                        [self.name, 'weight_norm_norm'])),
                    dtype=dtype,
                    persistable=False)
            abs_out = block.create_var(
                name=unique_name.generate(".".join(
                    [self.name, 'weight_norm_abs'])),
                dtype=dtype,
                persistable=False)
            block.append_op(
                type='abs', inputs={'X': x}, outputs={'Out': abs_out})
            pow_out = block.create_var(
                name=unique_name.generate(".".join(
                    [self.name, 'weight_norm_pow'])),
                dtype=dtype,
                persistable=False)
            block.append_op(
                type='pow',
                inputs={'X': abs_out},
                outputs={'Out': pow_out},
                attrs={'factor': float(p)})
            sum_out = block.create_var(
                name=unique_name.generate(".".join(
                    [self.name, 'weight_norm_sum'])),
                dtype=dtype,
                persistable=False)
            block.append_op(
                type='reduce_sum',
                inputs={'X': pow_out},
                outputs={'Out': sum_out},
                attrs={
                    'dim': dim,
                    'keep_dim': keep_dim,
                    'reduce_all': True if dim is None else False
                })
            block.append_op(
                type='pow',
                inputs={'X': sum_out},
                outputs={'Out': out},
                attrs={'factor': 1. / p})
            return out

        def __reshape_op(x,
                         shape,
                         out=None,
                         block=self.startup_program.global_block()):
            if out is None:
                out = block.create_var(
                    name=unique_name.generate(".".join(
                        [self.name, 'weight_norm_reshape'])),
                    dtype=dtype,
                    persistable=False)
            block.append_op(
                type='reshape',
                inputs={'X': x},
                outputs={'Out': out},
                attrs={'shape': shape})
            return out

        def __transpose_op(x,
                           axis,
                           out=None,
                           block=self.startup_program.global_block()):
            if out is None:
                out = block.create_var(
                    name=unique_name.generate(".".join(
                        [self.name, 'weight_norm_transpose'])),
                    dtype=dtype,
                    persistable=False)
            block.append_op(
                type='transpose',
                inputs={'X': x},
                outputs={'Out': out},
                attrs={'axis': axis})
            return out

        def __norm_except_dim(x,
                              out=None,
                              dim=None,
                              block=self.startup_program.global_block()):
            """Computes the norm over all dimensions except dim"""
            if out is None:
                out = block.create_var(
                    name=unique_name.generate(".".join(
                        [self.name, 'weight_norm_norm'])),
                    dtype=dtype,
                    persistable=False)
            if dim is None:
                __norm_op(x, out, dim=dim, block=block)
            elif dim == 0:
                out_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
                reshape = __reshape_op(x, shape=[x.shape[0], -1], block=block)
                norm = __norm_op(reshape, dim=1, block=block)
                __reshape_op(norm, out=out, shape=out_shape, block=block)
            elif dim == len(x.shape) - 1:
                out_shape = [1] * (len(x.shape) - 1) + [x.shape[-1]]
                reshape = __reshape_op(x, shape=[-1, x.shape[-1]], block=block)
                norm = __norm_op(reshape, dim=0, block=block)
                __reshape_op(norm, out=out, shape=out_shape, block=block)
            else:
                perm = list(range(len(x.shape)))
                perm[0], perm[dim] = dim, 0
                transpose = __transpose_op(x, perm, block=block)
                norm = __norm_op(transpose, dim=0, block=block)
                __transpose_op(norm, perm, out=out, block=block)
            return out

        def __weight_normalize(g, v, dim):
            """Calculations for weight normalization"""
            norm = __norm_except_dim(
                v, dim=dim, block=self.main_program.current_block())
            scale = elementwise_div(
                x=g, y=norm)  # The shapes of g and norm are the same.
            # Currently, elementwise_mul only support broadcast when the shape
            # of y is a subset of the shape of x. Thus, we reshape y to squeeze
            # to achive the subset.
            w = elementwise_mul(
                x=v,
                y=scale if dim is None else reshape(
                    x=scale, shape=[v.shape[dim]]),
                axis=-1 if dim is None else dim)
            # To serialize the original parameter for inference, maybe a
            # parameter rather than a variable should be returned.
            return w

        g_param_attr = copy.deepcopy(attr)
        g_param_attr.name = attr.name + '_g'
        g_param_shape = [1] * len(shape)
        if attr.dim is not None:
            g_param_shape[attr.dim] = shape[attr.dim]
        v_param_attr = copy.deepcopy(attr)
        v_param_attr.name = attr.name + '_v'
        v_param_shape = shape

        # Add to startup_program to initialize g and v.
        # Try to reconstruct the initializer of w by initializing g and v.
        # Set the initializers of g and v as below, then the distribution
        # of w is the same as initializing w with the given initializer.
        # For Data-Dependent Initialization, please compute the init-values
        # of g and v in external and then feed the values to g and v by
        # executing an extra program.
        g_param = self.startup_program.global_block().create_parameter(
            dtype=dtype,
            shape=g_param_shape,
            **g_param_attr._to_kwargs(with_initializer=False))
        v_param = self.startup_program.global_block().create_parameter(
            dtype=dtype,
            shape=v_param_shape,
            **v_param_attr._to_kwargs(with_initializer=True))
        __norm_except_dim(
            x=v_param,
            out=g_param,
            dim=attr.dim,
            block=self.startup_program.global_block())

        # Add weight normalization to main_program
        g_param = self.main_program.global_block().create_parameter(
            dtype=dtype, shape=g_param_shape, **g_param_attr._to_kwargs())
        v_param = self.main_program.global_block().create_parameter(
            dtype=dtype, shape=v_param_shape, **v_param_attr._to_kwargs())
        w_param = __weight_normalize(g_param, v_param, dim=attr.dim)
        return w_param

    def create_parameter(self,
                         attr,
                         shape,
                         dtype,
                         is_bias=False,
                         default_initializer=None):
        # Deepcopy the attr so that parameters can be shared in program
        attr = copy.deepcopy(attr)
        assert isinstance(attr, ParamAttr)
        suffix = 'b' if is_bias else 'w'
        if attr.name is None:
            attr.name = unique_name.generate(".".join([self.name, suffix]))

        if default_initializer is None and attr.initializer is None:
            if is_bias:
                attr._set_default_bias_initializer()
            else:
                attr._set_default_param_initializer()
        else:
            attr._set_default_initializer(default_initializer)

        # If weight normalization is set, insert extra parameters and ops.
        # Refer to https://arxiv.org/pdf/1602.07868.pdf
        if isinstance(attr, WeightNormParamAttr):
            param = self._create_weight_normalize(attr, shape, dtype)
            WeightNormParamAttr.params_with_weight_norm.append(param)
            return param

        self.startup_program.global_block().create_parameter(
            dtype=dtype, shape=shape, **attr._to_kwargs(with_initializer=True))
        return self.main_program.global_block().create_parameter(
            dtype=dtype, shape=shape, **attr._to_kwargs())

    def get_parameter(self, name):
        param = self.main_program.global_block().var(name)
        if not isinstance(param, Parameter):
            raise ValueError("no Parameter name %s found" % name)
        return param

    def create_tmp_variable(self, dtype, stop_gradient=False):
        return self.main_program.current_block().create_var(
            name=unique_name.generate(".".join([self.name, 'tmp'])),
            dtype=dtype,
            persistable=False,
            stop_gradient=stop_gradient)

    def create_variable(self, *args, **kwargs):
        return self.main_program.current_block().create_var(*args, **kwargs)

    def create_global_variable(self, persistable=False, *args, **kwargs):
        """
        create global variable, note that there is no initializer for this global variable.
        Args:
            persistable(bool): True if it is a checkpoint value.
            *args: See create_var's documentation
            **kwargs: See create_var's documentation

        Returns(Variable): the created variable.
        """
        return self.main_program.global_block().create_var(
            *args, persistable=persistable, **kwargs)

    def create_or_get_global_variable(self, name, *args, **kwargs):
        """
        Creates a global variable if not exists and returns the variable and
        a boolean flag which is true when it is a new variable.
        """
        if self.main_program.global_block().has_var(name):
            return self.main_program.global_block().var(name), False
        else:
            return self.create_global_variable(name=name, *args, **kwargs), True

    def set_variable_initializer(self, var, initializer):
        assert isinstance(var, Variable)
        self.startup_program.global_block().create_var(
            name=var.name,
            type=var.type,
            dtype=var.dtype,
            shape=var.shape,
            persistable=True,
            initializer=initializer)

    def append_bias_op(self, input_var, dim_start=1, dim_end=None):
        """
        Append bias operator and return its output. If the user does not set
        bias_attr, append_bias_op will return input_var

        :param input_var: the input variable. The len(input_var.shape) is
        larger or equal than 2.
        :bias_initializer: an instance of a subclass of Initializer used to
        initialize the bias
        :param dim_start:
        :param dim_end: the shape of the bias will be
        input_var.shape[dim_start:dim_end]. The bias is broadcasted to other
        dimensions and added to input_var to get the output
        """
        size = list(input_var.shape[dim_start:dim_end])
        bias_attr = self.bias_attr
        if not bias_attr:
            return input_var

        b = self.create_parameter(
            attr=bias_attr, shape=size, dtype=input_var.dtype, is_bias=True)
        tmp = self.create_tmp_variable(dtype=input_var.dtype)
        self.append_op(
            type='elementwise_add',
            inputs={'X': [input_var],
                    'Y': [b]},
            outputs={'Out': [tmp]},
            attrs={'axis': dim_start})
        return tmp

    def append_activation(self, input_var):
        act = self.kwargs.get('act', None)
        if act is None:
            return input_var
        if isinstance(act, str):
            act = {'type': act}

        if 'use_cudnn' in self.kwargs and self.kwargs.get('use_cudnn'):
            act['use_cudnn'] = self.kwargs.get('use_cudnn')
        if 'use_mkldnn' in self.kwargs:
            act['use_mkldnn'] = self.kwargs.get('use_mkldnn')
        act_type = act.pop('type')
        tmp = input_var
        # NOTE(dzhwinter): some activation support inplace compution.
        if not core.IsInplace(act_type):
            tmp = self.create_tmp_variable(dtype=input_var.dtype)
        self.append_op(
            type=act_type,
            inputs={"X": [input_var]},
            outputs={"Out": [tmp]},
            attrs=act)
        return tmp

    def _get_default_initializer(self, dtype):
        if dtype is None or dtype_is_floating(dtype) is True:
            return Xavier()
        else:
            # For integer and boolean types, initialize with all zeros
            return Constant()

    def is_instance(self, param_name, cls):
        param = self.kwargs.get(param_name, None)
        if not isinstance(param, cls):
            raise TypeError("The input {0} parameter of method {1} must be {2}",
                            param_name, self.layer_type, cls.__name__)
