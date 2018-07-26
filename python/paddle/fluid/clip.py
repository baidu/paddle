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

import functools
from . import layers
from . import framework
from . import core

__all__ = [
    'ErrorClipByValue',
    'GradientClipByValue',
    'GradientClipByNorm',
    'GradientClipByGlobalNorm',
]


class BaseErrorClipAttr(object):
    def __str__(self):
        raise NotImplementedError()

    def _append_clip_op(self, block, grad_name):
        raise NotImplementedError()


class ErrorClipByValue(BaseErrorClipAttr):
    """
    Clips tensor values to the range [min, max].

    Given a tensor t, this operation clips its value to min and max inplace.

    - Any values less than min are set to min.
    - Any values greater than max are set to max.

    Args:
        max (float): The maximum value to clip by.
        min (float, optional): The minimum value to clip by. if not set by user, \
        will be set to -max by framework.

    Examples:
        .. code-block:: python

            var = fluid.framework.Variable(..., error_clip=ErrorClipByValue(max=5.0), ...)
    """

    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def __str__(self):
        return "ByValue, min=%f, max=%f" % (self.min, self.max)

    def _append_clip_op(self, block, grad_name):
        clip_op_desc = block.desc.append_op()
        clip_op_desc.set_type("clip")
        clip_op_desc.set_input("X", [grad_name])
        clip_op_desc.set_output("Out", [grad_name])
        clip_op_desc.set_attr("min", self.min)
        clip_op_desc.set_attr("max", self.max)


def error_clip_callback(block, context):
    # the context is a grad_to_var map
    grad_to_var = context
    op_desc = block.desc.op(block.desc.op_size() - 1)
    for grad_n in [n for n in op_desc.output_arg_names() if n in grad_to_var]:
        fwd_var = block._var_recursive(grad_to_var[grad_n])
        error_clip = getattr(fwd_var, "error_clip", None)
        if not (error_clip is None or isinstance(error_clip,
                                                 BaseErrorClipAttr)):
            raise TypeError(
                "Variable's error_clip should be an instance of BaseErrorClipAttr or None."
            )
        if error_clip is not None:
            error_clip._append_clip_op(block, grad_n)


class BaseGradientClipAttr(object):
    def __str__(self):
        raise NotImplementedError()

    def _process_context(self, context, param, grad):
        raise NotImplementedError()

    def _create_operators(self, param, grad):
        raise NotImplementedError()


class NullGradientClipAttr(BaseGradientClipAttr):
    def __str__(self):
        return "Null"

    def _process_context(self, context, param, grad):
        pass

    def _create_operators(self, param, grad):
        return param, grad


class GradientClipByValue(BaseGradientClipAttr):
    """
    Clips gradient values to the range [min, max].

    Given a tensor t, this operation clips its value to min and max inplace.

    - Any values less than min are set to min.
    - Any values greater than max are set to max.

    Args:
        max (float): The maximum value to clip by.
        min (float, optional): The minimum value to clip by. if not set by user, \
        will be set to -max by framework.

    Examples:
        .. code-block:: python

            w_param_attrs = ParamAttr(name=None,
              initializer=UniformInitializer(low=-1.0, high=1.0, seed=0),
              learning_rate=1.0,
              regularizer=L1Decay(1.0),
              trainable=True,
              clip=GradientClipByValue(-1.0, 1.0))
            y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)
    """

    def __init__(self, max, min=None):
        max = float(max)
        if min is None:
            min = -max
        else:
            min = float(min)
        self.max = max
        self.min = min

    def __str__(self):
        return "ByValue, min=%f, max=%f" % (self.min, self.max)

    def _process_context(self, context, param, grad):
        pass

    def _create_operators(self, param, grad):
        new_grad = layers.clip(x=grad, min=self.min, max=self.max)
        return param, new_grad


class GradientClipByNorm(BaseGradientClipAttr):
    """
    Clips tensor values to a maximum L2-norm.

    This operator limits the L2 norm of the input :math:`X` within :math:`max\_norm`.
    If the L2 norm of :math:`X` is less than or equal to :math:`max\_norm`, :math:`Out`
    will be the same as :math:`X`. If the L2 norm of :math:`X` is greater than
    :math:`max\_norm`, :math:`X` will be linearly scaled to make the L2 norm of
    :math:`Out` equal to :math:`max\_norm`, as shown in the following formula:

    .. math::

        Out = \\frac{max\_norm * X}{norm(X)},

    where :math:`norm(X)` represents the L2 norm of :math:`X`.

    Args:
        clip_norm (float): The maximum norm value

    Examples:
        .. code-block:: python

            w_param_attrs = ParamAttr(name=None,
              initializer=UniformInitializer(low=-1.0, high=1.0, seed=0),
              learning_rate=1.0,
              regularizer=L1Decay(1.0),
              trainable=True,
              clip=GradientClipByNorm(clip_norm=2.0))
            y_predict = fluid.layers.fc(input=x, size=1, param_attr=w_param_attrs)

    """

    def __init__(self, clip_norm):
        self.clip_norm = clip_norm

    def __str__(self):
        return "ByNorm, clip_norm=%f" % self.clip_norm

    def _process_context(self, context, param, grad):
        pass

    def _create_operators(self, param, grad):
        new_grad = layers.clip_by_norm(x=grad, max_norm=self.clip_norm)
        return param, new_grad


class GradientClipByGlobalNorm(BaseGradientClipAttr):
    """
    Clips values of multiple tensors by the ratio of the sum of their norms.

    Given a list of tensors t_list, and a clipping ratio clip_norm, this
    operation returns a list of clipped tensors list_clipped and the global
    norm (global_norm) of all tensors in t_list.

    To perform the clipping, the values :math:`t\_list[i]` are set to:

    .. math::

        t\_list[i] = t\_list[i] * \\frac{clip\_norm}{\max(global\_norm, clip\_norm)}

    where:

    .. math::

        global\_norm = \sqrt{\sum_{i=0}^{N-1}(l2norm(t\_list[i]))^2}

    If :math:`clip\_norm > global\_norm` then the entries in t_list remain as they are,
    otherwise they're all shrunk by the global ratio.

    Args:
        clip_norm (float): The maximum norm value
        group_name (str, optional): The group name for this clip.

    Examples:
        .. code-block:: python

            p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

            with fluid.program_guard(main_program=prog_clip):
                fluid.clip.set_gradient_clip(
                    fluid.clip.GradientClipByGlobalNorm(clip_norm=2.0))
                p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)

    """

    def __init__(self, clip_norm, group_name="default_group"):
        if not isinstance(group_name, str):
            raise TypeError("'group_name' must be a basestring.")

        self.clip_norm = clip_norm
        self.group_name = group_name

    def __str__(self):
        return "ByGlobalNorm, group_name=%s, clip_norm=%f" % (self.group_name,
                                                              self.clip_norm)

    def _process_context(self, context, param, grad):
        if self.group_name not in context:
            context[self.group_name] = []
            context[self.group_name + "_clip_value"] = self.clip_norm
            context[self.group_name + "_clip"] = layers.fill_constant(
                shape=[1], dtype="float32", value=self.clip_norm)
        else:
            if not self.clip_norm == context[self.group_name + "_clip_value"]:
                raise ValueError(
                    "All parameters' 'clip_norm' of a same group should be the same"
                )

        local_norm_var = layers.reduce_sum(input=layers.pow(x=grad, factor=2.0))
        context[self.group_name].append(local_norm_var)

        self.context = context

    def _create_operators(self, param, grad):
        group_scale_name = self.group_name + "_scale"
        if group_scale_name not in self.context:
            group_norm_var = layers.sums(input=self.context[self.group_name])
            layers.sqrt(x=group_norm_var, out=group_norm_var)
            clip_var = self.context[self.group_name + "_clip"]
            group_scale_var = layers.elementwise_div(
                x=clip_var,
                y=layers.elementwise_max(
                    x=clip_var, y=group_norm_var))
            assert group_scale_var.shape == (1, )
            self.context[group_scale_name] = group_scale_var

        new_grad = layers.elementwise_mul(
            x=grad, y=self.context[group_scale_name])
        return param, new_grad


def set_gradient_clip(clip, param_list=None, program=None):
    """
    To specify parameters that require gradient clip.

    Args:
        clip(BaseGradientClipAttr): An instance of some derived class of BaseGradientClipAttr,
                which describes the type and detailed attributes of required gradient clip.
        param_list(list(Variable)): Parameters that require gradient clip.
                It can be a list of parameter or a list of parameter's name.
                When it's None, all parameters in the program will be included.
        program(Program): The program where parameters are.
                Will be the default main program when assigned with None.
    """
    if not isinstance(clip, BaseGradientClipAttr):
        raise TypeError(
            "'clip' should be an instance of BaseGradientClipAttr's derived class"
        )
    if program is None:
        program = framework.default_main_program()
    if param_list is None:
        param_list = program.block(0).all_parameters()
    if all(isinstance(elem, str) for elem in param_list):
        param_list = [program.block(0).var(elem) for elem in param_list]
    if not all(isinstance(elem, framework.Parameter) for elem in param_list):
        raise TypeError(
            "'param_list' should be a list of Parameter or basestring(parameter's name)."
        )

    for param in param_list:
        param.gradient_clip_attr = copy.deepcopy(clip)


def append_gradient_clip_ops(param_grads):
    context = dict()
    for p, g in param_grads:
        if g is None:
            continue
        with p.block.program.optimized_guard([p, g]):
            clip_attr = getattr(p, 'gradient_clip_attr', NullGradientClipAttr())
            if clip_attr is None:
                clip_attr = NullGradientClipAttr()
            if not isinstance(clip_attr, BaseGradientClipAttr):
                raise TypeError(
                    "clip attribute should be an instance of BaseGradientClipAttr"
                )

            clip_attr._process_context(context=context, param=p, grad=g)

    res = []
    for p, g in param_grads:
        if g is None:
            continue
        with p.block.program.optimized_guard([p, g]):
            res.append(clip_attr._create_operators(param=p, grad=g))

    return res


ClipByValue = GradientClipByValue
ClipByNorm = GradientClipByNorm
ClipByGlobalNorm = GradientClipByGlobalNorm
