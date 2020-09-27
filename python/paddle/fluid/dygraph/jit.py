# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import pickle
import warnings
import functools
from collections import OrderedDict
import six

import paddle

# deprecated module import
from paddle.fluid import core
from paddle.fluid.compiler import BuildStrategy, CompiledProgram, ExecutionStrategy
from paddle.fluid.data_feeder import check_type
from paddle.fluid.dygraph.base import program_desc_tracing_guard, switch_to_static_graph
from paddle.fluid.dygraph.dygraph_to_static import logging_utils
from paddle.fluid.dygraph.dygraph_to_static.logging_utils import set_code_level, set_verbosity
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator, StaticFunction, unwrap_decorators
from paddle.fluid.dygraph.io import EXTRA_VAR_INFO_FILENAME, VARIABLE_FILENAME, TranslatedLayer
from paddle.fluid.dygraph.layers import Layer
from paddle.fluid.executor import Executor, scope_guard
from paddle.fluid.framework import Block, ParamBase, Program, Variable
from paddle.fluid.framework import _current_expected_place, _dygraph_guard, _dygraph_tracer
from paddle.fluid.framework import dygraph_only, in_dygraph_mode
from paddle.fluid.wrapped_decorator import wrap_decorator

__all__ = [
    'TracedLayer', 'declarative', 'dygraph_to_static_func', 'set_code_level',
    'set_verbosity', 'save', 'load', 'SaveLoadConfig'
]


def create_program_from_desc(program_desc):
    program = Program()
    program.desc = program_desc
    program.blocks = [Block(program, 0)]
    program._sync_with_cpp()
    return program


def _extract_vars(inputs, result_list):
    if isinstance(inputs, Variable):
        result_list.append(inputs)
    elif isinstance(inputs, (list, tuple)):
        for var in inputs:
            _extract_vars(var, result_list)
    else:
        raise TypeError(
            "The type of 'each element of inputs' in fluid.dygraph.jit.TracedLayer.trace must be fluid.Variable, but received {}.".
            format(type(inputs)))


def extract_vars(inputs):
    result_list = []
    _extract_vars(inputs, result_list)
    return result_list


def _dygraph_to_static_func_(dygraph_func):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @dygraph_to_static_func only converts imperative dygraph APIs into
    declarative net-building APIs, which means it doesn't return immediate
    digital result as imperative mode. Users should handle Program and Executor
    by themselves.

    Note:
    This decorator is NOT our recommended way to transform imperative function
    to declarative function. We will remove this decorator after we finalize
    cleaning up code.

    Args:
        dygraph_func (callable): callable imperative function.

    Returns:
        Callable: converting imperative dygraph APIs into declarative
        net-building APIs.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          from paddle.fluid.dygraph.jit import dygraph_to_static_func

          @dygraph_to_static_func
          def func(x):
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1

               return x_v

          x = fluid.layers.fill_constant(shape=[3, 3], value=0, dtype='float64')

          x_v = func(x)
          exe = fluid.Executor(fluid.CPUPlace())
          out = exe.run(fetch_list=[x_v])
          print(out[0])
          # [[1. 1. 1.]
          #  [1. 1. 1.]
          #  [1. 1. 1.]]

    """

    # TODO: remove this decorator after we finalize training API
    def __impl__(*args, **kwargs):
        program_translator = ProgramTranslator()
        if in_dygraph_mode() or not program_translator.enable_to_static:
            logging_utils.warn(
                "The decorator 'dygraph_to_static_func' doesn't work in "
                "dygraph mode or set ProgramTranslator.enable to False. "
                "We will just return dygraph output.")
            return dygraph_func(*args, **kwargs)
        static_func = program_translator.get_func(dygraph_func)
        return static_func(*args, **kwargs)

    return __impl__


dygraph_to_static_func = wrap_decorator(_dygraph_to_static_func_)


def copy_decorator_attrs(original_func, decorated_obj):
    """
    Copies some necessary attributes from original function into decorated function.

    Args:
        original_func(callable): the original decorated function.
        decorated_obj(StaticFunction): the target decorated StaticFunction object.
    """
    decorator_name = "declarative"

    decorated_obj.__name__ = original_func.__name__
    decorated_obj._decorator_name = decorator_name
    decorated_obj.__wrapped__ = original_func
    decorated_obj.__doc__ = original_func.__doc__
    if hasattr(original_func, "__module__"):
        decorated_obj.__module__ = original_func.__module__

    return decorated_obj


def declarative(function=None, input_spec=None):
    """
    Converts imperative dygraph APIs into declarative function APIs. Decorator
    @declarative handles the Program and Executor of static mode and returns
    the result as dygraph Tensor(s). Users could use the returned dygraph
    Tensor(s) to do imperative training, inference, or other operations. If the
    decorated function calls other imperative function, the called one will be
    converted into declarative function as well.

    Args:
        function (callable): callable imperative function.
        input_spec(list[InputSpec]): list of InputSpec to specific the shape/dtype/name
            information of each input Tensor.

    Returns:
        Tensor(s): containing the numerical result.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy as np
          from paddle.fluid.dygraph.jit import declarative

          fluid.enable_dygraph()

          @declarative
          def func(x):
              x = fluid.dygraph.to_variable(x)
              if fluid.layers.mean(x) < 0:
                  x_v = x - 1
              else:
                  x_v = x + 1
              return x_v

          x = np.ones([1, 2])
          x_v = func(x)
          print(x_v.numpy()) # [[2. 2.]]

    """

    def decorated(python_func):
        """
        Decorates a python function into a StaticFunction object.
        """
        # Step 1. unwrap the function if it is already decorated.
        _, python_func = unwrap_decorators(python_func)

        # Step 2. copy some attributes from original python function.
        static_layer = copy_decorator_attrs(
            original_func=python_func,
            decorated_obj=StaticFunction(
                function=python_func, input_spec=input_spec))

        return static_layer

    # for usage: `declarative(foo, ...)`
    if function is not None:
        if isinstance(function, Layer):
            if isinstance(function.forward, StaticFunction):
                class_name = function.__class__.__name__
                logging_utils.warn(
                    "`{}.forward` has already been decorated somewhere. It will be redecorated to replace previous one.".
                    format(class_name))
            function.forward = decorated(function.forward)
            return function
        else:
            return decorated(function)

    # for usage: `@declarative`
    return decorated


class SaveLoadConfig(object):
    """
    The additional configuration options may be used in function 
    ``paddle.jit.save/load`` and ``paddle.load`` .
    
    Examples:
        1. Using ``SaveLoadConfig`` when saving model

        .. code-block:: python

            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt

            class SimpleNet(nn.Layer):
                def __init__(self, in_size, out_size):
                    super(SimpleNet, self).__init__()
                    self._linear = nn.Linear(in_size, out_size)

                @paddle.jit.to_static
                def forward(self, x):
                    y = self._linear(x)
                    z = self._linear(y)
                    return z

            # enable dygraph mode
            paddle.disable_static() 

            # train model
            net = SimpleNet(8, 8)
            adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
            x = paddle.randn([4, 8], 'float32')
            for i in range(10):
                out = net(x)
                loss = paddle.tensor.mean(out)
                loss.backward()
                adam.step()
                adam.clear_grad()

            # use SaveLoadconfig when saving model
            model_path = "simplenet.example.model"
            config = paddle.SaveLoadConfig()
            config.model_filename = "__simplenet__"
            paddle.jit.save(
                layer=net,
                model_path=model_path,
                config=config)

        2. Using ``SaveLoadConfig`` when loading model

        .. code-block:: python

            import paddle

            # enable dygraph mode
            paddle.disable_static() 

            # use SaveLoadconfig when loading model
            model_path = "simplenet.example.model"
            config = paddle.SaveLoadConfig()
            config.model_filename = "__simplenet__"
            infer_net = paddle.jit.load(model_path, config=config)
            # inference
            x = paddle.randn([4, 8], 'float32')
            pred = infer_net(x)
    """

    def __init__(self):
        self._output_spec = None
        self._model_filename = None
        self._params_filename = None
        self._separate_params = False
        # used for `paddle.load`
        self._keep_name_table = False

        # NOTE: Users rarely use following configs, so these configs are not open to users,
        # reducing user learning costs, but we retain the configuration capabilities

        # If True, programs are modified to only support direct inference deployment. 
        # Otherwise,more information will be stored for flexible optimization and re-training. 
        # Currently, only True is supported
        self._export_for_deployment = True

        # If True, It will save inference program only, and do not save params of Program
        self._program_only = False

    @property
    def output_spec(self):
        """
        Selects the output targets of the saved model ( ``paddle.jit.TranslatedLayer`` ).
        By default, all return variables of original Layer's forward function
        are kept as the output of the saved TranslatedLayer.

        The ``output_spec`` type should be list[Variable]. If the provided ``output_spec``
        list is not all output variables, the saved model will be pruned according to the
        given ``output_spec`` list.

        .. note::
            The ``output_spec`` is only used when saving model.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.nn as nn
                import paddle.optimizer as opt

                class SimpleNet(nn.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = nn.Linear(in_size, out_size)

                    @paddle.jit.to_static
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        loss = paddle.tensor.mean(z)
                        return z, loss

                # enable dygraph mode
                paddle.disable_static() 

                # train model
                net = SimpleNet(8, 8)
                adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
                x = paddle.randn([4, 8], 'float32')
                for i in range(10):
                    out, loss = net(x)
                    loss.backward()
                    adam.step()
                    adam.clear_grad()

                # use SaveLoadconfig.output_spec
                model_path = "simplenet.example.model.output_spec"
                config = paddle.SaveLoadConfig()
                config.output_spec = [out]
                paddle.jit.save(
                    layer=net,
                    model_path=model_path,
                    config=config)

                infer_net = paddle.jit.load(model_path)
                x = paddle.randn([4, 8], 'float32')
                pred = infer_net(x)
        """
        return self._output_spec

    @output_spec.setter
    def output_spec(self, spec):
        if not isinstance(spec, list):
            raise TypeError(
                "The SaveLoadConfig.output_spec should be 'list', but received input type is %s."
                % type(input))
            for var in spec:
                if not isinstance(var, core.VarBase):
                    raise TypeError(
                        "The element in SaveLoadConfig.output_spec list should be 'Variable', but received element's type is %s."
                        % type(var))
        self._output_spec = spec

    @property
    def model_filename(self):
        """
        The name of file to save the translated program of target Layer.
        Default filename is :code:`__model__` .

        Examples:
            .. code-block:: python

                import paddle
                import paddle.nn as nn
                import paddle.optimizer as opt

                class SimpleNet(nn.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = nn.Linear(in_size, out_size)

                    @paddle.jit.to_static
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        return z

                # enable dygraph mode
                paddle.disable_static() 

                # train model
                net = SimpleNet(8, 8)
                adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
                x = paddle.randn([4, 8], 'float32')
                for i in range(10):
                    out = net(x)
                    loss = paddle.tensor.mean(out)
                    loss.backward()
                    adam.step()
                    adam.clear_grad()

                # saving with configs.model_filename
                model_path = "simplenet.example.model.model_filename"
                config = paddle.SaveLoadConfig()
                config.model_filename = "__simplenet__"
                paddle.jit.save(
                    layer=net,
                    model_path=model_path,
                    config=config)

                # loading with configs.model_filename
                infer_net = paddle.jit.load(model_path, config=config)
                x = paddle.randn([4, 8], 'float32')
                pred = infer_net(x)
        """
        return self._model_filename

    @model_filename.setter
    def model_filename(self, filename):
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The SaveLoadConfig.model_filename should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError(
                "The SaveLoadConfig.model_filename is empty string.")
        self._model_filename = filename

    @property
    def params_filename(self):
        """
        The name of file to save all persistable variables in target Layer. 
        Default file name is :code:`__variables__` .
        
        Examples:
            .. code-block:: python

                import paddle
                import paddle.nn as nn
                import paddle.optimizer as opt

                class SimpleNet(nn.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = nn.Linear(in_size, out_size)

                    @paddle.jit.to_static
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        return z

                # enable dygraph mode
                paddle.disable_static() 

                # train model
                net = SimpleNet(8, 8)
                adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
                x = paddle.randn([4, 8], 'float32')
                for i in range(10):
                    out = net(x)
                    loss = paddle.tensor.mean(out)
                    loss.backward()
                    adam.step()
                    adam.clear_grad()

                model_path = "simplenet.example.model.params_filename"
                config = paddle.SaveLoadConfig()
                config.params_filename = "__params__"

                # saving with configs.params_filename
                paddle.jit.save(
                    layer=net,
                    model_path=model_path,
                    config=config)

                # loading with configs.params_filename
                infer_net = paddle.jit.load(model_path, config=config)
                x = paddle.randn([4, 8], 'float32')
                pred = infer_net(x)
        """
        return self._params_filename

    @params_filename.setter
    def params_filename(self, filename):
        if not isinstance(filename, six.string_types):
            raise TypeError(
                "The SaveLoadConfig.params_filename should be str, but received input's type is %s."
                % type(filename))
        if len(filename) == 0:
            raise ValueError(
                "The SaveLoadConfig.params_filename is empty string.")
        self._params_filename = filename

    # NOTE: [why not use params_filename=None control params saved separately]
    # The new save interface does not recommend parameters to be saved separately. 
    # Here, the concept should be separated as clearly as possible. 
    # Setting params_filename=None only means that the saved file name is set 
    # and without any other meaning. New separate_params control for file saved
    # separately can makes the concept clearer.
    @property
    def separate_params(self):
        """
        Configure whether to save the Layer parameters as separete files.
        (In order to be compatible with the behavior of ``paddle.static.save_inference_model`` )

        If True, each parameter will be saved to a file separately, the file name is the parameter name,
        and the SaveLoadConfig.params_filename configuration will not take effect. Default False.

        .. note::
            Only used for ``paddle.jit.save`` .

        Examples:
            .. code-block:: python

                import paddle
                import paddle.nn as nn
                import paddle.optimizer as opt

                class SimpleNet(nn.Layer):
                    def __init__(self, in_size, out_size):
                        super(SimpleNet, self).__init__()
                        self._linear = nn.Linear(in_size, out_size)

                    @paddle.jit.to_static
                    def forward(self, x):
                        y = self._linear(x)
                        z = self._linear(y)
                        return z

                # enable dygraph mode
                paddle.disable_static() 

                # train model
                net = SimpleNet(8, 8)
                adam = opt.Adam(learning_rate=0.1, parameters=net.parameters())
                x = paddle.randn([4, 8], 'float32')
                for i in range(10):
                    out = net(x)
                    loss = paddle.tensor.mean(out)
                    loss.backward()
                    adam.step()
                    adam.clear_grad()

                model_path = "simplenet.example.model.separate_params"
                config = paddle.SaveLoadConfig()
                config.separate_params = True

                # saving with configs.separate_params
                paddle.jit.save(
                    layer=net,
                    model_path=model_path,
                    config=config)
                # [result] the saved model directory contains:
                # linear_0.b_0  linear_0.w_0  __model__  __variables.info__

                # loading with configs.params_filename
                infer_net = paddle.jit.load(model_path, config=config)
                x = paddle.randn([4, 8], 'float32')
                pred = infer_net(x)
        """
        return self._separate_params

    @separate_params.setter
    def separate_params(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "The SaveLoadConfig.separate_params should be bool value, but received input's type is %s."
                % type(value))
        self._separate_params = value

    @property
    def keep_name_table(self):
        """
        Configures whether keep ``structured_name -> parameter_name`` dict in loaded state dict.
        This dict is the debugging information saved when call ``paddle.save`` . 
        It is generally only used for debugging and does not affect the actual training or inference. 
        By default, it will not be retained in ``paddle.load`` result. Default: False.
        
        .. note::
            Only used for ``paddle.load`` .

        Examples:
            .. code-block:: python

                import paddle
            
                paddle.disable_static()

                linear = paddle.nn.Linear(5, 1)

                state_dict = linear.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")

                config = paddle.SaveLoadConfig()
                config.keep_name_table = True
                para_state_dict = paddle.load("paddle_dy.pdparams", config)

                print(para_state_dict)
                # the name_table is 'StructuredToParameterName@@'
                # {'bias': array([0.], dtype=float32), 
                #  'StructuredToParameterName@@': 
                #     {'bias': u'linear_0.b_0', 'weight': u'linear_0.w_0'}, 
                #  'weight': array([[ 0.04230034],
                #     [-0.1222527 ],
                #     [ 0.7392676 ],
                #     [-0.8136974 ],
                #     [ 0.01211023]], dtype=float32)}
        """
        return self._keep_name_table

    @keep_name_table.setter
    def keep_name_table(self, value):
        if not isinstance(value, bool):
            raise TypeError(
                "The SaveLoadConfig.keep_name_table should be bool value, but received input's type is %s."
                % type(value))
        self._keep_name_table = value


# NOTE(chenweihang): change jit.save/load argument `configs` to `config`
def deprecate_save_load_configs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'configs' in kwargs:
            kwargs['config'] = kwargs['configs']
            kwargs.pop('configs')
        return func(*args, **kwargs)

    return wrapper


def _get_input_var_names(inputs, input_spec):
    name_none_error = "The %s's name is None. " \
        "When using jit.save, please set InputSepc's name in " \
        "to_static(input_spec=[]) and jit.save(input_spec=[]) " \
        "and make sure they are consistent."
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of InputSpec or example Tensor " \
        "in input_spec is the same as the name of InputSpec in " \
        "`to_static` decorated on the Layer.forward method."
    result_list = []
    input_var_names = [var.name for var in inputs if isinstance(var, Variable)]
    if input_spec is None:
        # no prune
        result_list = input_var_names
    elif input_spec is not None and len(input_spec) == len(input_var_names):
        # no prune
        result_list = input_var_names
        # if input spec name not in input_var_names, only raise warning 
        for spec in input_spec:
            if spec.name is None:
                warnings.warn(name_none_error % spec)
            elif spec.name not in input_var_names:
                warnings.warn(name_no_exists_error % spec.name)
            else:
                # do nothing
                pass
    else:
        # prune
        for spec in input_spec:
            if spec.name is None:
                # name is None, the input_spec only can be InputSpec
                raise ValueError(name_none_error % spec)
            elif spec.name not in input_var_names:
                # the input_spec can be `InputSpec` or `VarBase`
                raise ValueError(name_no_exists_error % spec.name)
            else:
                result_list.append(spec.name)

    return result_list


def _get_output_vars(outputs, output_spec):
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of example Tensor " \
        "in configs.output_spec is the output tensor of " \
        "Layer.forward method."
    result_list = []
    output_vars_dict = OrderedDict()
    for var in outputs:
        if isinstance(var, Variable):
            output_vars_dict[var.name] = var
    if output_spec is None:
        result_list = list(output_vars_dict.values())
    elif output_spec is not None and len(output_spec) == len(output_vars_dict):
        result_list = list(output_vars_dict.values())
        for var in output_spec:
            if var.name not in output_vars_dict:
                warnings.warn(name_no_exists_error % var.name)
    else:
        for var in output_spec:
            if var.name not in output_vars_dict:
                raise ValueError(name_no_exists_error % var.name)
            else:
                result_list.append(output_vars_dict[var.name])
    return result_list


def _infer_input_check(layer, input_spec):
    prog_translator = ProgramTranslator()
    if not prog_translator.enable_to_static:
        raise RuntimeError(
            "The paddle.jit.save doesn't work when setting ProgramTranslator.enable to False."
        )
    if not isinstance(layer, Layer):
        raise TypeError(
            "The input layer of paddle.jit.save should be 'Layer', but received layer type is %s."
            % type(layer))

    # avoid change user given input_spec
    inner_input_spec = None
    if input_spec is not None:
        if not isinstance(input_spec, list):
            raise TypeError(
                "The input input_spec should be 'list', but received input_spec's type is %s."
                % type(input_spec))
        inner_input_spec = []
        for var in input_spec:
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(var, (core.VarBase, Variable)):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var))
            else:
                raise TypeError(
                    "The element in input_spec list should be 'Variable' or `paddle.static.InputSpec`, but received element's type is %s."
                    % type(var))
    return inner_input_spec


def _get_concrete_program_from_layer(layer, inner_input_spec):
    # TODO(chenweihang): add support for other method, not only forward
    if isinstance(layer.forward, StaticFunction):
        concrete_program = layer.forward.concrete_program
    else:
        # transform in jit.save, if input_spec is incomplete, declarative will throw error
        static_forward = declarative(layer.forward, input_spec=inner_input_spec)
        concrete_program = static_forward.concrete_program
        # the input_spec has been used in declarative, which is equal to 
        # @declarative with input_spec and jit.save without input_spec,
        # avoid needless warning
        inner_input_spec = None
    return concrete_program


def _build_input_and_output(concrete_program, inner_input_spec, config):
    # NOTE(chenweihang): [ Get input variables name ]
    # There are two cases, whether to prune the inputs or not
    # - not prune inputs (recommend):
    #   - the len(input_spec) == len((concrete_program.inputs) - 1
    #   - here can use concrete_program.inputs directly
    # - prune inputs:
    #   - the input_spec length < len((concrete_program.inputs) - 1
    #   - the input_spec's name should be in concrete_program.inputs
    input_var_names = _get_input_var_names(concrete_program.inputs,
                                           inner_input_spec)

    # NOTE(chenweihang): [ Get output variables ]
    # the rule is like [ Get input variables name ]. For output var, 
    # we only support VarBase spec, and actually, we only need the 
    # var name of output, and we don't recommended to use output_spec
    output_vars = _get_output_vars(concrete_program.outputs, config.output_spec)

    return input_var_names, output_vars


# NOTE: This function is not exposed to users, only used for paddle2onnx now
@switch_to_static_graph
def get_inference_program(layer, input_spec=None, config=None):
    # 1. input check
    inner_input_spec = _infer_input_check(layer, input_spec)

    if config is None:
        config = SaveLoadConfig()

    # 2. get program from Layer
    concrete_program = _get_concrete_program_from_layer(layer, inner_input_spec)

    # 3. build input & output of save_infernece_model
    input_var_names, output_vars = _build_input_and_output(
        concrete_program, inner_input_spec, config)

    # 4. only get inference program
    inference_program = paddle.fluid.io.get_inference_program(
        input_var_names, output_vars, concrete_program.main_program.clone())

    return inference_program


@deprecate_save_load_configs
@switch_to_static_graph
def save(layer, model_path, input_spec=None, config=None):
    """
    Saves input declarative Layer as :ref:`api_imperative_TranslatedLayer` 
    format model, which can be used for inference or fine-tuning after loading.

    It will save the translated program and all related persistable 
    variables of input declarative Layer to given ``model_path``.
    
    The default saved translated program file name is ``__model__``,
    and the default saved persistable variables file name is ``__variables__``,
    and it also saved some additional variable description information to file 
    ``__variables.info__``, these additional information is used in fine-tuning.

    The saved model can be loaded by follow APIs:
      - :ref:`api_imperative_jit_load`
      - :ref:`api_fluid_io_load_inference_model` (need pass ``params_filename='__variables__'``)
      - Other C++ inference APIs

    Args:
        layer (Layer): the Layer to be saved. The Layer should be decorated by `@declarative`.
        model_path (str): the directory to save the model.
        input_spec (list[Variable], optional): Describes the input of the saved model. 
            It is the example inputs that will be passed to saved TranslatedLayer's forward
            function. If None, all input variables of the original Layer's forward function
            would be the inputs of the saved model. Default None.
        config (SaveLoadConfig, optional): :ref:`api_imperative_jit_saveLoadConfig` object
            that specifies additional configuration options. Default None.
    Returns:
        None

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt

            BATCH_SIZE = 16
            BATCH_NUM = 4
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(paddle.io.Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

                @paddle.jit.to_static
                def forward(self, x):
                    return self._linear(x)

            def train(layer, loader, loss_fn, opt):
                for epoch_id in range(EPOCH_NUM):
                    for batch_id, (image, label) in enumerate(loader()):
                        out = layer(image)
                        loss = loss_fn(out, label)
                        loss.backward()
                        opt.step()
                        opt.clear_grad()
                        print("Epoch {} batch {}: loss = {}".format(
                            epoch_id, batch_id, np.mean(loss.numpy())))

            # enable dygraph mode
            place = paddle.CPUPlace()
            paddle.disable_static(place) 

            # 1. train & save model.

            # create network
            layer = LinearNet()
            loss_fn = nn.CrossEntropyLoss()
            adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(dataset,
                places=place,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

            # train
            train(layer, loader, loss_fn, adam)

            # save
            model_path = "linear.example.model"
            paddle.jit.save(layer, model_path)
    """
    # 1. input check
    inner_input_spec = _infer_input_check(layer, input_spec)

    if config is None:
        config = SaveLoadConfig()

    # 2. get program from Layer
    concrete_program = _get_concrete_program_from_layer(layer, inner_input_spec)

    # 3. build input & output of save_infernece_model
    input_var_names, output_vars = _build_input_and_output(
        concrete_program, inner_input_spec, config)

    # NOTE(chenweihang): we maintain the mapping of variable name to
    # structured name, the buffer variable (non-persistable)
    # saved to inference program may not need by dygraph Layer, 
    # we only record the state_dict variable's structured name
    state_names_dict = dict()
    for structured_name, var in six.iteritems(layer.state_dict()):
        state_names_dict[var.name] = structured_name

    # 4. share parameters from Layer to scope & record var info
    scope = core.Scope()
    extra_var_info = dict()
    for param_or_buffer in concrete_program.parameters:
        # share to scope
        param_or_buffer_tensor = scope.var(param_or_buffer.name).get_tensor()
        src_tensor = param_or_buffer.value().get_tensor()
        param_or_buffer_tensor._share_data_with(src_tensor)
        # record var info
        extra_info_dict = dict()
        if param_or_buffer.name in state_names_dict:
            extra_info_dict['structured_name'] = state_names_dict[
                param_or_buffer.name]
        extra_info_dict['stop_gradient'] = param_or_buffer.stop_gradient
        if isinstance(param_or_buffer, ParamBase):
            extra_info_dict['trainable'] = param_or_buffer.trainable
        extra_var_info[param_or_buffer.name] = extra_info_dict

    # 5. save inference model
    from paddle.fluid.io import save_inference_model

    # VARIABLE_FILENAME keep nameing style consistent with '__model__'
    if config.params_filename is None:
        config.params_filename = VARIABLE_FILENAME

    with scope_guard(scope):
        save_inference_model(
            dirname=model_path,
            feeded_var_names=input_var_names,
            target_vars=output_vars,
            executor=Executor(_current_expected_place()),
            main_program=concrete_program.main_program.clone(),
            model_filename=config.model_filename,
            params_filename=None
            if config.separate_params else config.params_filename,
            export_for_deployment=config._export_for_deployment,
            program_only=config._program_only)

        # NOTE(chenweihang): [ Save extra variable info ]
        # save_inference_model will lose some important variable information, including:
        #   - Variable name and correspondence (when saved variables as one file)
        #   - Variable.stop_gradient information
        #   - Which persistent variable are parameter and which are not
        #   - Parameter.trainable information
        #
        # The lost information cannot be recovered when it is loaded again, 
        # so if we want to perform fine-tune after loading, we may need to 
        # configure redundant information to proceed.
        #
        # Due to compatibility issues, we cannot change the original storage structure, 
        # but we can save these information in `jit.save` without changing the original 
        # storage to improve user experience. So we save extra information into
        # file `__variables.info__`
        extra_var_info_path = os.path.join(model_path, EXTRA_VAR_INFO_FILENAME)
        with open(extra_var_info_path, 'wb') as f:
            pickle.dump(extra_var_info, f, protocol=2)


@deprecate_save_load_configs
@dygraph_only
def load(model_path, config=None):
    """
    :api_attr: imperative

    Load model saved by :ref:`api_imperative_jit_save` or :ref:`api_fluid_io_save_inference_model`
    as :ref:`api_imperative_TranslatedLayer`, then performing inference or fine-tune training.

    .. note::
        For some historical reasons, if you load model saved by :ref:`api_fluid_io_save_inference_model`,
        there will be the following limitations when using it in fine-tuning:
        1. Imperative mode do not support LoDTensor. All original model's feed targets or parametars that depend on LoD are temporarily unavailable.
        2. All saved model's feed targets need to be passed into TranslatedLayer's forward function.
        3. The variable's ``stop_gradient`` information is lost and can not be recovered.
        4. The parameter's ``trainable`` information is lost and can not be recovered.

    Args:
        model_path (str): The directory path where the model is saved.
        config (SaveLoadConfig, optional): :ref:`api_imperative_jit_saveLoadConfig` object that specifies 
            additional configuration options. Default None.

    Returns:
        TranslatedLayer: A Layer object can run saved translated model.

    Examples:
        1. Load model saved by :ref:`api_imperative_jit_save` then performing inference and fine-tune training.

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt

            BATCH_SIZE = 16
            BATCH_NUM = 4
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(paddle.io.Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            class LinearNet(nn.Layer):
                def __init__(self):
                    super(LinearNet, self).__init__()
                    self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

                @paddle.jit.to_static
                def forward(self, x):
                    return self._linear(x)

            def train(layer, loader, loss_fn, opt):
                for epoch_id in range(EPOCH_NUM):
                    for batch_id, (image, label) in enumerate(loader()):
                        out = layer(image)
                        loss = loss_fn(out, label)
                        loss.backward()
                        opt.step()
                        opt.clear_grad()
                        print("Epoch {} batch {}: loss = {}".format(
                            epoch_id, batch_id, np.mean(loss.numpy())))

            # enable dygraph mode
            place = paddle.CPUPlace()
            paddle.disable_static(place) 

            # 1. train & save model.

            # create network
            layer = LinearNet()
            loss_fn = nn.CrossEntropyLoss()
            adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(dataset,
                places=place,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

            # train
            train(layer, loader, loss_fn, adam)

            # save
            model_path = "linear.example.model"
            paddle.jit.save(layer, model_path)

            # 2. load model

            # load
            loaded_layer = paddle.jit.load(model_path)

            # inference
            loaded_layer.eval()
            x = paddle.randn([1, IMAGE_SIZE], 'float32')
            pred = loaded_layer(x)

            # fine-tune
            loaded_layer.train()
            adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
            train(loaded_layer, loader, loss_fn, adam)


        2. Load model saved by :ref:`api_fluid_io_save_inference_model` then performing and fine-tune training.

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.fluid as fluid
            import paddle.nn as nn
            import paddle.optimizer as opt

            BATCH_SIZE = 16
            BATCH_NUM = 4
            EPOCH_NUM = 4

            IMAGE_SIZE = 784
            CLASS_NUM = 10

            # define a random dataset
            class RandomDataset(paddle.io.Dataset):
                def __init__(self, num_samples):
                    self.num_samples = num_samples

                def __getitem__(self, idx):
                    image = np.random.random([IMAGE_SIZE]).astype('float32')
                    label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            image = fluid.data(name='image', shape=[None, 784], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            pred = fluid.layers.fc(input=image, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=pred, label=label)
            avg_loss = fluid.layers.mean(loss)

            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(dataset,
                feed_list=[image, label],
                places=place,
                batch_size=BATCH_SIZE, 
                shuffle=True,
                drop_last=True,
                num_workers=2)

            # 1. train and save inference model
            for data in loader():
                exe.run(
                    fluid.default_main_program(),
                    feed=data, 
                    fetch_list=[avg_loss])

            model_path = "fc.example.model"
            fluid.io.save_inference_model(
                model_path, ["image"], [pred], exe)

            # 2. load model

            # enable dygraph mode
            paddle.disable_static(place)

            # load
            fc = paddle.jit.load(model_path)

            # inference
            fc.eval()
            x = paddle.randn([1, IMAGE_SIZE], 'float32')
            pred = fc(x)

            # fine-tune
            fc.train()
            loss_fn = nn.CrossEntropyLoss()
            adam = opt.Adam(learning_rate=0.001, parameters=fc.parameters())
            loader = paddle.io.DataLoader(dataset,
                places=place,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)
            for epoch_id in range(EPOCH_NUM):
                for batch_id, (image, label) in enumerate(loader()):
                    out = fc(image)
                    loss = loss_fn(out, label)
                    loss.backward()
                    adam.step()
                    adam.clear_grad()
                    print("Epoch {} batch {}: loss = {}".format(
                        epoch_id, batch_id, np.mean(loss.numpy())))
    """
    return TranslatedLayer._construct(model_path, config)


@dygraph_only
def _trace(layer,
           inputs,
           feed_prefix='feed_',
           fetch_prefix='fetch_',
           tmp_prefix='t_'):
    assert isinstance(layer, Layer)

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    tracer = _dygraph_tracer()._get_program_desc_tracer()

    var_list = extract_vars(inputs)

    with program_desc_tracing_guard(True):
        original_outputs = layer(*inputs)
        if not isinstance(original_outputs, (list, tuple)):
            outputs = [original_outputs]
        else:
            outputs = original_outputs
        out_vars = [var for var in outputs]

        program_desc, feed_names, fetch_names, parameters = tracer.create_program_desc(
            var_list, feed_prefix, out_vars, fetch_prefix, tmp_prefix)
        tracer.reset()

    with _dygraph_guard(None):
        program = create_program_from_desc(program_desc)

    return original_outputs, program, feed_names, fetch_names, parameters


class TracedLayer(object):
    """
    :api_attr: imperative
    
    TracedLayer is used to convert a forward dygraph model to a static
    graph model. This is mainly used to save the dygraph model for online
    inference using C++. Besides, users can also do inference in Python
    using the converted static graph model, which usually has better
    performance than the original dygraph model.

    TracedLayer would run the static graph model using :code:`Executor`
    and :code:`CompiledProgram` . The static graph model would share
    parameters with the dygraph model.

    All TracedLayer objects should not be created by constructor and should
    be created by static method :code:`TracedLayer.trace(layer, inputs)` .

    The TracedLayer can only be used to convert the data-independent dygraph
    model into the static graph model, which means the dygraph model should
    be independent with the tensor data and shape.
    """

    def __init__(self, program, parameters, feed_names, fetch_names):
        self._program = program
        self._feed_names = feed_names
        self._fetch_names = fetch_names
        self._params = parameters

        self._place = _current_expected_place()

        self._scope = core.Scope()
        for p in parameters:
            src_tensor = p.value().get_tensor()
            dst_tensor = self._scope.var(p.name).get_tensor()
            dst_tensor._share_data_with(src_tensor)

        self._exe = Executor(self._place)
        self._compiled_program = None
        self._build_strategy = None
        self._exec_strategy = None

    @property
    def program(self):
        return self._program

    def _switch(self, is_test=True):
        for block_id in range(self._program.num_blocks):
            block = self._program.block(block_id)
            for op in block.ops:
                if op.has_attr("is_test"):
                    op._set_attr("is_test", is_test)

    @staticmethod
    @dygraph_only
    def trace(layer, inputs):
        """
        This method is the only allowed method to create TracedLayer object.
        It would call the :code:`layer(*inputs)` method to run the dygraph
        model and convert it into a static graph model.

        Args:
            layer (dygraph.Layer): the layer object to be traced.
            inputs (list(Tensor)|tuple(Tensor)|Tensor): the input tensors of
                the layer object.

        Returns:
            tuple: A tuple of 2 items, whose the first item is the output of
                :code:`layer(*inputs)` , and the second item is the created
                TracedLayer object.

        Examples:
            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                with fluid.dygraph.guard():
                    layer = ExampleLayer()
                    in_np = np.random.random([2, 3]).astype('float32')
                    in_var = to_variable(in_np)
                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])

                    # run the static graph model using Executor inside
                    out_static_graph = static_layer([in_var])

                    print(len(out_static_graph)) # 1
                    print(out_static_graph[0].shape) # (2, 10)

                    # save the static graph model for inference
                    static_layer.save_inference_model(dirname='./saved_infer_model')
        """
        assert isinstance(
            layer, Layer
        ), "The type of 'layer' in fluid.dygraph.jit.TracedLayer.trace must be fluid.dygraph.Layer, but received {}.".format(
            type(layer))
        outs, prog, feed, fetch, parameters = _trace(layer, inputs)
        traced = TracedLayer(prog, parameters, feed, fetch)
        return outs, traced

    def set_strategy(self, build_strategy=None, exec_strategy=None):
        """
        Set the strategies when running static graph model.

        Args:
            build_strategy (BuildStrategy, optional): build strategy of
                :code:`CompiledProgram` inside TracedLayer. Default None.
            exec_strategy (ExecutionStrategy, optional): execution strategy of
                :code:`CompiledProgram` inside TracedLayer. Default None.

        Returns:
            None

        Examples:
            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                with fluid.dygraph.guard():
                    layer = ExampleLayer()
                    in_np = np.random.random([2, 3]).astype('float32')
                    in_var = to_variable(in_np)

                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])

                    build_strategy = fluid.BuildStrategy()
                    build_strategy.enable_inplace = True

                    exec_strategy = fluid.ExecutionStrategy()
                    exec_strategy.num_threads = 2

                    static_layer.set_strategy(build_strategy=build_strategy, exec_strategy=exec_strategy)
                    out_static_graph = static_layer([in_var])
        """
        assert self._compiled_program is None, "Cannot set strategy after run"
        assert isinstance(
            build_strategy, (type(None), BuildStrategy)
        ), "The type of 'build_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.BuildStrategy, but received {}.".format(
            type(build_strategy))
        assert isinstance(
            exec_strategy, (type(None), ExecutionStrategy)
        ), "The type of 'exec_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.ExecutionStrategy, but received {}.".format(
            type(exec_strategy))
        self._build_strategy = build_strategy
        self._exec_strategy = exec_strategy

    @switch_to_static_graph
    def _compile(self):
        self._compiled_program = CompiledProgram(
            self._program).with_data_parallel(
                build_strategy=self._build_strategy,
                exec_strategy=self._exec_strategy,
                places=self._place)

    def _build_feed(self, inputs):
        assert isinstance(inputs, (list, tuple)), \
            "Inputs should be a list or tuple of variables"
        assert len(inputs) == len(self._feed_names)
        feed_dict = {}
        if in_dygraph_mode():
            for x, name in zip(inputs, self._feed_names):
                feed_dict[name] = x.value().get_tensor()
        else:
            for x, name in zip(inputs, self._feed_names):
                feed_dict[name] = x

        return feed_dict

    @switch_to_static_graph
    def _run(self, feed):
        return self._exe.run(self._compiled_program,
                             feed=feed,
                             fetch_list=self._fetch_names)

    def __call__(self, inputs):
        with scope_guard(self._scope):
            if self._compiled_program is None:
                self._compile()

            return self._run(self._build_feed(inputs))

    @switch_to_static_graph
    def save_inference_model(self, dirname, feed=None, fetch=None):
        """
        Save the TracedLayer to a model for inference. The saved
        inference model can be loaded by C++ inference APIs.

        Args:
            dirname (str): the directory to save the inference model.
            feed (list[int], optional): the input variable indices of the saved
                inference model. If None, all input variables of the
                TracedLayer object would be the inputs of the saved inference
                model. Default None.
            fetch (list[int], optional): the output variable indices of the
                saved inference model. If None, all output variables of the
                TracedLayer object would be the outputs of the saved inference
                model. Default None.

        Returns:
            None

        Examples:
            .. code-block:: python:

                import paddle.fluid as fluid
                from paddle.fluid.dygraph import Linear, to_variable, TracedLayer
                import numpy as np

                class ExampleLayer(fluid.dygraph.Layer):
                    def __init__(self):
                        super(ExampleLayer, self).__init__()
                        self._fc = Linear(3, 10)

                    def forward(self, input):
                        return self._fc(input)

                save_dirname = './saved_infer_model'
                in_np = np.random.random([2, 3]).astype('float32')

                with fluid.dygraph.guard():
                    layer = ExampleLayer()
                    in_var = to_variable(in_np)
                    out_dygraph, static_layer = TracedLayer.trace(layer, inputs=[in_var])
                    static_layer.save_inference_model(save_dirname, feed=[0], fetch=[0])

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                program, feed_vars, fetch_vars = fluid.io.load_inference_model(save_dirname,
                                                    exe)

                fetch, = exe.run(program, feed={feed_vars[0]: in_np}, fetch_list=fetch_vars)
                print(fetch.shape) # (2, 10)
        """
        check_type(dirname, "dirname", str,
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        check_type(feed, "feed", (type(None), list),
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        if isinstance(feed, list):
            for f in feed:
                check_type(f, "each element of feed", int,
                           "fluid.dygraph.jit.TracedLayer.save_inference_model")
        check_type(fetch, "fetch", (type(None), list),
                   "fluid.dygraph.jit.TracedLayer.save_inference_model")
        if isinstance(fetch, list):
            for f in fetch:
                check_type(f, "each element of fetch", int,
                           "fluid.dygraph.jit.TracedLayer.save_inference_model")

        from paddle.fluid.io import save_inference_model

        def get_feed_fetch(all_vars, partial_vars):
            if partial_vars is None:
                return all_vars

            return [all_vars[idx] for idx in partial_vars]

        with scope_guard(self._scope):
            feeded_var_names = get_feed_fetch(self._feed_names, feed)
            target_var_names = get_feed_fetch(self._fetch_names, fetch)
            target_vars = []
            for name in target_var_names:
                target_var = self._program.global_block().vars.get(name, None)
                assert target_var is not None, "{} cannot be found".format(name)
                target_vars.append(target_var)

            save_inference_model(
                dirname=dirname,
                feeded_var_names=feeded_var_names,
                target_vars=target_vars,
                executor=self._exe,
                main_program=self._program.clone())
