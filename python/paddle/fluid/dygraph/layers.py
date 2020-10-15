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

import collections
import contextlib
import sys
import numpy as np
import six
import re
import copy
import weakref
import warnings

from . import parallel_helper
from .. import unique_name
from paddle.fluid import core
from .layer_object_helper import LayerObjectHelper
from .base import program_desc_tracing_guard, param_guard
from paddle.fluid import framework
from ..param_attr import ParamAttr
from paddle.fluid.executor import Executor, global_scope
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.framework import _current_expected_place as _get_device

__all__ = ['Layer']

_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z])([A-Z])')


def _convert_camel_to_snake(name):
    s1 = _first_cap_re.sub(r'\1_\2', name)
    return _all_cap_re.sub(r'\1_\2', s1).lower()


class HookRemoveHelper(object):
    """ A HookRemoveHelper that can be used to remove hook. """

    next_hook_id = 0

    def __init__(self, hooks):
        self._hooks_ref = weakref.ref(hooks)
        self._hook_id = HookRemoveHelper.next_hook_id
        HookRemoveHelper.next_hook_id += 1

    def remove(self):
        hooks = self._hooks_ref()
        if hooks is not None and self._hook_id in hooks:
            del hooks[self._hook_id]


class Layer(core.Layer):
    """
    Dynamic graph Layer based on OOD, includes the parameters of the layer, the structure of the forward graph and so on.

    Parameters:
        name_scope (str, optional): prefix name used by the layer to name parameters.
            If prefix is "my_layer", parameter name in MyLayer
            can be "my_layer_0.w_n", where "w" is the parameter
            base name and "n" is an unique suffix auto-generated.
            If None, prefix name will be snake cased class name. Default: None.
        dtype(str, optional): data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16".
                Default: "float32"
    
    Returns:
        None
    """

    def __init__(self, name_scope=None, dtype="float32"):
        self.training = True
        if name_scope is None:
            name_scope = _convert_camel_to_snake(self.__class__.__name__)
        self._full_name = unique_name.generate(name_scope)
        self._helper = LayerObjectHelper(self._full_name)
        self._built = False
        self._dtype = dtype
        self._init_in_dynamic_mode = framework.in_dygraph_mode()

        self._parameters = collections.OrderedDict()
        # Buffers the variable (not parameter) created in layer
        self._buffers = collections.OrderedDict()
        self._non_persistable_buffer_names_set = set()
        self._sub_layers = collections.OrderedDict()
        self._loaddict_holder = collections.OrderedDict()

        self._forward_pre_hooks = collections.OrderedDict()
        self._forward_post_hooks = collections.OrderedDict()

    def train(self):
        """
        Sets this Layer and all its sublayers to training mode.
        This only effects certain modules like `Dropout` and `BatchNorm`.

        Returns:
            None

        Example::
            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self._linear = paddle.nn.Linear(1, 1)
                        self._dropout = paddle.nn.Dropout(p=0.5)

                    def forward(self, input):
                        temp = self._linear(input)
                        temp = self._dropout(temp)
                        return temp

                x = paddle.randn([10, 1], 'float32')
                mylayer = MyLayer()
                mylayer.eval()  # set mylayer._dropout to eval mode
                out = mylayer(x)
                mylayer.train()  # set mylayer._dropout to train mode
                out = mylayer(x)

        """
        # global setting
        framework._dygraph_tracer().train_mode()
        # Layer-level setting
        self.training = True
        for layer in self.sublayers():
            layer.train()

    def eval(self):
        """
        Sets this Layer and all its sublayers to evaluation mode.
        This only effects certain modules like `Dropout` and `BatchNorm`.

        Returns:
            None

        Example::
            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self._linear = paddle.nn.Linear(1, 1)
                        self._dropout = paddle.nn.Dropout(p=0.5)

                    def forward(self, input):
                        temp = self._linear(input)
                        temp = self._dropout(temp)
                        return temp

                x = paddle.randn([10, 1], 'float32')
                mylayer = MyLayer()
                mylayer.eval()  # set mylayer._dropout to eval mode
                out = mylayer(x)
                print(out)

        """
        # global setting
        framework._dygraph_tracer().eval_mode()
        # Layer-level setting
        self.training = False
        for layer in self.sublayers():
            layer.eval()

    def apply(self, fn):
        """
        Applies ``fn`` recursively to every sublayer (as returned by ``.sublayers()``)
        as well as self. Typical use includes initializing the parameters of a model.

        Parameters:
            fn (function): a function to be applied to each sublayer

        Returns:
            Layer: self

        Example::
            .. code-block:: python

              import paddle
              import paddle.nn as nn

              net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

              def init_weights(layer):
                  if type(layer) == nn.Linear:
                      print('before init weight:', layer.weight.numpy())
                      new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
                      layer.weight.set_value(new_weight)
                      print('after init weight:', layer.weight.numpy())

              net.apply(init_weights)

              print(net.state_dict())
        """
        for layer in self.children():
            layer.apply(fn)

        fn(self)

        return self

    def full_name(self):
        """Full name for this layer, composed by name_scope + "/" + MyLayer.__class__.__name__

        Returns:
            str: full name of this layer.

        Example::
            .. code-block:: python

                import paddle

                class LinearNet(paddle.nn.Layer):
                    def __init__(self):
                        super(LinearNet, self).__init__(name_scope = "demo_linear_net")
                        self._linear = paddle.nn.Linear(1, 1)

                    def forward(self, x):
                        return self._linear(x)

                linear_net = LinearNet()
                print(linear_net.full_name())   # demo_linear_net_0

        """
        return self._full_name

    def register_forward_post_hook(self, hook):
        """Register a forward post-hook for Layer. The hook will be called after `forward` function has been computed.

        It should have the following form, `input` and `output` of the `hook` is `input` and `output` of the `Layer` respectively.
        User can use forward post-hook to change the output of the Layer or perform information statistics tasks on the Layer.
 
        hook(Layer, input, output) -> None or modified output

        Parameters:
            hook(function): a function registered as a forward post-hook

        Returns:
            HookRemoveHelper: a HookRemoveHelper object that can be used to remove the added hook by calling `hook_remove_helper.remove()` .

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                # the forward_post_hook change the output of the layer: output = output * 2
                def forward_post_hook(layer, input, output):
                    # user can use layer, input and output for information statistis tasks

                    # change the output
                    return output * 2

                linear = paddle.nn.Linear(13, 5)

                # register the hook
                forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)

                value1 = np.arange(26).reshape(2, 13).astype("float32")
                in1 = paddle.to_tensor(value1)

                out0 = linear(in1)

                # remove the hook
                forward_post_hook_handle.remove()

                out1 = linear(in1)

                # hook change the linear's output to output * 2, so out0 is equal to out1 * 2.
                assert (out0.numpy() == (out1.numpy()) * 2).any()
        """
        hook_remove_helper = HookRemoveHelper(self._forward_post_hooks)
        self._forward_post_hooks[hook_remove_helper._hook_id] = hook
        return hook_remove_helper

    def register_forward_pre_hook(self, hook):
        """Register a forward pre-hook for Layer. The hook will be called before `forward` function has been computed.
        
        It should have the following form, `input` of the `hook` is `input` of the `Layer`,
        hook can either return a tuple or a single modified value in the hook. We will wrap the value into a tuple if 
        a single value is returned(unless that value is already a tuple).
        User can use forward pre-hook to change the input of the Layer or perform information statistics tasks on the Layer.

        hook(Layer, input) -> None or modified input

        Parameters:
            hook(function): a function registered as a forward pre-hook

        Returns:
            HookRemoveHelper: a HookRemoveHelper object that can be used to remove the added hook by calling `hook_remove_helper.remove()` .

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                # the forward_post_hook change the input of the layer: input = input * 2
                def forward_pre_hook(layer, input):
                    # user can use layer and input for information statistis tasks

                    # change the input
                    input_return = (input[0] * 2)
                    return input_return

                linear = paddle.nn.Linear(13, 5)

                # register the hook
                forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)

                value0 = np.arange(26).reshape(2, 13).astype("float32")
                in0 = paddle.to_tensor(value0)
                out0 = linear(in0)

                # remove the hook
                forward_pre_hook_handle.remove()

                value1 = value0 * 2
                in1 = paddle.to_tensor(value1)
                out1 = linear(in1)

                # hook change the linear's input to input * 2, so out0 is equal to out1.
                assert (out0.numpy() == out1.numpy()).any()
        """
        hook_remove_helper = HookRemoveHelper(self._forward_pre_hooks)
        self._forward_pre_hooks[hook_remove_helper._hook_id] = hook
        return hook_remove_helper

    def create_parameter(self,
                         shape,
                         attr=None,
                         dtype=None,
                         is_bias=False,
                         default_initializer=None):
        """Create parameters for this layer.
        
        Parameters:
            shape(list): Shape of the parameter.
            attr(ParamAttr, optional): Parameter attribute of weight. Please refer to :ref:`api_paddle_ParamAttr`. Default: None.
            dtype(str, optional): Data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16". Default: "float32".
            is_bias(bool, optional): if this is a bias parameter. Default: False.
            default_initializer(Initializer, optional): the default initializer for this parameter.
                If set None, default initializer will be set to paddle.nn.initializer.Xavier and paddle.nn.initializer.Constant
                for non-bias and bias parameter, respectively. Default: None.

        Returns:
            :Tensor, created parameter.

        Examples:
            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self._linear = paddle.nn.Linear(1, 1)
                        w_tmp = self.create_parameter([1,1])
                        self.add_parameter("w_tmp", w_tmp)

                    def forward(self, input):
                        return self._linear(input)

                mylayer = MyLayer()
                for name, param in mylayer.named_parameters():
                    print(name, param)      # will print w_tmp,_linear.weight,_linear.bias

        """
        temp_attr = copy.deepcopy(attr)
        if isinstance(temp_attr, six.string_types) and temp_attr == "":
            temp_attr = None
        return self._helper.create_parameter(temp_attr, shape, dtype, is_bias,
                                             default_initializer)

    # TODO: Add more parameter list when we need them
    def create_variable(self, name=None, persistable=None, dtype=None):
        """Create Variable for this layer.

        Parameters:
            name(str, optional): name of the variable. Please refer to :ref:`api_guide_Name` . Default: None
            persistable(bool, optional): if set this variable persistable. Default: False
            dtype(str, optional): data type of this parameter.
                If set str, it can be "bool",  "float16", "float32", "float64",
                "int8", "int16", "int32", "int64", "uint8" or "uint16".
                If set None, it will be "float32". Default: None

        Returns:
            Tensor, created Variable.

        Examples:
            .. code-block:: python

                import paddle

                class MyLinear(paddle.nn.Layer):
                    def __init__(self,
                                in_features,
                                out_features):
                        super(MyLinear, self).__init__()
                        self.linear = paddle.nn.Linear( 10, 10)
                            
                        self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)
                    
                    def forward(self, input):
                        out = self.linear(input)
                        paddle.assign( out, self.back_var)
                        
                        return out

        """
        if name is not None:
            var_name = ".".join([self._full_name, name])
        else:
            var_name = unique_name.generate(".".join(
                [self._full_name, "_generated_var"]))

        return self._helper.main_program.current_block().create_var(
            name=var_name,
            persistable=persistable,
            dtype=dtype,
            type=core.VarDesc.VarType.LOD_TENSOR)

    def parameters(self, include_sublayers=True):
        """Returns a list of all Parameters from current layer and its sub-layers.

        Parameters:
            include_sublayers(bool, optional): Whether include the parameters of sublayers. If True, also include the parameters from sublayers. Default: True

        Returns:
            list of Tensor : a list of Parameters.

        Examples:
            .. code-block:: python

            import paddle

            linear = paddle.nn.Linear(1,1)
            print(linear.parameters())  # print linear_0.w_0 and linear_0.b_0

        """
        ret = [
            param
            for _, param in self.named_parameters(
                include_sublayers=include_sublayers)
        ]
        return ret

    def children(self):
        """Returns an iterator over immediate children layers.

        Yields:
            Layer: a child layer

        Examples:
            .. code-block:: python

                import paddle

                linear1 = paddle.nn.Linear(10, 3)
                linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
                model = paddle.nn.Sequential(linear1, linear2)

                layer_list = list(model.children())

                print(layer_list)   # [<paddle.nn.layer.common.Linear object at 0x7f7b8113f830>, <paddle.nn.layer.common.Linear object at 0x7f7b8113f950>]

        """
        for _, layer in self.named_children():
            yield layer

    def named_children(self):
        """Returns an iterator over immediate children layers, yielding both
        the name of the layer as well as the layer itself.

        Yields:
            (string, Layer): Tuple containing a name and child layer

        Examples:
            .. code-block:: python

                import paddle

                linear1 = paddle.nn.Linear(10, 3)
                linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
                model = paddle.nn.Sequential(linear1, linear2)
                for prefix, layer in model.named_children():
                    print(prefix, layer)
                    # ('0', <paddle.nn.layer.common.Linear object at 0x7fb61ed85830>)
                    # ('1', <paddle.nn.layer.common.Linear object at 0x7fb61ed85950>)

        """
        memo = set()
        for name, layer in self._sub_layers.items():
            if layer is not None and layer not in memo:
                memo.add(layer)
                yield name, layer

    def sublayers(self, include_sublayers=True):
        """Returns a list of sub layers.

        Parameters:
            include_sublayers(bool, optional): Whether return the sublayers of sublayers. If True, also include the sublayers of sublayers. Default: True

        Returns:
            list of Layer : a list of sub layers.

        Examples:
            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self._linear = paddle.nn.Linear(1, 1)
                        self._dropout = paddle.nn.Dropout(p=0.5)

                    def forward(self, input):
                        temp = self._linear(input)
                        temp = self._dropout(temp)
                        return temp

                mylayer = MyLayer()
                print(mylayer.sublayers())  # [<paddle.nn.layer.common.Linear object at 0x7f44b58977d0>, <paddle.nn.layer.common.Dropout object at 0x7f44b58978f0>]

        """
        ret = [
            layer
            for _, layer in self.named_sublayers(
                include_sublayers=include_sublayers)
        ]
        return ret

    def named_parameters(self, prefix='', include_sublayers=True):
        """
        Returns an iterator over all parameters in the Layer, yielding tuple of name and parameter.

        Parameters:
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            include_sublayers(bool, optional): Whether include the parameters of sublayers.
                If True, also include the named parameters from sublayers. Default: True.

        Yields:
            (string, Parameter): Tuple of name and Parameter

        Examples:
            .. code-block:: python

                import paddle

                fc1 = paddle.nn.Linear(10, 3)
                fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
                model = paddle.nn.Sequential(fc1, fc2)
                for name, param in model.named_parameters():
                    print(name, param)

        """
        params_set = set()
        named_sublayers = self.named_sublayers(
            prefix=prefix,
            include_sublayers=include_sublayers,
            include_self=True)
        for layer_prefix, sublayer in named_sublayers:
            params = sublayer._parameters.items()
            for key, param in params:
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                name = layer_prefix + ('.' if layer_prefix else '') + key
                yield name, param

    def named_sublayers(self,
                        prefix='',
                        include_sublayers=True,
                        include_self=False,
                        layers_set=None):
        """
        Returns an iterator over all sublayers in the Layer, yielding tuple of name and sublayer.
        The duplicate sublayer will only be yielded once.

        Parameters:
            prefix(str, optional): Prefix to prepend to all parameter names. Default: ''.
            include_sublayers(bool, optional): Whether include the sublayers. Default: True.
            include_self(bool, optional): Whether include the Layer itself. Default: False.
            layers_set(set, optioanl): The set to record duplicate sublayers. Default: None.

        Yields:
            (string, Layer): Tuple of name and Layer

        Examples:
            .. code-block:: python

                import paddle

                fc1 = paddle.nn.Linear(10, 3)
                fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
                model = paddle.nn.Sequential(fc1, fc2)
                for prefix, layer in model.named_sublayers():
                    print(prefix, layer)

        """
        if layers_set is None:
            layers_set = set()
        if include_self and self not in layers_set:
            layers_set.add(self)
            yield prefix, self
        if include_sublayers:
            for key, layer in self._sub_layers.items():
                if layer is None:
                    continue
                layer_prefix = prefix + ('.' if prefix else '') + key
                for p, l in layer.named_sublayers(
                        prefix=layer_prefix,
                        include_sublayers=include_sublayers,
                        include_self=True,
                        layers_set=layers_set):
                    yield p, l

    def register_buffer(self, name, tensor, persistable=True):
        """
        Registers a tensor as buffer into the layer.

        `buffer` is a non-trainable tensor and will not be updated by optimizer,
        but is necessary for evaluation and inference. For example, the mean and variance in BatchNorm layers.
        The registered buffer is persistable by default, and will be saved into
        `state_dict` alongside parameters. If set persistable=False, it registers
        a non-persistable buffer, so that it will not be a part of `state_dict` .

        Buffers can be accessed as attributes using given names.

        Parameters:
            name (string): name of the buffer. The buffer can be accessed
                from this layer using the given name
            tensor (Tensor): the tensor to be registered as buffer.
            persistable (bool): whether the buffer is part of this layer's
                state_dict.

        Returns:
            None
        
        Examples:
            .. code-block:: python

                import numpy as np
                import paddle

                linear = paddle.nn.Linear(10, 3)
                value = np.array([0]).astype("float32")
                buffer = paddle.to_tensor(value)
                linear.register_buffer("buf_name", buffer, persistable=True)

                # get the buffer by attribute.
                print(linear.buf_name)

        """

        if '_buffers' not in self.__dict__:
            raise ValueError(
                "super(YourLayer, self).__init__() should be called first")
        elif not isinstance(name, six.string_types):
            raise TypeError(
                "The name of buffer should be a string, but received {}.".
                format(type(name).__name__))
        elif '.' in name:
            raise KeyError(
                "The name of buffer can not contain `.`, "
                "because when you access the newly added buffer in the "
                "form of `self.**.**`, it will cause AttributeError.")
        elif name == '':
            raise KeyError("The name of buffer can not be empty.")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists.".format(name))
        elif tensor is not None and not type(tensor) == core.VarBase:
            raise TypeError(
                "The registered buffer should be a core.VarBase, but received {}.".
                format(type(tensor).__name__))
        else:
            self._buffers[name] = tensor
            if persistable:
                self._non_persistable_buffer_names_set.discard(name)
            else:
                self._non_persistable_buffer_names_set.add(name)

    def buffers(self, include_sublayers=True):
        """
        Returns a list of all buffers from current layer and its sub-layers.

        Parameters:
            include_sublayers(bool, optional): Whether include the buffers of sublayers. If True, also include the buffers from sublayers. Default: True

        Returns:
            list of Tensor : a list of buffers.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle

                linear = paddle.nn.Linear(10, 3)
                value = np.array([0]).astype("float32")
                buffer = paddle.to_tensor(value)
                linear.register_buffer("buf_name", buffer, persistable=True)

                print(linear.buffers())     # == print([linear.buf_name])

        """
        ret = [
            buffer
            for _, buffer in self.named_buffers(
                include_sublayers=include_sublayers)
        ]
        return ret

    def named_buffers(self, prefix='', include_sublayers=True):
        """
        Returns an iterator over all buffers in the Layer, yielding tuple of name and Tensor.

        Parameters:
            prefix(str, optional): Prefix to prepend to all buffer names. Default: ''.
            include_sublayers(bool, optional): Whether include the buffers of sublayers.
                If True, also include the named buffers from sublayers. Default: True.

        Yields:
            (string, Tensor): Tuple of name and tensor

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle

                fc1 = paddle.nn.Linear(10, 3)
                buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))
                # register a tensor as buffer by specific `persistable`
                fc1.register_buffer("buf_name_1", buffer1, persistable=True)

                fc2 = paddle.nn.Linear(3, 10)
                buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))
                # register a buffer by assigning an attribute with Tensor.
                # The `persistable` can only be False by this way.
                fc2.buf_name_2 = buffer2

                model = paddle.nn.Sequential(fc1, fc2)

                # get all named buffers
                for name, buffer in model.named_buffers():
                    print(name, buffer)

        """
        buffers_set = set()
        named_sublayers = self.named_sublayers(
            prefix=prefix,
            include_sublayers=include_sublayers,
            include_self=True)
        for layer_prefix, sublayer in named_sublayers:
            buffers = sublayer._buffers.items()
            for key, buffer in buffers:
                if buffer is None or buffer in buffers_set:
                    continue
                buffers_set.add(buffer)
                name = layer_prefix + ('.' if layer_prefix else '') + key
                yield name, buffer

    def clear_gradients(self):
        """
        Clear the gradients of all parameters for this layer.
        
        Returns:
            None
        
        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)
                linear = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01,
                                            parameters=linear.parameters())
                out = linear(a)
                out.backward()
                adam.step()
                linear.clear_gradients()

        """
        for p in self.parameters():
            if p.trainable:
                p.clear_gradient()

    def _build_once(self, *args, **kwargs):
        pass

    def __call__(self, *inputs, **kwargs):
        for forward_pre_hook in self._forward_pre_hooks.values():
            hook_result = forward_pre_hook(self, inputs)
            if hook_result is not None:
                if not isinstance(hook_result, tuple):
                    hook_result = (hook_result, )
                inputs = hook_result

        if not self._built:
            with program_desc_tracing_guard(False):
                self._build_once(*inputs, **kwargs)
                if parallel_helper._is_data_parallel_mode():
                    parallel_helper._broadcast_parameters(
                        self._parameters.values())
            self._built = True

        with param_guard(self._parameters), param_guard(self._buffers):
            outputs = self.forward(*inputs, **kwargs)

        for forward_post_hook in self._forward_post_hooks.values():
            hook_result = forward_post_hook(self, inputs, outputs)
            if hook_result is not None:
                outputs = hook_result

        return outputs

    def forward(self, *inputs, **kwargs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.

        Parameters:
            *inputs(tuple): unpacked tuple arguments
            **kwargs(dict): unpacked dict arguments
        """
        raise NotImplementedError

    def backward(self, *inputs):
        raise ValueError("Layer shouldn't implement backward")

    def add_sublayer(self, name, sublayer):
        """Adds a sub Layer instance.

        Added sublayer can be accessed by self.name

        Parameters:
            name(str): name of this sublayer.
            sublayer(Layer): an instance of Layer.
        Returns:
            Layer: the sublayer passed in.
        
        Examples:
            .. code-block:: python

                import paddle

                class MySequential(paddle.nn.Layer):
                    def __init__(self, *layers):
                        super(MySequential, self).__init__()
                        if len(layers) > 0 and isinstance(layers[0], tuple):
                            for name, layer in layers:
                                self.add_sublayer(name, layer)
                        else:
                            for idx, layer in enumerate(layers):
                                self.add_sublayer(str(idx), layer)

                    def forward(self, input):
                        for layer in self._sub_layers.values():
                            input = layer(input)
                        return input

                fc1 = paddle.nn.Linear(10, 3)
                fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
                model = MySequential(fc1, fc2)
                for prefix, layer in model.named_sublayers():
                    print(prefix, layer)
        """
        assert isinstance(sublayer, core.Layer)

        self._sub_layers[name] = sublayer
        return sublayer

    def add_parameter(self, name, parameter):
        """Adds a Parameter instance.

        Added parameter can be accessed by self.name

        Parameters:
            name(str): name of this sublayer.
            parameter(Parameter): an instance of Parameter.
        Returns:
            Parameter: the parameter passed in.
        Examples:
            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self._linear = paddle.nn.Linear(1, 1)
                        w_tmp = self.create_parameter([1,1])
                        self.add_parameter("w_tmp", w_tmp)

                    def forward(self, input):
                        return self._linear(input)

                mylayer = MyLayer()
                for name, param in mylayer.named_parameters():
                    print(name, param)      # will print w_tmp,_linear.weight,_linear.bias

        """
        if '_parameters' not in self.__dict__:
            raise RuntimeError(
                "super(YourLayer, self).__init__() should be called firstly.")
        elif not isinstance(name, six.string_types):
            raise TypeError(
                "The name of parameter should be a string, but received {}.".
                format(type(name).__name__))
        elif '.' in name:
            raise KeyError(
                "The name of parameter can not contain `.`, "
                "because when you access the newly added parameter in the "
                "form of `self.**.**`, it will cause AttributeError.")
        elif name == '':
            raise KeyError("The name of parameter can not be empty.")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("The parameter '{}' already exists.".format(name))
        elif parameter is not None and not isinstance(parameter,
                                                      framework.Parameter):
            raise TypeError(
                "The parameter to be added should be a Parameter, but received {}.".
                format(type(parameter).__name__))
        else:
            if parameter is None:
                self._parameters[name] = None

            if len(self._loaddict_holder) > 0:
                assert parameter.name in self._loaddict_holder, "Parameter not found, Can't not find [ {} ] in state_dict".format(
                    parameter.name)

                parameter.set_value(self._loaddict_holder[parameter.name])

            self._parameters[name] = parameter
        return parameter

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._sub_layers:
            return self._sub_layers[name]
        elif name in self._buffers:
            return self._buffers[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        def _remove_if_exist(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
        params = self.__dict__.get('_parameters', None)
        if isinstance(value, framework.Parameter):
            if params is None:
                raise ValueError(
                    "super(YourLayer, self).__init__() should be called first")
            if len(self._loaddict_holder) > 0:
                assert value.name in self._loaddict_holder, "Parameter not found, Can't not find [ {} ] in state_dict".format(
                    value.name)

                value.set_value(self._loaddict_holder[value.name])

            _remove_if_exist(self.__dict__, self._buffers, self._sub_layers)
            params[name] = value
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    "assignment to parameter '{}' should be of type Parameter or None, but got '{}'"
                    .format(name, type(value).__name__))
            params[name] = None
        else:
            layers = self.__dict__.get('_sub_layers', None)
            if isinstance(value, core.Layer):
                if layers is None:
                    raise ValueError(
                        "super(YourLayer, self).__init__() should be called first"
                    )

                _remove_if_exist(self.__dict__, self._parameters, self._buffers)
                layers[name] = value
            elif layers is not None and name in layers:
                if value is not None:
                    raise TypeError(
                        "assignment to sublayer '{}' should be of type Layer or None, but got '{}'"
                        .format(name, type(value).__name__))
                layers[name] = None
            else:
                _buffers = self.__dict__.get('_buffers', None)
                if type(value) == core.VarBase:
                    if _buffers is None:
                        raise ValueError(
                            "super(YourLayer, self).__init__() should be called first"
                        )
                    _remove_if_exist(self.__dict__, self._parameters,
                                     self._sub_layers)
                    # Set persistable=False by default. Only `register_buffer` can
                    # add a persistable buffer.
                    if name not in self._buffers:
                        self._non_persistable_buffer_names_set.add(name)
                    _buffers[name] = value
                elif _buffers is not None and name in _buffers:
                    if value is not None:
                        raise TypeError(
                            "assignment to buffers '{}' should be of type core.VarBase or None, but got '{}'"
                            .format(name, type(value).__name__))
                    # Assigning None will remove the buffer, but if re-assign a new varBase to it,
                    # it will be remarked as a buffer with same `persistable` attribute.
                    _buffers[name] = None
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._sub_layers:
            del self._sub_layers[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistable_buffer_names_set.discard(name)
        else:
            object.__delattr__(self, name)

    def __dir__(self):
        """
        Return a list. Get all parameters, buffers(non-parameter variables), sublayers, method and attr of Layer.

        Examples:
            .. code-block:: python
                import paddle
                import numpy as np

                class Mylayer(paddle.nn.Layer):
                    def __init__(self):
                        super(Mylayer, self).__init__()
                        self.linear1 = paddle.nn.Linear(10, 10)
                        self.linear2 = paddle.nn.Linear(5, 5)
                        self.conv2d = paddle.nn.Conv2d(3, 2, 3)
                        self.embedding = paddle.nn.Embedding(128, 16)
                        self.h_0 = paddle.to_tensor(np.zeros([10, 10]).astype('float32'))

                mylayer = Mylayer()
                print(dir(mylayer))
                # only parts are shown, because of list have too much content
                # ['__call__', '__class__',  ... , 'conv2d', 'embedding', 'h_0', 'linear1', 'linear2', ... , 'sublayers', 'train']

        """
        method = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        sublayers = list(self._sub_layers.keys())
        buffers = list(self._buffers.keys())

        keys = method + attrs + parameters + sublayers + buffers

        return keys

    def state_dict(self,
                   destination=None,
                   include_sublayers=True,
                   structured_name_prefix=""):
        '''
        Get all parameters and persistable buffers of current layer and its sub-layers. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None
            include_sublayers(bool, optional) : If true, also include the parameters and persistable buffers from sublayers. Default: True

        Retruns:
            dict: a dict contains all the parameters and persistable buffers.

        Examples:
            .. code-block:: python

                import paddle

                emb = paddle.nn.Embedding(10, 10)

                state_dict = emb.state_dict()
                paddle.save( state_dict, "paddle_dy.pdparams")

        '''

        if destination is None:
            destination = collections.OrderedDict()
        for name, data in self._parameters.items():
            if data is not None:
                destination[structured_name_prefix + name] = data
        for name, buffer in self._buffers.items():
            if buffer is not None and name not in self._non_persistable_buffer_names_set:
                destination[structured_name_prefix + name] = buffer

        if include_sublayers:
            for layer_name, layer_item in self._sub_layers.items():
                if layer_item is not None:
                    destination_temp = destination.copy()
                    destination_temp.update(
                        layer_item.state_dict(
                            destination_temp, include_sublayers,
                            structured_name_prefix + layer_name + "."))
                    destination = destination_temp
        return destination

    @framework.deprecate_stat_dict
    def set_state_dict(self,
                       state_dict,
                       include_sublayers=True,
                       use_structured_name=True):
        '''
        Set parameters and persistable buffers from state_dict. All the parameters and buffers will be reset by the tensor in the state_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters and persistable buffers.
            include_sublayers(bool, optional) : If true, also include the parameters and peresistable buffers from sublayers. Default: True
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter or buffer name as key. 
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle

                emb = paddle.nn.Embedding(10, 10)

                state_dict = emb.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")
                para_state_dict = paddle.load("paddle_dy.pdparams")
                emb.set_state_dict(para_state_dict)

        '''

        def _check_match(key, param):
            state = state_dict.get(key, None)
            if state is None:
                raise ValueError("{} is not found in the provided dict.".format(
                    key))
            if list(state.shape) != list(param.shape):
                raise ValueError(
                    "{} receives a shape {}, but the expected shape is {}.".
                    format(key, list(state.shape), list(param.shape)))
            return param, state

        matched_param_state = []
        for key, param in self.state_dict().items():
            key_name = key if use_structured_name else param.name
            try:
                match_res = _check_match(key_name, param)
                matched_param_state.append(match_res)
            except ValueError as err:
                warnings.warn(("Skip loading for {}. ".format(key) + str(err)))

        if in_dygraph_mode():
            for param, state in matched_param_state:
                param.set_value(state)
        else:

            def _set_var(var, ndarray):
                t = global_scope().find_var(var.name).get_tensor()
                p = t._place()
                if p.is_cpu_place():
                    place = core.CPUPlace()
                elif p.is_cuda_pinned_place():
                    place = core.CUDAPinnedPlace()
                else:
                    p = core.Place()
                    p.set_place(t._place())
                    place = core.CUDAPlace(p.gpu_device_id())
                t.set(ndarray, place)

            executor = Executor(_get_device())._default_executor
            # restore parameter states
            core._create_loaded_parameter(
                [param for param, state in matched_param_state],
                global_scope(), executor)
            for param, state in matched_param_state:
                _set_var(param, state)

    # [aliases] Compatible with old method names
    set_dict = set_state_dict
    load_dict = set_state_dict
