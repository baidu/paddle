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

import numpy as np
import six
import logging
from collections import defaultdict

import paddle
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program, device_guard
from paddle.fluid.dygraph.parallel import apply_collective_grads

from ..fluid import framework
from ..fluid import layers
from ..fluid import unique_name
from ..fluid.backward import append_backward, _some_in_set_, _append_grad_suffix_, _get_no_grad_set_name
from ..fluid.clip import GradientClipBase, GradientClipByNorm, error_clip_callback, append_gradient_clip_ops
from ..fluid.framework import program_guard
from ..fluid.initializer import Constant
from ..fluid.layer_helper import LayerHelper
from ..fluid.layers import ops
from ..fluid.regularizer import append_regularization_ops
from ..fluid.dygraph import base as imperative_base
from ..fluid.dygraph import no_grad
from paddle.fluid import core
from paddle.fluid.layers import tensor
from functools import reduce
from ..fluid.wrapped_decorator import signature_safe_contextmanager
from .. import compat as cpt
from .lr import LRScheduler

__all__ = ['Optimizer']


class Optimizer(object):
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.

    Args:
        learning_rate (float|LRScheduler): The learning rate used to update ``Parameter``.
            It can be a float value or any subclass of ``LRScheduler`` .
        parameters (list, optional): List of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of \
            some derived class of ``GradientClipBase`` . There are three cliping strategies \
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , \
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Returns:
       Base class for optimizer. 
    
    Examples:
        .. code-block:: python

            #Take the subclass adam as an example
            import paddle
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
            out = linear(inp)
            loss = paddle.mean(out)
            adam = paddle.optimizer.Adam(learning_rate=0.1,
                    parameters=linear.parameters())
            out.backward()
            adam.step()
            adam.clear_grad()

    """

    @imperative_base.no_grad
    def __init__(self,
                 learning_rate,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        self._parameter_list = list(
            parameters) if parameters is not None else None
        self._name = name
        if framework.in_dygraph_mode():
            if self._parameter_list is None:
                raise AttributeError(
                    "parameters argument given to the Optimizer should not be None in dygraph mode."
                )
            if weight_decay is not None:
                for param in self._parameter_list:
                    if param.regularizer is not None:
                        logging.info(
                            "If regularizer of a Parameter has been set by 'paddle.ParamAttr' or 'static.WeightNormParamAttr' already. "
                            "The weight_decay[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            % weight_decay.__str__())
                        break
        if not isinstance(learning_rate, (float, LRScheduler)):
            raise TypeError(
                "learning rate should be float or LRScheduler, got %s here" %
                type(learning_rate))
        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipBase):
                raise TypeError(
                    "'grad_clip' should be an instance of GradientClipBase's derived class"
                )
        if isinstance(weight_decay, float):
            from ..fluid.regularizer import L2Decay
            self.regularization = L2Decay(weight_decay)
        else:
            self.regularization = weight_decay
        self._grad_clip = grad_clip
        self._learning_rate = learning_rate
        # the learning rate type should be inferenced from loss
        self._dtype = None
        # each program should have a independent learning rate
        # program -> tensor(learning_rate)
        self._learning_rate_map = dict()
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra tensors associated with the parameters
        # to train. These tensors are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())
        self.helper = None
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = dict()
        self.clear_gradients = self.clear_grad

    @framework.dygraph_only
    def state_dict(self):
        '''
        Get state dict information from optimizer. It contain all the tensor used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. If LRScheduler have been used, global_step will be include in state dict.
        If the optimizer never be called(minimize function), the state_dict is empty.

        Args: 
            None

        Returns:
            state_dict(dict) : dict contains all the Tensor used by optimizer
        
        Examples:
            .. code-block:: python

                import paddle
                emb = paddle.nn.Embedding(10, 10)

                adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
                state_dict = adam.state_dict()

        '''
        state_dict = {}
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                state_dict[var_tmp.name] = var_tmp
        # global step if use lr decay
        if isinstance(self._learning_rate, LRScheduler):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        '''
        Load optimizer state dict. For Adam optimizer, contains beta1, beta2, momentum etc. If LRScheduler have been used, global_step will be changed.

        Args: 
            state_dict(dict) : Dict contains all the Tensor needed by optimizer
        Return:
            None
        
        Examples:
            .. code-block:: python

                import paddle

                emb = paddle.nn.Embedding(10, 10)

                layer_state_dict = emb.state_dict()
                paddle.save(layer_state_dict, "emb.pdparams")

                scheduler = paddle.optimizer.lr.NoamDecay(	
                    d_model=0.01, warmup_steps=100, verbose=True)
                adam = paddle.optimizer.Adam(
                    learning_rate=scheduler,
                    parameters=emb.parameters())
                opt_state_dict = adam.state_dict()
                paddle.save(opt_state_dict, "adam.pdopt")

                opti_state_dict = paddle.load("adam.pdopt")
                adam.set_state_dict(opti_state_dict)

        '''
        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_state_dict(state_dict["LR_Scheduler"])

        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert var_tmp.name in state_dict, \
                        "optimizer Tensor {} not found".format( var_tmp.name )
                var = var_tmp.value()
                tensor = var.get_tensor()
                model_np = np.array(tensor)

                load_para = state_dict[var_tmp.name]

                if isinstance(load_para, Variable):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, core.VarBase):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, np.ndarray):
                    load_para_np = load_para
                else:
                    raise RuntimeError("State dict type {} not supprt".format(
                        str(type(load_para))))

                assert model_np.shape == load_para_np.shape,  \
                                          "Parameter shape not match, Dygraph Parameter [ {} ] need tensor with shape {} but load tensor with shape {}".format(
                                                 item.name, model_np.shape, load_para_np.shape)

                assert model_np.dtype == load_para_np.dtype, \
                                          "Parameter dtype not match, Dygraph Parameter [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                                                item.name, model_np.dtype, load_para_np.dtype)

                tensor.set(load_para_np, framework._current_expected_place())

    def get_opti_var_name_list(self):
        return self._opti_name_list

    def _create_global_learning_rate(self):
        if isinstance(self._learning_rate, LRScheduler):
            lr_var = self._global_learning_rate()
            # only create global lr_var once
            if not isinstance(lr_var, framework.Variable):
                lr_name = unique_name.generate('learning_rate')
                self._learning_rate._var_name = lr_name
                lr_var = self.helper.create_global_variable(
                    name=lr_name,
                    shape=[1],
                    persistable=True,
                    stop_gradient=True,
                    dtype=paddle.get_default_dtype()
                    if self._dtype is None else self._dtype)
                main_prog = framework.default_main_program()
                main_prog.lr_sheduler = self._learning_rate
                main_prog.lr_var = lr_var

                self._learning_rate_map[framework.default_main_program(
                )] = lr_var

            lr_value = float(self._learning_rate())
            self.helper.set_variable_initializer(
                lr_var, initializer=Constant(value=lr_value))
        elif isinstance(self._learning_rate, float):
            # only create global lr_var once
            lr = self._global_learning_rate()
            if isinstance(lr, framework.Variable):
                return
            else:
                self._learning_rate_map[framework.default_main_program(
                )] = layers.create_global_var(
                    name=unique_name.generate("learning_rate"),
                    shape=[1],
                    value=float(self._learning_rate),
                    dtype=paddle.get_default_dtype()
                    if self._dtype is None else self._dtype,
                    persistable=True)

    @framework.dygraph_only
    def set_lr(self, value):
        """
        :api_attr: imperative
        
        Set the value of the learning rate manually in the optimizer. If the optimizer use LRScheduler,
        this API cannot be invoked, because it will lead to conflict.

        Args:
            value (float): the value of learning rate

        Returns:
            None
          
        Examples:
            .. code-block:: python

                import paddle
                linear = paddle.nn.Linear(10, 10)

                adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

                # set learning rate manually by python float value
                lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                for i in range(5):
                    adam.set_lr(lr_list[i])
                    lr = adam.get_lr()
                    print("current lr is {}".format(lr))
                # Print:
                #    current lr is 0.2
                #    current lr is 0.3
                #    current lr is 0.4
                #    current lr is 0.5
                #    current lr is 0.6

        """
        if not isinstance(value, (int, float)):
            raise TypeError(
                "The type of 'value' in optimizer.set_lr must be float, but received %s."
                % (type(value)))
        if isinstance(self._learning_rate, LRScheduler):
            raise RuntimeError(
                "optimizer's learning rate can't be LRScheduler when invoke this API, because this will lead to conflict."
            )
        self._learning_rate = float(value)
        current_lr = self._global_learning_rate()
        if current_lr is not None:
            global_block = framework.default_main_program().global_block()
            global_block.append_op(
                type='fill_constant',
                outputs={'Out': [current_lr]},
                attrs={
                    'dtype': current_lr.dtype,
                    'shape': list(current_lr.shape),
                    'value': float(value)
                },
                stop_gradient=True)

    @framework.dygraph_only
    def get_lr(self):
        """
        :api_attr: imperative
        
        Get current step learning rate. The return value is all the same When LRScheduler is not used,
        otherwise return the current step learning rate.


        Returns:
            float: The learning rate of the current step.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                # example1: LRScheduler is not used, return value is all the same
                emb = paddle.nn.Embedding(10, 10)
                adam = paddle.optimizer.Adam(0.001, parameters = emb.parameters())
                lr = adam.get_lr()
                print(lr) # 0.001

                # example2: PiecewiseDecay is used, return the scheduled learning rate
                inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
                linear = paddle.nn.Linear(10, 10)
                inp = paddle.to_tensor(inp)
                out = linear(inp)
                loss = paddle.mean(out)
                
                bd = [2, 4, 6, 8]
                value = [0.2, 0.4, 0.6, 0.8, 1.0]
                scheduler = paddle.optimizer.lr.PiecewiseDecay(bd, value, 0)
                adam = paddle.optimizer.Adam(scheduler,
                                       parameters=linear.parameters())

                # first step: learning rate is 0.2
                np.allclose(adam.get_lr(), 0.2, rtol=1e-06, atol=0.0) # True

                # learning rate for different steps
                ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
                for i in range(12):
                    adam.step()
                    lr = adam.get_lr()
                    scheduler.step()
                    np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True

        """
        if isinstance(self._learning_rate, float):
            return self._learning_rate
        else:
            return self._learning_rate()

    def _global_learning_rate(self, program=None):
        """
        get global decayed learning rate
        :return:
        """
        if program is None:
            program = framework.default_main_program()
        return self._learning_rate_map.get(program, None)

    def _append_optimize_op(self, block, param_and_grad):
        """ append optimize operator to block and return all the added optimize_op
        """
        raise NotImplementedError(
            "Class \"Optimizer\" connot be used directly as an optimizer, please use its subclasses such as \"Adam\""
        )

    def _create_param_lr(self, param_and_grad):
        # create learning rate tensor for every parameter
        param = param_and_grad[0]
        param_lr = param.optimize_attr['learning_rate']
        if type(param_lr) == Variable:
            return param_lr
        else:
            if param_lr == 1.0:
                return self._global_learning_rate()
            else:
                with default_main_program()._lr_schedule_guard(
                        is_with_opt=True), framework.name_scope(
                            'scale_with_param_lr'):
                    return self._global_learning_rate() * param_lr

    def _create_accumulators(self, block, parameters):
        """Create all accumulators needed by the parameters

        Args:
            block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer
        """
        pass

    def _finish_update(self, block, parameters_and_grads):
        """Finish any custom updates needed
           before completing an optimization step

        Args:
            block: the block in which the loss tensor is present
            parameters: list of parameter tensors for the optimizer

        Returns:
            None
        """
        pass

    def _add_accumulator(self,
                         name,
                         param,
                         dtype=None,
                         fill_value=0.0,
                         shape=None,
                         type=None,
                         device=None):
        """Utility function to add an accumulator for a parameter

        Args:
            block: the block in which the loss tensor is present
            name: name of the accumulator
            param: parameter tensor for which accumulator is to be added
            dtype: data type of the accumulator tensor
            fill_value: value to initialize the accumulator tensor
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name in self._accumulators and
                param.name in self._accumulators[name]):
            if framework.in_dygraph_mode():
                return self._accumulators[name][param.name]
            raise Exception("Accumulator {} already exists for parameter {}".
                            format(name, param.name))
        if shape == None:
            shape = param.shape
        assert isinstance(self.helper, LayerHelper)

        var_name = param.name + "_" + name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype or param.dtype,
            type=param.type if type is None else type,
            shape=shape,
            belong_to_optimizer=True)
        if device is None:
            device = self._get_device_for_param(param.name)
        with device_guard(device):
            self.helper.set_variable_initializer(
                var, initializer=Constant(value=float(fill_value)))

        if framework.in_dygraph_mode():
            if len(self._accumulators_holder) > 0:
                assert var_name in self._accumulators_holder, \
                        "Optimizer set error, {} should in state dict".format( var_name )
                var.set_value(self._accumulators_holder[var_name])

        self._accumulators[name][param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter tensor for which accumulator is to be fetched

        Returns:
            accumulator tensor for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def _update_param_device_map(self, parameters_and_grads, target_block):
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].trainable is True:
                param_name = param_and_grad[0].name
                ops = target_block.ops
                device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName(
                )
                for op in ops:
                    input_arg_names = op.input_arg_names
                    if param_name in input_arg_names:
                        self._param_device_map[param_name] = op.attr(
                            device_attr_name)
                        break

    def _get_device_for_param(self, param_name):
        device = None
        if param_name in self._param_device_map:
            device = self._param_device_map[param_name]
        return device

    def _create_optimization_pass(self, parameters_and_grads):
        """Add optimization operators to update gradients to tensors.

        Args:
          parameters_and_grads(list(tuple(Tensor, Tensor))):
            a list of (tensor, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
            optimization. This will include parameter update ops, global step
            update ops and any other custom ops required by subclasses to manage
            their internal state.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Allways called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert current_block.backward_block_idx != -1, \
                "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)
        self._update_param_device_map(parameters_and_grads, target_block)
        self._create_accumulators(
            target_block,
            [p[0] for p in parameters_and_grads if p[0].trainable])
        self._create_global_learning_rate()

        if framework.in_dygraph_mode():
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].trainable is True:
                    self._append_optimize_op(target_block, param_and_grad)
        else:
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                with param_and_grad[0].block.program._optimized_guard(
                        param_and_grad), name_scope("optimizer"):
                    if param_and_grad[0].trainable is True:
                        device = self._get_device_for_param(param_and_grad[0]
                                                            .name)
                        with device_guard(device):
                            optimize_op = self._append_optimize_op(
                                target_block, param_and_grad)

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _append_dgc_ops(self, param_and_grad):
        pass

    def backward(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        The first part of ``minimize``, do auto-diff to append backward operations for
        the current program.

        Args:
            loss (Tensor): ``loss`` tensor to run optimizations.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameters (list, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.
            callbacks (list, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.

        Return:
            list: list of (param, grad) tensor pairs, param is ``Parameter``,
                grad is the gradient value corresponding to the parameter.

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np
                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)
                linear = paddle.nn.Linear(13, 5)
                # This can be any optimizer supported by dygraph.
                adam = paddle.optimizer.Adam(learning_rate = 0.01, 
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                adam.step()
                adam.clear_grad()
        """
        act_no_grad_set = None
        if framework.in_dygraph_mode():
            pass
        else:
            act_no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        self._dtype = loss.dtype
        if framework.in_dygraph_mode():
            parameter_list = parameters if parameters \
                else self._parameter_list

            if paddle.distributed.get_world_size() > 1:
                apply_collective_grads(parameter_list)

            params_grads = []
            for param in parameter_list:
                if not param.trainable:
                    continue
                if param._grad_ivar() is not None:
                    # create gradient tensor
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
        else:
            if callbacks is None:
                callbacks = [error_clip_callback]
            else:
                assert (isinstance(callbacks, list))
            program = loss.block.program
            assert len(loss.shape) == 1 and loss.shape[0] == 1, \
                "The loss.shape should be (1L,), but the current loss.shape is {}. " \
                "Maybe that you should call paddle.mean to process the current loss.".format(
                    loss.shape)
            parameter_list = parameters if parameters \
                else self._parameter_list
            with program_guard(program, startup_program):
                params_grads = append_backward(loss, parameter_list,
                                               act_no_grad_set, callbacks)
                # Note: since we can't use all_reduce_op now,
                #  dgc_op should be the last op of one grad.
                self._append_dgc_ops(params_grads)
        return params_grads

    def apply_gradients(self, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
                linear = paddle.nn.Linear(10, 10)
                inp = paddle.to_tensor(inp)
                out = linear(inp)
                loss = paddle.mean(out)
                optimizer = paddle.optimizer.Adam(learning_rate=0.1,
                        parameters=linear.parameters())
                params_grads = optimizer.backward(loss)
                optimizer.apply_gradients(params_grads)

        """

        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)
        else:

            params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = append_regularization_ops(params_grads,
                                                 self.regularization)

        optimize_ops = self._create_optimization_pass(params_grads)
        return optimize_ops

    def _apply_optimize(self, loss, startup_program, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.
        Args:
            loss (Tensor): loss tensor to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameters`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Returns:
            list: A list of operators appended to the current program.
        """
        if framework.in_dygraph_mode():
            with program_guard(framework.default_main_program(),
                               framework.default_startup_program()):
                if self._grad_clip is not None:
                    params_grads = self._grad_clip(params_grads)
                params_grads = append_regularization_ops(params_grads,
                                                         self.regularization)
                optimize_ops = self._create_optimization_pass(params_grads)
        else:
            program = loss.block.program
            with program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _get_no_grad_set(self, loss, no_grad_set=None):
        no_grad_set = _get_no_grad_set_name(no_grad_set)
        parameters = loss.block.program.global_block().all_parameters()
        param_no_trainable = set(
            [param.name for param in parameters if param.trainable is False])
        # If the parameter is no trainable, it should not have a gradient.
        no_grad_set.update(param_no_trainable)

        return no_grad_set

    @framework.dygraph_only
    def clear_grad(self):
        """
        Clear the gradients of all optimized parameters for model.
        
        Returns:
            None
        
        Examples:
            .. code-block:: python

                import numpy as np
                import paddle

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)
                linear = paddle.nn.Linear(13, 5)
                # This can be any optimizer supported by dygraph.
                adam = paddle.optimizer.Adam(learning_rate = 0.01, 
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                adam.step()
                adam.clear_grad()

        """
        for p in self._parameter_list:
            if p.trainable:
                p.clear_gradient()

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None):
        """
        Add operations to minimize ``loss`` by updating ``parameters``.

        Args:
            loss (Tensor): A ``Tensor`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameters``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameters (list, optional): List of ``Tensor`` or ``Tensor.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Tensor``  or ``Tensor.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) tensor pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            In static graph mode, the returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to 
            indicate program pruning. If so, the program will be pruned by ``feed`` and 
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:
            .. code-block:: python
 
                import paddle
                linear = paddle.nn.Linear(10, 10)
                input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
                out = linear(input)
                loss = paddle.mean(out)

                beta1 = paddle.to_tensor([0.9], dtype="float32")
                beta2 = paddle.to_tensor([0.99], dtype="float32")

                adam = paddle.optimizer.Adam(learning_rate=0.1,
                        parameters=linear.parameters(),
                        weight_decay=0.01)
                out.backward()
                adam.minimize(loss)
                adam.clear_grad()

        """
        assert isinstance(loss, Variable), "The loss should be an Tensor."

        parameter_list = parameters if parameters \
            else self._parameter_list

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameters=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self._apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads

    @framework.dygraph_only
    def step(self):
        """
        Execute the optimizer and update parameters once.
        
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle
                import numpy as np

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)
                linear = paddle.nn.Linear(13, 5)
                # This can be any optimizer supported by dygraph.
                adam = paddle.optimizer.Adam(learning_rate = 0.01, 
                                            parameters = linear.parameters())
                out = linear(a)
                out.backward()
                adam.step()
                adam.clear_grad()
        """
        if paddle.distributed.get_world_size() > 1:
            apply_collective_grads(self._parameter_list)

        self._dtype = None
        params_grads = []
        for param in self._parameter_list:
            if not param.trainable:
                continue
            if param._grad_ivar() is not None:
                grad_var = param._grad_ivar()
                params_grads.append((param, grad_var))

        self._apply_optimize(
            loss=None, startup_program=None, params_grads=params_grads)
