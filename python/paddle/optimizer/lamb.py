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

import warnings
from .optimizer import Optimizer
from ..fluid import core, framework, layers
from ..fluid import unique_name
from ..fluid.framework import Variable
from ..fluid.layer_helper import LayerHelper

__all__ = []


class Lamb(Optimizer):
    r"""
    LAMB (Layer-wise Adaptive Moments optimizer for Batching training) Optimizer.

    LAMB Optimizer is designed to scale up the batch size of training without losing
    accuracy, which supports adaptive element-wise updating and accurate layer-wise
    correction. For more information, please refer to `Large Batch Optimization for
    Deep Learning: Training BERT in 76 minutes <https://arxiv.org/abs/1904.00962>`_ .

    The updating of parameters follows:

    ..  math::

        m_t &= \\beta_1 m_{t - 1}+ (1 - \\beta_1)g_t

        v_t &= \\beta_2 v_{t - 1}  + (1 - \\beta_2)g_t^2

        m_t &= \\frac{m_t}{\\beta_1^t}

        v_t &= \\frac{v_t}{\\beta_2^t}

        r_t &= \\frac{m_t}{\\sqrt{v_t}+\\epsilon}

        w_t &= w_{t-1} -\\eta_t \\frac{\\left \| w_{t-1}\\right \|}{\\left \| r_t + \\lambda w_{t-1}\\right \|} (r_t + \\lambda w_{t-1})


    where :math:`m` is the 1st moment, and :math:`v` the 2nd moment, :math:`\\eta` the
    learning rate, :math:`\\lambda` the LAMB weight decay rate.

    Args:
        learning_rate (float|Variable, optional): the learning rate used to update parameters. \
            Can be a float value or a Variable with data type float32. Default 0.001.
        lamb_weight_decay (float, optional): The LAMB weight decay rate. Default 0.01. Remind that weight_decay should be None.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            Default 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            Default 0.999.
        epsilon (float, optional): A small float value for numerical stability. Default 1e-6.
        parameters (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_paddle_fluid_clip_ClipGradByGlobalNorm` , :ref:`api_paddle_fluid_clip_ClipGradByNorm` ,
            :ref:`api_paddle_fluid_clip_ClipGradByValue` ). If you want better convergence, it is recommended
            to use :ref:`api_paddle_fluid_clip_ClipGradByGlobalNorm` . Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        name(str|None): For detailed information, please refer to
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.
    Examples:
        .. code-block:: python
            
            import paddle
            import numpy as np
            inp = paddle.uniform(min=-0.1, max=0.1, shape=[10, 10], dtype='float32')
            linear = paddle.nn.Linear(10, 10)
            out = linear(inp)
            loss = paddle.mean(out)
            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.85], dtype="float32")
            lamb = paddle.optimizer.Lamb(learning_rate=0.002, parameters=linear.parameters(), lamb_weight_decay=0.01)
            back = out.backward()
            lamb.step()
            lamb.clear_grad()
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 lamb_weight_decay=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 parameters=None,
                 grad_clip=None,
                 exclude_from_weight_decay_fn=None,
                 multi_precision=False,
                 name=None):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(Lamb, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=None,
            grad_clip=grad_clip,
            name=name)
        self.type = "lamb"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lamb_weight_decay = lamb_weight_decay
        self._exclude_from_weight_decay_fn = exclude_from_weight_decay_fn
        self._multi_precision = multi_precision
        self._master_weights = {}

    def _create_master_weight(self, param):
        assert isinstance(self.helper, LayerHelper)

        var_name = param.name + "_fp32_master"
        var_name = unique_name.generate(var_name)
        var = layers.create_global_var(
            name=var_name,
            shape=param.shape,
            value=0,
            dtype='float32',
            persistable=True)
        block = self.helper.startup_program.global_block()
        block.append_op(
            type="cast",
            inputs={"X": [param]},
            outputs={"Out": [var]},
            attrs={
                "in_dtype": param.dtype,
                "out_dtype": core.VarDesc.VarType.FP32
            })
        self._master_weights[param.name] = var
        return var

    def _add_moments_pows(self, p):
        acc_dtype = p.dtype
        if acc_dtype == core.VarDesc.VarType.FP16:
            acc_dtype = core.VarDesc.VarType.FP32
        # Create accumulator tensors for first and second moments
        self._add_accumulator(self._moment1_acc_str, p)
        self._add_accumulator(self._moment2_acc_str, p)
        self._add_accumulator(
            name=self._beta1_pow_acc_str,
            param=p,
            fill_value=0.9 if isinstance(self._beta1, Variable) \
                    else self._beta1,
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
        self._add_accumulator(
            name=self._beta2_pow_acc_str,
            param=p,
            fill_value=0.999 if isinstance(self._beta2, Variable) \
                    else self._beta2,
            shape=[1],
            type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if self._multi_precision and p.dtype == core.VarDesc.VarType.FP16:
                master_p = self._create_master_weight(p)
                self._add_moments_pows(master_p)
                continue
            if p.dtype == core.VarDesc.VarType.FP16 and not self._multi_precision:
                warnings.warn(
                    "Accumulating with FP16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Momentum optimizer."
                )
            self._add_moments_pows(p)

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and param.dtype == core.VarDesc.VarType.FP16
        target_param = self._master_weights[
            param.name] if find_master else param
        target_name = target_param.name
        if (name not in self._accumulators or
                target_name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, target_name))
        return self._accumulators[name][target_name]

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        block.program._use_lamb = True

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])
        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        if self._exclude_from_weight_decay_fn is not None \
            and self._exclude_from_weight_decay_fn(param_and_grad[0]):
            weight_decay = 0.0
        else:
            weight_decay = self._lamb_weight_decay
        lr = self._create_param_lr(param_and_grad)

        if framework.in_dygraph_mode():
            _, _, _, _, _ = core.ops.lamb(
                param_and_grad[0], param_and_grad[1], lr, moment1, moment2,
                beta1_pow_acc, beta2_pow_acc, param_and_grad[0], moment1,
                moment2, beta1_pow_acc, beta2_pow_acc, 'beta1', self._beta1,
                'beta2', self._beta2, 'epsilon', self._epsilon, 'weight_decay',
                weight_decay)
            return None

        # create the lamb optimize op
        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "LearningRate": lr,
            "Moment1": moment1,
            "Moment2": moment2,
            "Beta1Pow": beta1_pow_acc,
            "Beta2Pow": beta2_pow_acc
        }
        outputs = {
            "ParamOut": param_and_grad[0],
            "Moment1Out": moment1,
            "Moment2Out": moment2,
            "Beta1PowOut": beta1_pow_acc,
            "Beta2PowOut": beta2_pow_acc
        }
        attrs = {
            "beta1": self._beta1,
            "beta2": self._beta2,
            "epsilon": self._epsilon,
            "weight_decay": weight_decay
        }

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        lamb_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return lamb_op
