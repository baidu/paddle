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

from .optimizer import Optimizer
from ..fluid import core
from ..fluid import framework
from ..fluid.framework import Variable, name_scope
from ..fluid.dygraph import no_grad
from ..fluid import layers
from ..fluid import unique_name
from ..fluid.layer_helper import LayerHelper
import warnings

__all__ = []


class SGD(Optimizer):
    r"""
    Optimizer of the stochastic gradient descent algorithm.

    .. math::

        param\_out = param - learning\_rate * grad

    Parameters:
        learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating. Default is false.
        rescale_grad (float, optional): Multiply the gradient with `rescale_grad` before updating. \
            Often choose to be ``1.0/batch_size``.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` . 
        
    Examples:
        .. code-block:: python

            import paddle
            import numpy as np
            inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.to_tensor(inp)
            out = linear(inp)
            loss = paddle.mean(out)
            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")
            sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
            back = out.backward()
            sgd.step()
            sgd.clear_grad()

    """

    def __init__(self,
                 learning_rate=0.001,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False,
                 rescale_grad=1.0,
                 name=None):
        if learning_rate is None:
            raise ValueError("learning_rate is not set")
        super(SGD, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)
        self.type = "sgd"
        self._rescale_grad = rescale_grad
        self._master_weights = {}

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        lr = self._create_param_lr(param_and_grad)
        if framework.in_dygraph_mode():
            core.ops.sgd(param_and_grad[0], lr, param_and_grad[1],
                         param_and_grad[0])
            return None

        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        assert isinstance(block, framework.Block)
        # create the optimize op
        sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": lr
            },
            outputs={"ParamOut": param_and_grad[0]},
            attrs={
                "multi_precision": find_master,
                "rescale_grad": self._rescale_grad
            },
            stop_gradient=True)

        return sgd_op

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
