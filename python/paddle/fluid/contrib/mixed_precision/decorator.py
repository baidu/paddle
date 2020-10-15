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

from ... import default_main_program
from ... import default_startup_program
from ... import layers
from ... import unique_name
from ... import program_guard
from . import fp16_utils
from .fp16_utils import rewrite_program
from .fp16_utils import update_role_var_grad
from .fp16_lists import AutoMixedPrecisionLists
from .amp_nn import check_finite_and_unscale
from .amp_nn import update_loss_scaling

__all__ = ["decorate"]


class OptimizerWithMixedPrecision(object):
    """
    Optimizer with mixed-precision (MP) training. This is a wrapper of a common 
    optimizer, plus the support of mixed-precision pre-training. The object
    of this class almost has the same behavior as the common optimizer, with the 
    methods `minimize()`, `backward()`, `apply_gradients()` implemented. 
    Additionally, it enables the MP training automatically, i.e, the creation 
    and maintenance of master parameters, scaling of loss, etc.

    Args:
        optimizer (Optimizer): A common Optimizer object.
        amp_lists (AutoMixedPrecisionLists): An AutoMixedPrecisionLists object.
        init_loss_scaling (float): The initial loss scaling factor.
        use_dynamic_loss_scaling (bool): Whether to use dynamic loss scaling.
        incr_every_n_steps(int): Increases loss scaling every n consecutive 
                                 steps with finite gradients.
        decr_every_n_nan_or_inf(int): Decreases loss scaling every n 
                                      accumulated steps with nan or 
                                      inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           the loss scaling.

    """

    def __init__(self, optimizer, amp_lists, init_loss_scaling,
                 use_dynamic_loss_scaling, incr_every_n_steps,
                 decr_every_n_nan_or_inf, incr_ratio, decr_ratio):
        self._optimizer = optimizer
        self._amp_lists = amp_lists
        self._param_grads = None
        self._train_program = None

        self._scaled_loss = None
        self._loss_scaling = None
        self._init_loss_scaling = init_loss_scaling
        self._use_dynamic_loss_scaling = use_dynamic_loss_scaling
        if self._use_dynamic_loss_scaling:
            self._incr_every_n_steps = incr_every_n_steps
            self._decr_every_n_nan_or_inf = decr_every_n_nan_or_inf
            self._incr_ratio = incr_ratio
            self._decr_ratio = decr_ratio
            self._num_good_steps = None
            self._num_bad_steps = None

    def get_loss_scaling(self):
        """Return the real-time loss scaling factor.
        """
        return self._loss_scaling

    def get_scaled_loss(self):
        """Return the scaled loss.
        It's useful when you feed customed loss into executor.
        """
        return self._scaled_loss

    def _init_amp_var(self):
        self._loss_scaling = layers.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=self._init_loss_scaling,
            dtype='float32',
            persistable=True)

        if self._use_dynamic_loss_scaling:
            self._num_good_steps = layers.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            self._num_bad_steps = layers.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

        # Ensure the data type of learning rate vars is float32 (same as the
        # master parameter dtype)
        if isinstance(self._optimizer._learning_rate, float):
            self._optimizer._learning_rate_map[default_main_program()] = \
                    layers.create_global_var(
                    name=unique_name.generate("learning_rate"),
                    shape=[1],
                    value=float(self._optimizer._learning_rate),
                    dtype='float32',
                    persistable=True)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Backward propagation or auto differentiation for gradients' computation.

        Args:
            loss (Variable): The loss Variable to minimize.
            startup_program (Program|None): The startup Program for initializing 
                                       parameters in `parameter_list`.
            parameter_list (list|None): A list of Variables to update.
            no_grad_set (set|None): A set of Variables should be ignored.
            callbacks (list|None): A list of callable objects to run when appending
                                   backward operator for one parameter.

        Returns:
            A list of (param, grad), which is a tuple of a parameter and its 
            gradient respectively, and the scaled loss.
        """
        train_program = loss.block.program
        self._train_program = train_program

        with program_guard(train_program, startup_program):
            self._init_amp_var()

            rewrite_program(train_program, self._amp_lists)
            self._scaled_loss = loss * self._loss_scaling
            params_grads = self._optimizer.backward(
                self._scaled_loss, startup_program, parameter_list, no_grad_set,
                callbacks)
            # Change the op_role_var attr for some ops, so that gradients
            # transferred across GPUs can be FP16.
            update_role_var_grad(train_program, params_grads)
        return params_grads

    def apply_gradients(self, params_grads):
        """
        Check scaled gradients to determine whether to update loss scaling and update 
        parameters by their scaled gradients, 
  
        Args:
            params_grads (list): A list of params and scaled grads.
    
        Returns:
            A list of optimize operators.
        """

        grads = [g for _, g in params_grads]
        with self._train_program._optimized_guard(grads):
            grads, found_inf = check_finite_and_unscale(
                grads, self._loss_scaling, name="find_infinite_scale")

        if self._use_dynamic_loss_scaling:
            with self._train_program._optimized_guard(grads):
                grads = update_loss_scaling(
                    grads,
                    found_inf,
                    self._loss_scaling,
                    self._num_good_steps,
                    self._num_bad_steps,
                    self._incr_every_n_steps,
                    self._decr_every_n_nan_or_inf,
                    self._incr_ratio,
                    self._decr_ratio,
                    name="update_loss_scaling")

        params_unscaled_grads = []
        for pg, new_g in zip(params_grads, grads):
            params_unscaled_grads.append((pg[0], new_g))
        # apply_gradient append all ops in global block, thus we shouldn't
        # apply gradient in the switch branch.
        optimize_ops = self._optimizer.apply_gradients(params_unscaled_grads)

        return optimize_ops

    def apply_optimize(self, loss, startup_program, params_grads):
        program = loss.block.program
        with program_guard(program, startup_program):
            optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Perform optimization by minimizing the given loss.

        Args:
            loss (Variable): The loss Variable.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.

        Returns:
            The scaled loss by scaling factor, the list of optimize ops, and a
            list of scaled parameters and gradients.
        """
        scaled_params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_optimize(loss, startup_program,
                                           scaled_params_grads)

        return optimize_ops, scaled_params_grads


def decorate(optimizer,
             amp_lists=None,
             init_loss_scaling=2**15,
             incr_every_n_steps=1000,
             decr_every_n_nan_or_inf=2,
             incr_ratio=2.0,
             decr_ratio=0.8,
             use_dynamic_loss_scaling=True):
    """ 
    Decorate the given optimizer to adapt to the mixed-precision training.

    Args:
        optimizer(Optimizer): A common Optimizer.
        amp_lists (AutoMixedPrecisionLists): An AutoMixedPrecisionLists object.
        init_loss_scaling(float): The initial loss scaling factor.
        incr_every_n_steps(int): Increases loss scaling every n consecutive 
                                 steps with finite gradients.
        decr_every_n_nan_or_inf(int): Decreases loss scaling every n 
                                      accumulated steps with nan or 
                                      inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           the loss scaling.
        use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling.

    Returns:
        An optimizer acting like a normal one but with mixed-precision training 
        enabled.

    Examples:
	.. code-block:: python

	    loss = network()
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
	
            mp_optimizer = fluid.contrib.mixed_precision.decorate(
	              optimizer=optimizer, init_loss_scaling=8.0)
	
            ops, param_grads = mp_optimizer.minimize(loss)
            scaled_loss = mp_optimizer.get_scaled_loss()
    """
    if amp_lists is None:
        amp_lists = AutoMixedPrecisionLists()
    mp_optimizer = OptimizerWithMixedPrecision(
        optimizer, amp_lists, init_loss_scaling, use_dynamic_loss_scaling,
        incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

    return mp_optimizer
