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
import re
from collections import defaultdict
from paddle.fluid.framework import Program, Variable
from . import framework
from . import layers
from .backward import append_backward
from .framework import program_guard
from . import unique_name
from .initializer import Constant
from .layer_helper import LayerHelper
from .regularizer import append_regularization_ops
from .clip import append_gradient_clip_ops, error_clip_callback
from contextlib import contextmanager

__all__ = [
    'SGD', 'Momentum', 'Adagrad', 'Adam', 'Adamax', 'DecayedAdagrad', 'Ftrl',
    'SGDOptimizer', 'MomentumOptimizer', 'AdagradOptimizer', 'AdamOptimizer',
    'AdamaxOptimizer', 'DecayedAdagradOptimizer', 'RMSPropOptimizer',
    'FtrlOptimizer', 'Adadelta', 'ModelAverage', 'RMSPropOptimizer'
]


class Optimizer(object):
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    def __init__(self,
                 learning_rate,
                 regularization=None,
                 LARS_weight_decay=0.0):
        if not isinstance(learning_rate, float) and \
                not isinstance(learning_rate, framework.Variable):
            raise TypeError("learning rate should be float or Variable")
        self.regularization = regularization
        self._learning_rate = learning_rate
        # the learning rate type should be inferenced from loss
        self._dtype = None
        # each program should have a independent learning rate
        # program -> Variable(learning_rate)
        self._learning_rate_map = dict()
        if isinstance(self._learning_rate, framework.Variable):
            self._learning_rate_map[framework.default_main_program(
            )] = self._learning_rate
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra variables associated with the parameters
        # to train. These variables are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())
        self.helper = None
        self._LARS_weight_decay = LARS_weight_decay

    def _create_global_learning_rate(self):
        lr = self._global_learning_rate()

        if isinstance(lr, framework.Variable):
            return
        else:
            if not isinstance(self._learning_rate, float):
                raise TypeError(
                    "learning rate variable is create outside optimizer,"
                    "can not create new learning rate variable for new program")

        # create learning rate in the current main program
        self._learning_rate_map[framework.default_main_program(
        )] = layers.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(self._learning_rate),
            dtype='float32' if self._dtype == None else self._dtype,
            persistable=True)

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
        raise NotImplementedError()

    def _create_param_lr(self, param_and_grad):
        # create learning rate variable for every parameter
        param = param_and_grad[0]
        param_lr = param.optimize_attr['learning_rate']
        if type(param_lr) == Variable:
            # param learning rate has been updated (LARS)
            print(("returns updated param lr ", param_lr))
            return param_lr
        else:
            if param_lr == 1.0:
                return self._global_learning_rate()
            else:
                return self._global_learning_rate() * param_lr

    def _create_accumulators(self, block, parameters):
        """Create all accumulators needed by the parameters

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer
        """
        pass

    def _finish_update(self, block):
        """Finish any custom updates needed
           before completing an optimization step

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer

        Returns:
            list of finish ops or None
        """
        pass

    def _add_accumulator(self,
                         name,
                         param,
                         dtype=None,
                         fill_value=0.0,
                         shape=None):
        """Utility function to add an accumulator for a parameter

        Args:
            block: the block in which the loss variable is present
            name: name of the accumulator
            param: parameter variable for which accumulator is to be added
            dtype: data type of the accumulator variable
            fill_value: value to initialize the accumulator variable
        """
        if (name in self._accumulators and
                param.name in self._accumulators[name]):
            raise Exception("Accumulator {} already exists for parameter {}".
                            format(name, param.name))
        if shape == None:
            shape = param.shape
        assert isinstance(self.helper, LayerHelper)
        var = self.helper.create_global_variable(
            name=unique_name.generate(name),
            persistable=True,
            dtype=dtype or param.dtype,
            type=param.type,
            shape=shape)
        self.helper.set_variable_initializer(
            var, initializer=Constant(value=float(fill_value)))
        self._accumulators[name][param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched

        Returns:
            accumulator variable for the parameter
        """
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def _create_optimization_pass(self,
                                  parameters_and_grads,
                                  loss,
                                  startup_program=None):
        """Add optimization operators to update gradients to variables.

        Args:
          loss(Variable): the target that this optimization is for.
          parameters_and_grads(list(tuple(Variable, Variable))):
          a list of (variable, gradient) pair to update.

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

        # Create any accumulators
        program = loss.block.program
        self._dtype = loss.dtype
        with program_guard(program, startup_program):
            global_block = framework.default_main_program().global_block()
            start = len(global_block.ops)
            self.helper = LayerHelper(self.__class__.__name__)
            self._create_accumulators(loss.block,
                                      [p[0] for p in parameters_and_grads])
            self._create_global_learning_rate()
            if self._LARS_weight_decay > 0.0:
                layers.append_LARS(parameters_and_grads,
                                   self._global_learning_rate(),
                                   self._LARS_weight_decay)

            optimize_ops = []
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                with param_and_grad[0].block.program.optimized_guard(
                        param_and_grad):
                    if param_and_grad[0].trainable is True:
                        optimize_op = self._append_optimize_op(loss.block,
                                                               param_and_grad)
                        optimize_ops.append(optimize_op)

            # Get custom finish ops for subclasses
            # FIXME: Need to fix this once we figure out how to handle dependencies
            self._finish_update(loss.block)

            end = len(global_block.ops)
            return global_block._slice_ops(start, end)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `append_backward()` and
        `create_optimization_pass()` into one.
        """
        params_grads = append_backward(loss, parameter_list, no_grad_set,
                                       [error_clip_callback])

        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = append_regularization_ops(params_grads,
                                                 self.regularization)

        optimize_ops = self._create_optimization_pass(params_grads, loss,
                                                      startup_program)
        return optimize_ops, params_grads


class SGDOptimizer(Optimizer):
    """
    Optimizer of the stochastic gradient descent algorithm.

    .. math::

        param\_out = param - learning\_rate * grad

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.

    Examples:
        .. code-block:: python

            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.2)
            sgd_optimizer.minimize(cost)
    """

    def __init__(self, learning_rate, **kwargs):
        assert learning_rate is not None
        super(SGDOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "sgd"

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        # create the optimize op
        sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0]})

        return sgd_op


class MomentumOptimizer(Optimizer):
    """

    Simple Momentum optimizer with velocity state

    This optimizer has a flag for Nestrov Momentum.

    The update equations are as follows:

    .. math::

        & velocity = mu * velocity + gradient

        & if (use\_nesterov):

        &\quad   param = param - (gradient + mu * velocity) * learning\_rate

        & else:

        &\quad   param = param - learning\_rate * velocity

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        momentum (float): momentum factor
        use_nesterov (bool): enables Nesterov momentum

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
            optimizer.minimize(cost)
    """
    _velocity_acc_str = "velocity"

    def __init__(self, learning_rate, momentum, use_nesterov=False, **kwargs):
        assert learning_rate is not None
        assert momentum is not None
        super(MomentumOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Velocity": velocity_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "VelocityOut": velocity_acc
            },
            attrs={"mu": self._momentum,
                   "use_nesterov": self._use_nesterov})

        return momentum_op


class AdagradOptimizer(Optimizer):
    """
    **Adaptive Gradient Algorithm (Adagrad)**

    The update is done as follows:

    .. math::

        moment\_out &= moment + grad * grad

        param\_out &= param - \\frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

    The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    does not have the epsilon attribute. It is added here in our implementation
    as also proposed here: http://cs231n.github.io/neural-networks-3/#ada
    for numerical stability to avoid the division by zero error.

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        epsilon (float): a small float value for numerical stability.

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Adagrad(learning_rate=0.2)
            optimizer.minimize(cost)
    """
    _moment_acc_str = "moment"

    def __init__(self, learning_rate, epsilon=1.0e-6, **kwargs):
        assert learning_rate is not None
        assert epsilon is not None
        super(AdagradOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "adagrad"
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])

        # Create the adagrad optimizer op
        adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0],
                     "MomentOut": moment_acc},
            attrs={"epsilon": self._epsilon})

        return adagrad_op


class AdamOptimizer(Optimizer):
    """
    This implements the Adam optimizer from Section 2 of the Adam
    paper : https://arxiv.org/abs/1412.6980.
    Adam is a first-order gradient-based optimization method based on
    adaptive estimates of lower-order moments.

    Adam updates:

    .. math::

        t & = t + 1

        moment\_1\_out & = {\\beta}_1 * moment\_1 + (1 - {\\beta}_1) * grad

        moment\_2\_out & = {\\beta}_2 * moment\_2 + (1 - {\\beta}_2) * grad * grad

        learning\_rate & = learning\_rate * \\
                          \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {\\beta}_1^t}

        param\_out & = param - learning\_rate * \\frac{moment\_1}{\sqrt{moment\_2} + \epsilon}

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        beta1 (float): The exponential decay rate for the 1st moment estimates.
        beta2 (float): The exponential decay rate for the 2nd moment estimates.
        epsilon (float): a small float value for numerical stability.

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Adam(learning_rate=0.2)
            optimizer.minimize(cost)

    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 **kwargs):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        main_block = block.program.global_block()
        # Create beta1 and beta2 power tensors
        beta_shape = [1]
        self._beta1_pow_acc = self.helper.create_global_variable(
            name=unique_name.generate('beta1_pow_acc'),
            dtype='float32' if self._dtype == None else self._dtype,
            shape=beta_shape,
            lod_level=0,
            persistable=True)
        self.helper.set_variable_initializer(
            self._beta1_pow_acc, initializer=Constant(self._beta1))

        self._beta2_pow_acc = self.helper.create_global_variable(
            name=unique_name.generate('beta2_pow_acc'),
            dtype='float32' if self._dtype == None else self._dtype,
            shape=beta_shape,
            lod_level=0,
            persistable=True)

        self.helper.set_variable_initializer(
            self._beta2_pow_acc, initializer=Constant(self._beta2))

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        # create the adam optimize op
        adam_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment1": moment1,
                "Moment2": moment2,
                "Beta1Pow": self._beta1_pow_acc,
                "Beta2Pow": self._beta2_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "Moment1Out": moment1,
                "Moment2Out": moment2
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon
            })

        return adam_op

    def _finish_update(self, block):
        """Update Beta1 and Beta2 Power accumulators
        """
        assert isinstance(block, framework.Block)
        main_block = block.program.global_block()
        scale_beta1 = main_block.append_op(
            type="scale",
            inputs={"X": self._beta1_pow_acc},
            outputs={"Out": self._beta1_pow_acc},
            attrs={"scale": self._beta1})

        scale_beta2 = main_block.append_op(
            type="scale",
            inputs={"X": self._beta2_pow_acc},
            outputs={"Out": self._beta2_pow_acc},
            attrs={"scale": self._beta2})

        return [scale_beta1, scale_beta2]


class AdamaxOptimizer(Optimizer):
    """
    We implement the Adamax optimizer from Section 7 of the Adam
    paper: https://arxiv.org/abs/1412.6980. Adamax is a variant of the
    Adam algorithm based on the infinity norm.

    Adamax updates:

    .. math::

        t & = t + 1

        moment\_out & = {\\beta}_1 * moment + (1 - {\\beta}_1) * grad

        inf\_norm\_out & = max({\\beta}_2 * inf\_norm + \epsilon, |grad|)

        learning\_rate & = \\frac{learning\_rate}{1 - {\\beta}_1^t}

        param\_out & = param - learning\_rate * \\frac{moment\_out}{inf\_norm\_out}


    The original paper does not have an epsilon attribute.
    However, it is added here for numerical stability to prevent the
    division by 0 error.

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        beta1 (float): The exponential decay rate for the 1st moment estimates.
        beta2 (float): The exponential decay rate for the 2nd moment estimates.
        epsilon (float): a small float value for numerical stability.

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Adamax(learning_rate=0.2)
            optimizer.minimize(cost)
    """
    _moment_acc_str = "moment"
    _inf_norm_acc_str = "inf_norm"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 **kwargs):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamaxOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "adamax"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        # Create beta1 power accumulator tensor
        beta_shape = [1]
        self._beta1_pow_acc = self.helper.create_global_variable(
            name=unique_name.generate('beta1_pow_acc'),
            dtype='float32' if self._dtype == None else self._dtype,
            shape=beta_shape,
            lod_level=0,
            persistable=True)
        self.helper.set_variable_initializer(
            self._beta1_pow_acc, initializer=Constant(self._beta1))

        # Create accumulator tensors for first moment and infinity norm
        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)
            self._add_accumulator(self._inf_norm_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment = self._get_accumulator(self._moment_acc_str, param_and_grad[0])
        inf_norm = self._get_accumulator(self._inf_norm_acc_str,
                                         param_and_grad[0])
        # create the adamax optimize op
        adamax_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment": moment,
                "InfNorm": inf_norm,
                "Beta1Pow": self._beta1_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": moment,
                "InfNormOut": inf_norm
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon
            })

        return adamax_op

    def _finish_update(self, block):
        """Update Beta1 Power accumulator
        """
        assert isinstance(block, framework.Block)
        main_block = block.program.global_block()
        scale_beta1 = main_block.append_op(
            type="scale",
            inputs={"X": self._beta1_pow_acc},
            outputs={"Out": self._beta1_pow_acc},
            attrs={"scale": self._beta1})

        return [scale_beta1]


class DecayedAdagradOptimizer(Optimizer):
    """
    **Decayed Adagrad Optimizer**

    The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

    The update is done as follows:

    .. math::

        moment\_out & = decay * moment + (1 - decay) * grad * grad

        param\_out & = param - \\frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

    The original paper(http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    does not have an epsilon attribute. It is added here for numerical
    stability to avoid the division by zero error.

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        decay (float): decay rate.
        epsilon (float): a small float value for numerical stability.

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.DecayedAdagrad(learning_rate=0.2)
            optimizer.minimize(cost)
    """
    _moment_acc_str = "moment"

    def __init__(self, learning_rate, decay=0.95, epsilon=1.0e-6, **kwargs):
        assert learning_rate is not None
        assert decay is not None
        assert epsilon is not None

        super(DecayedAdagradOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "decayed_adagrad"
        self._decay = decay
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])

        # Create the decayed adagrad optimizer op
        decayed_adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0],
                     "MomentOut": moment_acc},
            attrs={"epsilon": self._epsilon})

        return decayed_adagrad_op


class AdadeltaOptimizer(Optimizer):
    """
    **Adadelta Optimizer**

    Simple Adadelta optimizer with average squared grad state and
    average squared update state.
    The details of adadelta please refer to this
    `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
    <http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf>`_.

    ..  math::

        E(g_t^2) &= \\rho * E(g_{t-1}^2) + (1-\\rho) * g^2 \\\\
        learning\\_rate &= sqrt( ( E(dx_{t-1}^2) + \\epsilon ) / ( \\
                          E(g_t^2) + \\epsilon ) ) \\\\
        E(dx_t^2) &= \\rho * E(dx_{t-1}^2) + (1-\\rho) * (-g*learning\\_rate)^2

    Args:
        learning_rate(float): global learning rate
        rho(float): rho in equation
        epsilon(float): epsilon in equation

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Adadelta(
                learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)
            _, params_grads = optimizer.minimize(cost)
    """

    _avg_squared_grad_acc_str = "_avg_squared_grad"
    _avg_squared_update_acc_str = "_avg_squared_update"

    def __init__(self, learning_rate, epsilon=1.0e-6, rho=0.95, **kwargs):
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        super(AdadeltaOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "adadelta"
        self._epsilon = epsilon
        self._rho = rho

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._avg_squared_grad_acc_str, p)
            self._add_accumulator(self._avg_squared_update_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        avg_squared_grad_acc = self._get_accumulator(
            self._avg_squared_grad_acc_str, param_and_grad[0])
        avg_squared_update_acc = self._get_accumulator(
            self._avg_squared_update_acc_str, param_and_grad[0])

        # Create the adadelta optimizer op
        adadelta_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "AvgSquaredGrad": avg_squared_grad_acc,
                "AvgSquaredUpdate": avg_squared_update_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "AvgSquaredGradOut": avg_squared_grad_acc,
                "AvgSquaredUpdateOut": avg_squared_update_acc
            },
            attrs={"epsilon": self._epsilon,
                   "rho": self._rho})

        return adadelta_op


class RMSPropOptimizer(Optimizer):
    """
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
    rate method. The original slides proposed RMSProp: Slide 29 of
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .

    The original equation is as follows:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        w & = w - \\frac{\\eta} {\\sqrt{r(w,t) + \\epsilon}} \\nabla Q_{i}(w)

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`sqrt{v(w,t)}`.

    In some cases, adding a momentum term :math: `\\beta` is beneficial.
    In our implementation, Nesterov momentum is used:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        v(w, t) & = \\beta v(w, t-1) + \\frac{\\eta} {\\sqrt{v(w,t) +
            \\epsilon}} \\nabla Q_{i}(w)

        w & = w - v(w, t)

    where, :math:`\\rho` is a hyperparameter and typical values are 0.9, 0.95
    and so on. :math: `beta` is the momentum term. :math: `\\epsilon` is a
    smoothing term to avoid division by zero, usually set somewhere in range
    from 1e-4 to 1e-8.


    Args:
        learning_rate(float): global learning rate.
        rho(float): rho is :math: `\\rho` in equation, set 0.95 by default.
        epsilon(float): :math: `\\epsilon` in equation is smoothing term to
            avoid division by zero, set 1e-6 by default.
        momentum(float): :math:`\\beta` in equation is the momentum term,
            set 0.0 by default.

    Raises:
        ValueError: If learning_rate, rho, epsilon, momentum are None.

    Examples:
          .. code-block:: python

              optimizer = fluid.optimizer.RMSProp(0.0001)
              _, params_grads = optimizer.minimize(cost)
    """

    _momentum_acc_str = "momentum"
    _mean_square_acc_str = "mean_square"

    def __init__(self,
                 learning_rate,
                 rho=0.95,
                 epsilon=1.0e-6,
                 momentum=0.0,
                 **kwargs):
        super(RMSPropOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if momentum is None:
            raise ValueError("momentum is not set.")

        self.type = "rmsprop"
        self._rho = rho
        self._epsilon = epsilon
        self._momentum = momentum

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._momentum_acc_str, p)
            self._add_accumulator(self._mean_square_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        momentum_acc = self._get_accumulator(self._momentum_acc_str,
                                             param_and_grad[0])
        mean_square_acc = self._get_accumulator(self._mean_square_acc_str,
                                                param_and_grad[0])
        rmsprop_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": momentum_acc,
                "MeanSquare": mean_square_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": momentum_acc,
                "MeanSquareOut": mean_square_acc
            },
            attrs={
                "epsilon": self._epsilon,
                "decay": self._rho,
                "momentum": self._momentum
            })

        return rmsprop_op


class FtrlOptimizer(Optimizer):
    """
    FTRL (Follow The Regularized Leader) Optimizer.

    The paper that proposed Follow The Regularized Leader (FTRL):
    (https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

    ..  math::

        &new\_accum = squared\_accum + grad^2

        &if (lr\_power == -0.5):

        &\quad  linear\_accum += grad - \\frac{\\sqrt{new\_accum} - \\sqrt{squared\_accum}}{learning\_rate * param}

        &else:

        &\quad   linear\_accum += grad - \\frac{new\_accum^{-lr\_power} - accum^{-lr\_power}}{learning\_rate * param}


        &x = l1 * sign(linear\_accum) - linear\_accum

        &if (lr\_power == -0.5):

        &\quad   y = \\frac{\\sqrt{new\_accum}}{learning\_rate} + (2 * l2)

        &\quad   pre\_shrink = \\frac{x}{y}

        &\quad   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0)

        &else:

        &\quad   y = \\frac{new\_accum^{-lr\_power}}{learning\_rate} + (2 * l2)

        &\quad   pre\_shrink = \\frac{x}{y}

        &\quad   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0)

        &squared\_accum += grad^2

    Args:
        learning_rate (float|Variable): global learning rate.
        l1 (float):
        l2 (float):
        lr_power (float):

    Raises:
        ValueError: If learning_rate, rho, epsilon, momentum are None.

    Examples:
          .. code-block:: python

              optimizer = fluid.optimizer.Ftrl(0.0001)
              _, params_grads = optimizer.minimize(cost)
    """

    _squared_acc_str = "squared"
    _linear_acc_str = "linear"

    def __init__(self, learning_rate, l1=0.0, l2=0.0, lr_power=-0.5, **kwargs):
        super(FtrlOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")

        self.type = "ftrl"
        self._l1 = l1
        self._l2 = l2
        self._lr_power = lr_power

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._squared_acc_str, p)
            self._add_accumulator(self._linear_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        squared_acc = self._get_accumulator(self._squared_acc_str,
                                            param_and_grad[0])
        linear_acc = self._get_accumulator(self._linear_acc_str,
                                           param_and_grad[0])
        ftrl_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "SquaredAccumulator": squared_acc,
                "LinearAccumulator": linear_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "SquaredAccumOut": squared_acc,
                "LinearAccumOut": linear_acc
            },
            attrs={"l1": self._l1,
                   "l2": self._l1,
                   "lr_power": self._lr_power})

        return ftrl_op


# We short the class name, since users will use the optimizer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# sgd = fluid.optimizer.SGD(...)
#
# It is no need to add an `Optimizer` as the class suffix
SGD = SGDOptimizer
Momentum = MomentumOptimizer
Adagrad = AdagradOptimizer
Adam = AdamOptimizer
Adamax = AdamaxOptimizer
DecayedAdagrad = DecayedAdagradOptimizer
Adadelta = AdadeltaOptimizer
RMSProp = RMSPropOptimizer
Ftrl = FtrlOptimizer


class ModelAverage(Optimizer):
    """Accumulate the average of parameters whtin sliding window. The average
    result will be saved in temporary variables which can be applied to
    parameter variables of current model by calling 'apply()' method. And the
    'restore()' method is used to restored the parameter values of current model.

    The size of average window is determined by average_window_rate,
    min_average_window, max_average_window and current update times.

    Args:
        average_window_rate: The rate of average window.
        min_average_window: The minimum size of average window.
        max_average_window: The maximum size of average window.

    Examples:

      .. code-block:: python

        optimizer = fluid.optimizer.Momentum()
        optimizer.minimize(cost)
        model_average = fluid.optimizer.ModelAverage(0.15,
                                                min_average_window=10000,
                                                max_average_window=20000)
        for pass_id in range(args.pass_num):
            for data in train_reader():
                exe.run(fluid.default_main_program()...)

            with model_average.apply(exe):
                for data in test_reader():
                    exe.run(inference_program...)
    """

    def __init__(self,
                 average_window_rate,
                 min_average_window=10000,
                 max_average_window=10000,
                 **kwargs):
        super(ModelAverage, self).__init__(0.0, **kwargs)
        self.average_window = average_window_rate
        self.min_average_window = min_average_window
        self.max_average_window = max_average_window

        self.params_grads = []
        for param in framework.default_main_program().global_block(
        ).all_parameters():
            if param.do_model_average != False:
                grad = param.block.create_var(
                    name=unique_name.generate(".".join([param.name, 'tmp'])),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True)
                self.params_grads.append((param, grad))

        for param, grad in self.params_grads:
            self._append_average_accumulate_op(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            for param_grad in self.params_grads:
                self._add_average_apply_op(block, param_grad)

        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param_grad in self.params_grads:
                self._add_average_restore_op(block, param_grad)

    def _add_average_apply_op(self, block, param_grad):
        param = block._clone_variable(param_grad[0])
        grad = block._clone_variable(param_grad[1])
        sum_1 = block._clone_variable(self._get_accumulator('sum_1', param))
        sum_2 = block._clone_variable(self._get_accumulator('sum_2', param))
        sum_3 = block._clone_variable(self._get_accumulator('sum_3', param))
        num_accumulates = block._clone_variable(
            self._get_accumulator('num_accumulates', param))
        old_num_accumulates = block._clone_variable(
            self._get_accumulator('old_num_accumulates', param))
        num_updates = block._clone_variable(
            self._get_accumulator('num_updates', param))
        # backup param value to grad
        layers.assign(input=param, output=grad)
        # param = (sum_1 + sum_2 + sum_3) / (num_accumulates + old_num_accumulates)
        tmp = layers.sum(x=[num_accumulates, old_num_accumulates])
        sum = layers.sum(x=[sum_1, sum_2, sum_3])
        tmp = layers.cast(
            x=tmp, dtype='float32' if self._dtype == None else self._dtype)
        sum = layers.cast(
            x=sum, dtype='float32' if self._dtype == None else self._dtype)
        layers.elementwise_div(x=sum, y=tmp, out=param)

    def _add_average_restore_op(self, block, param_grad):
        param = block._clone_variable(param_grad[0])
        grad = block._clone_variable(param_grad[1])
        layers.assign(input=grad, output=param)

    def _append_average_accumulate_op(self, param):
        self.helper = LayerHelper("average_accumulate")
        sum_1 = self._add_accumulator('sum_1', param)
        sum_2 = self._add_accumulator('sum_2', param)
        sum_3 = self._add_accumulator('sum_3', param)
        num_accumulates = self._add_accumulator(
            'num_accumulates', param, dtype='int64', shape=[1])
        old_num_accumulates = self._add_accumulator(
            'old_num_accumulates', param, dtype='int64', shape=[1])
        num_updates = self._add_accumulator(
            'num_updates', param, dtype='int64', shape=[1])

        self.helper.append_op(
            type='average_accumulates',
            inputs={
                "param": param,
                "in_sum_1": sum_1,
                "in_sum_2": sum_2,
                "in_sum_3": sum_3,
                "in_num_accumulates": num_accumulates,
                "in_old_num_accumulates": old_num_accumulates,
                "in_num_updates": num_updates
            },
            outputs={
                "out_sum_1": sum_1,
                "out_sum_2": sum_2,
                "out_sum_3": sum_3,
                "out_num_accumulates": num_accumulates,
                "out_old_num_accumulates": old_num_accumulates,
                "out_num_updates": num_updates,
            },
            attrs={
                "average_window": self.average_window,
                "min_average_window": self.min_average_window,
                "max_average_window": self.max_average_window,
            })

    @contextmanager
    def apply(self, executor, need_restore=True):
        """Apply average values to parameters of current model.
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """Restore parameter values of current model.
        """
        executor.run(self.restore_program)
