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

from . import framework
from . import core

__all__ = ['L1Decay', 'L2Decay', 'L1DecayRegularizer', 'L2DecayRegularizer']


def append_regularization_ops(parameters_and_grads, regularization=None):
    """Create and add backward regularization Operators

    Creates and adds backward regularization operators in the BlockDesc.
    This will add gradients of the regularizer function to the gradients
    of the parameters and return these modified gradients. This is the
    same as implementing weight decay in optimizers for regularization.

    Args:
        parameters_and_grads: A list of (parameters, gradients) pairs
                              that need to be regularized.
        regularization: A global regularizer. If the parameter is not
                        set. It will be applied with regularizer.

    Returns:
        list[(Variable, Variable)]: list of (parameters, gradients) \
        pair with the regularized gradient

    Raises:
        Exception: Unknown regularization type
    """
    params_and_grads = []
    for param, grad in parameters_and_grads:
        # If no gradient then we don't need to do anything
        if grad is None:
            params_and_grads.append((param, grad))
            continue
        with param.block.program.optimized_guard([param, grad]):
            regularization_term = None
            if param.regularizer is not None:
                # Add variable for regularization term in grad block
                regularization_term = param.regularizer(param, grad, grad.block)
            elif regularization is not None:
                regularization_term = regularization(param, grad, grad.block)

            # If no regularization specified, then we don't need to do anything
            if regularization_term is None:
                params_and_grads.append((param, grad))
                continue

            assert grad.shape == regularization_term.shape

            grad.block.append_op(
                type='elementwise_add',
                inputs={"X": grad,
                        "Y": regularization_term},
                outputs={"Out": grad})
            params_and_grads.append((param, grad))

    return params_and_grads


class WeightDecayRegularizer(object):
    """Base class for weight decay regularizers

    Defines the common interface of weight-decay regularizers.
    Weight-decay regularizers are added only during the backward
    pass for faster regularization. They add operations to the network
    that correspond to gradient of the regularization function.
    Users should not use this class directly, but need to use one
    of its implementations
    """

    def __init__(self):
        pass

    def __call__(self, param, grad, block):
        """Add corresponding weight decay operations to the network
        """
        raise NotImplementedError()

    def __str__(self):
        """Debug string
        """
        raise NotImplementedError()


class L2DecayRegularizer(WeightDecayRegularizer):
    """Implements the L2 Weight Decay Regularization

    Small values of L2 can help prevent over fitting the training data.

    .. math::

        L2WeightDecay = reg\_coeff * parameter

    Args:
        regularization_coeff(float): regularization coeff

    Examples:
        .. code-block:: python

            optimizer = fluid.optimizer.Adagrad(
                learning_rate=1e-4,
                regularization=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_cost)
    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L2DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

    def __call__(self, param, grad, block):
        """Add L2 weight decay ops to network

        Adds L2 weight decay ops.
        L2WeightDecay = reg_coeff * parameter

        Args:
            param: parameter variable for which regularization is applied
            block: block in which variable is to be created

        Returns:
            new variable for weight decay
        """
        assert isinstance(param, framework.Parameter)
        assert isinstance(block, framework.Block)

        decay = block.create_var(
            dtype="float32", shape=param.shape, lod_level=param.lod_level)

        if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
            decay = block.create_var(
                dtype="float32",
                shape=param.shape,
                type=core.VarDesc.VarType.SELECTED_ROWS)
            block.append_op(
                type='lookup_table',
                inputs={'W': param,
                        'Ids': grad},
                outputs={'Out': decay},
                attrs={'is_sparse': True})
            param = decay

        # Append Op to calculate decay
        block.append_op(
            type='scale',
            inputs={"X": param},
            outputs={"Out": decay},
            attrs={"scale": self._regularization_coeff})

        return decay

    def __str__(self):
        return "L2Decay, regularization_coeff=%f" % self._regularization_coeff


class L1DecayRegularizer(WeightDecayRegularizer):
    """Implements the L1 Weight Decay Regularization

    L1 regularization encourages sparsity.

    .. math::

        L1WeightDecay = reg\_coeff * sign(parameter)

    Args:
        regularization_coeff(float): regularization coeff

    Examples:
        .. code-block:: python

            program = fluid.framework.Program()
            block = program.global_block()
            mul_x = block.create_parameter(
                dtype="float32",
                shape=[5, 10],
                lod_level=0,
                name="mul.x",
                regularizer=fluid.regularizer.L1DecayRegularizer(0.5))
    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L1DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

    def __call__(self, param, grad, block):
        """Add L1 weight decay ops to network

        Adds L1 weight decay ops.
        L1WeightDecay = reg_coeff * sign(parameter)

        Args:
            param: parameter variable for which regularization is applied
            block: block in which variable is to be created

        Returns:
            new variable for weight decay
        """
        assert isinstance(param, framework.Parameter)
        assert isinstance(block, framework.Block)
        decay = block.create_var(
            dtype="float32", shape=param.shape, lod_level=param.lod_level)

        if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
            decay = block.create_var(
                dtype="float32",
                shape=param.shape,
                type=core.VarDesc.VarType.SELECTED_ROWS)
            block.append_op(
                type='lookup_table',
                inputs={'W': param,
                        'Ids': grad},
                outputs={'Out': decay},
                attrs={'is_sparse': True})

        # Append sign op
        block.append_op(
            type='sign', inputs={"X": param}, outputs={"Out": decay})

        # Append scale op to the output of sign op
        block.append_op(
            type='scale',
            inputs={"X": decay},
            outputs={"Out": decay},
            attrs={"scale": self._regularization_coeff})

        return decay

    def __str__(self):
        return "L1Decay, regularization_coeff=%f" % self._regularization_coeff


# We short the class name, since users will use the regulaizer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# hidden = fluid.layers.fc(...,
#                          param_attr=fluid.regularizer.Xavier())
#
# It is no need to add a `Regularizer` as the class suffix
L1Decay = L1DecayRegularizer
L2Decay = L2DecayRegularizer
