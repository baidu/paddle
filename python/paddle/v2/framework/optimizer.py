import paddle.v2.framework.framework as framework

__all__ = ['SGDOptimizer']


def grad_var_name(name):
    return name + "@GRAD"


class Optimizer(object):
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly, but need to use one of it's implementation.
    """

    def __init__(self):
        pass

    def _append_optimize_op(self, block, param_and_grad):
        """ append optimize operator to block and return all the added optimize_op
        """
        raise NotImplementedError()

    def create_backward_pass(self, loss, parameter_list=None, no_grad_set=None):
        """
        create and add gradient Operators in BlockDesc to Compute gradients of `loss`
        for parameters in parameter_list

        Args:
          loss: an variable generated by cost function.
          no_grad_set: variable that should not create gradient
          parameter_list: parameters that need to compute gradient and update to optimize the lost.

        Returns:
          list of (parameters, gradients) pair.
        """

        assert isinstance(loss, framework.Variable)
        loss.block.program.append_backward(loss, no_grad_set or set())
        if parameter_list is not None:
            parameters = parameter_list
        else:
            parameters = loss.block.program.parameters
        params_and_grads = []
        for param in parameters:
            grad = grad_var_name(param)
            if loss.block.has_var(grad):
                params_and_grads.append((param, grad))
            else:
                params_and_grads.append((param, None))
        return params_and_grads

    def create_optimization_pass(self, parameters_and_grads, loss):
        """Add optimization operators to update gradients to variables.

        Args:
          loss: the target that this optimization is for.
          parameters_and_grads: a list of (variable, gradient) pair to update.

        Returns:
          optmization_op_list: a list of optimization operator that will update parameter using gradient.
        """
        optimize_ops = []
        for param_and_grad in parameters_and_grads:
            if param_and_grad[2] is not None:
                optimize_op = self._append_optimize_op(loss.block,
                                                       param_and_grad)
                optimize_ops.append(optimize_op)
        return optimize_ops

    def minimize(self, loss, parameter_list=None, no_grad_set=None):
        """Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `create_backward_pass()` and
        `create_optimization_pass()` into one.
        """
        params_grads = self.create_backward_pass(loss, parameter_list,
                                                 no_grad_set or set())
        optimize_ops = self.create_optimization_pass(params_grads, loss)
        return optimize_ops


class SGDOptimizer(Optimizer):
    """ Simple SGD optimizer without any state.
    """

    def __init__(self, learning_rate):
        assert learning_rate is not None
        super(Optimizer, self).__init__()
        self.type = "sgd"
        self._learning_rate = learning_rate

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        lr_shape = [1]
        # create a var for learning_rate
        lr = block.create_var(dtype="float32", shape=lr_shape, lod_level=0)

        # create an op to init the learning_rate
        init_op = block.append_op(
            type="fill_constant",
            outputs={"Out": lr.name},
            attrs={"shape": lr_shape,
                   "value": self._learning_rate})

        # create the optimize op
        sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param", param_and_grad[0], "Grad", param_and_grad[1],
                "LearningRate", lr.name()
            },
            outputs={"Out", param_and_grad[0]},
            attrs={"shape": [1],
                   "value": self._learning_rate})

        return sgd_op
