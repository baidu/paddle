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

from .initializer import Initializer, Xavier, Constant
from .regularizer import WeightDecayRegularizer

__all__ = [
    'ParamAttr',
    'WeightNormParamAttr',
]


class ParamAttr(object):
    """
    Parameter attributes object. To fine-tuning network training process, user
    can set parameter's attributes to control training details. Such as learning rate,
    regularization, trainable, do_model_average and the method to initialize param.


    Args:
        name(str): The parameter's name. Default None.
        initializer(Initializer): The method to initial this parameter. Default None.
        learning_rate(float): The parameter's learning rate. The learning rate when
            optimize is :math:`global\_lr * parameter\_lr * scheduler\_factor`.
            Default 1.0.
        regularizer(WeightDecayRegularizer): Regularization factor. Default None.
        trainable(bool): Whether this parameter is trainable. Default True.
        gradient_clip(BaseGradientClipAttr): The method to clip this parameter's
            gradient. Default None.
        do_model_average(bool): Whether this parameter should do model average.
            Default False.

    Examples:
        .. code-block:: python

            w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                            learning_rate=0.5,
                                            regularizer=fluid.L2Decay(1.0),
                                            trainable=True)
            y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)
    """

    def __init__(self,
                 name=None,
                 initializer=None,
                 learning_rate=1.0,
                 regularizer=None,
                 trainable=True,
                 gradient_clip=None,
                 do_model_average=False):
        self.name = name
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.trainable = trainable
        self.gradient_clip = gradient_clip
        self.model_average = do_model_average

    def _set_default_initializer(self, initializer):
        """
        Set the default initializer, the initializer should be Constant,
        Uniform, Normal, Xavier, MSRA.

        Args:
            initializer(Initializer): the initializer to set.

        Returns:
            None
        """
        if initializer is None:
            if self.initializer is None:
                raise ValueError("ParamAttr.initializer is not set")
            return

        if self.initializer is not None:
            return

        self.initializer = initializer

    def _set_default_param_initializer(self):
        """
        Set the default initializer for the parameter with Xavier.

        Args:
            None.

        Returns:
            None.
        """
        self._set_default_initializer(Xavier())

    def _set_default_bias_initializer(self):
        """
        Set the default initializer for the bias with Constant(0.0).

        Args:
            None.

        Returns:
            None.
        """
        self._set_default_initializer(Constant(0.0))

    @staticmethod
    def _to_attr(arg):
        """
        Create ParamAttr[s].

        Args:
            arg: Arguments to initialize ParamAttr[s]. arg's type can be
                str, Initializer, float, WeightDecayRegularizer, BaseGradientClipAttr,
                bool, ParamAttr, or a list of above type.

        Returns:
            ParamAttr[s]: ParamAttr[s] initialized with arg.

        Raises:
            arg can not initialize a ParamAttr.
        """
        if arg is None:
            return ParamAttr()
        elif isinstance(arg, list) or isinstance(arg, tuple):
            return [ParamAttr._to_attr(a) for a in arg]
        elif isinstance(arg, ParamAttr):
            return arg
        elif isinstance(arg, str) or isinstance(arg, str):
            return ParamAttr(name=arg)
        elif isinstance(arg, Initializer):
            return ParamAttr(initializer=arg)
        elif isinstance(arg, WeightDecayRegularizer):
            return ParamAttr(regularizer=arg)
        elif isinstance(arg, bool):
            return ParamAttr._to_attr(None) if arg else False
        else:
            raise TypeError("{0} cast to ParamAttr".format(type(arg)))

    def _to_kwargs(self, with_initializer=False):
        """
        Returns the attributes of this parameter.

        Args:
            with_initializer(bool): Whether to add initializer attr.

        Returns:
            Parameter attributes(map): The attributes of this parameter.
        """
        kwargs = {
            'name': self.name,
            'optimize_attr': {
                'learning_rate': self.learning_rate
            },
            'regularizer': self.regularizer,
            'trainable': self.trainable,
            'gradient_clip_attr': self.gradient_clip,
            'model_average': self.model_average
        }
        if with_initializer:
            kwargs['initializer'] = self.initializer
        return kwargs


class WeightNormParamAttr(ParamAttr):
    """
    Used for weight Norm. Weight Norm is a reparameterization of the weight vectors
    in a neural network that decouples the length of those weight vectors from
    their direction. Weight Norm has been implemented as discussed in this
    paper: `Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks
    <https://arxiv.org/pdf/1602.07868.pdf>`_.

    Args:
        dim(list): The parameter's name. Default None.
        kwargs: Any field in ParamAttr. Default None.

    Examples:
        .. code-block:: python

            data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
            fc = fluid.layers.fc(input=data,
                                 size=1000,
                                 param_attr=WeightNormParamAttr(
                                      dim=None,
                                      name='weight_norm_param'))

    """
    # List to record the parameters reparameterized by weight normalization.
    # If these parameters are treated as Variable rather than Parameter,
    # it can be used to discriminate these parameters and help to serialize
    # these paramters for inference.
    params_with_weight_norm = []

    def __init__(self, dim=None, **kwargs):
        super(WeightNormParamAttr, self).__init__(**kwargs)
        self.dim = dim
