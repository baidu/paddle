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
from .layer_function_generator import generate_layer_fn

__activations__ = [
    'sigmoid',
    'logsigmoid',
    'exp',
    'tanh',
    'tanh_shrink',
    'softshrink',
    'sqrt',
    'abs',
    'ceil',
    'floor',
    'cos',
    'sin',
    'round',
    'reciprocal',
    'square',
    'softplus',
    'softsign',
    'brelu',
    'leaky_relu',
    'soft_relu',
    'elu',
    'relu6',
    'pow',
    'stanh',
    'hard_sigmoid',
    'swish',
]

__all__ = [
    'mean',
    'mul',
    'scale',
    'sigmoid_cross_entropy_with_logits',
    'elementwise_add',
    'elementwise_div',
    'elementwise_sub',
    'elementwise_mul',
    'elementwise_max',
    'elementwise_min',
    'elementwise_pow',
    'clip',
    'clip_by_norm',
    'logical_and',
    'logical_or',
    'logical_xor',
    'logical_not',
    'uniform_random_batch_size_like',
    'gaussian_random',
    'gaussian_random_batch_size_like',
    'scatter',
    'sum',
    'slice',
    'polygon_box_transform',
    'shape',
    'iou_similarity',
    'maxout',
] + __activations__

for _OP in set(__all__):
    globals()[_OP] = generate_layer_fn(_OP)

__all__ += ["uniform_random"]

_uniform_random_ = generate_layer_fn('uniform_random')


def uniform_random(shape, dtype=None, min=None, max=None, seed=None):
    kwargs = dict()
    for name in locals():
        val = locals()[name]
        if val is not None:
            kwargs[name] = val
    return _uniform_random_(**kwargs)


uniform_random.__doc__ = _uniform_random_.__doc__ + """
Examples:

    >>> result = fluid.layers.uniform_random(shape=[32, 784])
"""

__all__ += ['hard_shrink']

_hard_shrink_ = generate_layer_fn('hard_shrink')


def hard_shrink(x, threshold=None):
    kwargs = dict()
    for name in locals():
        val = locals()[name]
        if val is not None:
            kwargs[name] = val
    return _hard_shrink_(**kwargs)


hard_shrink.__doc__ = _hard_shrink_.__doc__ + """
Examples:

    >>> data = fluid.layers.data(name="input", shape=[784])
    >>> result = fluid.layers.hard_shrink(x=data, threshold=0.3)
"""

__all__ += ['cumsum']

_cum_sum_ = generate_layer_fn('cumsum')


def cumsum(x, axis=None, exclusive=None, reverse=None):
    kwargs = dict()
    for name in locals():
        val = locals()[name]
        if val is not None:
            kwargs[name] = val

    return _cum_sum_(**kwargs)


cumsum.__doc__ = _cum_sum_.__doc__ + """
Examples:

    >>> data = fluid.layers.data(name="input", shape=[32, 784])
    >>> result = fluid.layers.cumsum(data, axis=0)
"""

__all__ += ['thresholded_relu']

_thresholded_relu_ = generate_layer_fn('thresholded_relu')


def thresholded_relu(x, threshold=None):
    kwargs = dict()
    for name in locals():
        val = locals()[name]
        if val is not None:
            kwargs[name] = val

    _thresholded_relu_(**kwargs)


thresholded_relu.__doc__ = _thresholded_relu_.__doc__ + """
Examples:

    >>> data = fluid.layers.data(name="input", shape=[1])
    >>> result = fluid.layers.thresholded_relu(data, threshold=0.4)
"""
