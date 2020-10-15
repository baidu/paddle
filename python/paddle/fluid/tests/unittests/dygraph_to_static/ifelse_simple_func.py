#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid


def add_fn(x):
    x = x + 1
    return x


def loss_fn(x, lable):
    loss = fluid.layers.cross_entropy(x, lable)
    return loss


def dyfunc_with_if_else(x_v, label=None):
    if fluid.layers.mean(x_v).numpy()[0] > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    # plain if in python
    if label is not None:
        loss = fluid.layers.cross_entropy(x_v, label)
        return loss
    return x_v


def dyfunc_with_if_else2(x, col=100):
    row = 0
    if abs(col) > x.shape[-1]:
        # TODO: Don't support return non-Tensor in Tensor-dependent `if` stament currently.
        #  `x` is Tensor, `col` is not Tensor, and `col` is the return value of `true_fn` after transformed.
        # col = -1
        col = fluid.layers.fill_constant(shape=[1], value=-1, dtype="int64")
    if fluid.layers.reduce_mean(x).numpy()[0] > x.numpy()[row][col]:
        y = fluid.layers.relu(x)
    else:
        x_pow = fluid.layers.pow(x, 2)
        y = fluid.layers.tanh(x_pow)
    return y


def dyfunc_with_if_else3(x):
    # Create new var in parent scope, return it in true_fn and false_fn.
    # The var is created only in one of If.body or If.orelse node, and it used as gast.Load firstly after gast.If node.
    # The transformed code:
    """
    q = paddle.jit.dy2static.
        data_layer_not_check(name='q', shape=[-1], dtype='float32')
    z = paddle.jit.dy2static.
            data_layer_not_check(name='z', shape=[-1], dtype='float32')

    def true_fn_0(q, x, y):
        x = x + 1
        z = x + 2
        q = x + 3
        return q, x, y, z

    def false_fn_0(q, x, y):
        y = y + 1
        z = x - 2
        m = x + 2
        n = x + 3
        return q, x, y, z
    q, x, y, z = fluid.layers.cond(fluid.layers.mean(x)[0] < 5, lambda :
        paddle.jit.dy2static.convert_call(true_fn_0)(q, x, y),
        lambda : paddle.jit.dy2static.convert_call(false_fn_0)(q,
        x, y))
    """
    y = x + 1
    # NOTE: x_v[0] < 5 is True
    if fluid.layers.mean(x).numpy()[0] < 5:
        x = x + 1
        z = x + 2
        q = x + 3
    else:
        y = y + 1
        z = x - 2
        m = x + 2
        n = x + 3

    q = q + 1
    n = q + 2
    x = n
    return x


def nested_if_else(x_v):
    batch_size = 16
    feat_size = x_v.shape[-1]
    bias = fluid.layers.fill_constant([feat_size], dtype='float32', value=1)
    if x_v.shape[0] != batch_size:
        # TODO: Don't support return non-Tensor in Tensor-dependent `if` stament currently.
        #  `x_v.shape[0]` is not Tensor, and `batch_size` is the return value of `true_fn` after transformed.
        # col = -1
        # batch_size = x_v.shape[0]
        batch_size = fluid.layers.shape(x_v)[0]

    # if tensor.shape is [1], now support to compare with numpy.
    if fluid.layers.mean(x_v).numpy() < 0:
        y = x_v + bias
        w = fluid.layers.fill_constant([feat_size], dtype='float32', value=10)
        if y.numpy()[0] < 10:
            tmp = y * w
            y = fluid.layers.relu(tmp)
            if fluid.layers.mean(y).numpy()[0] < batch_size:
                y = fluid.layers.abs(y)
            else:
                tmp = fluid.layers.fill_constant(
                    [feat_size], dtype='float32', value=-1)
                y = y - tmp
    else:
        y = x_v - bias
    return y


def nested_if_else_2(x):
    y = fluid.layers.reshape(x, [-1, 1])
    b = 2
    if b < 1:
        # var `z` is not visible for outer scope
        z = y
    x_shape_0 = x.shape[0]
    if x_shape_0 < 1:
        if fluid.layers.shape(y).numpy()[0] < 1:
            res = fluid.layers.fill_constant(
                value=2, shape=x.shape, dtype="int32")
            # `z` is a new var here.
            z = y + 1
        else:
            res = fluid.layers.fill_constant(
                value=3, shape=x.shape, dtype="int32")
    else:
        res = x
    return res


def nested_if_else_3(x):
    y = fluid.layers.reshape(x, [-1, 1])
    b = 2
    # var `z` is visible for func.body
    if b < 1:
        z = y
    else:
        z = x

    if b < 1:
        res = x
        # var `out` is only visible for current `if`
        if b > 1:
            out = x + 1
        else:
            out = x - 1
    else:
        y_shape = fluid.layers.shape(y)
        if y_shape.numpy()[0] < 1:
            res = fluid.layers.fill_constant(
                value=2, shape=x.shape, dtype="int32")
            # `z` is created in above code block.
            z = y + 1
        else:
            res = fluid.layers.fill_constant(
                value=3, shape=x.shape, dtype="int32")
            # `out` is a new var.
            out = x + 1
    return res


class NetWithControlFlowIf(fluid.dygraph.Layer):
    def __init__(self, hidden_dim=16):
        super(NetWithControlFlowIf, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = fluid.dygraph.Linear(
            input_dim=hidden_dim,
            output_dim=5,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.99)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.5)))
        self.alpha = 10.
        self.constant_vars = {}

    def forward(self, input):
        hidden_dim = input.shape[-1]
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "hidden_dim {} of input is not equal to FC.weight[0]: {}"
                .format(hidden_dim, self.hidden_dim))

        self.constant_vars['bias'] = fluid.layers.fill_constant(
            [5], dtype='float32', value=1)
        # Control flow `if` statement
        fc_out = self.fc(input)
        if fluid.layers.mean(fc_out).numpy()[0] < 0:
            y = fc_out + self.constant_vars['bias']
            self.constant_vars['w'] = fluid.layers.fill_constant(
                [5], dtype='float32', value=10)
            if y.numpy()[0] < self.alpha:
                # Create new var, but is not used.
                x = 10
                tmp = y * self.constant_vars['w']
                y = fluid.layers.relu(tmp)
                # Nested `if/else`
                if y.numpy()[-1] < self.alpha:
                    # Modify variable of class
                    self.constant_vars['w'] = fluid.layers.fill_constant(
                        [hidden_dim], dtype='float32', value=9)
                    y = fluid.layers.abs(y)
                else:
                    tmp = fluid.layers.fill_constant(
                        [5], dtype='float32', value=-1)
                    y = y - tmp
        else:
            y = fc_out - self.constant_vars['bias']

        loss = fluid.layers.mean(y)
        return loss


def if_with_and_or(x_v, label=None):
    batch_size = fluid.layers.shape(x_v)
    if x_v is not None and (fluid.layers.mean(x_v).numpy()[0] > 0 or
                            label is not None) and batch_size[0] > 1 and True:
        x_v = x_v - 1
    else:
        x_v = x_v + 1

    if label is not None:
        loss = fluid.layers.cross_entropy(x_v, label)
        return loss
    return x_v


def if_with_and_or_1(x, y=None):
    batch_size = fluid.layers.shape(x)
    if batch_size[0] > 1 and y is not None:
        x = x + 1
    if y is not None or batch_size[0] > 1:
        x = x - 1
    return x


def if_with_and_or_2(x, y=None):
    batch_size = fluid.layers.shape(x)
    if x is not None and batch_size[0] > 1 and y is not None:
        x = x + 1
    if batch_size[0] > 1 or y is not None or x is not None:
        x = x - 1
    return x


def if_with_and_or_3(x, y=None):
    batch_size = fluid.layers.shape(x)
    mean_res = fluid.layers.mean(x)
    if x is not None and batch_size[0] > 1 and y is not None and mean_res.numpy(
    )[0] > 0:
        x = x + 1
    if mean_res.numpy()[0] > 0 and (x is not None and batch_size[0] > 1) and y:
        x = x - 1
    return x


def if_with_and_or_4(x, y=None):
    batch_size = fluid.layers.shape(x)
    mean_res = fluid.layers.mean(x)
    if (x is not None and batch_size[0] > 1) or (y is not None and
                                                 mean_res.numpy()[0] > 0):
        x = x + 1
    if (x is not None or batch_size[0] > 1) and (y is not None or
                                                 mean_res.numpy()[0] > 0):
        x = x - 1
    return x


def if_with_class_var(x, y=None):
    class Foo(object):
        def __init__(self):
            self.a = 1
            self.b = 2

    foo = Foo()
    batch_size = fluid.layers.shape(x)
    mean_res = fluid.layers.mean(x)

    if batch_size[0] > foo.a:
        x = x + foo.b
    else:
        x = x - foo.b
    return x


def if_tensor_case(x):
    x = fluid.dygraph.to_variable(x)

    mean = fluid.layers.mean(x)
    # It is equivalent to `if mean != 0`
    if mean:
        for i in range(0, 10):
            if i > 5:
                x += 1
                break
            x += 1
    else:
        for i in range(0, 37):
            x += 1
            break
            x += i

    # join `and`/`or`
    if fluid.layers.mean(x) + 1 and mean > 1 and x is not None or 2 > 1:
        x -= 1

    # `not` statement
    if not (x[0][0] and (mean * x)[0][0]):
        x += 1

    return x
