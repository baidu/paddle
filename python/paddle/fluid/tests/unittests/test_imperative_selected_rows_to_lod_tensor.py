#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import six

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope

DTYPE_LIST = ["float32"] if core.is_compiled_with_rocm(
) else ["float32", "float64"]

paddle.enable_static()


class SimpleNet(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_steps=20,
                 init_scale=0.1,
                 is_sparse=False,
                 dtype='float32'):
        super(SimpleNet, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_steps = num_steps
        self.embedding = Embedding(
            size=[vocab_size, hidden_size],
            dtype=dtype,
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.hidden_size],
            dtype=dtype,
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size],
            dtype=dtype,
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label):
        x_emb = self.embedding(input)
        fc = fluid.layers.matmul(x_emb, self.softmax_weight)
        fc = fluid.layers.elementwise_add(fc, self.softmax_bias)
        projection = fluid.layers.matmul(
            fc, fluid.layers.transpose(
                self.embedding.weight, perm=[1, 0]))
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss


class TestDygraphSimpleNet(unittest.TestCase):
    def test_simple_net(self):
        for is_sparse in [True, False]:
            for dtype in DTYPE_LIST:
                self.simple_net_float(is_sparse, dtype)

    def simple_net_float(self, is_sparse, dtype):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for place in places:
            seed = 90
            hidden_size = 10
            vocab_size = 1000
            num_steps = 3
            init_scale = 0.1
            batch_size = 4
            batch_num = 200

            for is_sort_sum_gradient in [True, False]:
                traced_layer = None
                with fluid.dygraph.guard(place):
                    paddle.seed(seed)
                    paddle.framework.random._manual_program_seed(seed)

                    simple_net = SimpleNet(
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        num_steps=num_steps,
                        init_scale=init_scale,
                        is_sparse=is_sparse,
                        dtype=dtype)

                    sgd = SGDOptimizer(
                        learning_rate=1e-3,
                        parameter_list=simple_net.parameters())
                    dy_param_updated = dict()
                    dy_param_init = dict()
                    dy_loss = None

                    fluid.set_flags({
                        'FLAGS_sort_sum_gradient': is_sort_sum_gradient
                    })

                    for i in range(batch_num):
                        x_data = np.arange(12).reshape(4, 3).astype('int64')
                        y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                        x_data = x_data.reshape((-1, num_steps))
                        y_data = y_data.reshape((-1, 1))

                        x = to_variable(x_data)
                        y = to_variable(y_data)
                        outs = simple_net(x, y)
                        dy_loss = outs
                        if i == 0:
                            for param in simple_net.parameters():
                                dy_param_init[param.name] = param.numpy()
                        dy_loss.backward()
                        sgd.minimize(dy_loss)
                        sgd.clear_gradients()
                        if i == batch_num - 1:
                            for param in simple_net.parameters():
                                dy_param_updated[param.name] = param.numpy()
                    dy_loss_value = dy_loss.numpy()

                with new_program_scope():
                    paddle.seed(seed)
                    paddle.framework.random._manual_program_seed(seed)

                    simple_net = SimpleNet(
                        hidden_size=hidden_size,
                        vocab_size=vocab_size,
                        num_steps=num_steps,
                        is_sparse=is_sparse,
                        dtype=dtype)

                    exe = fluid.Executor(place)
                    sgd = SGDOptimizer(learning_rate=1e-3)
                    x = fluid.layers.data(
                        name="x", shape=[-1, num_steps], dtype='int64')
                    y = fluid.layers.data(name="y", shape=[-1, 1], dtype=dtype)

                    static_loss = simple_net(x, y)
                    sgd.minimize(static_loss)
                    static_param_updated = dict()
                    static_param_init = dict()
                    static_param_name_list = list()
                    for param in simple_net.parameters():
                        static_param_name_list.append(param.name)

                    out = exe.run(framework.default_startup_program(),
                                  fetch_list=static_param_name_list)
                    for i in range(len(static_param_name_list)):
                        static_param_init[static_param_name_list[i]] = out[i]
                    static_loss_value = None
                    for i in range(batch_num):
                        x_data = np.arange(12).reshape(4, 3).astype('int64')
                        y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                        x_data = x_data.reshape((-1, num_steps))
                        y_data = y_data.reshape((-1, 1))
                        fetch_list = [static_loss]
                        fetch_list.extend(static_param_name_list)
                        out = exe.run(fluid.default_main_program(),
                                      feed={"x": x_data,
                                            "y": y_data},
                                      fetch_list=fetch_list)
                        static_loss_value = out[0]

                        if i == batch_num - 1:
                            for k in range(3, len(out)):
                                static_param_updated[static_param_name_list[
                                    k - 1]] = out[k]

                self.assertTrue(
                    np.array_equal(static_loss_value, dy_loss_value))
                for key, value in six.iteritems(static_param_init):
                    self.assertTrue(np.array_equal(value, dy_param_init[key]))
                for key, value in six.iteritems(static_param_updated):
                    self.assertTrue(
                        np.array_equal(value, dy_param_updated[key]))


if __name__ == '__main__':
    unittest.main()
