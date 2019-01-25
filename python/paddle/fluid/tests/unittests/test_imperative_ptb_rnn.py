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

from __future__ import print_function

import unittest
import paddle.fluid as fluid
from paddle.fluid.imperative.nn import EMBEDDING
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.imperative.base import to_variable
import numpy as np
from paddle.fluid.backward import append_backward


class SimpleLSTMRNN(fluid.imperative.Layer):
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self.input = None
        self.num_steps = num_steps

    def _build_once(self, input_embedding, init_hidden=None, init_cell=None):
        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.hidden_array = []
        self.cell_array = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = fluid.layers.create_parameter(
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(weight_1)
            bias_1 = fluid.layers.create_parameter(
                [self._hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(bias_1)

            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        res = []
        for index in range(self.num_steps):
            self.input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self.input = fluid.layers.reshape(
                self.input, shape=[-1, self._hidden_size])
            for k in range(self._num_layers):
                pre_hidden = self.hidden_array[k]
                print("pre_hidden shape is:{}".format(pre_hidden.shape))
                print("input shape is:{}".format(self.input.shape))
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([self.input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                print("gate_input shape is: {}".format(gate_input.shape))
                print("gate_input value is :{}".format(gate_input._numpy()))
                print("gate_input desc is :{}".format(gate_input))
                # i, j, f, o = fluid.layers.split(gate_input, num_or_sections=4, dim=-1)
            #         #
            #         #         c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
            #         #             i) * fluid.layers.tanh(j)
            #         #         m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
            #         #
            #         #         self.hidden_array[k] = m
            #         #         self.cell_array[k] = c
            #         #         self.input = m
            #         #
            #         #         if self.dropout is not None and self.dropout > 0.0:
            #         #             self.input = fluid.layers.dropout(
            #         #                 self.input,
            #         #                 dropout_prob=self.dropout,
            #         #                 dropout_implementation='upscale_in_train')
            #         #
            #         #     res.append(
            #         #         fluid.layers.reshape(
            #         #             input, shape=[1, -1, self._hidden_size]))
            #         # real_res = fluid.layers.concat(res, 0)
            #         # real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
            #         # last_hidden = fluid.layers.concat(self.hidden_array, 1)
            #         # last_hidden = fluid.layers.reshape(
            #         #     last_hidden, shape=[-1, self._num_layers, self._hidden_size])
            #         # last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
            #         # last_cell = fluid.layers.concat(self.cell_array, 1)
            #         # last_cell = fluid.layers.reshape(
            #         #     last_cell, shape=[-1, self._num_layers, self._hidden_size])
            #         # last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
            #         #
            # return real_res, last_hidden, last_cell
        return [1], [2], [3]


class PtbModel(fluid.imperative.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 dropout=None):
        super(PtbModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        self.embedding = EMBEDDING(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))

    def _build_once(self, input, label, init_hidden, init_cell):
        self.softmax_weight = fluid.layers.create_parameter(
            [self.hidden_size, self.vocab_size],
            dtype="float32",
            name="softmax_weight",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = fluid.layers.create_parameter(
            [self.vocab_size],
            dtype="float32",
            name='softmax_bias',
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label, init_hidden, init_cell):

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)
        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.drop_out,
                dropout_implementation='upscale_in_train')
        print("init_c is {}".format(init_c))
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)
        rnn_out = fluid.layers.reshape(
            rnn_out, shape=[-1, self.num_steps, self.hidden_size])
        projection = fluid.layers.reshape(rnn_out, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.vocab_size])
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        loss.permissions = True

        return loss, last_hidden, last_cell


class TestImperativePtbRnn(unittest.TestCase):
    def test_mnist_cpu_float32(self):
        seed = 90
        hidden_size = 10
        vocab_size = 1000
        num_layers = 1
        num_steps = 3
        init_scale = 0.1
        batch_size = 4

        with fluid.imperative.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            # TODO: marsyang1993 Change seed to
            ptb_model = PtbModel(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                num_steps=num_steps,
                init_scale=init_scale)

            sgd = SGDOptimizer(learning_rate=1e-3)
            print("q")
            for i in range(2):
                x_data = np.arange(12).reshape(4, 3).astype('int64')
                y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
                x_data = x_data.reshape((-1, num_steps, 1))
                y_data = y_data.reshape((-1, 1))
                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                x = to_variable(x_data)
                y = to_variable(y_data)
                init_hidden = to_variable(init_hidden_data)
                init_cell = to_variable(init_cell_data)
                dy_loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                            init_cell)
                dy_param_init = dict()
                if i == 0:
                    for param in fluid.default_main_program().global_block(
                    ).all_parameters():
                        dy_param_init[param.name] = param._numpy()
                dy_loss._backward()
                sgd.minimize(dy_loss)
                dy_param_updated = dict()
                for param in fluid.default_main_program().global_block(
                ).all_parameters():
                    dy_param_updated[param.name] = param._numpy()


if __name__ == '__main__':
    unittest.main()
