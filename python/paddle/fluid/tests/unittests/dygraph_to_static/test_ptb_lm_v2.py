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

from __future__ import absolute_import, division, print_function

import logging
import time
import unittest

import numpy as np
import paddle

PRINT_STEP = 20
SEED = 2020

program_translator = paddle.jit.ProgramTranslator()


class SimpleLSTMRNN(paddle.nn.Layer):
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
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(
                    low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Uniform(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(
                    low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        cell_array = []
        hidden_array = []

        for i in range(self._num_layers):
            hidden_array.append(init_hidden[i])
            cell_array.append(init_cell[i])

        res = []
        for index in range(self._num_steps):
            step_input = input_embedding[:, index, :]
            for k in range(self._num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = paddle.concat(x=[step_input, pre_hidden], axis=1)
                gate_input = paddle.matmul(x=nn, y=weight_1)

                gate_input = paddle.add(x=gate_input, y=bias)
                i, j, f, o = paddle.split(
                    x=gate_input, num_or_sections=4, axis=-1)
                c = pre_cell * paddle.nn.functional.sigmoid(
                    f) + paddle.nn.functional.sigmoid(i) * paddle.tanh(j)
                m = paddle.tanh(c) * paddle.nn.functional.sigmoid(o)
                hidden_array[k] = m
                cell_array[k] = c
                step_input = m

                if self._dropout is not None and self._dropout > 0.0:
                    step_input = paddle.nn.functional.dropout(
                        step_input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
            res.append(step_input)
        real_res = paddle.concat(x=res, axis=1)
        real_res = paddle.reshape(real_res,
                                  [-1, self._num_steps, self._hidden_size])
        last_hidden = paddle.concat(x=hidden_array, axis=1)
        last_hidden = paddle.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = paddle.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = paddle.concat(x=cell_array, axis=1)
        last_cell = paddle.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = paddle.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(paddle.nn.Layer):
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
        self.embedding = paddle.fluid.dygraph.nn.Embedding(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=paddle.ParamAttr(
                name='embedding_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=paddle.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=paddle.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(
                low=-self.init_scale, high=self.init_scale))

    def build_once(self, input, label, init_hidden, init_cell):
        pass

    @paddle.jit.to_static
    def forward(self, input, label, init_hidden, init_cell):

        init_h = paddle.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = paddle.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)

        x_emb = paddle.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = paddle.nn.functional.dropout(
                x_emb,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)

        projection = paddle.matmul(x=rnn_out, y=self.softmax_weight)
        projection = paddle.add(x=projection, y=self.softmax_bias)

        loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = paddle.reshape(loss, shape=[-1, self.num_steps])
        loss = paddle.mean(loss, axis=[0])
        loss = paddle.fluid.layers.reduce_sum(loss)

        return loss, last_hidden, last_cell

    def debug_emb(self):

        np.save("emb_grad", self.x_emb.gradient())


def train(place):

    num_layers = 1
    batch_size = 4
    hidden_size = 10
    num_steps = 3
    init_scale = 0.1
    max_epoch = 1
    dropout = 0.0
    vocab_size = 1000
    batch_num = 200

    paddle.disable_static(place)
    paddle.manual_seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    ptb_model = PtbModel(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_steps=num_steps,
        init_scale=init_scale,
        dropout=dropout)

    sgd = paddle.optimizer.SGD(learning_rate=1e-3,
                               parameters=ptb_model.parameters())

    for epoch_id in range(max_epoch):

        total_loss = 0.0
        iters = 0.0
        total_sample = 0

        init_hidden_data = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')
        init_cell_data = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')

        init_hidden = paddle.to_tensor(
            data=init_hidden_data, dtype=None, place=None, stop_gradient=True)
        init_cell = paddle.to_tensor(
            data=init_cell_data, dtype=None, place=None, stop_gradient=True)
        for step_id in range(batch_num):
            x_data = np.arange(12).reshape(4, 3).astype('int64')
            y_data = np.arange(1, 13).reshape(4, 3).astype('int64')
            y_data = y_data.reshape((-1, 1))

            x_data = x_data.reshape((-1, num_steps, 1))
            y_data = y_data.reshape((-1, num_steps, 1))

            x = paddle.to_tensor(
                data=x_data, dtype=None, place=None, stop_gradient=True)
            y = paddle.to_tensor(
                data=y_data, dtype=None, place=None, stop_gradient=True)

            dy_loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                        init_cell)
            out_loss = dy_loss.numpy()

            dy_loss.backward()
            sgd.minimize(dy_loss)
            ptb_model.clear_gradients()

            total_loss += out_loss
            iters += num_steps
            total_sample += 1
            if step_id % PRINT_STEP == 0:
                if step_id == 0:
                    logging.info("epoch %d | step %d, loss %0.3f" %
                                 (epoch_id, step_id, total_loss / total_sample))
                    avg_batch_time = time.time()
                else:
                    speed = PRINT_STEP / (time.time() - avg_batch_time)
                    logging.info(
                        "epoch %d | step %d, loss %0.3f, speed %.3f steps/s" %
                        (epoch_id, step_id, total_loss / total_sample, speed))
                    avg_batch_time = time.time()

    ret = out_loss, last_hidden.numpy(), last_cell.numpy()
    paddle.enable_static()
    return ret


def train_dygraph(place):
    program_translator.enable(False)
    return train(place)


def train_static(place):
    program_translator.enable(True)
    return train(place)


class TestPtb(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def test_check_result(self):
        loss_1, hidden_1, cell_1 = train_static(self.place)
        loss_2, hidden_2, cell_2 = train_dygraph(self.place)

        self.assertTrue(
            np.allclose(loss_1, loss_2),
            msg="static loss: {} \ndygraph loss: {}".format(loss_1, loss_2))
        self.assertTrue(
            np.allclose(hidden_1, hidden_2),
            msg="static hidden: {} \ndygraph acc1: {}".format(hidden_1,
                                                              hidden_2))
        self.assertTrue(
            np.allclose(cell_1, cell_2),
            msg="static cell: {} \ndygraph cell: {}".format(cell_1, cell_2))


if __name__ == '__main__':
    unittest.main()
