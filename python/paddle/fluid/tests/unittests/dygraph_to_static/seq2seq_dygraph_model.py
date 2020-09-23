# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid import layers
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.nn import Embedding
from seq2seq_utils import Seq2SeqModelHyperParams as args

INF = 1. * 1e5
alpha = 0.6
uniform_initializer = lambda x: fluid.initializer.UniformInitializer(low=-x, high=x)
zero_constant = fluid.initializer.Constant(0.0)


class BasicLSTMUnit(Layer):
    def __init__(self,
                 hidden_size,
                 input_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation=None,
                 activation=None,
                 forget_bias=1.0,
                 dtype='float32'):
        super(BasicLSTMUnit, self).__init__(dtype)

        self._hiden_size = hidden_size
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._gate_activation = gate_activation or layers.sigmoid
        self._activation = activation or layers.tanh
        self._forget_bias = forget_bias
        self._dtype = dtype
        self._input_size = input_size

        self._weight = self.create_parameter(
            attr=self._param_attr,
            shape=[self._input_size + self._hiden_size, 4 * self._hiden_size],
            dtype=self._dtype)

        self._bias = self.create_parameter(
            attr=self._bias_attr,
            shape=[4 * self._hiden_size],
            dtype=self._dtype,
            is_bias=True)

    def forward(self, input, pre_hidden, pre_cell):
        concat_input_hidden = layers.concat([input, pre_hidden], 1)
        gate_input = layers.matmul(x=concat_input_hidden, y=self._weight)

        gate_input = layers.elementwise_add(gate_input, self._bias)
        i, j, f, o = layers.split(gate_input, num_or_sections=4, dim=-1)
        new_cell = layers.elementwise_add(
            layers.elementwise_mul(pre_cell,
                                   layers.sigmoid(f + self._forget_bias)),
            layers.elementwise_mul(layers.sigmoid(i), layers.tanh(j)))

        new_hidden = layers.tanh(new_cell) * layers.sigmoid(o)

        return new_hidden, new_cell


class BaseModel(fluid.dygraph.Layer):
    def __init__(self,
                 hidden_size,
                 src_vocab_size,
                 tar_vocab_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None,
                 beam_size=1,
                 beam_start_token=1,
                 beam_end_token=2,
                 beam_max_step_num=2,
                 mode='train'):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dropout = dropout
        self.beam_size = beam_size
        self.beam_start_token = beam_start_token
        self.beam_end_token = beam_end_token
        self.beam_max_step_num = beam_max_step_num
        self.mode = mode
        self.kinf = 1e9

        param_attr = ParamAttr(initializer=uniform_initializer(self.init_scale))
        bias_attr = ParamAttr(initializer=zero_constant)
        forget_bias = 1.0

        self.src_embeder = Embedding(
            size=[self.src_vocab_size, self.hidden_size],
            param_attr=fluid.ParamAttr(
                initializer=uniform_initializer(init_scale)))

        self.tar_embeder = Embedding(
            size=[self.tar_vocab_size, self.hidden_size],
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                initializer=uniform_initializer(init_scale)))

        self.enc_units = []
        for i in range(num_layers):
            self.enc_units.append(
                self.add_sublayer(
                    "enc_units_%d" % i,
                    BasicLSTMUnit(
                        hidden_size=self.hidden_size,
                        input_size=self.hidden_size,
                        param_attr=param_attr,
                        bias_attr=bias_attr,
                        forget_bias=forget_bias)))

        self.dec_units = []
        for i in range(num_layers):
            self.dec_units.append(
                self.add_sublayer(
                    "dec_units_%d" % i,
                    BasicLSTMUnit(
                        hidden_size=self.hidden_size,
                        input_size=self.hidden_size,
                        param_attr=param_attr,
                        bias_attr=bias_attr,
                        forget_bias=forget_bias)))

        self.fc = fluid.dygraph.nn.Linear(
            self.hidden_size,
            self.tar_vocab_size,
            param_attr=param_attr,
            bias_attr=False)

    def _transpose_batch_time(self, x):
        return fluid.layers.transpose(x, [1, 0] + list(range(2, len(x.shape))))

    def _merge_batch_beams(self, x):
        return fluid.layers.reshape(x, shape=(-1, x.shape[2]))

    def _split_batch_beams(self, x):
        return fluid.layers.reshape(x, shape=(-1, self.beam_size, x.shape[1]))

    def _expand_to_beam_size(self, x):
        x = fluid.layers.unsqueeze(x, [1])
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = fluid.layers.expand(x, expand_times)
        return x

    def _real_state(self, state, new_state, step_mask):
        new_state = fluid.layers.elementwise_mul(new_state, step_mask, axis=0) - \
                    fluid.layers.elementwise_mul(state, (step_mask - 1), axis=0)
        return new_state

    def _gather(self, x, indices, batch_pos):
        topk_coordinates = fluid.layers.stack([batch_pos, indices], axis=2)
        return fluid.layers.gather_nd(x, topk_coordinates)

    @declarative
    def forward(self, inputs):
        inputs = [fluid.dygraph.to_variable(np_inp) for np_inp in inputs]
        src, tar, label, src_sequence_length, tar_sequence_length = inputs
        if src.shape[0] < self.batch_size:
            self.batch_size = src.shape[0]

        src_emb = self.src_embeder(self._transpose_batch_time(src))

        # NOTE: modify model code about `enc_hidden` and `enc_cell` to transforme dygraph code successfully.
        # Because nested list can't be transformed now.
        enc_hidden_0 = to_variable(
            np.zeros(
                (self.batch_size, self.hidden_size), dtype='float32'))
        enc_cell_0 = to_variable(
            np.zeros(
                (self.batch_size, self.hidden_size), dtype='float32'))
        zero = fluid.layers.zeros(shape=[1], dtype="int64")
        enc_hidden = fluid.layers.create_array(dtype="float32")
        enc_cell = fluid.layers.create_array(dtype="float32")
        for i in range(self.num_layers):
            index = zero + i
            enc_hidden = fluid.layers.array_write(
                enc_hidden_0, index, array=enc_hidden)
            enc_cell = fluid.layers.array_write(
                enc_cell_0, index, array=enc_cell)

        max_seq_len = src_emb.shape[0]

        enc_len_mask = fluid.layers.sequence_mask(
            src_sequence_length, maxlen=max_seq_len, dtype="float32")
        enc_len_mask = fluid.layers.transpose(enc_len_mask, [1, 0])

        # TODO: Because diff exits if call while_loop in static graph.
        # In while block, a Variable created in parent block participates in the calculation of gradient,
        # the gradient is wrong because each step scope always returns the same value generated by last step.
        # NOTE: Replace max_seq_len(Tensor src_emb.shape[0]) with args.max_seq_len(int) to avoid this bug temporarily.
        for k in range(args.max_seq_len):
            enc_step_input = src_emb[k]
            step_mask = enc_len_mask[k]
            new_enc_hidden, new_enc_cell = [], []
            for i in range(self.num_layers):
                enc_new_hidden, enc_new_cell = self.enc_units[i](
                    enc_step_input, enc_hidden[i], enc_cell[i])
                if self.dropout != None and self.dropout > 0.0:
                    enc_step_input = fluid.layers.dropout(
                        enc_new_hidden,
                        dropout_prob=self.dropout,
                        dropout_implementation='upscale_in_train')
                else:
                    enc_step_input = enc_new_hidden

                new_enc_hidden.append(
                    self._real_state(enc_hidden[i], enc_new_hidden, step_mask))
                new_enc_cell.append(
                    self._real_state(enc_cell[i], enc_new_cell, step_mask))

            enc_hidden, enc_cell = new_enc_hidden, new_enc_cell

        dec_hidden, dec_cell = enc_hidden, enc_cell
        tar_emb = self.tar_embeder(self._transpose_batch_time(tar))
        max_seq_len = tar_emb.shape[0]
        dec_output = []
        for step_idx in range(max_seq_len):
            j = step_idx + 0
            step_input = tar_emb[j]
            new_dec_hidden, new_dec_cell = [], []
            for i in range(self.num_layers):
                new_hidden, new_cell = self.dec_units[i](
                    step_input, dec_hidden[i], dec_cell[i])
                new_dec_hidden.append(new_hidden)
                new_dec_cell.append(new_cell)
                if self.dropout != None and self.dropout > 0.0:
                    step_input = fluid.layers.dropout(
                        new_hidden,
                        dropout_prob=self.dropout,
                        dropout_implementation='upscale_in_train')
                else:
                    step_input = new_hidden
            dec_output.append(step_input)

        dec_output = fluid.layers.stack(dec_output)
        dec_output = self.fc(self._transpose_batch_time(dec_output))
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=dec_output, label=label, soft_label=False)
        loss = fluid.layers.squeeze(loss, axes=[2])
        max_tar_seq_len = fluid.layers.shape(tar)[1]
        tar_mask = fluid.layers.sequence_mask(
            tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
        loss = loss * tar_mask
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss

    @declarative
    def beam_search(self, inputs):
        src, tar, label, src_sequence_length, tar_sequence_length = inputs
        if src.shape[0] < self.batch_size:
            self.batch_size = src.shape[0]

        src_emb = self.src_embeder(self._transpose_batch_time(src))
        enc_hidden_0 = to_variable(
            np.zeros(
                (self.batch_size, self.hidden_size), dtype='float32'))
        enc_cell_0 = to_variable(
            np.zeros(
                (self.batch_size, self.hidden_size), dtype='float32'))
        zero = fluid.layers.zeros(shape=[1], dtype="int64")
        enc_hidden = fluid.layers.create_array(dtype="float32")
        enc_cell = fluid.layers.create_array(dtype="float32")
        for j in range(self.num_layers):
            index = zero + j
            enc_hidden = fluid.layers.array_write(
                enc_hidden_0, index, array=enc_hidden)
            enc_cell = fluid.layers.array_write(
                enc_cell_0, index, array=enc_cell)

        max_seq_len = src_emb.shape[0]

        enc_len_mask = fluid.layers.sequence_mask(
            src_sequence_length, maxlen=max_seq_len, dtype="float32")
        enc_len_mask = fluid.layers.transpose(enc_len_mask, [1, 0])

        for k in range(args.max_seq_len):
            enc_step_input = src_emb[k]
            step_mask = enc_len_mask[k]

            new_enc_hidden, new_enc_cell = [], []

            for i in range(self.num_layers):
                enc_new_hidden, enc_new_cell = self.enc_units[i](
                    enc_step_input, enc_hidden[i], enc_cell[i])
                if self.dropout != None and self.dropout > 0.0:
                    enc_step_input = fluid.layers.dropout(
                        enc_new_hidden,
                        dropout_prob=self.dropout,
                        dropout_implementation='upscale_in_train')
                else:
                    enc_step_input = enc_new_hidden

                new_enc_hidden.append(
                    self._real_state(enc_hidden[i], enc_new_hidden, step_mask))
                new_enc_cell.append(
                    self._real_state(enc_cell[i], enc_new_cell, step_mask))

            enc_hidden, enc_cell = new_enc_hidden, new_enc_cell

        # beam search
        batch_beam_shape = (self.batch_size, self.beam_size)
        vocab_size_tensor = to_variable(np.full((
            1), self.tar_vocab_size)).astype("int64")
        start_token_tensor = to_variable(
            np.full(
                batch_beam_shape, self.beam_start_token, dtype='int64'))
        end_token_tensor = to_variable(
            np.full(
                batch_beam_shape, self.beam_end_token, dtype='int64'))
        step_input = self.tar_embeder(start_token_tensor)
        beam_finished = to_variable(
            np.full(
                batch_beam_shape, 0, dtype='float32'))
        beam_state_log_probs = to_variable(
            np.array(
                [[0.] + [-self.kinf] * (self.beam_size - 1)], dtype="float32"))
        beam_state_log_probs = fluid.layers.expand(beam_state_log_probs,
                                                   [self.batch_size, 1])
        dec_hidden, dec_cell = enc_hidden, enc_cell
        dec_hidden = [self._expand_to_beam_size(ele) for ele in dec_hidden]
        dec_cell = [self._expand_to_beam_size(ele) for ele in dec_cell]

        batch_pos = fluid.layers.expand(
            fluid.layers.unsqueeze(
                to_variable(np.arange(
                    0, self.batch_size, 1, dtype="int64")), [1]),
            [1, self.beam_size])
        predicted_ids = []
        parent_ids = []

        for step_idx in range(self.beam_max_step_num):
            if fluid.layers.reduce_sum(1 - beam_finished).numpy()[0] == 0:
                break
            step_input = self._merge_batch_beams(step_input)
            new_dec_hidden, new_dec_cell = [], []
            state = 0
            dec_hidden = [
                self._merge_batch_beams(state) for state in dec_hidden
            ]
            dec_cell = [self._merge_batch_beams(state) for state in dec_cell]

            for i in range(self.num_layers):
                new_hidden, new_cell = self.dec_units[i](
                    step_input, dec_hidden[i], dec_cell[i])
                new_dec_hidden.append(new_hidden)
                new_dec_cell.append(new_cell)
                if self.dropout != None and self.dropout > 0.0:
                    step_input = fluid.layers.dropout(
                        new_hidden,
                        dropout_prob=self.dropout,
                        dropout_implementation='upscale_in_train')
                else:
                    step_input = new_hidden
            cell_outputs = self._split_batch_beams(step_input)
            cell_outputs = self.fc(cell_outputs)

            step_log_probs = fluid.layers.log(
                fluid.layers.softmax(cell_outputs))
            noend_array = [-self.kinf] * self.tar_vocab_size
            noend_array[self.beam_end_token] = 0
            noend_mask_tensor = to_variable(
                np.array(
                    noend_array, dtype='float32'))

            step_log_probs = fluid.layers.elementwise_mul(
                fluid.layers.expand(fluid.layers.unsqueeze(beam_finished, [2]), [1, 1, self.tar_vocab_size]),
                noend_mask_tensor, axis=-1) - \
                             fluid.layers.elementwise_mul(step_log_probs, (beam_finished - 1), axis=0)
            log_probs = fluid.layers.elementwise_add(
                x=step_log_probs, y=beam_state_log_probs, axis=0)
            scores = fluid.layers.reshape(
                log_probs, [-1, self.beam_size * self.tar_vocab_size])
            topk_scores, topk_indices = fluid.layers.topk(
                input=scores, k=self.beam_size)

            beam_indices = fluid.layers.elementwise_floordiv(topk_indices,
                                                             vocab_size_tensor)
            token_indices = fluid.layers.elementwise_mod(topk_indices,
                                                         vocab_size_tensor)
            next_log_probs = self._gather(scores, topk_indices, batch_pos)

            x = 0
            new_dec_hidden = [
                self._split_batch_beams(state) for state in new_dec_hidden
            ]
            new_dec_cell = [
                self._split_batch_beams(state) for state in new_dec_cell
            ]
            new_dec_hidden = [
                self._gather(x, beam_indices, batch_pos) for x in new_dec_hidden
            ]
            new_dec_cell = [
                self._gather(x, beam_indices, batch_pos) for x in new_dec_cell
            ]

            new_dec_hidden = [
                self._gather(x, beam_indices, batch_pos) for x in new_dec_hidden
            ]
            new_dec_cell = [
                self._gather(x, beam_indices, batch_pos) for x in new_dec_cell
            ]
            next_finished = self._gather(beam_finished, beam_indices, batch_pos)
            next_finished = fluid.layers.cast(next_finished, "bool")
            next_finished = fluid.layers.logical_or(
                next_finished,
                fluid.layers.equal(token_indices, end_token_tensor))
            next_finished = fluid.layers.cast(next_finished, "float32")

            dec_hidden, dec_cell = new_dec_hidden, new_dec_cell
            beam_finished = next_finished
            beam_state_log_probs = next_log_probs
            step_input = self.tar_embeder(token_indices)
            predicted_ids.append(token_indices)
            parent_ids.append(beam_indices)

        predicted_ids = fluid.layers.stack(predicted_ids)
        parent_ids = fluid.layers.stack(parent_ids)
        predicted_ids = fluid.layers.gather_tree(predicted_ids, parent_ids)
        predicted_ids = self._transpose_batch_time(predicted_ids)
        return predicted_ids


class AttentionModel(fluid.dygraph.Layer):
    def __init__(self,
                 hidden_size,
                 src_vocab_size,
                 tar_vocab_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None,
                 beam_size=1,
                 beam_start_token=1,
                 beam_end_token=2,
                 beam_max_step_num=2,
                 mode='train'):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dropout = dropout
        self.beam_size = beam_size
        self.beam_start_token = beam_start_token
        self.beam_end_token = beam_end_token
        self.beam_max_step_num = beam_max_step_num
        self.mode = mode
        self.kinf = 1e9

        param_attr = ParamAttr(initializer=uniform_initializer(self.init_scale))
        bias_attr = ParamAttr(initializer=zero_constant)
        forget_bias = 1.0

        self.src_embeder = Embedding(
            size=[self.src_vocab_size, self.hidden_size],
            param_attr=fluid.ParamAttr(
                name='source_embedding',
                initializer=uniform_initializer(init_scale)))

        self.tar_embeder = Embedding(
            size=[self.tar_vocab_size, self.hidden_size],
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='target_embedding',
                initializer=uniform_initializer(init_scale)))

        self.enc_units = []
        for i in range(num_layers):
            self.enc_units.append(
                self.add_sublayer(
                    "enc_units_%d" % i,
                    BasicLSTMUnit(
                        hidden_size=self.hidden_size,
                        input_size=self.hidden_size,
                        param_attr=param_attr,
                        bias_attr=bias_attr,
                        forget_bias=forget_bias)))

        self.dec_units = []
        for i in range(num_layers):
            if i == 0:
                self.dec_units.append(
                    self.add_sublayer(
                        "dec_units_%d" % i,
                        BasicLSTMUnit(
                            hidden_size=self.hidden_size,
                            input_size=self.hidden_size * 2,
                            param_attr=ParamAttr(
                                name="dec_units_%d" % i,
                                initializer=uniform_initializer(
                                    self.init_scale)),
                            bias_attr=bias_attr,
                            forget_bias=forget_bias)))
            else:
                self.dec_units.append(
                    self.add_sublayer(
                        "dec_units_%d" % i,
                        BasicLSTMUnit(
                            hidden_size=self.hidden_size,
                            input_size=self.hidden_size,
                            param_attr=ParamAttr(
                                name="dec_units_%d" % i,
                                initializer=uniform_initializer(
                                    self.init_scale)),
                            bias_attr=bias_attr,
                            forget_bias=forget_bias)))

        self.attn_fc = fluid.dygraph.nn.Linear(
            self.hidden_size,
            self.hidden_size,
            param_attr=ParamAttr(
                name="self_attn_fc",
                initializer=uniform_initializer(self.init_scale)),
            bias_attr=False)

        self.concat_fc = fluid.dygraph.nn.Linear(
            2 * self.hidden_size,
            self.hidden_size,
            param_attr=ParamAttr(
                name="self_concat_fc",
                initializer=uniform_initializer(self.init_scale)),
            bias_attr=False)

        self.fc = fluid.dygraph.nn.Linear(
            self.hidden_size,
            self.tar_vocab_size,
            param_attr=ParamAttr(
                name="self_fc",
                initializer=uniform_initializer(self.init_scale)),
            bias_attr=False)

    def _transpose_batch_time(self, x):
        return fluid.layers.transpose(x, [1, 0] + list(range(2, len(x.shape))))

    def _merge_batch_beams(self, x):
        return fluid.layers.reshape(x, shape=(-1, x.shape[2]))

    def tile_beam_merge_with_batch(self, x):
        x = fluid.layers.unsqueeze(x, [1])  # [batch_size, 1, ...]
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = fluid.layers.expand(x, expand_times)  # [batch_size, beam_size, ...]
        x = fluid.layers.transpose(x, list(range(2, len(x.shape))) +
                                   [0, 1])  # [..., batch_size, beam_size]
        # use 0 to copy to avoid wrong shape
        x = fluid.layers.reshape(
            x, shape=[0] *
            (len(x.shape) - 2) + [-1])  # [..., batch_size * beam_size]
        x = fluid.layers.transpose(
            x, [len(x.shape) - 1] +
            list(range(0, len(x.shape) - 1)))  # [batch_size * beam_size, ...]
        return x

    def _split_batch_beams(self, x):
        return fluid.layers.reshape(x, shape=(-1, self.beam_size, x.shape[1]))

    def _expand_to_beam_size(self, x):
        x = fluid.layers.unsqueeze(x, [1])
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = fluid.layers.expand(x, expand_times)
        return x

    def _real_state(self, state, new_state, step_mask):
        new_state = fluid.layers.elementwise_mul(new_state, step_mask, axis=0) - \
                    fluid.layers.elementwise_mul(state, (step_mask - 1), axis=0)
        return new_state

    def _gather(self, x, indices, batch_pos):
        topk_coordinates = fluid.layers.stack([batch_pos, indices], axis=2)
        return fluid.layers.gather_nd(x, topk_coordinates)

    def attention(self, query, enc_output, mask=None):
        query = fluid.layers.unsqueeze(query, [1])
        memory = self.attn_fc(enc_output)
        attn = fluid.layers.matmul(query, memory, transpose_y=True)

        if mask is not None:
            attn = fluid.layers.transpose(attn, [1, 0, 2])
            attn = fluid.layers.elementwise_add(attn, mask * 1000000000, -1)
            attn = fluid.layers.transpose(attn, [1, 0, 2])
        weight = fluid.layers.softmax(attn)
        weight_memory = fluid.layers.matmul(weight, memory)

        return weight_memory

    def _change_size_for_array(self, func, array):
        print(" ^" * 10, "_change_size_for_array")
        print("array : ", array)
        for i, state in enumerate(array):
            fluid.layers.array_write(func(state), i, array)

        return array

    @declarative
    def forward(self, inputs):
        inputs = [fluid.dygraph.to_variable(np_inp) for np_inp in inputs]
        src, tar, label, src_sequence_length, tar_sequence_length = inputs
        if src.shape[0] < self.batch_size:
            self.batch_size = src.shape[0]

        src_emb = self.src_embeder(self._transpose_batch_time(src))

        # NOTE: modify model code about `enc_hidden` and `enc_cell` to transform dygraph code successfully.
        # Because nested list can't be transformed now.
        enc_hidden_0 = to_variable(
            np.zeros(
                (self.batch_size, self.hidden_size), dtype='float32'))
        enc_hidden_0.stop_gradient = True
        enc_cell_0 = to_variable(
            np.zeros(
                (self.batch_size, self.hidden_size), dtype='float32'))
        enc_hidden_0.stop_gradient = True
        zero = fluid.layers.zeros(shape=[1], dtype="int64")
        enc_hidden = fluid.layers.create_array(dtype="float32")
        enc_cell = fluid.layers.create_array(dtype="float32")
        for i in range(self.num_layers):
            index = zero + i
            enc_hidden = fluid.layers.array_write(
                enc_hidden_0, index, array=enc_hidden)
            enc_cell = fluid.layers.array_write(
                enc_cell_0, index, array=enc_cell)

        max_seq_len = src_emb.shape[0]

        enc_len_mask = fluid.layers.sequence_mask(
            src_sequence_length, maxlen=max_seq_len, dtype="float32")
        enc_padding_mask = (enc_len_mask - 1.0)
        enc_len_mask = fluid.layers.transpose(enc_len_mask, [1, 0])

        enc_outputs = []
        # TODO: Because diff exits if call while_loop in static graph.
        # In while block, a Variable created in parent block participates in the calculation of gradient,
        # the gradient is wrong because each step scope always returns the same value generated by last step.
        for p in range(max_seq_len):
            k = 0 + p
            enc_step_input = src_emb[k]
            step_mask = enc_len_mask[k]
            new_enc_hidden, new_enc_cell = [], []
            for i in range(self.num_layers):
                enc_new_hidden, enc_new_cell = self.enc_units[i](
                    enc_step_input, enc_hidden[i], enc_cell[i])
                if self.dropout != None and self.dropout > 0.0:
                    enc_step_input = fluid.layers.dropout(
                        enc_new_hidden,
                        dropout_prob=self.dropout,
                        dropout_implementation='upscale_in_train')
                else:
                    enc_step_input = enc_new_hidden

                new_enc_hidden.append(
                    self._real_state(enc_hidden[i], enc_new_hidden, step_mask))
                new_enc_cell.append(
                    self._real_state(enc_cell[i], enc_new_cell, step_mask))
            enc_outputs.append(enc_step_input)
            enc_hidden, enc_cell = new_enc_hidden, new_enc_cell

        enc_outputs = fluid.layers.stack(enc_outputs)
        enc_outputs = self._transpose_batch_time(enc_outputs)

        # train
        input_feed = to_variable(
            np.zeros(
                (self.batch_size, self.hidden_size), dtype='float32'))
        # NOTE: set stop_gradient here, otherwise grad var is null
        input_feed.stop_gradient = True
        dec_hidden, dec_cell = enc_hidden, enc_cell
        tar_emb = self.tar_embeder(self._transpose_batch_time(tar))
        max_seq_len = tar_emb.shape[0]
        dec_output = []

        for step_idx in range(max_seq_len):
            j = step_idx + 0
            step_input = tar_emb[j]
            step_input = fluid.layers.concat([step_input, input_feed], 1)
            new_dec_hidden, new_dec_cell = [], []
            for i in range(self.num_layers):
                new_hidden, new_cell = self.dec_units[i](
                    step_input, dec_hidden[i], dec_cell[i])
                new_dec_hidden.append(new_hidden)
                new_dec_cell.append(new_cell)
                if self.dropout != None and self.dropout > 0.0:
                    step_input = fluid.layers.dropout(
                        new_hidden,
                        dropout_prob=self.dropout,
                        dropout_implementation='upscale_in_train')
                else:
                    step_input = new_hidden
            dec_att = self.attention(step_input, enc_outputs, enc_padding_mask)
            dec_att = fluid.layers.squeeze(dec_att, [1])
            concat_att_out = fluid.layers.concat([dec_att, step_input], 1)
            out = self.concat_fc(concat_att_out)
            input_feed = out
            dec_output.append(out)
            dec_hidden, dec_cell = new_dec_hidden, new_dec_cell

        dec_output = fluid.layers.stack(dec_output)
        dec_output = self.fc(self._transpose_batch_time(dec_output))
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=dec_output, label=label, soft_label=False)
        loss = fluid.layers.squeeze(loss, axes=[2])
        max_tar_seq_len = fluid.layers.shape(tar)[1]
        tar_mask = fluid.layers.sequence_mask(
            tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
        loss = loss * tar_mask
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss
