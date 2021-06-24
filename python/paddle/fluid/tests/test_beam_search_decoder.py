#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
"""
A simple machine translation demo using beam search decoder.
"""

from __future__ import print_function

import contextlib
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
from paddle.fluid.contrib.decoder.beam_search_decoder import *
import unittest
import os

paddle.enable_static()

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 32
word_dim = 32
decoder_size = hidden_dim
IS_SPARSE = True
batch_size = 2
max_length = 8
topk_size = 50
trg_dic_size = 10000
beam_size = 2


def encoder():
    # encoder
    src_word = layers.data(
        name="src_word", shape=[1], dtype='int64', lod_level=1)
    src_embedding = layers.embedding(
        input=src_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE)

    fc1 = layers.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = layers.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = layers.sequence_last_step(input=lstm_hidden0)
    return encoder_out


def decoder_state_cell(context):
    h = InitState(init=context, need_reorder=True)
    state_cell = StateCell(inputs={'x': None}, states={'h': h}, out_state='h')

    @state_cell.state_updater
    def updater(state_cell):
        current_word = state_cell.get_input('x')
        prev_h = state_cell.get_state('h')
        # make sure lod of h heritted from prev_h
        h = layers.fc(input=[prev_h, current_word],
                      size=decoder_size,
                      act='tanh')
        state_cell.set_state('h', h)

    return state_cell


def decoder_train(state_cell):
    # decoder
    trg_language_word = layers.data(
        name="target_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = layers.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE)

    decoder = TrainingDecoder(state_cell)

    with decoder.block():
        current_word = decoder.step_input(trg_embedding)
        decoder.state_cell.compute_state(inputs={'x': current_word})
        current_score = layers.fc(input=decoder.state_cell.get_state('h'),
                                  size=target_dict_dim,
                                  act='softmax')
        decoder.state_cell.update_states()
        decoder.output(current_score)

    return decoder()


def decoder_decode(state_cell):
    init_ids = layers.data(
        name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = layers.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    decoder = BeamSearchDecoder(
        state_cell=state_cell,
        init_ids=init_ids,
        init_scores=init_scores,
        target_dict_dim=target_dict_dim,
        word_dim=word_dim,
        input_var_dict={},
        topk_size=topk_size,
        sparse_emb=IS_SPARSE,
        max_len=max_length,
        beam_size=beam_size,
        end_id=1,
        name=None)
    decoder.decode()
    translation_ids, translation_scores = decoder()

    return translation_ids, translation_scores


def train_main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    context = encoder()
    state_cell = decoder_state_cell(context)
    rnn_out = decoder_train(state_cell)
    label = layers.data(
        name="target_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = layers.mean(x=cost)

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-3)
    optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)
    feed_order = ['src_word', 'target_word', 'target_next_word']

    exe = Executor(place)

    def train_loop(main_program):
        exe.run(framework.default_startup_program())

        feed_list = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list, place)

        for pass_id in range(1):
            for batch_id, data in enumerate(train_reader()):
                outs = exe.run(main_program,
                               feed=feeder.feed(data),
                               fetch_list=[avg_cost])
                avg_cost_val = np.array(outs[0])
                print('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                      " avg_cost=" + str(avg_cost_val))
                if batch_id > 3:
                    break

    train_loop(framework.default_main_program())


def decode_main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    context = encoder()
    state_cell = decoder_state_cell(context)
    translation_ids, translation_scores = decoder_decode(state_cell)

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    init_ids_data = np.array([0 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_lod = [1] * batch_size
    init_lod = [init_lod, init_lod]

    init_ids = fluid.create_lod_tensor(init_ids_data, init_lod, place)
    init_scores = fluid.create_lod_tensor(init_scores_data, init_lod, place)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    feed_order = ['src_word']
    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    data = next(train_reader())
    feed_dict = feeder.feed([[x[0]] for x in data])
    feed_dict['init_ids'] = init_ids
    feed_dict['init_scores'] = init_scores

    result_ids, result_scores = exe.run(
        framework.default_main_program(),
        feed=feed_dict,
        fetch_list=[translation_ids, translation_scores],
        return_numpy=False)
    print(result_ids.lod())


class TestBeamSearchDecoder(unittest.TestCase):
    pass


@contextlib.contextmanager
def scope_prog_guard():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            yield


def inject_test_train(use_cuda):
    f_name = 'test_{0}_train'.format('cuda' if use_cuda else 'cpu')

    def f(*args):
        with scope_prog_guard():
            train_main(use_cuda)

    setattr(TestBeamSearchDecoder, f_name, f)


def inject_test_decode(use_cuda, decorator=None):
    f_name = 'test_{0}_decode'.format('cuda' if use_cuda else 'cpu')

    def f(*args):
        with scope_prog_guard():
            decode_main(use_cuda)

    if decorator is not None:
        f = decorator(f)

    setattr(TestBeamSearchDecoder, f_name, f)


for _use_cuda_ in (False, True):
    inject_test_train(_use_cuda_)

for _use_cuda_ in (False, True):
    _decorator_ = None
    inject_test_decode(use_cuda=_use_cuda_, decorator=_decorator_)

if __name__ == '__main__':
    unittest.main()
