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
import contextlib

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as pd
from paddle.fluid.executor import Executor
import unittest
import os

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
hidden_dim = 32
word_dim = 16
batch_size = 2
max_length = 8
topk_size = 50
trg_dic_size = 10000
beam_size = 2

decoder_size = hidden_dim


def encoder(is_sparse):
    # encoder
    src_word_id = pd.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = pd.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = pd.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = pd.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = pd.sequence_last_step(input=lstm_hidden0)
    return encoder_out


def decoder_train(context, is_sparse):
    # decoder
    trg_language_word = pd.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = pd.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr=fluid.ParamAttr(name='vemb'))

    rnn = pd.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        pre_state = rnn.memory(init=context)
        current_state = pd.fc(input=[current_word, pre_state],
                              size=decoder_size,
                              act='tanh')

        current_score = pd.fc(input=current_state,
                              size=target_dict_dim,
                              act='softmax')
        rnn.update_memory(pre_state, current_state)
        rnn.output(current_score)

    return rnn()


def decoder_decode(context, is_sparse):
    init_state = context
    array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)
    counter = pd.zeros(shape=[1], dtype='int64', force_cpu=True)

    # fill the first element with init_state
    state_array = pd.create_array('float32')
    pd.array_write(init_state, array=state_array, i=counter)

    # ids, scores as memory
    ids_array = pd.create_array('int64')
    scores_array = pd.create_array('float32')

    init_ids = pd.data(name="init_ids", shape=[1], dtype="int64", lod_level=2)
    init_scores = pd.data(
        name="init_scores", shape=[1], dtype="float32", lod_level=2)

    pd.array_write(init_ids, array=ids_array, i=counter)
    pd.array_write(init_scores, array=scores_array, i=counter)

    cond = pd.less_than(x=counter, y=array_len)

    while_op = pd.While(cond=cond)
    with while_op.block():
        pre_ids = pd.array_read(array=ids_array, i=counter)
        pre_state = pd.array_read(array=state_array, i=counter)
        pre_score = pd.array_read(array=scores_array, i=counter)

        # expand the recursive_sequence_lengths of pre_state to be the same with pre_score
        pre_state_expanded = pd.sequence_expand(pre_state, pre_score)

        pre_ids_emb = pd.embedding(
            input=pre_ids,
            size=[dict_size, word_dim],
            dtype='float32',
            is_sparse=is_sparse)

        # use rnn unit to update rnn
        current_state = pd.fc(input=[pre_state_expanded, pre_ids_emb],
                              size=decoder_size,
                              act='tanh')
        current_state_with_lod = pd.lod_reset(x=current_state, y=pre_score)
        # use score to do beam search
        current_score = pd.fc(input=current_state_with_lod,
                              size=target_dict_dim,
                              act='softmax')
        topk_scores, topk_indices = pd.topk(current_score, k=beam_size)
        # calculate accumulated scores after topk to reduce computation cost
        accu_scores = pd.elementwise_add(
            x=pd.log(topk_scores), y=pd.reshape(
                pre_score, shape=[-1]), axis=0)
        selected_ids, selected_scores = pd.beam_search(
            pre_ids,
            pre_score,
            topk_indices,
            accu_scores,
            beam_size,
            end_id=10,
            level=0)

        pd.increment(x=counter, value=1, in_place=True)

        # update the memories
        pd.array_write(current_state, array=state_array, i=counter)
        pd.array_write(selected_ids, array=ids_array, i=counter)
        pd.array_write(selected_scores, array=scores_array, i=counter)

        # update the break condition: up to the max length or all candidates of
        # source sentences have ended.
        length_cond = pd.less_than(x=counter, y=array_len)
        finish_cond = pd.logical_not(pd.is_empty(x=selected_ids))
        pd.logical_and(x=length_cond, y=finish_cond, out=cond)

    translation_ids, translation_scores = pd.beam_search_decode(
        ids=ids_array, scores=scores_array, beam_size=beam_size, end_id=10)

    # return init_ids, init_scores

    return translation_ids, translation_scores


def train_main(use_cuda, is_sparse, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    context = encoder(is_sparse)
    rnn_out = decoder_train(context, is_sparse)
    label = pd.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = pd.cross_entropy(input=rnn_out, label=label)
    avg_cost = pd.mean(cost)

    optimizer = fluid.optimizer.Adagrad(
        learning_rate=1e-4,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=0.1))
    optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    feed_order = [
        'src_word_id', 'target_language_word', 'target_language_next_word'
    ]

    exe = Executor(place)

    def train_loop(main_program):
        exe.run(framework.default_startup_program())

        feed_list = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]
        feeder = fluid.DataFeeder(feed_list, place)

        batch_id = 0
        for pass_id in range(1):
            for data in train_data():
                outs = exe.run(main_program,
                               feed=feeder.feed(data),
                               fetch_list=[avg_cost])
                avg_cost_val = np.array(outs[0])
                print(('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                       " avg_cost=" + str(avg_cost_val)))
                if batch_id > 3:
                    break
                batch_id += 1

    if is_local:
        train_loop(framework.default_main_program())
    else:
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def decode_main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    context = encoder(is_sparse)
    translation_ids, translation_scores = decoder_decode(context, is_sparse)

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    init_ids_data = np.array([1 for _ in range(batch_size)], dtype='int64')
    init_scores_data = np.array(
        [1. for _ in range(batch_size)], dtype='float32')
    init_ids_data = init_ids_data.reshape((batch_size, 1))
    init_scores_data = init_scores_data.reshape((batch_size, 1))
    init_recursive_seq_lens = [1] * batch_size
    init_recursive_seq_lens = [init_recursive_seq_lens, init_recursive_seq_lens]

    init_ids = fluid.create_lod_tensor(init_ids_data, init_recursive_seq_lens,
                                       place)
    init_scores = fluid.create_lod_tensor(init_scores_data,
                                          init_recursive_seq_lens, place)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    feed_order = ['src_word_id']
    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    for data in train_data():
        feed_dict = feeder.feed([[x[0]] for x in data])
        feed_dict['init_ids'] = init_ids
        feed_dict['init_scores'] = init_scores

        result_ids, result_scores = exe.run(
            framework.default_main_program(),
            feed=feed_dict,
            fetch_list=[translation_ids, translation_scores],
            return_numpy=False)
        print((result_ids.recursive_sequence_lengths()))
        break


class TestMachineTranslation(unittest.TestCase):
    pass


@contextlib.contextmanager
def scope_prog_guard():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            yield


def inject_test_train(use_cuda, is_sparse):
    f_name = 'test_{0}_{1}_train'.format('cuda' if use_cuda else 'cpu', 'sparse'
                                         if is_sparse else 'dense')

    def f(*args):
        with scope_prog_guard():
            train_main(use_cuda, is_sparse)

    setattr(TestMachineTranslation, f_name, f)


def inject_test_decode(use_cuda, is_sparse, decorator=None):
    f_name = 'test_{0}_{1}_decode'.format('cuda'
                                          if use_cuda else 'cpu', 'sparse'
                                          if is_sparse else 'dense')

    def f(*args):
        with scope_prog_guard():
            decode_main(use_cuda, is_sparse)

    if decorator is not None:
        f = decorator(f)

    setattr(TestMachineTranslation, f_name, f)


for _use_cuda_ in (False, True):
    for _is_sparse_ in (False, True):
        inject_test_train(_use_cuda_, _is_sparse_)

for _use_cuda_ in (False, True):
    for _is_sparse_ in (False, True):

        _decorator_ = None
        if _use_cuda_:
            _decorator_ = unittest.skip(
                reason='Beam Search does not support CUDA!')

        inject_test_decode(
            is_sparse=_is_sparse_, use_cuda=_use_cuda_, decorator=_decorator_)

if __name__ == '__main__':
    unittest.main()
