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
import math
import numpy as np
import os
import time
import unittest

import paddle
import paddle.dataset.conll05 as conll05
import paddle.fluid as fluid

word_dict, verb_dict, label_dict = conll05.get_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
pred_dict_len = len(verb_dict)

mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8
mix_hidden_lr = 1e-3

IS_SPARSE = True
PASS_NUM = 10
BATCH_SIZE = 10

embedding_name = 'emb'


def load_parameter(file_name, h, w):
    with open(file_name, 'rb') as f:
        f.read(16)  # skip header.
        return np.fromfile(f, dtype=np.float32).reshape(h, w)


def db_lstm(word, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, mark,
            **ignored):
    # 8 features
    predicate_embedding = fluid.layers.embedding(
        input=predicate,
        size=[pred_dict_len, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr='vemb')

    mark_embedding = fluid.layers.embedding(
        input=mark,
        size=[mark_dict_len, mark_dim],
        dtype='float32',
        is_sparse=IS_SPARSE)

    word_input = [word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2]
    emb_layers = [
        fluid.layers.embedding(
            size=[word_dict_len, word_dim],
            input=x,
            param_attr=fluid.ParamAttr(
                name=embedding_name, trainable=False)) for x in word_input
    ]
    emb_layers.append(predicate_embedding)
    emb_layers.append(mark_embedding)

    hidden_0_layers = [
        fluid.layers.fc(input=emb, size=hidden_dim) for emb in emb_layers
    ]

    hidden_0 = fluid.layers.sums(input=hidden_0_layers)

    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim)
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    feature_out = fluid.layers.sums(input=[
        fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')
    ])

    return feature_out


def train(use_cuda, save_dirname=None, is_local=True):
    # define network topology
    word = fluid.layers.data(
        name='word_data', shape=[1], dtype='int64', lod_level=1)
    predicate = fluid.layers.data(
        name='verb_data', shape=[1], dtype='int64', lod_level=1)
    ctx_n2 = fluid.layers.data(
        name='ctx_n2_data', shape=[1], dtype='int64', lod_level=1)
    ctx_n1 = fluid.layers.data(
        name='ctx_n1_data', shape=[1], dtype='int64', lod_level=1)
    ctx_0 = fluid.layers.data(
        name='ctx_0_data', shape=[1], dtype='int64', lod_level=1)
    ctx_p1 = fluid.layers.data(
        name='ctx_p1_data', shape=[1], dtype='int64', lod_level=1)
    ctx_p2 = fluid.layers.data(
        name='ctx_p2_data', shape=[1], dtype='int64', lod_level=1)
    mark = fluid.layers.data(
        name='mark_data', shape=[1], dtype='int64', lod_level=1)
    feature_out = db_lstm(**locals())
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)
    crf_cost = fluid.layers.linear_chain_crf(
        input=feature_out,
        label=target,
        param_attr=fluid.ParamAttr(
            name='crfw', learning_rate=mix_hidden_lr))
    avg_cost = fluid.layers.mean(crf_cost)

    # TODO(qiao)
    # check other optimizers and check why out will be NAN
    sgd_optimizer = fluid.optimizer.SGD(
        learning_rate=fluid.layers.exponential_decay(
            learning_rate=0.01,
            decay_steps=100000,
            decay_rate=0.5,
            staircase=True))
    sgd_optimizer.minimize(avg_cost)

    # TODO(qiao)
    # add dependency track and move this config before optimizer
    crf_decode = fluid.layers.crf_decoding(
        input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.conll05.test(), buf_size=8192),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    feeder = fluid.DataFeeder(
        feed_list=[
            word, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2, predicate, mark, target
        ],
        place=place)
    exe = fluid.Executor(place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        embedding_param = fluid.global_scope().find_var(
            embedding_name).get_tensor()
        embedding_param.set(
            load_parameter(conll05.get_embedding(), word_dict_len, word_dim),
            place)

        start_time = time.time()
        batch_id = 0
        for pass_id in range(PASS_NUM):
            for data in train_data():
                cost = exe.run(main_program,
                               feed=feeder.feed(data),
                               fetch_list=[avg_cost])
                cost = cost[0]

                if batch_id % 10 == 0:
                    print(("avg_cost:" + str(cost)))
                    if batch_id != 0:
                        print(("second per batch: " + str(
                            (time.time() - start_time) / batch_id)))
                    # Set the threshold low to speed up the CI test
                    if float(cost) < 60.0:
                        if save_dirname is not None:
                            # TODO(liuyiqun): Change the target to crf_decode
                            fluid.io.save_inference_model(save_dirname, [
                                'word_data', 'verb_data', 'ctx_n2_data',
                                'ctx_n1_data', 'ctx_0_data', 'ctx_p1_data',
                                'ctx_p2_data', 'mark_data'
                            ], [feature_out], exe)
                        return

                batch_id = batch_id + 1

    if is_local:
        train_loop(fluid.default_main_program())
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


def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

        # Setup input by creating LoDTensor to represent sequence of words.
        # Here each word is the basic element of the LoDTensor and the shape of 
        # each word (base_shape) should be [1] since it is simply an index to 
        # look up for the corresponding word vector.
        # Suppose the recursive_sequence_lengths info is set to [[3, 4, 2]],
        # which has only one level of detail. Then the created LoDTensor will have only 
        # one higher level structure (sequence of words, or sentence) than the basic 
        # element (word). Hence the LoDTensor will hold data for three sentences of 
        # length 3, 4 and 2, respectively. 
        # Note that recursive_sequence_lengths should be a list of lists.
        recursive_seq_lens = [[3, 4, 2]]
        base_shape = [1]
        # The range of random integers is [low, high]
        word = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=word_dict_len - 1)
        pred = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=pred_dict_len - 1)
        ctx_n2 = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=word_dict_len - 1)
        ctx_n1 = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=word_dict_len - 1)
        ctx_0 = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=word_dict_len - 1)
        ctx_p1 = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=word_dict_len - 1)
        ctx_p2 = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=word_dict_len - 1)
        mark = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=mark_dict_len - 1)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        assert feed_target_names[0] == 'word_data'
        assert feed_target_names[1] == 'verb_data'
        assert feed_target_names[2] == 'ctx_n2_data'
        assert feed_target_names[3] == 'ctx_n1_data'
        assert feed_target_names[4] == 'ctx_0_data'
        assert feed_target_names[5] == 'ctx_p1_data'
        assert feed_target_names[6] == 'ctx_p2_data'
        assert feed_target_names[7] == 'mark_data'

        results = exe.run(inference_program,
                          feed={
                              feed_target_names[0]: word,
                              feed_target_names[1]: pred,
                              feed_target_names[2]: ctx_n2,
                              feed_target_names[3]: ctx_n1,
                              feed_target_names[4]: ctx_0,
                              feed_target_names[5]: ctx_p1,
                              feed_target_names[6]: ctx_p2,
                              feed_target_names[7]: mark
                          },
                          fetch_list=fetch_targets,
                          return_numpy=False)
        print((results[0].recursive_sequence_lengths()))
        np_data = np.array(results[0])
        print(("Inference Shape: ", np_data.shape))


def main(use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "label_semantic_roles.inference.model"

    train(use_cuda, save_dirname, is_local)
    infer(use_cuda, save_dirname)


class TestLabelSemanticRoles(unittest.TestCase):
    def test_cuda(self):
        with self.scope_prog_guard():
            main(use_cuda=True)

    def test_cpu(self):
        with self.scope_prog_guard():
            main(use_cuda=False)

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
