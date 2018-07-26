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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
import math
import sys

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 32
word_dim = 16
IS_SPARSE = True
batch_size = 10
max_length = 50
topk_size = 50
trg_dic_size = 10000

decoder_size = hidden_dim

# need to fix random seed and training data to compare the loss
# value accurately calculated by the default and the memory optimization
# version.
fluid.default_startup_program().random_seed = 111


def encoder_decoder():
    # encoder
    src_word_id = layers.data(
        name="src_word_id", shape=[1], dtype='int64', lod_level=1)
    src_embedding = layers.embedding(
        input=src_word_id,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    fc1 = fluid.layers.fc(input=src_embedding, size=hidden_dim * 4, act='tanh')
    lstm_hidden0, lstm_0 = layers.dynamic_lstm(input=fc1, size=hidden_dim * 4)
    encoder_out = layers.sequence_last_step(input=lstm_hidden0)

    # decoder
    trg_language_word = layers.data(
        name="target_language_word", shape=[1], dtype='int64', lod_level=1)
    trg_embedding = layers.embedding(
        input=trg_language_word,
        size=[dict_size, word_dim],
        dtype='float32',
        is_sparse=IS_SPARSE,
        param_attr=fluid.ParamAttr(name='vemb'))

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        mem = rnn.memory(init=encoder_out)
        fc1 = fluid.layers.fc(input=[current_word, mem],
                              size=decoder_size,
                              act='tanh')
        out = fluid.layers.fc(input=fc1, size=target_dict_dim, act='softmax')
        rnn.update_memory(mem, fc1)
        rnn.output(out)

    return rnn()


def main():
    rnn_out = encoder_decoder()
    label = layers.data(
        name="target_language_next_word", shape=[1], dtype='int64', lod_level=1)
    cost = layers.cross_entropy(input=rnn_out, label=label)
    avg_cost = fluid.layers.mean(cost)

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-4)
    optimizer.minimize(avg_cost)

    # fluid.memory_optimize(fluid.default_main_program())
    fluid.release_memory(fluid.default_main_program())

    # fix the order of training data
    train_data = paddle.batch(
        paddle.dataset.wmt14.train(dict_size), batch_size=batch_size)

    # train_data = paddle.batch(
    #     paddle.reader.shuffle(
    #         paddle.dataset.wmt14.train(dict_size), buf_size=1000),
    #     batch_size=batch_size)

    place = core.CPUPlace()
    exe = Executor(place)

    exe.run(framework.default_startup_program())

    feed_order = [
        'src_word_id', 'target_language_word', 'target_language_next_word'
    ]

    feed_list = [
        fluid.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    batch_id = 0
    for pass_id in range(10):
        for data in train_data():
            outs = exe.run(fluid.default_main_program(),
                           feed=feeder.feed(data),
                           fetch_list=[avg_cost])
            avg_cost_val = np.array(outs[0])
            print(('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                   " avg_cost=" + str(avg_cost_val)))
            if batch_id > 2:
                exit(0)
            if math.isnan(float(avg_cost_val)):
                sys.exit("got NaN loss, training failed.")
            batch_id += 1


if __name__ == '__main__':
    main()
