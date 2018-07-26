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
import contextlib
import math
import sys
import unittest
from paddle.fluid.executor import Executor

dict_size = 30000
source_dict_dim = target_dict_dim = dict_size
src_dict, trg_dict = paddle.dataset.wmt14.get_dict(dict_size)
hidden_dim = 32
embedding_dim = 16
batch_size = 10
max_length = 50
topk_size = 50
encoder_size = decoder_size = hidden_dim
IS_SPARSE = True
USE_PEEPHOLES = False


def bi_lstm_encoder(input_seq, hidden_size):
    input_forward_proj = fluid.layers.fc(input=input_seq,
                                         size=hidden_size * 4,
                                         bias_attr=True)
    forward, _ = fluid.layers.dynamic_lstm(
        input=input_forward_proj,
        size=hidden_size * 4,
        use_peepholes=USE_PEEPHOLES)
    input_backward_proj = fluid.layers.fc(input=input_seq,
                                          size=hidden_size * 4,
                                          bias_attr=True)
    backward, _ = fluid.layers.dynamic_lstm(
        input=input_backward_proj,
        size=hidden_size * 4,
        is_reverse=True,
        use_peepholes=USE_PEEPHOLES)

    forward_last = fluid.layers.sequence_last_step(input=forward)
    backward_first = fluid.layers.sequence_first_step(input=backward)

    return forward_last, backward_first


# FIXME(peterzhang2029): Replace this function with the lstm_unit_op.
def lstm_step(x_t, hidden_t_prev, cell_t_prev, size):
    def linear(inputs):
        return fluid.layers.fc(input=inputs, size=size, bias_attr=True)

    forget_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    input_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    output_gate = fluid.layers.sigmoid(x=linear([hidden_t_prev, x_t]))
    cell_tilde = fluid.layers.tanh(x=linear([hidden_t_prev, x_t]))

    cell_t = fluid.layers.sums(input=[
        fluid.layers.elementwise_mul(
            x=forget_gate, y=cell_t_prev), fluid.layers.elementwise_mul(
                x=input_gate, y=cell_tilde)
    ])

    hidden_t = fluid.layers.elementwise_mul(
        x=output_gate, y=fluid.layers.tanh(x=cell_t))

    return hidden_t, cell_t


def lstm_decoder_without_attention(target_embedding, decoder_boot, context,
                                   decoder_size):
    rnn = fluid.layers.DynamicRNN()

    cell_init = fluid.layers.fill_constant_batch_size_like(
        input=decoder_boot,
        value=0.0,
        shape=[-1, decoder_size],
        dtype='float32')
    cell_init.stop_gradient = False

    with rnn.block():
        current_word = rnn.step_input(target_embedding)
        context = rnn.static_input(context)

        hidden_mem = rnn.memory(init=decoder_boot, need_reorder=True)
        cell_mem = rnn.memory(init=cell_init)
        decoder_inputs = fluid.layers.concat(
            input=[context, current_word], axis=1)
        h, c = lstm_step(decoder_inputs, hidden_mem, cell_mem, decoder_size)
        rnn.update_memory(hidden_mem, h)
        rnn.update_memory(cell_mem, c)
        out = fluid.layers.fc(input=h,
                              size=target_dict_dim,
                              bias_attr=True,
                              act='softmax')
        rnn.output(out)
    return rnn()


def seq_to_seq_net():
    """Construct a seq2seq network."""

    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)

    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[source_dict_dim, embedding_dim],
        dtype='float32')

    src_forward_last, src_backward_first = bi_lstm_encoder(
        input_seq=src_embedding, hidden_size=encoder_size)

    encoded_vector = fluid.layers.concat(
        input=[src_forward_last, src_backward_first], axis=1)

    decoder_boot = fluid.layers.fc(input=src_backward_first,
                                   size=decoder_size,
                                   bias_attr=False,
                                   act='tanh')

    trg_word_idx = fluid.layers.data(
        name='target_sequence', shape=[1], dtype='int64', lod_level=1)

    trg_embedding = fluid.layers.embedding(
        input=trg_word_idx,
        size=[target_dict_dim, embedding_dim],
        dtype='float32')

    prediction = lstm_decoder_without_attention(trg_embedding, decoder_boot,
                                                encoded_vector, decoder_size)
    label = fluid.layers.data(
        name='label_sequence', shape=[1], dtype='int64', lod_level=1)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)

    return avg_cost, prediction


def train(use_cuda, save_dirname=None):
    [avg_cost, prediction] = seq_to_seq_net()

    optimizer = fluid.optimizer.Adagrad(learning_rate=1e-4)
    optimizer.minimize(avg_cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(dict_size), buf_size=1000),
        batch_size=batch_size)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    feed_order = ['source_sequence', 'target_sequence', 'label_sequence']
    feed_list = [
        framework.default_main_program().global_block().var(var_name)
        for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    batch_id = 0
    for pass_id in range(2):
        for data in train_data():
            outs = exe.run(framework.default_main_program(),
                           feed=feeder.feed(data),
                           fetch_list=[avg_cost])

            avg_cost_val = np.array(outs[0])
            print(('pass_id=' + str(pass_id) + ' batch=' + str(batch_id) +
                   " avg_cost=" + str(avg_cost_val)))
            if math.isnan(float(avg_cost_val[0])):
                sys.exit("got NaN loss, training failed.")
            if batch_id > 3:
                if save_dirname is not None:
                    fluid.io.save_inference_model(
                        save_dirname, ['source_sequence',
                                       'target_sequence'], [prediction], exe)
                return

            batch_id += 1


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
        # Suppose the recursive_sequence_lengths info is set to [[4, 6]],
        # which has only one level of detail. Then the created LoDTensor will have only 
        # one higher level structure (sequence of words, or sentence) than the basic 
        # element (word). Hence the LoDTensor will hold data for two sentences of 
        # length 4 and 6, respectively. 
        # Note that recursive_sequence_lengths should be a list of lists.
        recursive_seq_lens = [[4, 6]]
        base_shape = [1]
        # The range of random integers is [low, high]
        word_data = fluid.create_random_int_lodtensor(
            recursive_seq_lens, base_shape, place, low=0, high=1)
        trg_word = fluid.create_random_int_lodtensor(
            recursive_seq_lens, base_shape, place, low=0, high=1)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        assert feed_target_names[0] == 'source_sequence'
        assert feed_target_names[1] == 'target_sequence'
        results = exe.run(inference_program,
                          feed={
                              feed_target_names[0]: word_data,
                              feed_target_names[1]: trg_word,
                          },
                          fetch_list=fetch_targets,
                          return_numpy=False)
        print((results[0].recursive_sequence_lengths()))
        np_data = np.array(results[0])
        print(("Inference shape: ", np_data.shape))
        print(("Inference results: ", np_data))


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "rnn_encoder_decoder.inference.model"

    train(use_cuda, save_dirname)
    infer(use_cuda, save_dirname)


class TestRnnEncoderDecoder(unittest.TestCase):
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
