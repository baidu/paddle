# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.layers.device import get_places
import unittest
import paddle.fluid as fluid
import paddle
import contextlib
import math
import numpy as np
import sys
import os


def convolution_net(data, label, input_dim, class_dim=2, emb_dim=32,
                    hid_dim=32):
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=3,
        act="tanh",
        pool_type="sqrt")
    conv_4 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=4,
        act="tanh",
        pool_type="sqrt")
    prediction = fluid.layers.fc(input=[conv_3, conv_4],
                                 size=class_dim,
                                 act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, accuracy, prediction


def dyn_rnn_lstm(data, label, input_dim, class_dim=2, emb_dim=32,
                 lstm_size=128):
    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)
    sentence = fluid.layers.fc(input=emb, size=lstm_size, act='tanh')

    rnn = fluid.layers.DynamicRNN()
    with rnn.block():
        word = rnn.step_input(sentence)
        prev_hidden = rnn.memory(value=0.0, shape=[lstm_size])
        prev_cell = rnn.memory(value=0.0, shape=[lstm_size])

        def gate_common(ipt, hidden, size):
            gate0 = fluid.layers.fc(input=ipt, size=size, bias_attr=True)
            gate1 = fluid.layers.fc(input=hidden, size=size, bias_attr=False)
            return gate0 + gate1

        forget_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                         lstm_size))
        input_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                        lstm_size))
        output_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                         lstm_size))
        cell_gate = fluid.layers.sigmoid(x=gate_common(word, prev_hidden,
                                                       lstm_size))

        cell = forget_gate * prev_cell + input_gate * cell_gate
        hidden = output_gate * fluid.layers.tanh(x=cell)
        rnn.update_memory(prev_cell, cell)
        rnn.update_memory(prev_hidden, hidden)
        rnn.output(hidden)

    last = fluid.layers.sequence_last_step(rnn())
    prediction = fluid.layers.fc(input=last, size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, accuracy, prediction


def stacked_lstm_net(data,
                     label,
                     input_dim,
                     class_dim=2,
                     emb_dim=128,
                     hid_dim=512,
                     stacked_num=3):
    assert stacked_num % 2 == 1

    emb = fluid.layers.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)
    # add bias attr

    # TODO(qijun) linear act
    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    prediction = fluid.layers.fc(input=[fc_last, lstm_last],
                                 size=class_dim,
                                 act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, accuracy, prediction


def train(word_dict,
          net_method,
          use_cuda,
          parallel=False,
          save_dirname=None,
          is_local=True):
    BATCH_SIZE = 128
    PASS_NUM = 5
    dict_dim = len(word_dict)
    class_dim = 2

    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    if not parallel:
        cost, acc_out, prediction = net_method(
            data, label, input_dim=dict_dim, class_dim=class_dim)
    else:
        places = get_places()
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            cost, acc, _ = net_method(
                pd.read_input(data),
                pd.read_input(label),
                input_dim=dict_dim,
                class_dim=class_dim)
            pd.write_output(cost)
            pd.write_output(acc)

        cost, acc = pd()
        cost = fluid.layers.mean(cost)
        acc_out = fluid.layers.mean(acc)
        prediction = None
        assert save_dirname is None

    adagrad = fluid.optimizer.Adagrad(learning_rate=0.002)
    adagrad.minimize(cost)

    train_data = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.imdb.train(word_dict), buf_size=1000),
        batch_size=BATCH_SIZE)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())

        for pass_id in range(PASS_NUM):
            for data in train_data():
                cost_val, acc_val = exe.run(main_program,
                                            feed=feeder.feed(data),
                                            fetch_list=[cost, acc_out])
                print(("cost=" + str(cost_val) + " acc=" + str(acc_val)))
                if cost_val < 0.4 and acc_val > 0.8:
                    if save_dirname is not None:
                        fluid.io.save_inference_model(save_dirname, ["words"],
                                                      prediction, exe)
                    return
                if math.isnan(float(cost_val)):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError("Cost is too large for {0}".format(
            net_method.__name__))

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


def infer(word_dict, use_cuda, save_dirname=None):
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

        word_dict_len = len(word_dict)

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
        tensor_words = fluid.create_random_int_lodtensor(
            recursive_seq_lens,
            base_shape,
            place,
            low=0,
            high=word_dict_len - 1)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        assert feed_target_names[0] == "words"
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_words},
                          fetch_list=fetch_targets,
                          return_numpy=False)
        print((results[0].recursive_sequence_lengths()))
        np_data = np.array(results[0])
        print(("Inference Shape: ", np_data.shape))
        print(("Inference results: ", np_data))


def main(word_dict, net_method, use_cuda, parallel=False, save_dirname=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    train(
        word_dict,
        net_method,
        use_cuda,
        parallel=parallel,
        save_dirname=save_dirname)
    infer(word_dict, use_cuda, save_dirname)


class TestUnderstandSentiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.word_dict = paddle.dataset.imdb.word_dict()

    @contextlib.contextmanager
    def new_program_scope(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield

    def test_conv_cpu(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=False,
                save_dirname="understand_sentiment_conv.inference.model")

    def test_conv_cpu_parallel(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=False,
                parallel=True)

    @unittest.skip(reason="make CI faster")
    def test_stacked_lstm_cpu(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=stacked_lstm_net,
                use_cuda=False,
                save_dirname="understand_sentiment_stacked_lstm.inference.model")

    def test_stacked_lstm_cpu_parallel(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=stacked_lstm_net,
                use_cuda=False,
                parallel=True)

    def test_conv_gpu(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=True,
                save_dirname="understand_sentiment_conv.inference.model")

    def test_conv_gpu_parallel(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=convolution_net,
                use_cuda=True,
                parallel=True)

    @unittest.skip(reason="make CI faster")
    def test_stacked_lstm_gpu(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=stacked_lstm_net,
                use_cuda=True,
                save_dirname="understand_sentiment_stacked_lstm.inference.model")

    def test_stacked_lstm_gpu_parallel(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=stacked_lstm_net,
                use_cuda=True,
                parallel=True)

    @unittest.skip(reason='make CI faster')
    def test_dynrnn_lstm_gpu(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=dyn_rnn_lstm,
                use_cuda=True,
                parallel=False)

    def test_dynrnn_lstm_gpu_parallel(self):
        with self.new_program_scope():
            main(
                self.word_dict,
                net_method=dyn_rnn_lstm,
                use_cuda=True,
                parallel=True)


if __name__ == '__main__':
    unittest.main()
