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

import paddle.fluid as fluid
import paddle
import unittest
import numpy

from paddle.fluid.framework import Program, program_guard
from paddle.fluid.layers.control_flow import lod_rank_table
from paddle.fluid.layers.control_flow import max_sequence_len
from paddle.fluid.layers.control_flow import lod_tensor_to_array
from paddle.fluid.layers.control_flow import array_to_lod_tensor
from paddle.fluid.layers.control_flow import shrink_memory
from fake_reader import fake_imdb_reader

numpy.random.seed(2020)


class TestDynamicRNN(unittest.TestCase):
    def setUp(self):
        self.word_dict_len = 5147
        self.BATCH_SIZE = 2
        reader = fake_imdb_reader(self.word_dict_len, self.BATCH_SIZE * 100)
        self.train_data = paddle.batch(reader, batch_size=self.BATCH_SIZE)

    def _train(self,
               main_program,
               startup_program,
               feed_list,
               fetch_list,
               is_nested=False,
               max_iters=1):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        data = next(self.train_data())

        for iter_id in range(max_iters):
            fetch_outs = exe.run(main_program,
                                 feed=feeder.feed(data),
                                 fetch_list=fetch_list,
                                 return_numpy=False)
            if len(fetch_list) == 3:
                rnn_in_seq = fetch_outs[0]
                rnn_out_seq = fetch_outs[1]
                if not is_nested:
                    # Check for lod set in runtime. When lod_level is 1,
                    # the lod of DynamicRNN's output should be the same as input.
                    self.assertEqual(rnn_in_seq.lod(), rnn_out_seq.lod())

                loss_i = numpy.array(fetch_outs[2])
            elif len(fetch_list) == 1:
                loss_i = numpy.array(fetch_outs[0])
            #print(loss_i)

            self.assertEqual((1, ), loss_i.shape)
            self.assertFalse(numpy.isnan(loss_i))
            if iter_id == 0:
                loss_0 = loss_i

        if max_iters > 10:
            # loss should be small after 10 mini-batch
            self.assertLess(loss_i[0], loss_0[0])

    def test_plain_while_op(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=1)
            sent_emb = fluid.layers.embedding(
                input=sentence, size=[self.word_dict_len, 32], dtype='float32')

            rank_table = lod_rank_table(x=sent_emb)
            sent_emb_array = lod_tensor_to_array(x=sent_emb, table=rank_table)

            seq_len = max_sequence_len(rank_table=rank_table)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            i.stop_gradient = False

            boot_mem = fluid.layers.fill_constant_batch_size_like(
                input=fluid.layers.array_read(
                    array=sent_emb_array, i=i),
                value=0,
                shape=[-1, 100],
                dtype='float32')
            boot_mem.stop_gradient = False
            mem_array = fluid.layers.array_write(x=boot_mem, i=i)

            cond = fluid.layers.less_than(x=i, y=seq_len)
            cond.stop_gradient = False
            while_op = fluid.layers.While(cond=cond)
            out = fluid.layers.create_array(dtype='float32')

            with while_op.block():
                mem = fluid.layers.array_read(array=mem_array, i=i)
                ipt = fluid.layers.array_read(array=sent_emb_array, i=i)

                mem = shrink_memory(x=mem, i=i, table=rank_table)

                hidden = fluid.layers.fc(input=[mem, ipt], size=100, act='tanh')

                fluid.layers.array_write(x=hidden, i=i, array=out)
                fluid.layers.increment(x=i, in_place=True)
                fluid.layers.array_write(x=hidden, i=i, array=mem_array)
                fluid.layers.less_than(x=i, y=seq_len, cond=cond)

            result_all_timesteps = array_to_lod_tensor(x=out, table=rank_table)
            last = fluid.layers.sequence_last_step(input=result_all_timesteps)

            logits = fluid.layers.fc(input=last, size=1, act=None)
            label = fluid.layers.data(name='label', shape=[1], dtype='float32')
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(1e-4)
            sgd.minimize(loss=loss)

        # Check for lod_level set in compile-time.
        self.assertEqual(sent_emb.lod_level, result_all_timesteps.lod_level)

        self._train(
            main_program=main_program,
            startup_program=startup_program,
            feed_list=[sentence, label],
            fetch_list=[sent_emb, result_all_timesteps, loss],
            is_nested=False,
            max_iters=1)

    def test_train_dynamic_rnn(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        main_program.random_seed = 10
        startup_program.random_seed = 10
        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=1)
            sent_emb = fluid.layers.embedding(
                input=sentence, size=[self.word_dict_len, 32], dtype='float32')

            drnn = fluid.layers.DynamicRNN()
            with drnn.block():
                in_ = drnn.step_input(sent_emb)
                mem = drnn.memory(shape=[100], dtype='float32')
                out_ = fluid.layers.fc(input=[in_, mem], size=100, act='tanh')
                drnn.update_memory(mem, out_)
                drnn.output(out_)

            drnn_result = drnn()
            last = fluid.layers.sequence_last_step(input=drnn_result)
            logits = fluid.layers.fc(input=last, size=1, act=None)

            label = fluid.layers.data(name='label', shape=[1], dtype='float32')
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.Adam(1e-3)
            sgd.minimize(loss=loss)

        # Check for lod_level set in compile-time.
        self.assertEqual(sent_emb.lod_level, drnn_result.lod_level)

        self._train(
            main_program=main_program,
            startup_program=startup_program,
            feed_list=[sentence, label],
            fetch_list=[sent_emb, drnn_result, loss],
            is_nested=False,
            max_iters=100)

    def _fake_reader(self):
        seq_len, label = [[2, 2]], [0, 1]
        data = []
        for ele in seq_len:
            for j in ele:
                data.append([numpy.random.randint(30) for _ in range(j)])

        while True:
            yield data, label

    # this unit test is just used to the two layer nested dyn_rnn.
    def test_train_nested_dynamic_rnn(self):
        word_dict = [i for i in range(30)]

        main_program = fluid.Program()
        startup_program = fluid.Program()
        main_program.random_seed = 10
        startup_program.random_seed = 10
        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=2)
            label = fluid.layers.data(
                name='label', shape=[1], dtype='float32', lod_level=1)

            drnn0 = fluid.layers.DynamicRNN()
            with drnn0.block():
                in_0 = drnn0.step_input(sentence)
                assert in_0.lod_level == 1, "the lod level of in_ should be 1"
                sentence_emb = fluid.layers.embedding(
                    input=in_0, size=[len(word_dict), 32], dtype='float32')
                out_0 = fluid.layers.fc(input=sentence_emb,
                                        size=100,
                                        act='tanh')

                drnn1 = fluid.layers.DynamicRNN()
                with drnn1.block():
                    in_1 = drnn1.step_input(out_0)
                    assert in_1.lod_level == 0, "the lod level of in_1 should be 0"
                    out_1 = fluid.layers.fc(input=[in_1], size=100, act='tanh')
                    drnn1.output(out_1)

                drnn1_result = drnn1()
                last_1 = fluid.layers.sequence_last_step(input=drnn1_result)
                drnn0.output(last_1)

            last = drnn0()
            logits = fluid.layers.fc(input=last, size=1, act=None)
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(1e-3)
            sgd.minimize(loss=loss)

        train_data_orig = self.train_data
        self.train_data = paddle.batch(self._fake_reader, batch_size=2)
        self._train(
            main_program=main_program,
            startup_program=startup_program,
            feed_list=[sentence, label],
            fetch_list=[loss],
            is_nested=True,
            max_iters=100)
        self.train_data = train_data_orig

    # this unit test is just used to the two layer nested dyn_rnn.
    def test_train_nested_dynamic_rnn2(self):
        word_dict = [i for i in range(30)]

        hidden_size = 32
        main_program = fluid.Program()
        startup_program = fluid.Program()
        main_program.random_seed = 10
        startup_program.random_seed = 10
        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=2)
            label = fluid.layers.data(
                name='label', shape=[1], dtype='float32', lod_level=1)

            drnn0 = fluid.layers.DynamicRNN()
            with drnn0.block():
                in_0 = drnn0.step_input(sentence)
                sentence_emb = fluid.layers.embedding(
                    input=in_0,
                    size=[len(word_dict), hidden_size],
                    dtype='float32')
                input_forward_proj = fluid.layers.fc(input=sentence_emb,
                                                     size=hidden_size * 4,
                                                     act=None,
                                                     bias_attr=False)
                forward, _ = fluid.layers.dynamic_lstm(
                    input=input_forward_proj,
                    size=hidden_size * 4,
                    use_peepholes=False)

                drnn1 = fluid.layers.DynamicRNN()
                with drnn1.block():
                    in_1 = drnn1.step_input(forward)
                    out_1 = fluid.layers.fc(input=[in_1], size=100, act='tanh')
                    drnn1.output(out_1)

                last = fluid.layers.sequence_last_step(input=drnn1())
                drnn0.output(last)

            last = drnn0()
            logits = fluid.layers.fc(input=last, size=1, act=None)
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=logits, label=label)
            loss = fluid.layers.mean(loss)
            sgd = fluid.optimizer.SGD(1e-3)
            sgd.minimize(loss=loss)

        train_data_orig = self.train_data
        self.train_data = paddle.batch(self._fake_reader, batch_size=2)
        self._train(
            main_program=main_program,
            startup_program=startup_program,
            feed_list=[sentence, label],
            fetch_list=[loss],
            is_nested=True,
            max_iters=100)
        self.train_data = train_data_orig


class TestDynamicRNNErrors(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            init = fluid.layers.zeros(shape=[1], dtype='float32')
            shape = 'shape'
            sentence = fluid.data(
                name='sentence', shape=[None, 32], dtype='float32', lod_level=1)

            # The type of Input(shape) in API(memory) must be list or tuple
            def input_shape_type_of_memory():
                drnn = fluid.layers.DynamicRNN()
                with drnn.block():
                    res = drnn.memory(init, shape)

            self.assertRaises(TypeError, input_shape_type_of_memory)

            # The type of element of Input(*outputs) in API(output) must be Variable.
            def outputs_type_of_output():
                drnn = fluid.layers.DynamicRNN()
                with drnn.block():
                    word = drnn.step_input(sentence)
                    memory = drnn.memory(shape=[10], dtype='float32', value=0)
                    hidden = fluid.layers.fc(input=[word, memory],
                                             size=10,
                                             act='tanh')
                    out = numpy.ones(1).astype('float32')
                    drnn.update_memory(ex_mem=memory, new_mem=hidden)
                    drnn.output(hidden, out)

                self.assertRaises(TypeError, outputs_type_of_output)


if __name__ == '__main__':
    unittest.main()
