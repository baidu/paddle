import paddle.v2.fluid as fluid
import paddle.v2 as paddle
import unittest
import numpy


class TestDynRNN(unittest.TestCase):
    def setUp(self):
        self.word_dict = paddle.dataset.imdb.word_dict()
        self.BATCH_SIZE = 100
        self.train_data = paddle.batch(
            paddle.dataset.imdb.train(self.word_dict),
            batch_size=self.BATCH_SIZE)

    def test_plain_while_op(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            sentence = fluid.layers.data(
                name='word', shape=[1], dtype='int64', lod_level=1)
            sent_emb = fluid.layers.embedding(
                input=sentence, size=[65535, 32], dtype='float32')

            label = fluid.layers.data(name='label', shape=[1], dtype='float32')

            rank_table = fluid.layers.lod_rank_table(x=sent_emb)

            sent_emb_array = fluid.layers.lod_tensor_to_array(
                x=sent_emb, table=rank_table)

            seq_len = fluid.layers.max_sequence_len(rank_table=rank_table)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)

            boot_mem = fluid.layers.fill_constant_batch_size_like(
                input=fluid.layers.array_read(
                    array=sent_emb_array, i=i),
                value=0,
                shape=[-1, 100],
                dtype='float32')

            mem_array = fluid.layers.array_write(x=boot_mem, i=i)

            cond = fluid.layers.less_than(x=i, y=seq_len)
            while_op = fluid.layers.While(cond=cond)
            out = fluid.layers.create_array(dtype='float32')

            with while_op.block():
                mem = fluid.layers.array_read(array=mem_array, i=i)
                ipt = fluid.layers.array_read(array=sent_emb_array, i=i)

                mem = fluid.layers.shrink_memory(x=mem, i=i, table=rank_table)

                hidden = fluid.layers.fc(input=[mem, ipt], size=100, act='tanh')
                fluid.layers.array_write(x=hidden, i=i, array=out)
                fluid.layers.increment(x=i, in_place=True)
                fluid.layers.array_write(x=hidden, i=i, array=mem_array)
                fluid.layers.less_than(x=i, y=seq_len, cond=cond)

            all_timesteps = fluid.layers.array_to_lod_tensor(
                x=out, table=rank_table)
            last = fluid.layers.sequence_pool(
                input=all_timesteps, pool_type='last')
            logits = fluid.layers.fc(input=last, size=1, act=None)

        cpu = fluid.CPUPlace()
        exe = fluid.Executor(cpu)
        exe.run(startup_program)
        feeder = fluid.DataFeeder(feed_list=[sentence, label], place=cpu)

        data = next(self.train_data())
        val = exe.run(main_program, feed=feeder.feed(data),
                      fetch_list=[logits])[0]
        self.assertEqual((self.BATCH_SIZE, 1), val.shape)
        val = val.sum()
        self.assertFalse(numpy.isnan(val))


if __name__ == '__main__':
    unittest.main()
