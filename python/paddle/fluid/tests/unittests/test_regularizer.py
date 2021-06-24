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

import unittest
from functools import partial
import contextlib
import numpy as np
import random
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.optimizer as optimizer
import paddle.fluid.regularizer as regularizer
from paddle.fluid.backward import append_backward


class TestL2DecayRegularizer(unittest.TestCase):
    def test_l2decay_regularizer(self):
        paddle.enable_static()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            regularizer=regularizer.L2DecayRegularizer(0.5))
        self.assertTrue(mul_x.regularizer is not None)
        self.assertTrue(
            isinstance(mul_x.regularizer, regularizer.L2DecayRegularizer))
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        count_ops = len(block.ops)
        optimizer = paddle.optimizer.Adam()
        params_grads = optimizer.append_regularization_ops(params_grads)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(block.ops), count_ops + 2)
        self.assertEqual(block.ops[-1].type, 'sum')
        self.assertEqual(block.ops[-2].type, 'scale')


class TestL1DecayRegularizer(unittest.TestCase):
    def test_l2decay_regularizer(self):
        paddle.enable_static()
        program = framework.Program()
        block = program.global_block()
        mul_x = block.create_parameter(
            dtype="float32",
            shape=[5, 10],
            lod_level=0,
            name="mul.x",
            regularizer=regularizer.L1DecayRegularizer(0.5))
        self.assertTrue(mul_x.regularizer is not None)
        self.assertTrue(
            isinstance(mul_x.regularizer, regularizer.L1DecayRegularizer))
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        block.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})
        mean_out = block.create_var(
            dtype="float32", shape=[1], lod_level=0, name="mean.out")
        block.append_op(
            type="mean", inputs={"X": mul_out}, outputs={"Out": mean_out})
        params_grads = append_backward(mean_out)
        self.assertEqual(len(params_grads), 1)
        count_ops = len(block.ops)
        optimizer = paddle.optimizer.Adam()
        params_grads = optimizer.append_regularization_ops(params_grads)
        self.assertEqual(len(params_grads), 1)
        self.assertEqual(len(block.ops), count_ops + 3)
        self.assertEqual(block.ops[-1].type, 'sum')
        self.assertEqual(block.ops[-2].type, 'scale')
        self.assertEqual(block.ops[-3].type, 'sign')


def bow_net(data,
            label,
            dict_dim,
            is_sparse=False,
            emb_dim=8,
            hid_dim=8,
            hid_dim2=6,
            class_dim=2):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    fluid/PaddleNLP/text_classification/nets.py
    """
    emb = fluid.layers.embedding(
        input=data, is_sparse=is_sparse, size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    return avg_cost


class TestRegularizer(unittest.TestCase):
    def setUp(self):
        self.word_len = 1500
        self.train_data = [[(random.sample(range(1000), 10), [0])]
                           for _ in range(2)]

    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    @contextlib.contextmanager
    def scope_prog_guard(self, main_prog, startup_prog):
        scope = fluid.core.Scope()
        with fluid.unique_name.guard():
            with fluid.scope_guard(scope):
                with fluid.program_guard(main_prog, startup_prog):
                    yield

    def run_program(self, place, feed_list):
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        exe.run(fluid.default_startup_program())

        main_prog = fluid.default_main_program()
        param_list = [var.name for var in main_prog.block(0).all_parameters()]

        param_sum = []
        for data in self.train_data:
            out = exe.run(main_prog,
                          feed=feeder.feed(data),
                          fetch_list=param_list)
            p_sum = 0
            for v in out:
                p_sum += np.sum(np.abs(v))
            param_sum.append(p_sum)
        return param_sum

    def check_l2decay_regularizer(self, place, model):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_prog = fluid.framework.Program()
        startup_prog = fluid.framework.Program()
        with self.scope_prog_guard(
                main_prog=main_prog, startup_prog=startup_prog):
            data = fluid.layers.data(
                name="words", shape=[1], dtype="int64", lod_level=1)
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")

            avg_cost = model(data, label, self.word_len)

            optimizer = fluid.optimizer.Adagrad(
                learning_rate=0.1,
                regularization=fluid.regularizer.L2Decay(1.0))
            optimizer.minimize(avg_cost)
            param_sum = self.run_program(place, [data, label])
        return param_sum

    def check_l2decay(self, place, model):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_prog = fluid.framework.Program()
        startup_prog = fluid.framework.Program()

        with self.scope_prog_guard(
                main_prog=main_prog, startup_prog=startup_prog):
            data = fluid.layers.data(
                name="words", shape=[1], dtype="int64", lod_level=1)
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")

            avg_cost_l2 = model(data, label, self.word_len)

            param_list = fluid.default_main_program().block(0).all_parameters()
            para_sum = []
            for para in param_list:
                para_mul = fluid.layers.square(x=para)
                para_sum.append(fluid.layers.reduce_sum(input=para_mul))
            avg_cost_l2 += fluid.layers.sums(para_sum) * .5

            optimizer = fluid.optimizer.Adagrad(learning_rate=0.1)
            optimizer.minimize(avg_cost_l2)
            param_sum = self.run_program(place, [data, label])
        return param_sum

    def test_l2(self):
        for place in self.get_places():
            dense_sparse_p_sum = []
            for sparse in [True, False]:
                model = partial(bow_net, is_sparse=sparse)
                framework_l2 = self.check_l2decay_regularizer(place, model)
                l2 = self.check_l2decay(place, model)
                assert len(l2) == len(framework_l2)
                for i in range(len(l2)):
                    assert np.isclose(a=framework_l2[i], b=l2[i], rtol=5e-5)
                dense_sparse_p_sum.append(framework_l2)

            assert len(dense_sparse_p_sum[0]) == len(dense_sparse_p_sum[1])
            for i in range(len(dense_sparse_p_sum[0])):
                assert np.isclose(
                    a=dense_sparse_p_sum[0][i],
                    b=dense_sparse_p_sum[1][i],
                    rtol=5e-5)

    def test_repeated_regularization(self):
        l1 = fluid.regularizer.L1Decay(regularization_coeff=0.1)
        l2 = fluid.regularizer.L2Decay(regularization_coeff=0.01)
        fc_param_attr = fluid.ParamAttr(regularizer=l1)
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.layers.uniform_random([2, 2, 3])
            out = fluid.layers.fc(x, 5, param_attr=fc_param_attr)
            loss = fluid.layers.reduce_sum(out)
            sgd = fluid.optimizer.SGD(learning_rate=0.1, regularization=l2)
            sgd.minimize(loss)
        with fluid.dygraph.guard():
            input = fluid.dygraph.to_variable(
                np.random.randn(3, 2).astype('float32'))
            paddle.seed(1)
            paddle.framework.random._manual_program_seed(1)

            linear1 = fluid.dygraph.Linear(
                2, 2, param_attr=fc_param_attr, bias_attr=fc_param_attr)
            linear2 = fluid.dygraph.Linear(
                2, 2, param_attr=fc_param_attr, bias_attr=fc_param_attr)

            loss1 = linear1(input)
            loss1.backward()
            # set l2 regularizer in optimizer, but l1 in fluid.ParamAttr

            fluid.optimizer.SGD(parameter_list=linear1.parameters(),
                                learning_rate=1e-2,
                                regularization=l2).minimize(loss1)
            # only set l1 in fluid.ParamAttr
            loss2 = linear2(input)
            loss2.backward()
            fluid.optimizer.SGD(parameter_list=linear2.parameters(),
                                learning_rate=1e-2).minimize(loss2)
            # they should both be applied by l1, and keep the same
            self.assertTrue(
                np.allclose(linear1.weight.numpy(), linear2.weight.numpy()),
                "weight should use the regularization in fluid.ParamAttr!")
            self.assertTrue(
                np.allclose(linear1.bias.numpy(), linear2.bias.numpy()),
                "bias should use the regularization in fluid.ParamAttr!")


if __name__ == '__main__':
    unittest.main()
