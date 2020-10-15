# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import six
from fake_reader import fake_imdb_reader


def bow_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    fluid/PaddleNLP/text_classification/nets.py
    """
    emb = fluid.layers.embedding(
        input=data, is_sparse=True, size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return avg_cost


class TestGradientClip(unittest.TestCase):
    def setUp(self):
        self.word_dict_len = 5147
        self.BATCH_SIZE = 2
        reader = fake_imdb_reader(self.word_dict_len, self.BATCH_SIZE * 100)
        self.train_data = paddle.batch(reader, batch_size=self.BATCH_SIZE)
        self.init()

    def init(self):
        pass

    def get_places(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        return places

    def clip_gradient(self, params_grads):
        pass

    def check_clip_result(self, out, out_clip):
        pass

    def check_gradient_clip(self, place):
        prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            image = fluid.data(name="a", shape=[-1, 784], dtype='float32')
            label = fluid.data(name="b", shape=[-1, 1], dtype='int64')
            hidden = fluid.layers.fc(input=image, size=32, act='relu')
            predict = fluid.layers.fc(input=hidden, size=10, act='softmax')

            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(cost)

        prog_clip = prog.clone()
        avg_cost_clip = prog_clip.block(0).var(avg_cost.name)

        p_g = fluid.backward.append_backward(loss=avg_cost)
        p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

        p_g = sorted(p_g, key=lambda x: x[0].name)
        p_g_clip = sorted(p_g_clip, key=lambda x: x[0].name)
        with fluid.program_guard(
                main_program=prog_clip, startup_program=startup_program):
            p_g_clip = self.clip_gradient(p_g_clip)

        grad_list = [elem[1] for elem in p_g]
        grad_clip_list = [elem[1] for elem in p_g_clip]

        train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=3)
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
        exe.run(startup_program)

        data = next(train_reader())
        out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
        out_clip = exe.run(prog_clip,
                           feed=feeder.feed(data),
                           fetch_list=grad_clip_list)
        self.check_clip_result(out, out_clip)

    def check_sparse_gradient_clip(self, place):
        prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            data = fluid.data(
                name="words", shape=[-1, 1], dtype="int64", lod_level=1)
            label = fluid.data(name="label", shape=[-1, 1], dtype="int64")
            cost = bow_net(data, label, self.word_dict_len)

            self.backward_and_optimize(cost)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
        exe.run(startup_program)

        data = next(self.train_data())
        val = exe.run(prog, feed=feeder.feed(data), fetch_list=[cost])[0]
        self.assertEqual((1, ), val.shape)
        print(val)
        self.assertFalse(np.isnan(val))

    def backward_and_optimize(cost):
        pass


class TestGradientClipByGlobalNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def clip_gradient(self, params_grads):
        clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=self.clip_norm)
        print(clip)
        return clip(params_grads)

    def check_clip_result(self, out, out_clip):
        global_norm = 0
        for v in out:
            global_norm += np.sum(np.power(v, 2))
        global_norm = np.sqrt(global_norm)
        scale = self.clip_norm / np.maximum(self.clip_norm, global_norm)
        res = []
        for i in range(len(out)):
            out[i] = scale * out[i]

        for u, v in zip(out, out_clip):
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-5, atol=1e-8),
                "gradient clip by global norm has wrong results!")

    # test whether the ouput is right when use 'set_gradient_clip'
    def test_old_gradient_clip(self):
        def func(params_grads):
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=self.clip_norm)
            fluid.clip.set_gradient_clip(clip)
            return fluid.clip.append_gradient_clip_ops(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(fluid.CPUPlace())

    # test whether the ouput is right when use grad_clip
    def test_new_gradient_clip(self):
        def func(params_grads):
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=self.clip_norm)
            print(clip)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(fluid.CPUPlace())

    # invoke 'set_gradient_clip' in a wrong order
    def test_wrong_API_order(self):
        def backward_func(cost):
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0)
            fluid.clip.set_gradient_clip(clip)
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01,
                                                grad_clip=clip)
            # if 'set_gradient_clip' and 'optimize(grad_clip)' together, 'set_gradient_clip' will be ineffective
            sgd_optimizer.minimize(cost)
            # 'set_gradient_clip' must before 'minimize', otherwise, 'set_gradient_clip' will be ineffective
            fluid.clip.set_gradient_clip(clip)

        self.backward_and_optimize = backward_func
        for place in self.get_places():
            self.check_sparse_gradient_clip(place)

    # if grad is None or not need clip
    def test_none_grad(self):
        clip = fluid.clip.GradientClipByGlobalNorm(self.clip_norm)
        x = fluid.default_main_program().global_block().create_parameter(
            name="x", shape=[2, 3], dtype="float32")
        y = fluid.default_main_program().global_block().create_parameter(
            name="y", shape=[2, 3], dtype="float32")

        # (x, None) should not be returned
        params_grads = [(x, None), (x, y), (y, x)]
        params_grads = clip(params_grads)
        self.assertTrue(
            len(clip(params_grads)) == 2,
            "ClipByGlobalNorm: when grad is None, it shouldn't be returned by gradient clip!"
        )
        self.assertTrue(
            params_grads[0][1].name != 'y',
            "ClipByGlobalNorm: param_grad (x, y) should be clipped!")

    # raise typeError
    def test_tpyeError(self):
        # the type of optimizer(grad_clip=) must be an instance of GradientClipBase's derived class
        with self.assertRaises(TypeError):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1,
                                                grad_clip="test")


class TestGradientClipByNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def clip_gradient(self, params_grads):
        clip = fluid.clip.GradientClipByNorm(clip_norm=self.clip_norm)
        print(clip)
        return clip(params_grads)

    def check_clip_result(self, out, out_clip):
        for u, v in zip(out, out_clip):
            norm = np.sqrt(np.sum(np.power(u, 2)))
            scale = self.clip_norm / np.maximum(self.clip_norm, norm)
            u = u * scale
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-5, atol=1e-8),
                "gradient clip by norm has wrong results!")

    # test whether the ouput is right when use grad_clip
    def test_gradient_clip(self):
        self.check_gradient_clip(fluid.CPUPlace())

    # if grad is None or not need clip
    def test_none_grad(self):
        clip = fluid.clip.GradientClipByNorm(self.clip_norm)
        x = fluid.default_main_program().global_block().create_parameter(
            name="x", shape=[2, 3], dtype="float32", need_clip=False)
        y = fluid.default_main_program().global_block().create_parameter(
            name="y", shape=[2, 3], dtype="float32", need_clip=False)

        # (x, None) should not be returned
        params_grads = [(x, None), (x, y)]
        params_grads = clip(params_grads)
        self.assertTrue(
            len(clip(params_grads)) == 1,
            "ClipGradByNorm: when grad is None, it shouldn't be returned by gradient clip!"
        )
        self.assertTrue(
            params_grads[0][1].name == 'y',
            "ClipGradByNorm: grad should not be clipped when filtered out!")


class TestGradientClipByValue(TestGradientClip):
    def init(self):
        self.max = 0.2
        self.min = 0.1

    def clip_gradient(self, params_grads):
        clip = fluid.clip.GradientClipByValue(max=self.max, min=self.min)
        print(clip)
        return clip(params_grads)

    def check_clip_result(self, out, out_clip):
        for i, v in enumerate(out):
            out[i] = np.clip(v, self.min, self.max)
        for u, v in zip(out, out_clip):
            u = np.clip(u, self.min, self.max)
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-6, atol=1e-8),
                "gradient clip by value has wrong results!")

    # test whether the ouput is right when use grad_clip
    def test_gradient_clip(self):
        self.check_gradient_clip(fluid.CPUPlace())

    # if grad is None or not need clip
    def test_none_grad(self):
        clip = fluid.clip.GradientClipByValue(self.max, self.min)
        x = fluid.default_main_program().global_block().create_parameter(
            name="x", shape=[2, 3], dtype="float32", need_clip=False)
        y = fluid.default_main_program().global_block().create_parameter(
            name="y", shape=[2, 3], dtype="float32", need_clip=False)

        # (x, None) should not be returned
        params_grads = [(x, None), (x, y)]
        params_grads = clip(params_grads)
        self.assertTrue(
            len(clip(params_grads)) == 1,
            "ClipGradByValue: when grad is None, it shouldn't be returned by gradient clip!"
        )
        self.assertTrue(
            params_grads[0][1].name == 'y',
            "ClipGradByValue: grad should not be clipped when filtered out!")


class TestDygraphGradientClip(unittest.TestCase):
    def test_gradient_clip(self):
        with fluid.dygraph.guard():
            linear = fluid.dygraph.Linear(5, 5)
            inputs = fluid.layers.uniform_random(
                [16, 5], min=-10, max=10).astype('float32')
            out = linear(fluid.dygraph.to_variable(inputs))
            loss = fluid.layers.reduce_mean(out)
            loss.backward()
            sgd_optimizer = fluid.optimizer.SGD(
                learning_rate=0.0,
                parameter_list=linear.parameters(),
                grad_clip=fluid.clip.GradientClipByGlobalNorm(0.1))
            self.check_clip_result(loss, sgd_optimizer)

    def check_clip_result(self, loss, optimizer):
        pass


class TestDygraphGradientClipByGlobalNorm(TestDygraphGradientClip):
    def setUp(self):
        self.clip_norm = 0.8
        self.clip1 = fluid.clip.GradientClipByGlobalNorm(
            clip_norm=self.clip_norm)
        self.clip2 = fluid.clip.GradientClipByGlobalNorm(
            clip_norm=self.clip_norm)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = fluid.dygraph.to_variable(
            np.array([2, 3]).astype("float32"), name="x")
        y = fluid.dygraph.to_variable(
            np.array([3, 4]).astype("float32"), name="y")
        assert len(self.clip1([(x, x), (x, y), (x, None)])) == 2
        # get params and grads from network
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip2(params_grads)
        _, grads_clip = zip(*params_grads)

        global_norm = 0
        for u in grads:
            u = u.numpy()
            global_norm += np.sum(np.power(u, 2))
        global_norm = np.sqrt(global_norm)

        global_norm_clip = 0
        for v in grads_clip:
            v = v.numpy()
            global_norm_clip += np.sum(np.power(v, 2))
        global_norm_clip = np.sqrt(global_norm_clip)

        a = np.minimum(global_norm, self.clip_norm)
        b = global_norm_clip
        self.assertTrue(
            np.isclose(
                a=a, b=b, rtol=1e-6, atol=1e-8),
            "gradient clip by global norm has wrong results, expetcd:%f, but recieved:%f"
            % (a, b))


class TestDygraphGradientClipByNorm(TestDygraphGradientClip):
    def setUp(self):
        self.clip_norm = 0.8
        self.clip = fluid.clip.GradientClipByNorm(clip_norm=self.clip_norm)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = fluid.dygraph.to_variable(np.array([2, 3]).astype("float32"))
        assert len(self.clip([(x, None)])) == 0
        # get params and grads from network
        self.clip([(fluid.dygraph.to_variable(np.array([2, 3])), None)])
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)

        for u, v in zip(grads, grads_clip):
            u = u.numpy()
            v = v.numpy()
            a = np.sqrt(np.sum(np.power(u, 2)))
            a = np.minimum(a, self.clip_norm)
            b = np.sqrt(np.sum(np.power(v, 2)))
            self.assertTrue(
                np.isclose(
                    a=a, b=b, rtol=1e-6, atol=1e-8),
                "gradient clip by norm has wrong results, expetcd:%f, but recieved:%f"
                % (a, b))


class TestDygraphGradientClipByValue(TestDygraphGradientClip):
    def setUp(self):
        self.max = 0.2
        self.min = 0.1
        self.clip = fluid.clip.GradientClipByValue(max=self.max, min=self.min)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = fluid.dygraph.to_variable(np.array([2, 3]).astype("float32"))
        assert len(self.clip([(x, None)])) == 0
        # get params and grads from network
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)
        for u, v in zip(grads, grads_clip):
            u = np.clip(u.numpy(), self.min, self.max)
            v = v.numpy()
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=1e-6, atol=1e-8),
                "gradient clip by value has wrong results!")


if __name__ == '__main__':
    unittest.main()
