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

import contextlib
import numpy as np
from decorator_helper import prog_scope
import inspect
from six.moves import filter

import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.device import get_places
import paddle.fluid.nets as nets
from paddle.fluid.framework import Program, program_guard, default_main_program
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid import core
from paddle.fluid.initializer import Constant
import paddle.fluid.layers as layers
from test_imperative_base import new_program_scope
from paddle.fluid.dygraph import nn
from paddle.fluid.dygraph import base
from paddle.fluid.dygraph import to_variable


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        with new_program_scope():
            paddle.manual_seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield

    def get_static_graph_result(self,
                                feed,
                                fetch_list,
                                with_lod=False,
                                force_to_use_cpu=False):
        exe = fluid.Executor(self._get_place(force_to_use_cpu))
        exe.run(fluid.default_startup_program())
        return exe.run(fluid.default_main_program(),
                       feed=feed,
                       fetch_list=fetch_list,
                       return_numpy=(not with_lod))

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with fluid.dygraph.guard(
                self._get_place(force_to_use_cpu=force_to_use_cpu)):
            paddle.manual_seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield


class TestLayer(LayerTest):
    def test_custom_layer_with_kwargs(self):
        class CustomLayer(fluid.Layer):
            def __init__(self, input_size, linear1_size=4):
                super(CustomLayer, self).__init__()
                self.linear1 = nn.Linear(
                    input_size, linear1_size, bias_attr=False)
                self.linear2 = nn.Linear(linear1_size, 1, bias_attr=False)

            def forward(self, x, do_linear2=False):
                ret = self.linear1(x)
                if do_linear2:
                    ret = self.linear2(ret)
                return ret

        with self.dynamic_graph():
            inp = np.ones([3, 3], dtype='float32')
            x = base.to_variable(inp)
            custom = CustomLayer(input_size=3, linear1_size=2)
            ret = custom(x, do_linear2=False)
            self.assertTrue(np.array_equal(ret.numpy().shape, [3, 2]))
            ret = custom(x, do_linear2=True)
            self.assertTrue(np.array_equal(ret.numpy().shape, [3, 1]))

    def test_dropout(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            dropout = nn.Dropout(p=0.35, seed=1, is_test=False)
            ret = dropout(t)
            ret2 = fluid.layers.dropout(
                t, dropout_prob=0.35, seed=1, is_test=False)
            static_ret, static_ret2 = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret, ret2])
        with self.dynamic_graph():
            t = base.to_variable(inp)
            dropout = nn.Dropout(p=0.35, seed=1, is_test=False)
            dy_ret = dropout(t)
            dy_ret2 = fluid.layers.dropout(
                t, dropout_prob=0.35, seed=1, is_test=False)
            dy_ret_value = dy_ret.numpy()
            dy_ret2_value = dy_ret2.numpy()

        self.assertTrue(np.array_equal(static_ret, static_ret2))
        self.assertTrue(np.array_equal(dy_ret_value, dy_ret2_value))
        self.assertTrue(np.array_equal(static_ret, dy_ret_value))

    def test_linear(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            linear = nn.Linear(
                32, 4, bias_attr=fluid.initializer.ConstantInitializer(value=1))
            ret = linear(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret])[0]
        with self.dynamic_graph():
            t = base.to_variable(inp)
            linear = nn.Linear(
                32, 4, bias_attr=fluid.initializer.ConstantInitializer(value=1))
            dy_ret = linear(t)
            dy_ret_value = dy_ret.numpy()

        self.assertTrue(np.array_equal(static_ret, dy_ret_value))

        with self.static_graph():

            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1))
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1))
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    def test_Flatten(self):
        inp = np.ones([3, 4, 4, 5], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 4, 4, 5],
                dtype='float32',
                append_batch_size=False)
            flatten = nn.Flatten()
            ret = flatten(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret])[0]
        with self.dynamic_graph():
            t = base.to_variable(inp)
            flatten = nn.Flatten()
            dy_ret = flatten(t)
            dy_ret_value = dy_ret.numpy()

        self.assertTrue(np.array_equal(static_ret, dy_ret_value))

        with self.static_graph():

            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1))
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = nn.Linear(
                    32,
                    4,
                    bias_attr=fluid.initializer.ConstantInitializer(value=1))
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    def test_layer_norm(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            ret = layers.layer_norm(
                t,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret])[0]
        with self.static_graph():
            t = layers.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            lm = nn.LayerNorm(
                normalized_shape=[32, 32],
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')
            ret = lm(t)
            static_ret2 = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret])[0]
        with self.dynamic_graph():
            lm = nn.LayerNorm(
                normalized_shape=[32, 32],
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')
            dy_ret = lm(base.to_variable(inp))
            dy_ret_value = dy_ret.numpy()
        with self.dynamic_graph():
            lm = nn.LayerNorm(
                normalized_shape=[32, 32],
                shift=False,
                scale=False,
                param_attr=fluid.initializer.ConstantInitializer(value=1),
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')
            lm(base.to_variable(inp))

            self.assertFalse(hasattr(lm, "_scale_w"))
            self.assertFalse(hasattr(lm, "_bias_w"))

        self.assertTrue(np.array_equal(static_ret, static_ret2))
        self.assertTrue(np.array_equal(dy_ret_value, static_ret2))

        with self.dynamic_graph():
            lm = nn.LayerNorm(
                normalized_shape=[16, 32],
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')
            with self.assertRaises(ValueError):
                lm(base.to_variable(inp))

    def test_SyncBatchNorm(self):
        if core.is_compiled_with_cuda():
            with self.static_graph():
                t = layers.data(name='t', shape=[-1, 3, 5, 5], dtype='float32')
                my_sync_bn = paddle.nn.SyncBatchNorm(3)
                ret = my_sync_bn(t)
                static_ret = self.get_static_graph_result(
                    feed={'t': np.ones(
                        [3, 3, 5, 5], dtype='float32')},
                    fetch_list=[ret])[0]

            with self.dynamic_graph():
                t = np.ones([3, 3, 5, 5], dtype='float32')
                my_syncbn = paddle.nn.SyncBatchNorm(3)
                dy_ret = my_syncbn(base.to_variable(t))
                dy_ret_value = dy_ret.numpy()
            self.assertTrue(np.array_equal(static_ret, dy_ret_value))

    def test_relu(self):
        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            ret = layers.relu(t)
            static_ret = self.get_static_graph_result(
                feed={'t': np.ones(
                    [3, 3], dtype='float32')}, fetch_list=[ret])[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            dy_ret = layers.relu(base.to_variable(t))
            dy_ret_value = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, dy_ret_value))

    def test_pad2d(self):
        with self.static_graph():
            t = layers.data(name='t', shape=[-1, 3, 5, 5], dtype='float32')
            ret = layers.pad2d(t, paddings=[1, 1, 1, 1])
            static_ret = self.get_static_graph_result(
                feed={'t': np.ones(
                    [3, 3, 5, 5], dtype='float32')},
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            t = np.ones([3, 3, 5, 5], dtype='float32')
            my_pad2d = paddle.nn.layer.Pad2D(paddings=1)
            dy_ret = my_pad2d(base.to_variable(t))
            dy_ret_value = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, dy_ret_value))

    def test_matmul(self):
        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            t2 = layers.data(name='t2', shape=[3, 3], dtype='float32')
            ret = layers.matmul(t, t2)
            static_ret = self.get_static_graph_result(
                feed={
                    't': np.ones(
                        [3, 3], dtype='float32'),
                    't2': np.ones(
                        [3, 3], dtype='float32')
                },
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            t2 = np.ones([3, 3], dtype='float32')
            dy_ret = layers.matmul(base.to_variable(t), base.to_variable(t2))
            dy_ret_value = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, dy_ret_value))

    def test_conv2d(self):
        with self.static_graph():
            images = layers.data(name='pixel', shape=[3, 5, 5], dtype='float32')
            ret = layers.conv2d(input=images, num_filters=3, filter_size=[2, 2])
            static_ret = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 5, 5], dtype='float32')},
                fetch_list=[ret])[0]

        with self.static_graph():
            images = layers.data(name='pixel', shape=[3, 5, 5], dtype='float32')
            conv2d = nn.Conv2D(
                num_channels=3, num_filters=3, filter_size=[2, 2])
            ret = conv2d(images)
            static_ret2 = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 5, 5], dtype='float32')},
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            images = np.ones([2, 3, 5, 5], dtype='float32')
            conv2d = nn.Conv2D(
                num_channels=3, num_filters=3, filter_size=[2, 2])
            dy_ret = conv2d(base.to_variable(images))
            dy_ret_value = dy_ret.numpy()

        with self.dynamic_graph():
            images = np.ones([2, 3, 5, 5], dtype='float32')
            conv2d = nn.Conv2D(
                num_channels=3,
                num_filters=3,
                filter_size=[2, 2],
                bias_attr=False)
            dy_ret = conv2d(base.to_variable(images))
            self.assertTrue(conv2d.bias is None)

        with self.static_graph():
            # the input of Conv2D must be Variable.
            def test_Variable():
                images = np.ones([2, 3, 5, 5], dtype='float32')
                conv2d = nn.Conv2D(
                    num_channels=3, num_filters=3, filter_size=[2, 2])
                conv2d_ret1 = conv2d(images)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Conv2D must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                images = layers.data(
                    name='pixel', shape=[3, 5, 5], dtype='int32')
                conv2d = nn.Conv2D(
                    num_channels=3, num_filters=3, filter_size=[2, 2])
                conv2d_ret2 = conv2d(images)

            self.assertRaises(TypeError, test_type)

        self.assertTrue(np.allclose(static_ret, dy_ret_value))
        self.assertTrue(np.allclose(static_ret, static_ret2))

        with self.dynamic_graph():
            images = np.ones([2, 3, 5, 5], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            conv2d1 = nn.Conv2D(
                num_channels=3, num_filters=3, filter_size=[2, 2])
            conv2d2 = nn.Conv2D(
                num_channels=3,
                num_filters=3,
                filter_size=[2, 2],
                param_attr=weight_attr)
            dy_ret1 = conv2d1(base.to_variable(images))
            dy_ret2 = conv2d2(base.to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv2d1_weight_np = conv2d1.weight.numpy()
            conv2d1_bias = conv2d1.bias
            self.assertFalse(
                np.array_equal(conv2d1_weight_np, conv2d2.weight.numpy()))
            conv2d2.weight.set_value(conv2d1_weight_np)
            self.assertTrue(
                np.array_equal(conv2d1_weight_np, conv2d2.weight.numpy()))
            conv2d2.bias.set_value(conv2d1_bias)
            dy_ret1 = conv2d1(base.to_variable(images))
            dy_ret2 = conv2d2(base.to_variable(images))
            self.assertTrue(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv2d2.weight = conv2d1.weight
            conv2d2.bias = conv2d1.bias
            self.assertTrue(
                np.array_equal(conv2d1.weight.numpy(), conv2d2.weight.numpy()))
            self.assertTrue(
                np.array_equal(conv2d1.bias.numpy(), conv2d2.bias.numpy()))

    def test_gru_unit(self):
        lod = [[2, 4, 3]]
        D = 5
        T = sum(lod[0])
        N = len(lod[0])

        input = np.random.rand(T, 3 * D).astype('float32')
        hidden_input = np.random.rand(T, D).astype('float32')

        with self.static_graph():
            x = layers.data(name='x', shape=[-1, D * 3], dtype='float32')
            hidden = layers.data(name='hidden', shape=[-1, D], dtype='float32')
            updated_hidden, reset_hidden_pre, gate = layers.gru_unit(
                input=x, hidden=hidden, size=D * 3)
            static_ret = self.get_static_graph_result(
                feed={'x': input,
                      'hidden': hidden_input},
                fetch_list=[updated_hidden, reset_hidden_pre, gate])

        with self.static_graph():
            x = layers.data(name='x', shape=[-1, D * 3], dtype='float32')
            hidden = layers.data(name='hidden', shape=[-1, D], dtype='float32')
            updated_hidden, reset_hidden_pre, gate = layers.gru_unit(
                input=x, hidden=hidden, size=D * 3)
            gru = nn.GRUUnit(size=D * 3)
            updated_hidden, reset_hidden_pre, gate = gru(x, hidden)

            static_ret2 = self.get_static_graph_result(
                feed={'x': input,
                      'hidden': hidden_input},
                fetch_list=[updated_hidden, reset_hidden_pre, gate])

        with self.dynamic_graph():
            gru = nn.GRUUnit(size=D * 3)
            dy_ret = gru(
                base.to_variable(input), base.to_variable(hidden_input))
            dy_ret_value = []
            for i in range(len(static_ret)):
                dy_ret_value.append(dy_ret[i].numpy())

        for i in range(len(static_ret)):
            self.assertTrue(np.allclose(static_ret[i], static_ret2[i]))
            self.assertTrue(np.allclose(static_ret[i], dy_ret_value[i]))

        with self.dynamic_graph():
            custom_weight = np.random.randn(D, D * 3).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            gru1 = nn.GRUUnit(size=D * 3)
            gru2 = nn.GRUUnit(size=D * 3, param_attr=weight_attr)
            dy_ret1 = gru1(
                base.to_variable(input), base.to_variable(hidden_input))
            dy_ret2 = gru2(
                base.to_variable(input), base.to_variable(hidden_input))
            self.assertFalse(
                np.array_equal(gru1.weight.numpy(), gru2.weight.numpy()))
            for o1, o2 in zip(dy_ret1, dy_ret2):
                self.assertFalse(np.array_equal(o1.numpy(), o2.numpy()))
            gru2.weight.set_value(gru1.weight.numpy())
            gru2.bias.set_value(gru1.bias)
            dy_ret1 = gru1(
                base.to_variable(input), base.to_variable(hidden_input))
            dy_ret2 = gru2(
                base.to_variable(input), base.to_variable(hidden_input))
            for o1, o2 in zip(dy_ret1, dy_ret2):
                self.assertTrue(np.array_equal(o1.numpy(), o2.numpy()))

            gru2.weight = gru1.weight
            gru2.bias = gru1.bias
            self.assertTrue(
                np.array_equal(gru1.weight.numpy(), gru2.weight.numpy()))
            self.assertTrue(
                np.array_equal(gru1.bias.numpy(), gru2.bias.numpy()))

    def test_elementwise_math(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 1.1
        n3 = np.ones([3, 3], dtype='float32') * 2
        n4 = np.ones([3, 3], dtype='float32') * 3
        n5 = np.ones([3, 3], dtype='float32') * 4
        n6 = np.ones([3, 3], dtype='float32') * 5

        with self.static_graph():
            t = layers.data(name='t', shape=[3, 3], dtype='float32')
            t2 = layers.data(name='t2', shape=[3, 3], dtype='float32')
            t3 = layers.data(name='t3', shape=[3, 3], dtype='float32')
            t4 = layers.data(name='t4', shape=[3, 3], dtype='float32')
            t5 = layers.data(name='t5', shape=[3, 3], dtype='float32')
            t6 = layers.data(name='t6', shape=[3, 3], dtype='float32')

            ret = layers.elementwise_add(t, t2)
            ret = layers.elementwise_pow(ret, t3)
            ret = layers.elementwise_div(ret, t4)
            ret = layers.elementwise_sub(ret, t5)
            ret = layers.elementwise_mul(ret, t6)

            static_ret = self.get_static_graph_result(
                feed={
                    't': n,
                    't2': n2,
                    't3': n3,
                    't4': n4,
                    't5': n5,
                    't6': n6
                },
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            ret = layers.elementwise_add(to_variable(n), to_variable(n2))
            ret = layers.elementwise_pow(ret, to_variable(n3))
            ret = layers.elementwise_div(ret, to_variable(n4))
            ret = layers.elementwise_sub(ret, to_variable(n5))
            dy_ret = layers.elementwise_mul(ret, to_variable(n6))
            dy_ret_value = dy_ret.numpy()
        self.assertTrue(np.allclose(static_ret, dy_ret_value))

    def test_elementwise_minmax(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 2

        with self.dynamic_graph():
            min_ret = layers.elementwise_min(to_variable(n), to_variable(n2))
            max_ret = layers.elementwise_max(to_variable(n), to_variable(n2))
            min_ret_value = min_ret.numpy()
            max_ret_value = max_ret.numpy()

        self.assertTrue(np.allclose(n, min_ret_value))
        self.assertTrue(np.allclose(n2, max_ret_value))

    def test_sequence_conv(self):
        inp_np = np.arange(12).reshape([3, 4]).astype('float32')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with self.static_graph():
            seq = layers.data(
                name='seq_in',
                shape=[3, 4],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            out = layers.sequence_conv(seq, 2, act='sigmoid')
            static_rlt = self.get_static_graph_result(
                feed={
                    "seq_in": fluid.create_lod_tensor(
                        data=inp_np,
                        recursive_seq_lens=[[1, 1, 1]],
                        place=place)
                },
                fetch_list=[out],
                with_lod=True)[0]

        with self.static_graph():
            seq = layers.data(
                name='seq_in',
                shape=[3, 4],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            seq_conv = nn.SequenceConv('seq_conv', num_filters=2, act='sigmoid')
            out = seq_conv(seq)
            static_rlt2 = self.get_static_graph_result(
                feed={
                    "seq_in": fluid.create_lod_tensor(
                        data=inp_np,
                        recursive_seq_lens=[[1, 1, 1]],
                        place=place)
                },
                fetch_list=[out],
                with_lod=True)[0]
        self.assertTrue(
            np.array_equal(np.array(static_rlt), np.array(static_rlt2)))

    def test_conv2d_transpose(self):
        inp_np = np.arange(0, 24).reshape([2, 3, 2, 2]).astype('float32')
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            out = layers.conv2d_transpose(
                input=img,
                num_filters=10,
                filter_size=27,
                act='sigmoid',
                bias_attr=fluid.initializer.ConstantInitializer(value=1))
            static_rlt = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out])[0]
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2], dtype='float32')
            conv2d_transpose = nn.Conv2DTranspose(
                num_channels=3,
                num_filters=10,
                filter_size=27,
                act='sigmoid',
                bias_attr=fluid.initializer.ConstantInitializer(value=1))
            out = conv2d_transpose(img)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out])[0]
        with self.dynamic_graph():
            conv2d_transpose = nn.Conv2DTranspose(
                num_channels=3,
                num_filters=10,
                filter_size=27,
                act='sigmoid',
                bias_attr=fluid.initializer.ConstantInitializer(value=1))
            dy_rlt = conv2d_transpose(base.to_variable(inp_np))
            dy_rlt_value = dy_rlt.numpy()
        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt_value, static_rlt2))

        with self.dynamic_graph():
            images = np.ones([2, 3, 5, 5], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            conv2d1 = nn.Conv2DTranspose(
                num_channels=3, num_filters=3, filter_size=[2, 2])
            conv2d2 = nn.Conv2DTranspose(
                num_channels=3,
                num_filters=3,
                filter_size=[2, 2],
                param_attr=weight_attr)
            dy_ret1 = conv2d1(base.to_variable(images))
            dy_ret2 = conv2d2(base.to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv2d1_weight_np = conv2d1.weight.numpy()
            conv2d1_bias = conv2d1.bias
            self.assertFalse(
                np.array_equal(conv2d1_weight_np, conv2d2.weight.numpy()))
            conv2d2.weight.set_value(conv2d1_weight_np)
            self.assertTrue(
                np.array_equal(conv2d1_weight_np, conv2d2.weight.numpy()))
            conv2d2.bias.set_value(conv2d1_bias)
            dy_ret1 = conv2d1(base.to_variable(images))
            dy_ret2 = conv2d2(base.to_variable(images))
            self.assertTrue(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv2d2.weight = conv2d1.weight
            conv2d2.bias = conv2d1.bias
            self.assertTrue(
                np.array_equal(conv2d1.weight.numpy(), conv2d2.weight.numpy()))
            self.assertTrue(
                np.array_equal(conv2d1.bias.numpy(), conv2d2.bias.numpy()))

        with self.static_graph():

            # the input of Conv2DTranspose must be Variable.
            def test_Variable():
                images = np.ones([2, 3, 5, 5], dtype='float32')
                conv2d = nn.Conv2DTranspose(
                    num_channels=3, num_filters=3, filter_size=[2, 2])
                conv2d_ret1 = conv2d(images)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Conv2DTranspose must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                images = layers.data(
                    name='pixel', shape=[3, 5, 5], dtype='int32')
                conv2d = nn.Conv2DTranspose(
                    num_channels=3, num_filters=3, filter_size=[2, 2])
                conv2d_ret2 = conv2d(images)

            self.assertRaises(TypeError, test_type)

    def test_bilinear_tensor_product(self):
        inp_np_x = np.array([[1, 2, 3]]).astype('float32')
        inp_np_y = np.array([[4, 5, 6]]).astype('float32')

        with self.static_graph():
            data_x = layers.data(
                name='x',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            data_y = layers.data(
                name='y',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            out = layers.bilinear_tensor_product(
                data_x,
                data_y,
                6,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')

            static_rlt = self.get_static_graph_result(
                feed={'x': inp_np_x,
                      'y': inp_np_y}, fetch_list=[out])[0]

        with self.static_graph():
            data_x = layers.data(
                name='x',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            data_y = layers.data(
                name='y',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            btp = nn.BilinearTensorProduct(
                3,
                3,
                6,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')
            out = btp(data_x, data_y)
            static_rlt2 = self.get_static_graph_result(
                feed={'x': inp_np_x,
                      'y': inp_np_y}, fetch_list=[out])[0]
        with self.dynamic_graph():
            btp = nn.BilinearTensorProduct(
                3,
                3,
                6,
                bias_attr=fluid.initializer.ConstantInitializer(value=1),
                act='sigmoid')
            dy_rlt = btp(base.to_variable(inp_np_x), base.to_variable(inp_np_y))
            dy_rlt_value = dy_rlt.numpy()
        with self.dynamic_graph():
            btp2 = nn.BilinearTensorProduct(3, 3, 6, act='sigmoid')
            dy_rlt2 = btp2(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y))
            dy_rlt2_value = dy_rlt2.numpy()
        with self.static_graph():
            data_x2 = layers.data(
                name='x',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            data_y2 = layers.data(
                name='y',
                shape=[1, 3],
                dtype="float32",
                append_batch_size=False)
            out2 = layers.bilinear_tensor_product(
                data_x2, data_y2, 6, act='sigmoid')

            static_rlt3 = self.get_static_graph_result(
                feed={'x': inp_np_x,
                      'y': inp_np_y}, fetch_list=[out2])[0]

        self.assertTrue(np.array_equal(dy_rlt2_value, static_rlt3))
        self.assertTrue(np.array_equal(static_rlt2, static_rlt))
        self.assertTrue(np.array_equal(dy_rlt_value, static_rlt))

        with self.dynamic_graph():
            custom_weight = np.random.randn(6, 3, 3).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            btp1 = nn.BilinearTensorProduct(3, 3, 6, act='sigmoid')
            btp2 = nn.BilinearTensorProduct(
                3, 3, 6, act='sigmoid', param_attr=weight_attr)
            dy_rlt1 = btp1(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y))
            dy_rlt2 = btp2(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y))
            self.assertFalse(np.array_equal(dy_rlt1.numpy(), dy_rlt2.numpy()))
            btp2.weight.set_value(btp1.weight.numpy())
            btp2.bias.set_value(btp1.bias)
            dy_rlt1 = btp1(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y))
            dy_rlt2 = btp2(
                base.to_variable(inp_np_x), base.to_variable(inp_np_y))
            self.assertTrue(np.array_equal(dy_rlt1.numpy(), dy_rlt2.numpy()))

            btp2.weight = btp1.weight
            btp2.bias = btp1.bias
            self.assertTrue(
                np.array_equal(btp1.weight.numpy(), btp2.weight.numpy()))
            self.assertTrue(
                np.array_equal(btp1.bias.numpy(), btp2.bias.numpy()))

    def prelu_test(self, mode):
        inp_np = np.ones([5, 200, 100, 100]).astype('float32')
        with self.static_graph():
            data_t = layers.data(
                name="input",
                shape=[5, 200, 100, 100],
                dtype="float32",
                append_batch_size=False)
            out = layers.prelu(
                data_t, mode, param_attr=ParamAttr(initializer=Constant(1.0)))
            static_rlt = self.get_static_graph_result(
                feed={"input": inp_np}, fetch_list=[out])[0]

        with self.static_graph():
            data_t = layers.data(
                name="input",
                shape=[5, 200, 100, 100],
                dtype="float32",
                append_batch_size=False)
            prelu = nn.PRelu(
                mode=mode,
                channel=inp_np.shape[1],
                input_shape=data_t.shape,
                param_attr=ParamAttr(initializer=Constant(1.0)))
            out = prelu(data_t)
            static_rlt2 = self.get_static_graph_result(
                feed={"input": inp_np}, fetch_list=[out])[0]

        with self.dynamic_graph():
            prelu = nn.PRelu(
                mode=mode,
                channel=inp_np.shape[1],
                input_shape=inp_np.shape,
                param_attr=ParamAttr(initializer=Constant(1.0)))
            dy_rlt = prelu(base.to_variable(inp_np))
            dy_rlt_value = dy_rlt.numpy()

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt_value, static_rlt))

        with self.dynamic_graph():
            inp_np = np.random.randn(5, 200, 100, 100).astype("float32")
            inp = base.to_variable(inp_np)
            prelu1 = nn.PRelu(
                mode=mode,
                channel=inp_np.shape[1],
                input_shape=inp_np.shape,
                param_attr=ParamAttr(initializer=Constant(2.0)))
            prelu2 = nn.PRelu(
                mode=mode,
                channel=inp_np.shape[1],
                input_shape=inp_np.shape,
                param_attr=ParamAttr(initializer=Constant(1.0)))
            dy_rlt1 = prelu1(inp)
            dy_rlt2 = prelu2(inp)
            self.assertFalse(
                np.array_equal(prelu1.weight.numpy(), prelu2.weight.numpy()))
            self.assertFalse(np.array_equal(dy_rlt1.numpy(), dy_rlt2.numpy()))
            prelu2.weight.set_value(prelu1.weight.numpy())
            dy_rlt1 = prelu1(inp)
            dy_rlt2 = prelu2(inp)
            self.assertTrue(np.array_equal(dy_rlt1.numpy(), dy_rlt2.numpy()))

            prelu2.weight = prelu1.weight
            self.assertTrue(
                np.array_equal(prelu1.weight.numpy(), prelu2.weight.numpy()))

    def test_prelu(self):
        self.prelu_test("channel")
        self.prelu_test("element")
        self.prelu_test("all")

    def test_embeding(self):
        inp_word = np.array([[[1]]]).astype('int64')
        dict_size = 20
        with self.static_graph():
            data_t = layers.data(name='word', shape=[1], dtype='int64')
            emb = layers.embedding(
                input=data_t,
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False)
            static_rlt = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb])[0]
        with self.static_graph():
            data_t = layers.data(name='word', shape=[1], dtype='int64')
            emb2 = nn.Embedding(
                size=[dict_size, 32], param_attr='emb.w', is_sparse=False)
            emb_rlt = emb2(data_t)
            static_rlt2 = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb_rlt])[0]
        with self.dynamic_graph():
            emb2 = nn.Embedding(
                size=[dict_size, 32], param_attr='emb.w', is_sparse=False)
            dy_rlt = emb2(base.to_variable(inp_word))
            dy_rlt_value = dy_rlt.numpy()

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt_value, static_rlt))

        with self.dynamic_graph():
            custom_weight = np.random.randn(dict_size, 32).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            emb1 = nn.Embedding(size=[dict_size, 32], is_sparse=False)
            emb2 = nn.Embedding(
                size=[dict_size, 32], param_attr=weight_attr, is_sparse=False)
            rep1 = emb1(base.to_variable(inp_word))
            rep2 = emb2(base.to_variable(inp_word))
            self.assertFalse(np.array_equal(emb1.weight.numpy(), custom_weight))
            self.assertTrue(np.array_equal(emb2.weight.numpy(), custom_weight))
            self.assertFalse(np.array_equal(rep1.numpy(), rep2.numpy()))
            emb2.weight.set_value(emb1.weight.numpy())
            rep2 = emb2(base.to_variable(inp_word))
            self.assertTrue(np.array_equal(rep1.numpy(), rep2.numpy()))

            emb2.weight = emb1.weight
            self.assertTrue(
                np.array_equal(emb1.weight.numpy(), emb2.weight.numpy()))

    def test_nce(self):
        window_size = 5
        dict_size = 20
        label_word = int(window_size // 2) + 1
        inp_word = np.array([[1], [2], [3], [4], [5]]).astype('int64')
        nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')
        seed = 1
        with self.static_graph():
            words = []
            for i in range(window_size):
                words.append(
                    layers.data(
                        name='word_{0}'.format(i), shape=[None], dtype='int64'))
            sample_weights = layers.fill_constant(
                shape=[5, 1], dtype='float32', value=1)
            embs = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb = fluid.embedding(
                    input=words[i],
                    size=[dict_size, 32],
                    param_attr='emb.w',
                    is_sparse=False)
                embs.append(emb)

            embs = layers.concat(input=embs, axis=1)
            wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
            nce_loss = layers.nce(input=embs,
                                  label=wl,
                                  num_total_classes=dict_size,
                                  num_neg_samples=2,
                                  sampler="custom_dist",
                                  custom_dist=nid_freq_arr.tolist(),
                                  seed=seed,
                                  param_attr='nce.w',
                                  bias_attr='nce.b',
                                  sample_weight=sample_weights)
            feed_dict = dict()
            for i in range(window_size):
                feed_dict['word_{0}'.format(i)] = inp_word[i]
            static_rlt = self.get_static_graph_result(
                feed=feed_dict, fetch_list=[nce_loss])[0]
        with self.static_graph():
            words = []
            for i in range(window_size):
                words.append(
                    layers.data(
                        name='word_{0}'.format(i), shape=[None], dtype='int64'))
            sample_weights = layers.fill_constant(
                shape=[5, 1], dtype='float32', value=1)
            emb = nn.Embedding(
                size=[dict_size, 32], param_attr='emb.w', is_sparse=False)

            embs2 = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb_rlt = emb(words[i])
                embs2.append(emb_rlt)

            embs2 = layers.concat(input=embs2, axis=1)
            nce = nn.NCE(num_total_classes=dict_size,
                         dim=embs2.shape[1],
                         num_neg_samples=2,
                         sampler="custom_dist",
                         custom_dist=nid_freq_arr.tolist(),
                         seed=seed,
                         param_attr='nce.w',
                         bias_attr='nce.b',
                         sample_weight=sample_weights)

            wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
            nce_loss2 = nce(embs2, wl)
            feed_dict = dict()
            for i in range(len(words)):
                feed_dict['word_{0}'.format(i)] = inp_word[i]

            static_rlt2 = self.get_static_graph_result(
                feed=feed_dict, fetch_list=[nce_loss2])[0]

        with self.dynamic_graph():
            words = []
            for i in range(window_size):
                words.append(base.to_variable(inp_word[i]))
            sample_weights = layers.fill_constant(
                shape=[5, 1], dtype='float32', value=1)
            emb = nn.Embedding(
                size=[dict_size, 32], param_attr='emb.w', is_sparse=False)

            embs3 = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb_rlt = emb(words[i])
                embs3.append(emb_rlt)

            embs3 = layers.concat(
                input=embs3, axis=fluid.dygraph.to_variable(np.array([1])))
            nce = nn.NCE(num_total_classes=dict_size,
                         dim=embs3.shape[1],
                         num_neg_samples=2,
                         sampler="custom_dist",
                         custom_dist=nid_freq_arr.tolist(),
                         seed=seed,
                         param_attr='nce.w',
                         bias_attr='nce.b',
                         sample_weight=sample_weights)

            wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
            dy_rlt = nce(embs3, wl)
            dy_rlt_value = dy_rlt.numpy()

        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt_value, static_rlt))

        with self.dynamic_graph():
            custom_weight = np.random.randn(dict_size, 128).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            words = []
            for i in range(window_size):
                words.append(base.to_variable(inp_word[i]))
            sample_weights = layers.fill_constant(
                shape=fluid.dygraph.to_variable(np.array([5, 1])),
                dtype='float32',
                value=1)
            emb = nn.Embedding(
                size=[dict_size, 32], param_attr='emb.w', is_sparse=False)

            embs3 = []
            for i in range(window_size):
                if i == label_word:
                    continue

                emb_rlt = emb(words[i])
                embs3.append(emb_rlt)

            embs3 = layers.concat(input=embs3, axis=1)
            nce1 = nn.NCE(num_total_classes=dict_size,
                          dim=embs3.shape[1],
                          num_neg_samples=2,
                          sampler="custom_dist",
                          custom_dist=nid_freq_arr.tolist(),
                          seed=seed,
                          param_attr='nce1.w',
                          bias_attr='nce1.b',
                          sample_weight=sample_weights)

            nce2 = nn.NCE(num_total_classes=dict_size,
                          dim=embs3.shape[1],
                          num_neg_samples=2,
                          sampler="custom_dist",
                          custom_dist=nid_freq_arr.tolist(),
                          seed=seed,
                          param_attr=weight_attr,
                          bias_attr='nce2.b',
                          sample_weight=sample_weights)

            wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
            nce1_loss = nce1(embs3, wl)
            nce2_loss = nce2(embs3, wl)
            self.assertFalse(
                np.array_equal(nce1_loss.numpy(), nce2_loss.numpy()))
            nce2.weight.set_value(nce1.weight.numpy())
            nce2.bias.set_value(nce1.bias)
            nce1_loss = nce1(embs3, wl)
            nce2_loss = nce2(embs3, wl)
            self.assertTrue(
                np.array_equal(nce1_loss.numpy(), nce2_loss.numpy()))

            nce2.weight = nce1.weight
            nce2.bias = nce1.bias
            self.assertTrue(
                np.array_equal(nce1.weight.numpy(), nce2.weight.numpy()))
            self.assertTrue(
                np.array_equal(nce1.bias.numpy(), nce2.bias.numpy()))

    def test_one_hot(self):
        with self.dynamic_graph():
            label = fluid.dygraph.to_variable(np.array([[1], [1], [3], [0]]))
            one_hot_label1 = fluid.layers.one_hot(input=label, depth=4)
            one_hot_label2 = fluid.layers.one_hot(
                input=label, depth=fluid.dygraph.to_variable(np.array([4])))
            self.assertTrue(
                np.array_equal(one_hot_label1.numpy(), one_hot_label2.numpy()))

    def test_split(self):
        with self.dynamic_graph():
            input = fluid.dygraph.to_variable(np.random.random((3, 8, 5)))
            x0, x1 = fluid.layers.split(input, num_or_sections=2, dim=1)
            x00, x11 = fluid.layers.split(
                input,
                num_or_sections=2,
                dim=fluid.dygraph.to_variable(np.array([1])))
            self.assertTrue(np.array_equal(x0.numpy(), x00.numpy()))
            self.assertTrue(np.array_equal(x1.numpy(), x11.numpy()))

    def test_topk(self):
        with self.dynamic_graph():
            input = fluid.dygraph.to_variable(np.random.random((13, 11)))
            top5_values1, top5_indices1 = layers.topk(input, k=5)
            top5_values2, top5_indices2 = layers.topk(
                input, k=fluid.dygraph.to_variable(np.array([5])))
            self.assertTrue(
                np.array_equal(top5_values1.numpy(), top5_values2.numpy()))
            self.assertTrue(
                np.array_equal(top5_indices1.numpy(), top5_indices2.numpy()))

    def test_conv3d(self):
        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 6, 6, 6], dtype='float32')
            ret = layers.conv3d(input=images, num_filters=3, filter_size=2)
            static_ret = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret])[0]

        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 6, 6, 6], dtype='float32')
            conv3d = nn.Conv3D(num_channels=3, num_filters=3, filter_size=2)
            ret = conv3d(images)
            static_ret2 = self.get_static_graph_result(
                feed={'pixel': np.ones(
                    [2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret])[0]

        with self.dynamic_graph():
            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            conv3d = nn.Conv3D(num_channels=3, num_filters=3, filter_size=2)
            dy_ret = conv3d(base.to_variable(images))
            dy_rlt_value = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, dy_rlt_value))
        self.assertTrue(np.allclose(static_ret, static_ret2))

        with self.dynamic_graph():
            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            conv3d1 = nn.Conv3D(num_channels=3, num_filters=3, filter_size=2)
            conv3d2 = nn.Conv3D(
                num_channels=3,
                num_filters=3,
                filter_size=2,
                param_attr=weight_attr)
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d1_weight_np = conv3d1.weight.numpy()
            conv3d1_bias = conv3d1.bias
            self.assertFalse(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy()))
            conv3d2.weight.set_value(conv3d1_weight_np)
            self.assertTrue(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy()))
            conv3d1.bias.set_value(conv3d1_bias)
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            self.assertTrue(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d2.weight = conv3d1.weight
            conv3d2.bias = conv3d1.bias
            self.assertTrue(
                np.array_equal(conv3d1.weight.numpy(), conv3d2.weight.numpy()))
            self.assertTrue(
                np.array_equal(conv3d1.bias.numpy(), conv3d2.bias.numpy()))

    def test_row_conv(self):
        input = np.arange(15).reshape([3, 5]).astype('float32')
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with self.static_graph():
            x = layers.data(
                name='X',
                shape=[3, 5],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            ret = layers.row_conv(input=x, future_context_size=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.static_graph():
            x = layers.data(
                name='X',
                shape=[3, 5],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            rowConv = nn.RowConv('RowConv', future_context_size=2)
            ret = rowConv(x)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        # TODO: dygraph can't support LODTensor

        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_group_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            X = fluid.layers.data(
                name='X',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            ret = layers.group_norm(
                input=X,
                groups=2,
                param_attr=fluid.initializer.Uniform(
                    low=-0.5, high=0.5),
                bias_attr=fluid.initializer.ConstantInitializer(value=1))
            static_ret = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.static_graph():
            X = fluid.layers.data(
                name='X',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            groupNorm = nn.GroupNorm(
                channels=shape[1],
                groups=2,
                param_attr=fluid.initializer.Uniform(
                    low=-0.5, high=0.5),
                bias_attr=fluid.initializer.ConstantInitializer(value=1))
            ret = groupNorm(X)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'X': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.dynamic_graph():
            groupNorm = nn.GroupNorm(
                channels=shape[1],
                groups=2,
                param_attr=fluid.initializer.Uniform(
                    low=-0.5, high=0.5),
                bias_attr=fluid.initializer.ConstantInitializer(value=1))
            dy_ret = groupNorm(base.to_variable(input))
            dy_rlt_value = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, dy_rlt_value))
        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_instance_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            X = fluid.layers.data(
                name='X', shape=shape, dtype='float32', append_batch_size=False)
            ret = layers.instance_norm(input=X)
            static_ret = self.get_static_graph_result(
                feed={'X': input}, fetch_list=[ret])[0]

        with self.static_graph():
            X = fluid.layers.data(
                name='X', shape=shape, dtype='float32', append_batch_size=False)
            instanceNorm = nn.InstanceNorm(num_channels=shape[1])
            ret = instanceNorm(X)
            static_ret2 = self.get_static_graph_result(
                feed={'X': input}, fetch_list=[ret])[0]

        with self.dynamic_graph():
            instanceNorm = nn.InstanceNorm(num_channels=shape[1])
            dy_ret = instanceNorm(base.to_variable(input))
            dy_rlt_value = dy_ret.numpy()

        with self.dynamic_graph():
            instanceNorm = nn.InstanceNorm(num_channels=shape[1])
            dy_ret = instanceNorm(base.to_variable(input))
            dy_rlt_value2 = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, dy_rlt_value))
        self.assertTrue(np.allclose(static_ret, dy_rlt_value2))
        self.assertTrue(np.allclose(static_ret, static_ret2))

        with self.static_graph():
            # the input of InstanceNorm must be Variable.
            def test_Variable():
                instanceNorm = nn.InstanceNorm(num_channels=shape[1])
                ret1 = instanceNorm(input)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of InstanceNorm must be float32 or float64
            def test_type():
                input = np.random.random(shape).astype('int32')
                instanceNorm = nn.InstanceNorm(num_channels=shape[1])
                ret2 = instanceNorm(input)

            self.assertRaises(TypeError, test_type)

    def test_spectral_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            Weight = fluid.layers.data(
                name='Weight',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            ret = layers.spectral_norm(weight=Weight, dim=1, power_iters=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'Weight': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place),
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.static_graph():
            Weight = fluid.layers.data(
                name='Weight',
                shape=shape,
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            spectralNorm = nn.SpectralNorm(shape, dim=1, power_iters=2)
            ret = spectralNorm(Weight)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'Weight': fluid.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place)
                },
                fetch_list=[ret],
                with_lod=True)[0]

        with self.dynamic_graph():
            spectralNorm = nn.SpectralNorm(shape, dim=1, power_iters=2)
            dy_ret = spectralNorm(base.to_variable(input))
            dy_rlt_value = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, dy_rlt_value))
        self.assertTrue(np.allclose(static_ret, static_ret2))

    def test_tree_conv(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        adj_array = [1, 2, 1, 3, 1, 4, 1, 5, 2, 6, 2, 7, 2, 8, 4, 9, 4, 10]
        adj = np.array(adj_array).reshape((1, 9, 2)).astype('int32')
        adj = np.tile(adj, (1, 1, 1))
        vectors = np.random.random((1, 10, 5)).astype('float32')
        with self.static_graph():
            NodesVector = fluid.layers.data(
                name='NodesVector',
                shape=(1, 10, 5),
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            EdgeSet = fluid.layers.data(
                name='EdgeSet',
                shape=(1, 9, 2),
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            ret = fluid.contrib.layers.tree_conv(
                nodes_vector=NodesVector,
                edge_set=EdgeSet,
                output_size=6,
                num_filters=1,
                max_depth=2)
            static_ret = self.get_static_graph_result(
                feed={
                    'NodesVector': fluid.create_lod_tensor(
                        data=vectors, recursive_seq_lens=[[1]], place=place),
                    'EdgeSet': fluid.create_lod_tensor(
                        data=adj, recursive_seq_lens=[[1]], place=place)
                },
                fetch_list=[ret],
                with_lod=False)[0]

        with self.static_graph():
            NodesVector = fluid.layers.data(
                name='NodesVector',
                shape=(1, 10, 5),
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            EdgeSet = fluid.layers.data(
                name='EdgeSet',
                shape=(1, 9, 2),
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            treeConv = nn.TreeConv(
                feature_size=5, output_size=6, num_filters=1, max_depth=2)
            ret = treeConv(NodesVector, EdgeSet)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'NodesVector': fluid.create_lod_tensor(
                        data=vectors, recursive_seq_lens=[[1]], place=place),
                    'EdgeSet': fluid.create_lod_tensor(
                        data=adj, recursive_seq_lens=[[1]], place=place)
                },
                fetch_list=[ret],
                with_lod=False)[0]

        with self.dynamic_graph():
            treeConv = nn.TreeConv(
                feature_size=5, output_size=6, num_filters=1, max_depth=2)
            dy_ret = treeConv(base.to_variable(vectors), base.to_variable(adj))
            dy_rlt_value = dy_ret.numpy()

        self.assertTrue(np.allclose(static_ret, static_ret2))
        self.assertTrue(np.allclose(static_ret, dy_rlt_value))

        with self.dynamic_graph():
            custom_weight = np.random.randn(5, 3, 6, 1).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            treeConv1 = nn.TreeConv(
                feature_size=5,
                output_size=6,
                num_filters=1,
                max_depth=2,
                bias_attr='tc1_b')
            treeConv2 = nn.TreeConv(
                feature_size=5,
                output_size=6,
                num_filters=1,
                max_depth=2,
                param_attr=weight_attr,
                bias_attr='tc2_b')
            dy_ret1 = treeConv1(
                base.to_variable(vectors), base.to_variable(adj))
            dy_ret2 = treeConv2(
                base.to_variable(vectors), base.to_variable(adj))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))
            treeConv2.weight.set_value(treeConv1.weight.numpy())
            treeConv2.bias.set_value(treeConv1.bias)
            dy_ret1 = treeConv1(
                base.to_variable(vectors), base.to_variable(adj))
            dy_ret2 = treeConv2(
                base.to_variable(vectors), base.to_variable(adj))
            self.assertTrue(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            treeConv2.weight = treeConv1.weight
            treeConv2.bias = treeConv1.bias
            self.assertTrue(
                np.array_equal(treeConv1.weight.numpy(),
                               treeConv2.weight.numpy()))
            self.assertTrue(
                np.array_equal(treeConv1.bias.numpy(), treeConv2.bias.numpy()))

    def test_conv3d_transpose(self):
        input_array = np.arange(0, 48).reshape(
            [2, 3, 2, 2, 2]).astype('float32')

        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2, 2], dtype='float32')
            out = layers.conv3d_transpose(
                input=img, num_filters=12, filter_size=12, use_cudnn=False)
            static_rlt = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out])[0]
        with self.static_graph():
            img = layers.data(name='pixel', shape=[3, 2, 2, 2], dtype='float32')
            conv3d_transpose = nn.Conv3DTranspose(
                num_channels=3, num_filters=12, filter_size=12, use_cudnn=False)
            out = conv3d_transpose(img)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out])[0]
        with self.dynamic_graph():
            conv3d_transpose = nn.Conv3DTranspose(
                num_channels=3, num_filters=12, filter_size=12, use_cudnn=False)
            dy_rlt = conv3d_transpose(base.to_variable(input_array))
            dy_rlt_value = dy_rlt.numpy()
        self.assertTrue(np.allclose(static_rlt2, static_rlt))
        self.assertTrue(np.allclose(dy_rlt_value, static_rlt))

        with self.dynamic_graph():
            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
            weight_attr = fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    custom_weight))
            conv3d1 = nn.Conv3DTranspose(
                num_channels=3,
                num_filters=3,
                filter_size=2,
                bias_attr='conv3d1_b',
                use_cudnn=False)
            conv3d2 = nn.Conv3DTranspose(
                num_channels=3,
                num_filters=3,
                filter_size=2,
                param_attr=weight_attr,
                bias_attr='conv3d2_b',
                use_cudnn=False)
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d1_weight_np = conv3d1.weight.numpy()
            conv3d1_bias = conv3d1.bias
            self.assertFalse(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy()))
            conv3d2.weight.set_value(conv3d1_weight_np)
            self.assertTrue(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy()))
            conv3d1.bias.set_value(conv3d1_bias)
            dy_ret1 = conv3d1(base.to_variable(images))
            dy_ret2 = conv3d2(base.to_variable(images))
            self.assertTrue(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d2.weight = conv3d1.weight
            conv3d2.bias = conv3d1.bias
            self.assertTrue(
                np.array_equal(conv3d1.weight.numpy(), conv3d2.weight.numpy()))
            self.assertTrue(
                np.array_equal(conv3d1.bias.numpy(), conv3d2.bias.numpy()))

    def test_eye_op(self):
        np_eye = np.eye(3, 2)
        array_rlt1 = [np_eye for _ in range(3)]
        stack_rlt1 = np.stack(array_rlt1, axis=0)
        array_rlt2 = [stack_rlt1 for _ in range(4)]
        stack_rlt2 = np.stack(array_rlt2, axis=0)

        with self.dynamic_graph():
            eye_tensor = layers.eye(num_rows=3, num_columns=2)
            eye_tensor_rlt1 = layers.eye(num_rows=3,
                                         num_columns=2,
                                         batch_shape=[3])
            eye_tensor_rlt2 = layers.eye(num_rows=3,
                                         num_columns=2,
                                         batch_shape=[4, 3])
            diag_tensor = layers.eye(20)
            eye_tensor_value = eye_tensor.numpy()
            eye_tensor_rlt1_value = eye_tensor_rlt1.numpy()
            eye_tensor_rlt2_value = eye_tensor_rlt2.numpy()
            diag_tensor_value = diag_tensor.numpy()
        self.assertTrue(np.allclose(eye_tensor_value, np_eye))
        self.assertTrue(np.allclose(eye_tensor_rlt1_value, stack_rlt1))
        self.assertTrue(np.allclose(eye_tensor_rlt2_value, stack_rlt2))
        self.assertTrue(np.allclose(diag_tensor_value, np.eye(20)))

        with self.assertRaises(TypeError):
            layers.eye(num_rows=3.1)
        with self.assertRaises(TypeError):
            layers.eye(num_rows=3, num_columns=2.2)
        with self.assertRaises(TypeError):
            layers.eye(num_rows=3, batch_shape=2)
        with self.assertRaises(TypeError):
            layers.eye(num_rows=3, batch_shape=[-1])

    def test_while_loop(self):
        with self.static_graph():
            i = layers.fill_constant(shape=[1], dtype='int64', value=0)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)

            def cond(i):
                return layers.less_than(i, ten)

            def body(i):
                return i + 1

            out = layers.while_loop(cond, body, [i])
            static_ret = self.get_static_graph_result(feed={}, fetch_list=out)

        with self.dynamic_graph():
            i = layers.fill_constant(shape=[1], dtype='int64', value=0)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)

            def cond(i):
                return layers.less_than(i, ten)

            def body(i):
                return i + 1

            dy_ret = layers.while_loop(cond, body, [i])
            with self.assertRaises(ValueError):
                j = layers.fill_constant(shape=[1], dtype='int64', value=0)

                def body2(i):
                    return i + 1, i + 2

                layers.while_loop(cond, body2, [j])

        self.assertTrue(np.array_equal(static_ret[0], dy_ret[0].numpy()))

    def test_compare(self):
        value_a = np.arange(3)
        value_b = np.arange(3)
        # less than
        with self.static_graph():
            a = layers.data(name='a', shape=[1], dtype='int64')
            b = layers.data(name='b', shape=[1], dtype='int64')
            cond = layers.less_than(x=a, y=b)
            static_ret = self.get_static_graph_result(
                feed={"a": value_a,
                      "b": value_b}, fetch_list=[cond])[0]
        with self.dynamic_graph():
            da = base.to_variable(value_a)
            db = base.to_variable(value_b)
            dcond = layers.less_than(x=da, y=db)

            for i in range(len(static_ret)):
                self.assertTrue(dcond.numpy()[i] == static_ret[i])

        # less equal
        with self.static_graph():
            a1 = layers.data(name='a1', shape=[1], dtype='int64')
            b1 = layers.data(name='b1', shape=[1], dtype='int64')
            cond1 = layers.less_equal(x=a1, y=b1)
            static_ret1 = self.get_static_graph_result(
                feed={"a1": value_a,
                      "b1": value_b}, fetch_list=[cond1])[0]
        with self.dynamic_graph():
            da1 = base.to_variable(value_a)
            db1 = base.to_variable(value_b)
            dcond1 = layers.less_equal(x=da1, y=db1)

            for i in range(len(static_ret1)):
                self.assertTrue(dcond1.numpy()[i] == static_ret1[i])

        #greater than
        with self.static_graph():
            a2 = layers.data(name='a2', shape=[1], dtype='int64')
            b2 = layers.data(name='b2', shape=[1], dtype='int64')
            cond2 = layers.greater_than(x=a2, y=b2)
            static_ret2 = self.get_static_graph_result(
                feed={"a2": value_a,
                      "b2": value_b}, fetch_list=[cond2])[0]
        with self.dynamic_graph():
            da2 = base.to_variable(value_a)
            db2 = base.to_variable(value_b)
            dcond2 = layers.greater_than(x=da2, y=db2)

            for i in range(len(static_ret2)):
                self.assertTrue(dcond2.numpy()[i] == static_ret2[i])

        #greater equal
        with self.static_graph():
            a3 = layers.data(name='a3', shape=[1], dtype='int64')
            b3 = layers.data(name='b3', shape=[1], dtype='int64')
            cond3 = layers.greater_equal(x=a3, y=b3)
            static_ret3 = self.get_static_graph_result(
                feed={"a3": value_a,
                      "b3": value_b}, fetch_list=[cond3])[0]
        with self.dynamic_graph():
            da3 = base.to_variable(value_a)
            db3 = base.to_variable(value_b)
            dcond3 = layers.greater_equal(x=da3, y=db3)

            for i in range(len(static_ret3)):
                self.assertTrue(dcond3.numpy()[i] == static_ret3[i])

        # equal
        with self.static_graph():
            a4 = layers.data(name='a4', shape=[1], dtype='int64')
            b4 = layers.data(name='b4', shape=[1], dtype='int64')
            cond4 = layers.equal(x=a4, y=b4)
            static_ret4 = self.get_static_graph_result(
                feed={"a4": value_a,
                      "b4": value_b}, fetch_list=[cond4])[0]
        with self.dynamic_graph():
            da4 = base.to_variable(value_a)
            db4 = base.to_variable(value_b)
            dcond4 = layers.equal(x=da4, y=db4)

            for i in range(len(static_ret4)):
                self.assertTrue(dcond4.numpy()[i] == static_ret4[i])

        # not equal
        with self.static_graph():
            a5 = layers.data(name='a5', shape=[1], dtype='int64')
            b5 = layers.data(name='b5', shape=[1], dtype='int64')
            cond5 = layers.equal(x=a5, y=b5)
            static_ret5 = self.get_static_graph_result(
                feed={"a5": value_a,
                      "b5": value_b}, fetch_list=[cond5])[0]
        with self.dynamic_graph():
            da5 = base.to_variable(value_a)
            db5 = base.to_variable(value_b)
            dcond5 = layers.equal(x=da5, y=db5)

            for i in range(len(static_ret5)):
                self.assertTrue(dcond5.numpy()[i] == static_ret5[i])

    def test_cond(self):
        def less_than_branch(a, b):
            return fluid.layers.elementwise_add(a, b)

        def greater_equal_branch(a, b):
            return fluid.layers.elementwise_sub(a, b)

        with self.static_graph():
            a = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0.1)
            b = fluid.layers.fill_constant(
                shape=[1], dtype='float32', value=0.23)
            out = fluid.layers.cond(a >= b, lambda: greater_equal_branch(a, b),
                                    lambda: less_than_branch(a, b))
            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            ret = exe.run(fetch_list=[out])
            static_res = ret[0]

        with self.dynamic_graph():
            a = fluid.dygraph.to_variable(np.array([0.1]).astype('float32'))
            b = fluid.dygraph.to_variable(np.array([0.23]).astype('float32'))
            out = layers.cond(a < b, lambda: less_than_branch(a, b),
                              lambda: greater_equal_branch(a, b))
            out2 = layers.cond(a >= b, lambda: greater_equal_branch(a, b),
                               lambda: less_than_branch(a, b))
            dynamic_res = out.numpy()
            dynamic_res2 = out2.numpy()
            self.assertTrue(np.array_equal(dynamic_res, dynamic_res2))
            with self.assertRaises(TypeError):
                layers.cond(a < b, 'str', 'str')
            with self.assertRaises(TypeError):
                layers.cond(a >= b, 'str', 'str')

        self.assertTrue(np.array_equal(static_res, dynamic_res))

    def test_case(self):
        def fn_1():
            return layers.fill_constant(shape=[1, 2], dtype='float32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[2, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[3], dtype='int32', value=3)

        with self.static_graph():
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)

            pred_1 = layers.less_than(z, x)  # true: 0.2 < 0.3
            pred_2 = layers.less_than(x, y)  # false: 0.3 < 0.1
            pred_3 = layers.equal(x, y)  # false: 0.3 == 0.1

            out_1 = layers.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)
            out_2 = layers.case(pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)])

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            static_res1, static_res2 = exe.run(fetch_list=[out_1, out_2])

        with self.dynamic_graph():
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)

            pred_1 = layers.less_than(z, x)  # true: 0.2 < 0.3
            pred_2 = layers.less_than(x, y)  # false: 0.3 < 0.1
            pred_3 = layers.equal(x, y)  # false: 0.3 == 0.1

            out_1 = layers.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)
            out_2 = layers.case(pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)])
            dynamic_res1 = out_1.numpy()
            dynamic_res2 = out_2.numpy()

        self.assertTrue(np.array_equal(static_res1, dynamic_res1))
        self.assertTrue(np.array_equal(static_res2, dynamic_res2))

    def test_switch_case(self):
        def fn_1():
            return layers.fill_constant(shape=[1, 2], dtype='float32', value=1)

        def fn_2():
            return layers.fill_constant(shape=[2, 2], dtype='int32', value=2)

        def fn_3():
            return layers.fill_constant(shape=[3], dtype='int32', value=3)

        with self.static_graph():
            index_1 = layers.fill_constant(shape=[1], dtype='int32', value=1)
            index_2 = layers.fill_constant(shape=[1], dtype='int32', value=2)

            out_1 = layers.switch_case(
                branch_index=index_1,
                branch_fns={1: fn_1,
                            2: fn_2},
                default=fn_3)
            out_2 = layers.switch_case(
                branch_index=index_2,
                branch_fns=[(1, fn_1), (2, fn_2)],
                default=fn_3)
            out_3 = layers.switch_case(
                branch_index=index_2,
                branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)])

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)
            static_res1, static_res2, static_res3 = exe.run(
                fetch_list=[out_1, out_2, out_3])

        with self.dynamic_graph():
            index_1 = layers.fill_constant(shape=[1], dtype='int32', value=1)
            index_2 = layers.fill_constant(shape=[1], dtype='int32', value=2)

            out_1 = layers.switch_case(
                branch_index=index_1,
                branch_fns={1: fn_1,
                            2: fn_2},
                default=fn_3)
            out_2 = layers.switch_case(
                branch_index=index_2,
                branch_fns=[(1, fn_1), (2, fn_2)],
                default=fn_3)
            out_3 = layers.switch_case(
                branch_index=index_2,
                branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)])

            dynamic_res1 = out_1.numpy()
            dynamic_res2 = out_2.numpy()
            dynamic_res3 = out_3.numpy()

        self.assertTrue(np.array_equal(static_res1, dynamic_res1))
        self.assertTrue(np.array_equal(static_res2, dynamic_res2))
        self.assertTrue(np.array_equal(static_res3, dynamic_res3))

    def test_crop_tensor(self):
        with self.static_graph():
            x = fluid.layers.data(name="x1", shape=[6, 5, 8])

            dim1 = fluid.layers.data(
                name="dim1", shape=[1], append_batch_size=False)
            dim2 = fluid.layers.data(
                name="dim2", shape=[1], append_batch_size=False)
            crop_shape1 = (1, 2, 4, 4)
            crop_shape2 = fluid.layers.data(
                name="crop_shape", shape=[4], append_batch_size=False)
            crop_shape3 = [-1, dim1, dim2, 4]
            crop_offsets1 = [0, 0, 1, 0]
            crop_offsets2 = fluid.layers.data(
                name="crop_offset", shape=[4], append_batch_size=False)
            crop_offsets3 = [0, dim1, dim2, 0]

            out1 = fluid.layers.crop_tensor(
                x, shape=crop_shape1, offsets=crop_offsets1)
            out2 = fluid.layers.crop_tensor(
                x, shape=crop_shape2, offsets=crop_offsets2)
            out3 = fluid.layers.crop_tensor(
                x, shape=crop_shape3, offsets=crop_offsets3)

            self.assertIsNotNone(out1)
            self.assertIsNotNone(out2)
            self.assertIsNotNone(out3)

    def test_shard_index(self):
        with self.static_graph():
            x = fluid.layers.data(name="label", shape=[4, 1], dtype='int64')
            shard_label = fluid.layers.shard_index(
                input=x, index_num=20, nshards=2, shard_id=0)

        self.assertIsNotNone(shard_label)

    def test_accuracy(self):
        x = np.random.rand(3, 32, 32).astype("float32")
        y = np.array([[1], [0], [1]])
        with self.static_graph():
            data = fluid.data(name="input", shape=[-1, 32, 32], dtype="float32")
            label = fluid.data(name="label", shape=[-1, 1], dtype="int")
            fc_out = fluid.layers.fc(input=data, size=10)
            predict = fluid.layers.softmax(input=fc_out)
            result = fluid.layers.accuracy(input=predict, label=label, k=5)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(fluid.default_startup_program())
            # x = np.random.rand(3, 32, 32).astype("float32")
            # y = np.array([[1], [0], [1]])
            static_out = exe.run(feed={"input": x,
                                       "label": y},
                                 fetch_list=result[0])

        with self.dynamic_graph(force_to_use_cpu=True):
            data = base.to_variable(x)
            label = base.to_variable(y)
            fc_out = fluid.layers.fc(data, size=10)
            predict = fluid.layers.softmax(fc_out)
            dynamic_out = fluid.layers.accuracy(input=predict, label=label, k=5)

        self.assertTrue(np.array_equal(static_out[0], dynamic_out.numpy()))


class TestBook(LayerTest):
    def setUp(self):
        self.only_static_set = set({"make_word_embedding"})
        self.not_compare_static_dygraph_set = set({
            "make_gaussian_random", "make_gaussian_random_batch_size_like",
            "make_kldiv_loss", "make_prelu",
            "make_sampled_softmax_with_cross_entropy", "make_sampling_id",
            "make_uniform_random_batch_size_like"
        })
        self.all_close_compare = set({"make_spectral_norm"})

    def test_all_layers(self):
        attrs = (getattr(self, name) for name in dir(self))
        methods = filter(inspect.ismethod, attrs)
        for method in methods:
            if not method.__name__.startswith('make_'):
                continue
            self._low_data_bound = 0
            self._high_data_bound = 2
            self._batch_size = 2
            self._feed_dict = {}
            self._force_to_use_cpu = False
            with self.static_graph():
                static_var = method()
                if isinstance(static_var, tuple):
                    static_var = static_var[0]

                if static_var is not None:
                    fetch_list = [static_var.name]
                    static_result = self.get_static_graph_result(
                        feed=self._feed_dict,
                        fetch_list=fetch_list,
                        force_to_use_cpu=self._force_to_use_cpu)

                else:
                    assert method.__name__ in ('make_get_places')
                    continue
            if method.__name__ in self.only_static_set:
                continue

            with self.dynamic_graph(self._force_to_use_cpu):
                dy_result = method()
                if isinstance(dy_result, tuple):
                    dy_result = dy_result[0]
                dy_result_value = dy_result.numpy()

            if method.__name__ in self.all_close_compare:
                self.assertTrue(
                    np.allclose(
                        static_result[0], dy_result_value, atol=0, rtol=1e-05),
                    "Result of function [{}] compare failed".format(
                        method.__name__))
                continue

            if method.__name__ not in self.not_compare_static_dygraph_set:
                self.assertTrue(
                    np.array_equal(static_result[0], dy_result_value),
                    "Result of function [{}] not equal".format(method.__name__))

    def _get_np_data(self, shape, dtype, append_batch_size=True):
        np.random.seed(self.seed)
        if append_batch_size:
            shape = [self._batch_size] + shape
        if dtype == 'float32':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'float64':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'int32':
            return np.random.randint(self._low_data_bound,
                                     self._high_data_bound, shape).astype(dtype)
        elif dtype == 'int64':
            return np.random.randint(self._low_data_bound,
                                     self._high_data_bound, shape).astype(dtype)

    def _get_data(self,
                  name,
                  shape,
                  dtype,
                  set_feed_dict=True,
                  append_batch_size=True):
        if base.enabled():
            return base.to_variable(
                value=self._get_np_data(shape, dtype, append_batch_size),
                name=name,
                zero_copy=False)
        else:
            if set_feed_dict:
                self._feed_dict[name] = self._get_np_data(shape, dtype,
                                                          append_batch_size)
            return layers.data(
                name=name,
                shape=shape,
                dtype=dtype,
                append_batch_size=append_batch_size)

    def make_sampled_softmax_with_cross_entropy(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            logits = self._get_data(name='Logits', shape=[256], dtype='float32')
            label = self._get_data(name='Label', shape=[1], dtype='int64')
            num_samples = 25
            output = layers.sampled_softmax_with_cross_entropy(logits, label,
                                                               num_samples)
            return (output)

    def make_fit_a_line(self):
        with program_guard(
                fluid.default_main_program(),
                startup_program=fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[13], dtype='float32')
            y_predict = layers.fc(input=x, size=1, act=None)
            y = self._get_data(name='y', shape=[1], dtype='float32')
            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)
            return (avg_cost)

    def make_recognize_digits_mlp(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            # Change g_program, so the rest layers use `g_program`
            images = self._get_data(name='pixel', shape=[784], dtype='float32')
            label = self._get_data(name='label', shape=[1], dtype='int64')
            hidden1 = layers.fc(input=images, size=128, act='relu')
            hidden2 = layers.fc(input=hidden1, size=64, act='relu')
            predict = layers.fc(input=[hidden2, hidden1],
                                size=10,
                                act='softmax',
                                param_attr=["sftmax.w1", "sftmax.w2"])
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(cost)
            return (avg_cost)

    def make_conv2d_transpose(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            img = self._get_data(name='pixel', shape=[3, 2, 2], dtype='float32')
            return layers.conv2d_transpose(
                input=img, num_filters=10, output_size=28)

    def make_recognize_digits_conv(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            images = self._get_data(
                name='pixel', shape=[1, 28, 28], dtype='float32')
            label = self._get_data(name='label', shape=[1], dtype='int64')
            conv_pool_1 = nets.simple_img_conv_pool(
                input=images,
                filter_size=5,
                num_filters=2,
                pool_size=2,
                pool_stride=2,
                act="relu")
            conv_pool_2 = nets.simple_img_conv_pool(
                input=conv_pool_1,
                filter_size=5,
                num_filters=4,
                pool_size=2,
                pool_stride=2,
                act="relu")

            predict = layers.fc(input=conv_pool_2, size=10, act="softmax")
            cost = layers.cross_entropy(input=predict, label=label)
            avg_cost = layers.mean(cost)
            return avg_cost

    def make_word_embedding(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            dict_size = 10000
            embed_size = 32
            first_word = self._get_data(name='firstw', shape=[1], dtype='int64')
            second_word = self._get_data(
                name='secondw', shape=[1], dtype='int64')
            third_word = self._get_data(name='thirdw', shape=[1], dtype='int64')
            forth_word = self._get_data(name='forthw', shape=[1], dtype='int64')
            next_word = self._get_data(name='nextw', shape=[1], dtype='int64')

            embed_first = layers.embedding(
                input=first_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_second = layers.embedding(
                input=second_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            embed_third = layers.embedding(
                input=third_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')
            embed_forth = layers.embedding(
                input=forth_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w')

            concat_embed = layers.concat(
                input=[embed_first, embed_second, embed_third, embed_forth],
                axis=1)

            hidden1 = layers.fc(input=concat_embed, size=256, act='sigmoid')
            predict_word = layers.fc(input=hidden1,
                                     size=dict_size,
                                     act='softmax')
            cost = layers.cross_entropy(input=predict_word, label=next_word)
            avg_cost = layers.mean(cost)
            return (avg_cost)

    def make_sigmoid_cross_entropy(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            dat = self._get_data(name='data', shape=[10], dtype='float32')
            lbl = self._get_data(name='label', shape=[10], dtype='float32')
            ignore_index = -1
            return (layers.sigmoid_cross_entropy_with_logits(
                x=dat, label=lbl, ignore_index=ignore_index))

    def make_hsigmoid(self):
        self._force_to_use_cpu = True
        with fluid.framework._dygraph_place_guard(place=fluid.CPUPlace()):
            x = self._get_data(name='x', shape=[2], dtype='float32')
            y = self._get_data(name='y', shape=[2], dtype='int64')
            return (layers.hsigmoid(input=x, label=y, num_classes=2))

        # test hsigmod with custom tree structure
        program2 = Program()
        with program_guard(program2):
            x2 = self._get_data(name='x2', shape=[4, 8], dtype='float32')
            y2 = self._get_data(name='y2', shape=[4], dtype='int64')
            path_table = self._get_data(
                name='path_table', shape=[4, 6], dtype='int64')
            path_code = self._get_data(
                name='path_code', shape=[4, 6], dtype='int64')
            return (layers.hsigmoid(
                input=x2,
                label=y2,
                num_classes=6,
                path_table=path_table,
                path_code=path_code,
                is_custom=True))

    def make_pool2d(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 224, 224], dtype='float32')
            return (layers.pool2d(
                x, pool_size=[5, 3], pool_stride=[1, 2], pool_padding=(2, 1)))

    def make_pool2d_infershape(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            theta = self._get_data("theta", shape=[2, 3], dtype='float32')
            x = fluid.layers.affine_grid(theta, out_shape=[2, 3, 244, 244])
            return (layers.pool2d(
                x, pool_size=[5, 3], pool_stride=[1, 2], pool_padding=(2, 1)))

    def make_pool3d(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(
                name='x', shape=[3, 244, 244, 244], dtype='float32')
            return (layers.pool3d(
                x,
                pool_size=[5, 3, 2],
                pool_stride=[1, 2, 3],
                pool_padding=(2, 1, 1)))

    def make_adaptive_pool2d(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 224, 224], dtype='float32')
            return (layers.adaptive_pool2d(x, [3, 3], pool_type='avg'))
            pool, mask = layers.adaptive_pool2d(x, [3, 3], require_index=True)
            return (pool)
            return (mask)
            return (layers.adaptive_pool2d(x, 3, pool_type='avg'))
            pool, mask = layers.adaptive_pool2d(x, 3, require_index=True)
            return (pool)
            return (mask)

    def make_adaptive_pool3d(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(
                name='x', shape=[3, 244, 224, 224], dtype='float32')
            return (layers.adaptive_pool3d(x, [3, 3, 3], pool_type='avg'))
            pool, mask = layers.adaptive_pool3d(
                x, [3, 3, 3], require_index=True)
            return (pool)
            return (mask)
            return (layers.adaptive_pool3d(x, 3, pool_type='avg'))
            pool, mask = layers.adaptive_pool3d(x, 3, require_index=True)
            return (pool)
            return (mask)

    def make_lstm_unit(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x_t_data = self._get_data(
                name='x_t_data', shape=[10, 10], dtype='float32')
            x_t = layers.fc(input=x_t_data, size=10)
            prev_hidden_data = self._get_data(
                name='prev_hidden_data', shape=[10, 30], dtype='float32')
            prev_hidden = layers.fc(input=prev_hidden_data, size=30)
            prev_cell_data = self._get_data(
                name='prev_cell', shape=[10, 30], dtype='float32')
            prev_cell = layers.fc(input=prev_cell_data, size=30)
            return (layers.lstm_unit(
                x_t=x_t, hidden_t_prev=prev_hidden, cell_t_prev=prev_cell))

    def make_softmax(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(name='data', shape=[10], dtype='float32')
            hid = layers.fc(input=data, size=20)
            return (layers.softmax(hid, axis=1))

    def make_space_to_depth(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(
                name='data',
                shape=[32, 9, 6, 6],
                append_batch_size=False,
                dtype='float32')
            return (layers.space_to_depth(data, 3))

    def make_lrn(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(name='data', shape=[6, 2, 2], dtype='float32')
            return (layers.lrn(data))

    def make_get_places(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            get_places(device_count=1)

    @prog_scope()
    def make_nce(self):
        window_size = 5
        words = []
        for i in range(window_size):
            words.append(
                self._get_data(
                    name='word_{0}'.format(i), shape=[1], dtype='int64'))

        dict_size = 10000
        label_word = int(window_size // 2) + 1

        embs = []
        for i in range(window_size):
            if i == label_word:
                continue

            emb = layers.embedding(
                input=words[i],
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=True)

            embs.append(emb)

        embs = layers.concat(input=embs, axis=1)
        loss = layers.nce(input=embs,
                          label=words[label_word],
                          num_total_classes=dict_size,
                          param_attr='nce.w',
                          bias_attr='nce.b')
        avg_loss = layers.mean(loss)
        return (avg_loss)

    def make_multiplex(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x1 = self._get_data(name='x1', shape=[4], dtype='float32')
            x2 = self._get_data(name='x2', shape=[4], dtype='float32')
            index = self._get_data(name='index', shape=[1], dtype='int32')
            out = layers.multiplex(inputs=[x1, x2], index=index)
            return (out)

    def make_softmax_with_cross_entropy(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[16], dtype='float32')
            y = self._get_data(name='label', shape=[1], dtype='int64')
            loss, softmax = layers.softmax_with_cross_entropy(
                x, y, return_softmax=True)
            self.assertIsNotNone(loss)
            self.assertIsNotNone(softmax)

            loss = layers.softmax_with_cross_entropy(x, y)
            self.assertIsNotNone(loss)

            x1 = self._get_data(name='x1', shape=[16, 32, 64], dtype='float32')
            y1 = self._get_data(name='label1', shape=[1, 32, 64], dtype='int64')
            y2 = self._get_data(name='label2', shape=[16, 1, 64], dtype='int64')
            y3 = self._get_data(name='label3', shape=[16, 32, 1], dtype='int64')
            loss1 = layers.softmax_with_cross_entropy(x1, y1, axis=1)
            loss2 = layers.softmax_with_cross_entropy(x1, y2, axis=2)
            loss3 = layers.softmax_with_cross_entropy(x1, y3, axis=3)
            loss4 = layers.softmax_with_cross_entropy(x1, y3, axis=-1)
            self.assertIsNotNone(loss1)
            self.assertIsNotNone(loss2)
            self.assertIsNotNone(loss3)
            self.assertIsNotNone(loss4)
            return (loss4)

    def make_smooth_l1(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[4], dtype='float32')
            y = self._get_data(name='label', shape=[4], dtype='float32')
            loss = layers.smooth_l1(x, y)
            return (loss)

    def make_scatter(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(
                name='x',
                shape=[3, 3],
                append_batch_size=False,
                dtype='float32')
            idx = self._get_data(
                name='idx', shape=[2], append_batch_size=False, dtype='int32')
            updates = self._get_data(
                name='updates',
                shape=[2, 3],
                append_batch_size=False,
                dtype='float32')
            out = layers.scatter(input=x, index=idx, updates=updates)
            return (out)

    def make_one_hot(self):
        with fluid.framework._dygraph_place_guard(place=fluid.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            return (one_hot_label)

    def make_label_smooth(self):
        # TODO(minqiyang): support gpu ut
        self._force_to_use_cpu = True
        with fluid.framework._dygraph_place_guard(place=fluid.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = layers.one_hot(input=label, depth=10)
            smooth_label = layers.label_smooth(
                label=one_hot_label, epsilon=0.1, dtype="int32")
            return (smooth_label)

    def make_topk(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(name="label", shape=[200], dtype="float32")
            values, indices = layers.topk(data, k=5)
            return (values)
            return (indices)

    def make_resize_bilinear(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 9, 6], dtype="float32")
            output = layers.resize_bilinear(x, out_shape=[12, 12])
            return (output)

    def make_resize_bilinear_by_scale(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 9, 6], dtype="float32")
            output = layers.resize_bilinear(x, scale=1.5)
            return (output)

    def make_resize_nearest(self):
        try:
            with program_guard(fluid.default_main_program(),
                               fluid.default_startup_program()):
                x = self._get_data(name='x1', shape=[3, 9, 6], dtype="float32")
                output = layers.resize_nearest(x, out_shape=[12, 12])
        except ValueError:
            pass

        try:
            with program_guard(fluid.default_main_program(),
                               fluid.default_startup_program()):
                x = self._get_data(
                    name='x2', shape=[3, 9, 6, 7], dtype="float32")
                output = layers.resize_nearest(x, out_shape=[12, 12, 12])
        except ValueError:
            pass

        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 9, 6], dtype="float32")
            output = layers.resize_nearest(x, out_shape=[12, 12])
            return (output)

    def make_resize_nearest_by_scale(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x1', shape=[3, 9, 6], dtype="float32")
            output = layers.resize_nearest(x, scale=1.8)
            return (output)

    def make_resize_trilinear(self):
        try:
            with program_guard(fluid.default_main_program(),
                               fluid.default_startup_program()):
                x = self._get_data(name='x2', shape=[3, 9, 6], dtype="float32")
                output = layers.resize_trilinear(x, out_shape=[12, 12, 12])
        except ValueError:
            pass

        try:
            with program_guard(fluid.default_main_program(),
                               fluid.default_startup_program()):
                x = self._get_data(
                    name='x', shape=[3, 9, 6, 7], dtype="float32")
                output = layers.resize_trilinear(x, out_shape=[12, 12])
        except ValueError:
            pass

        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 9, 6, 7], dtype="float32")
            output = layers.resize_trilinear(x, out_shape=[12, 12, 12])
            return (output)

    def make_resize_trilinear_by_scale(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 9, 6, 7], dtype="float32")
            output = layers.resize_trilinear(x, scale=2.1)
            return (output)

    def make_polygon_box_transform(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[8, 4, 4], dtype="float32")
            output = layers.polygon_box_transform(input=x)
            return (output)

    def make_l2_normalize(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[8, 7, 10], dtype="float32")
            output = layers.l2_normalize(x, axis=1)
            return output

    def make_crop(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 5], dtype="float32")
            y = self._get_data(name='y', shape=[2, 3], dtype="float32")
            output = layers.crop(x, shape=y)
            return (output)

    def make_mean_iou(self):
        with fluid.framework._dygraph_place_guard(place=fluid.CPUPlace()):
            x = self._get_data(name='x', shape=[16], dtype='int32')
            y = self._get_data(name='label', shape=[16], dtype='int32')
            iou = layers.mean_iou(x, y, self._high_data_bound)
            return (iou)

    def make_argsort(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(name='x', shape=[2, 3, 3], dtype="float32")
            out, ids = layers.argsort(input=data, axis=1)
            return (out)
            return (ids)

    def make_rank_loss(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            label = self._get_data(
                name='label',
                append_batch_size=False,
                shape=[16, 1],
                dtype="float32")
            left = self._get_data(
                name='left',
                append_batch_size=False,
                shape=[16, 1],
                dtype="float32")
            right = self._get_data(
                name='right',
                append_batch_size=False,
                shape=[16, 1],
                dtype="float32")
            out = layers.rank_loss(label, left, right, name="rank_loss")
            return (out)

    def make_shape(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32")
            out = layers.shape(input)
            return (out)

    def make_pad2d(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32")
            paddings = layers.fill_constant(shape=[4], dtype='int32', value=1)
            out = layers.pad2d(
                input,
                paddings=[1, 2, 3, 4],
                mode='reflect',
                data_format='NCHW',
                name="shape")
            out_1 = layers.pad2d(
                input,
                paddings=paddings,
                mode='reflect',
                data_format='NCHW',
                name="shape")
            return (out)
            return (out_1)

    def make_prelu(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[5, 200, 100, 100], dtype="float32")
            mode = 'channel'
            out = layers.prelu(
                input,
                mode,
                param_attr=ParamAttr(initializer=Constant(1.0)),
                name='prelu')
            return (out)

    def make_soft_relu(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.soft_relu(input, threshold=30.0, name='soft_relu')
            return (out)

    def make_sigmoid(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.sigmoid(input, name='sigmoid')
            return (out)

    def make_exp(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.exp(input, name='exp')
            return (out)

    def make_tanh(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.tanh(input, name='tanh')
            return (out)

    def make_tanh_shrink(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.tanh_shrink(input, name='tanh_shrink')
            return (out)

    def make_sqrt(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.sqrt(input, name='sqrt')
            return (out)

    def make_abs(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.abs(input, name='abs')
            return (out)

    def make_ceil(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.ceil(input, name='ceil')
            return (out)

    def make_floor(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.floor(input, name='floor')
            return (out)

    def make_cos(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.cos(input, name='cos')
            return (out)

    def make_sin(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.sin(input, name='sin')
            return (out)

    def make_round(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.round(input, name='round')
            return (out)

    def make_reciprocal(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.reciprocal(input, name='reciprocal')
            return (out)

    def make_square(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.square(input, name='square')
            return (out)

    def make_softplus(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.softplus(input, name='softplus')
            return (out)

    def make_softsign(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.softsign(input, name='softsign')
            return (out)

    def make_mish(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.mish(input, name='mish')
            return (out)

    def make_cross_entropy(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="x", shape=[30, 10], dtype="float32")
            label = self._get_data(name="label", shape=[30, 1], dtype="int64")
            mode = 'channel'
            out = layers.cross_entropy(x, label, False, 4)
            return (out)

    def make_bpr_loss(self):
        self._force_to_use_cpu = True
        with fluid.framework._dygraph_place_guard(place=fluid.CPUPlace()):
            x = self._get_data(name="x", shape=[30, 10], dtype="float32")
            label = self._get_data(name="label", shape=[30, 1], dtype="int64")
            out = layers.bpr_loss(x, label)
            return (out)

    def make_expand(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="input", shape=[10], dtype='int32')
            out = layers.expand(x, [1, 2])
            return out

    def make_uniform_random_batch_size_like(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32')
            out = layers.uniform_random_batch_size_like(input, [-1, 11])
            return (out)

    def make_gaussian_random(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            out = layers.gaussian_random(shape=[20, 30])
            return (out)

    def make_sampling_id(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(
                name="X",
                shape=[13, 11],
                dtype='float32',
                append_batch_size=False)

            out = layers.sampling_id(x)
            return (out)

    def make_gaussian_random_batch_size_like(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32')

            out = layers.gaussian_random_batch_size_like(
                input, shape=[-1, 11], mean=1.0, std=2.0)
            return (out)

    def make_sum(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32')

            out = layers.sum(input)
            return (out)

    def make_slice(self):
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        axes = [0, 1, 2]

        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32')

            out = layers.slice(input, axes=axes, starts=starts, ends=ends)
            return out

    def make_scale_variable(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32')
            scale_var = self._get_data(
                name="scale",
                shape=[1],
                dtype='float32',
                append_batch_size=False)

            out = layers.scale(input, scale=scale_var)
            return out

    def make_softshrink(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = layers.softshrink(input, alpha=0.3)
            return (out)

    def make_iou_similarity(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="x", shape=[4], dtype="float32")
            y = self._get_data(name="y", shape=[4], dtype="float32")
            out = layers.iou_similarity(x, y, name='iou_similarity')
            return (out)

    def make_grid_sampler(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name='x', shape=[3, 5, 7], dtype='float32')
            grid = self._get_data(name='grid', shape=[5, 7, 2], dtype='float32')
            out = layers.grid_sampler(x, grid)
            return (out)

    def make_bilinear_tensor_product_layer(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(name='data', shape=[4], dtype="float32")

            theta = self._get_data(name="theta", shape=[5], dtype="float32")
            out = layers.bilinear_tensor_product(data, theta, 6)
            return (out)

    def make_batch_norm(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32")
            out = layers.batch_norm(data)
            return (out)

    def make_batch_norm_momentum_variable(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32")
            momentum = self._get_data(
                name='momentum',
                shape=[1],
                dtype='float32',
                append_batch_size=False)
            out = layers.batch_norm(data, momentum=momentum)
            return (out)

    def make_inplace_abn(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32")
            out = layers.inplace_abn(data, act='leaky_relu', act_alpha=0.2)
            return (out)

    def make_inplace_abn_momentum_variable(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32")
            momentum = self._get_data(
                name='momentum',
                shape=[1],
                dtype='float32',
                append_batch_size=False)
            out = layers.inplace_abn(
                data, momentum=momentum, act='elu', act_alpha=2.0)
            return (out)

    def make_range(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            layers.range(0, 10, 2, 'int32')
            layers.range(0.1, 10.0, 0.2, 'float32')
            layers.range(0.1, 10.0, 0.2, 'float64')
            start = layers.fill_constant(shape=[1], value=0.1, dtype="float32")
            end = layers.fill_constant(shape=[1], value=10.0, dtype="float32")
            step = layers.fill_constant(shape=[1], value=0.2, dtype="float32")
            y = layers.range(start, end, step, 'float64')
            return y

    def make_spectral_norm(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            weight = self._get_data(
                name='weight',
                shape=[2, 3, 32, 32],
                dtype="float32",
                append_batch_size=False)
            out = layers.spectral_norm(weight, dim=1, power_iters=1)
            return (out)

    def make_kldiv_loss(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(
                name='x',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False)
            target = self._get_data(
                name='target',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False)
            loss = layers.kldiv_loss(x=x, target=target, reduction='batchmean')
            return (loss)

    def make_temporal_shift(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="X", shape=[16, 4, 4], dtype="float32")
            out = layers.temporal_shift(x, seg_num=2, shift_ratio=0.2)
            return (out)

    def make_shuffle_channel(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="X", shape=[16, 4, 4], dtype="float32")
            out = layers.shuffle_channel(x, group=4)
            return (out)

    def make_fsp_matrix(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="X", shape=[16, 4, 4], dtype="float32")
            y = self._get_data(name="Y", shape=[8, 4, 4], dtype="float32")
            out = layers.fsp_matrix(x, y)
            return (out)

    def make_pixel_shuffle(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="X", shape=[9, 4, 4], dtype="float32")
            out = layers.pixel_shuffle(x, upscale_factor=3)
            return (out)

    def make_mse_loss(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = layers.mse_loss(input=x, label=y)
            return (out)

    def make_square_error_cost(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = layers.square_error_cost(input=x, label=y)
            return (out)

    def test_dynamic_lstmp(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            hidden_dim, proj_dim = 16, 8
            seq_data = layers.data(
                name='seq_data', shape=[10, 10], dtype='float32', lod_level=1)
            fc_out = layers.fc(input=seq_data, size=4 * hidden_dim)
            self.assertIsNotNone(
                layers.dynamic_lstmp(
                    input=fc_out, size=4 * hidden_dim, proj_size=proj_dim))

    def test_linear_chain_crf(self):
        with self.static_graph():
            label_dict_len = 10
            feature = layers.data(name='feature', shape=[784], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='int64')
            emission = layers.fc(input=feature, size=10)
            crf = layers.linear_chain_crf(
                input=emission, label=label, param_attr=ParamAttr(name="crfw"))
            crf_decode = layers.crf_decoding(
                input=emission, param_attr=ParamAttr(name="crfw"))
            self.assertFalse(crf is None)
            self.assertFalse(crf_decode is None)
            return layers.chunk_eval(
                input=crf_decode,
                label=label,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) // 2)

    def test_linear_chain_crf_padding(self):
        with self.static_graph():
            label_dict_len, max_len = 10, 20
            feature = layers.data(
                name='feature', shape=[max_len, 784], dtype='float32')
            label = layers.data(name='label', shape=[max_len], dtype='int64')
            length = layers.data(name='length', shape=[1], dtype='int64')
            emission = layers.fc(input=feature, size=10, num_flatten_dims=2)
            crf = layers.linear_chain_crf(
                input=emission,
                label=label,
                length=length,
                param_attr=ParamAttr(name="crfw"))
            crf_decode = layers.crf_decoding(
                input=emission,
                length=length,
                param_attr=ParamAttr(name="crfw"))
            self.assertFalse(crf is None)
            self.assertFalse(crf_decode is None)
            return layers.chunk_eval(
                input=crf_decode,
                label=label,
                seq_length=length,
                chunk_scheme="IOB",
                num_chunk_types=(label_dict_len - 1) // 2)

    def test_im2sequence(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[3, 128, 128], dtype='float32')
            y = layers.data(name='y', shape=[], dtype='float32')
            output = layers.im2sequence(
                input=x,
                input_image_size=y,
                stride=[1, 1],
                filter_size=[2, 2],
                out_stride=[1, 1])
            return (output)

    def test_lod_reset(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            # case 1
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2)
            z = layers.lod_reset(x=x, y=y)
            self.assertTrue(z.lod_level == 2)
            # case 2
            lod_tensor_in = layers.data(name='lod_in', shape=[1], dtype='int32')
            z = layers.lod_reset(x=x, y=lod_tensor_in)
            self.assertTrue(z.lod_level == 1)
            # case 3
            z = layers.lod_reset(x=x, target_lod=[1, 2, 3])
            self.assertTrue(z.lod_level == 1)
            return z

    def test_affine_grid(self):
        with self.static_graph():
            data = layers.data(name='data', shape=[2, 3, 3], dtype="float32")
            out, ids = layers.argsort(input=data, axis=1)

            theta = layers.data(name="theta", shape=[2, 3], dtype="float32")
            out_shape = layers.data(name="out_shape", shape=[-1], dtype="int32")
            data_0 = layers.affine_grid(theta, out_shape)
            data_1 = layers.affine_grid(theta, [5, 3, 28, 28])

            self.assertIsNotNone(data_0)
            self.assertIsNotNone(data_1)

    def test_stridedslice(self):
        axes = [0, 1, 2]
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        strides = [1, 1, 1]
        with self.static_graph():
            x = layers.data(name="x", shape=[245, 30, 30], dtype="float32")
            out = layers.strided_slice(
                x, axes=axes, starts=starts, ends=ends, strides=strides)
            return out

    def test_fill_constant_batch_size_like(self):
        with self.static_graph():
            like = fluid.layers.fill_constant(
                shape=[1, 200], value=10, dtype='int64')
            out = layers.fill_constant_batch_size_like(
                input=like, shape=[2, 3300], value=1315454564656, dtype='int64')
            return out

    def test_psroi_pool(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name="x", shape=[245, 30, 30], dtype="float32")
            rois = layers.data(
                name="rois", shape=[4], dtype="float32", lod_level=1)
            output = layers.psroi_pool(x, rois, 5, 0.25, 7, 7)
            return (output)

    def test_sequence_expand(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[10], dtype='float32')
            y = layers.data(
                name='y', shape=[10, 20], dtype='float32', lod_level=2)
            return (layers.sequence_expand(x=x, y=y, ref_level=1))

    def test_sequence_reshape(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[8], dtype='float32', lod_level=1)
            out = layers.sequence_reshape(input=x, new_dim=16)
            return (out)

    def test_sequence_unpad(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[10, 5], dtype='float32')
            length = layers.data(name='length', shape=[], dtype='int64')
            return (layers.sequence_unpad(x=x, length=length))

    def test_sequence_softmax(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            seq_data = layers.data(
                name='seq_data', shape=[10, 10], dtype='float32', lod_level=1)
            seq = layers.fc(input=seq_data, size=20)
            return (layers.sequence_softmax(seq))

    def test_sequence_unsqueeze(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[8, 2], dtype='float32')
            out = layers.unsqueeze(input=x, axes=[1])
            return (out)

    def test_sequence_scatter(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(
                name='x',
                shape=[3, 6],
                append_batch_size=False,
                dtype='float32')
            idx = layers.data(
                name='idx',
                shape=[12, 1],
                append_batch_size=False,
                dtype='int32',
                lod_level=1)
            updates = layers.data(
                name='updates',
                shape=[12, 1],
                append_batch_size=False,
                dtype='float32',
                lod_level=1)
            out = layers.sequence_scatter(input=x, index=idx, updates=updates)
            return (out)

    def test_sequence_slice(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            import numpy as np
            seqs = layers.data(
                name='x', shape=[10, 5], dtype='float32', lod_level=1)
            offset = layers.assign(input=np.array([[0, 1]]).astype('int32'))
            length = layers.assign(input=np.array([[2, 1]]).astype('int32'))
            out = layers.sequence_slice(
                input=seqs, offset=offset, length=length)
            return (out)

    def test_filter_by_instag(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x1 = layers.data(
                name='Ins', shape=[32, 1], dtype='float32', lod_level=0)
            x2 = layers.data(
                name='Ins_tag',
                shape=[32, 1],
                dtype='int64',
                lod_level=0,
                stop_gradient=True)
            x3 = layers.create_global_var(
                shape=[1, 1],
                value=20,
                dtype='int64',
                persistable=True,
                force_cpu=True,
                name='Filter_tag')
            out1, out2 = layers.filter_by_instag(x1, x2, x3, is_lod=True)

    def test_shuffle_batch(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(
                name='X', shape=[4, 50], dtype='float32', lod_level=0)
            out1 = fluid.contrib.layers.shuffle_batch(x)
            default_main_program().random_seed = 1000
            out2 = fluid.contrib.layers.shuffle_batch(x)
            self.assertIsNotNone(out1)
            self.assertIsNotNone(out2)
            return (out1)

    def test_partial_sum(self):
        with self.static_graph():
            x = fluid.data(name="x", shape=[None, 3], dtype="float32")
            y = fluid.data(name="y", shape=[None, 3], dtype="float32")
            sum = fluid.contrib.layers.partial_sum(
                [x, y], start_index=0, length=2)
            return (sum)

    def test_batch_fc(self):
        with self.static_graph():
            input = fluid.data(name="input", shape=[16, 2, 3], dtype="float32")
            out = fluid.contrib.layers.batch_fc(
                input=input,
                param_size=[16, 3, 10],
                param_attr=fluid.ParamAttr(
                    learning_rate=1.0,
                    name="w_0",
                    initializer=fluid.initializer.Xavier(uniform=False)),
                bias_size=[16, 10],
                bias_attr=fluid.ParamAttr(
                    learning_rate=1.0,
                    name="b_0",
                    initializer=fluid.initializer.Xavier(uniform=False)),
                act="relu")
        return (out)

    def test_rank_attention(self):
        with self.static_graph():
            input = fluid.data(name="input", shape=[None, 2], dtype="float32")
            rank_offset = fluid.data(
                name="rank_offset", shape=[None, 7], dtype="int32")
            out = fluid.contrib.layers.rank_attention(
                input=input,
                rank_offset=rank_offset,
                rank_param_shape=[18, 3],
                rank_param_attr=fluid.ParamAttr(
                    learning_rate=1.0,
                    name="ubm_rank_param.w_0",
                    initializer=fluid.initializer.Xavier(uniform=False)),
                max_rank=3)
            return (out)

    def test_roi_pool(self):
        x_np = np.random.rand(2, 3, 8, 8).astype('float32')
        rois_np = np.random.rand(3, 4).astype('float32')
        rois_num_np = np.array([1, 2]).astype('int32')

        with self.static_graph():
            x = layers.data(name="x", shape=[3, 8, 8], dtype="float32")
            rois = layers.data(name="rois", shape=[4], dtype="float32")
            rois_num = fluid.data(name="rois_num", shape=[None], dtype="int32")
            output = layers.roi_pool(x, rois, 4, 4, 0.5, rois_num=rois_num)
            static_res = self.get_static_graph_result(
                feed={'x': x_np,
                      'rois': rois_np,
                      'rois_num': rois_num_np},
                fetch_list=[output])[0]

        with self.dynamic_graph():
            x_dy = base.to_variable(x_np)
            rois_dy = base.to_variable(rois_np)
            rois_num_dy = base.to_variable(rois_num_np)
            dy_res = layers.roi_pool(
                x_dy, rois_dy, 4, 4, 0.5, rois_num=rois_num_dy)
            dy_res_value = dy_res[0].numpy()
        self.assertTrue(np.array_equal(static_res, dy_res_value))

    def test_sequence_enumerate(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name="input", shape=[1], dtype='int32', lod_level=1)
            out = layers.sequence_enumerate(input=x, win_size=2, pad_value=0)

    def test_roi_align(self):
        x_np = np.random.rand(2, 3, 8, 8).astype('float32')
        rois_np = np.random.rand(3, 4).astype('float32')
        rois_num_np = np.array([1, 2]).astype('int32')

        with self.static_graph():
            x = layers.data(name="x", shape=[3, 8, 8], dtype="float32")
            rois = layers.data(name="rois", shape=[4], dtype="float32")
            rois_num = fluid.data(name="rois_num", shape=[None], dtype="int32")
            output = layers.roi_align(x, rois, 4, 4, 0.5, 2, rois_num=rois_num)
            static_res = self.get_static_graph_result(
                feed={'x': x_np,
                      'rois': rois_np,
                      'rois_num': rois_num_np},
                fetch_list=[output])[0]

        with self.dynamic_graph():
            x_dy = base.to_variable(x_np)
            rois_dy = base.to_variable(rois_np)
            rois_num_dy = base.to_variable(rois_num_np)
            dy_res = layers.roi_align(
                x_dy, rois_dy, 4, 4, 0.5, 2, rois_num=rois_num_dy)
            dy_res_value = dy_res.numpy()
        self.assertTrue(np.array_equal(static_res, dy_res_value))

    def test_roi_perspective_transform(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name="x", shape=[256, 30, 30], dtype="float32")
            rois = layers.data(
                name="rois", shape=[8], dtype="float32", lod_level=1)
            output = layers.roi_perspective_transform(x, rois, 7, 7, 0.6)
            return (output)

    def test_row_conv(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = layers.data(name='x', shape=[16], dtype='float32', lod_level=1)
            out = layers.row_conv(input=x, future_context_size=2)
            return (out)

    def test_simple_conv2d(self):
        # TODO(minqiyang): dygraph do not support layers with param now
        with self.static_graph():
            images = layers.data(
                name='pixel', shape=[3, 48, 48], dtype='float32')
            return layers.conv2d(
                input=images, num_filters=3, filter_size=[4, 4])

    def test_squeeze(self):
        # TODO(minqiyang): dygraph do not support layers with param now
        with self.static_graph():
            x = layers.data(name='x', shape=[1, 1, 4], dtype='float32')
            out = layers.squeeze(input=x, axes=[2])
            return (out)

    def test_flatten(self):
        # TODO(minqiyang): dygraph do not support op without kernel now
        with self.static_graph():
            x = layers.data(
                name='x',
                append_batch_size=False,
                shape=[4, 4, 3],
                dtype="float32")
            out = layers.flatten(x, axis=1, name="flatten")
            return (out)

    def test_linspace(self):
        program = Program()
        with program_guard(program):
            out = layers.linspace(20, 10, 5, 'float64')
            self.assertIsNotNone(out)
        print(str(program))

    def test_deformable_conv(self):
        with self.static_graph():
            input = layers.data(
                name='input',
                append_batch_size=False,
                shape=[2, 3, 32, 32],
                dtype="float32")
            offset = layers.data(
                name='offset',
                append_batch_size=False,
                shape=[2, 18, 32, 32],
                dtype="float32")
            mask = layers.data(
                name='mask',
                append_batch_size=False,
                shape=[2, 9, 32, 32],
                dtype="float32")
            out = layers.deformable_conv(
                input=input,
                offset=offset,
                mask=mask,
                num_filters=2,
                filter_size=3,
                padding=1)
            return (out)

    def test_deformable_conv2(self):
        with self.static_graph():
            input = fluid.data(
                name='input', shape=[None, 3, None, None], dtype="float32")
            offset = fluid.data(
                name='offset', shape=[None, 18, None, None], dtype="float32")
            mask = fluid.data(
                name='mask', shape=[None, 9, None, None], dtype="float32")
            out = layers.deformable_conv(
                input=input,
                offset=offset,
                mask=mask,
                num_filters=2,
                filter_size=3,
                padding=1)
            return (out)

    def test_unfold(self):
        with self.static_graph():
            x = layers.data(name='x', shape=[3, 20, 20], dtype='float32')
            out = layers.unfold(x, [3, 3], 1, 1, 1)
            return (out)

    def test_partial_concat(self):
        with self.static_graph():
            x = fluid.data(name="x", shape=[None, 3], dtype="float32")
            y = fluid.data(name="y", shape=[None, 3], dtype="float32")
            concat1 = fluid.contrib.layers.partial_concat(
                [x, y], start_index=0, length=2)
            concat2 = fluid.contrib.layers.partial_concat(
                x, start_index=0, length=-1)
            return concat1, concat2

    def test_deform_roi_pooling(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = layers.data(
                name='input',
                shape=[2, 3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            rois = layers.data(
                name="rois", shape=[4], dtype='float32', lod_level=1)
            trans = layers.data(
                name="trans",
                shape=[2, 3, 32, 32],
                dtype='float32',
                append_batch_size=False)
            out = layers.deformable_roi_pooling(
                input=input,
                rois=rois,
                trans=trans,
                no_trans=False,
                spatial_scale=1.0,
                group_size=(1, 1),
                pooled_height=8,
                pooled_width=8,
                part_size=(8, 8),
                sample_per_part=4,
                trans_std=0.1)
        return (out)

    def test_deformable_conv_v1(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = layers.data(
                name='input',
                append_batch_size=False,
                shape=[2, 3, 32, 32],
                dtype="float32")
            offset = layers.data(
                name='offset',
                append_batch_size=False,
                shape=[2, 18, 32, 32],
                dtype="float32")
            out = layers.deformable_conv(
                input=input,
                offset=offset,
                mask=None,
                num_filters=2,
                filter_size=3,
                padding=1,
                modulated=False)
            return (out)

    def test_retinanet_target_assign(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            bbox_pred = layers.data(
                name='bbox_pred',
                shape=[1, 100, 4],
                append_batch_size=False,
                dtype='float32')
            cls_logits = layers.data(
                name='cls_logits',
                shape=[1, 100, 10],
                append_batch_size=False,
                dtype='float32')
            anchor_box = layers.data(
                name='anchor_box',
                shape=[100, 4],
                append_batch_size=False,
                dtype='float32')
            anchor_var = layers.data(
                name='anchor_var',
                shape=[100, 4],
                append_batch_size=False,
                dtype='float32')
            gt_boxes = layers.data(
                name='gt_boxes',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            gt_labels = layers.data(
                name='gt_labels',
                shape=[10, 1],
                append_batch_size=False,
                dtype='int32')
            is_crowd = layers.data(
                name='is_crowd',
                shape=[1],
                append_batch_size=False,
                dtype='int32')
            im_info = layers.data(
                name='im_info',
                shape=[1, 3],
                append_batch_size=False,
                dtype='float32')
            return (layers.retinanet_target_assign(
                bbox_pred, cls_logits, anchor_box, anchor_var, gt_boxes,
                gt_labels, is_crowd, im_info, 10))

    def test_sigmoid_focal_loss(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = layers.data(
                name='data',
                shape=[10, 80],
                append_batch_size=False,
                dtype='float32')
            label = layers.data(
                name='label',
                shape=[10, 1],
                append_batch_size=False,
                dtype='int32')
            fg_num = layers.data(
                name='fg_num',
                shape=[1],
                append_batch_size=False,
                dtype='int32')
            out = fluid.layers.sigmoid_focal_loss(
                x=input, label=label, fg_num=fg_num, gamma=2., alpha=0.25)
            return (out)

    def test_addmm(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            input = layers.data(
                name='input_data',
                shape=[3, 3],
                append_batch_size=False,
                dtype='float32')
            x = layers.data(
                name='x',
                shape=[3, 2],
                append_batch_size=False,
                dtype='float32')
            y = layers.data(
                name='y',
                shape=[2, 3],
                append_batch_size=False,
                dtype='float32')

            out = paddle.addmm(input=input, x=x, y=y)
            return (out)

    def test_retinanet_detection_output(self):
        with program_guard(fluid.default_main_program(),
                           fluid.default_startup_program()):
            bboxes = layers.data(
                name='bboxes',
                shape=[1, 21, 4],
                append_batch_size=False,
                dtype='float32')
            scores = layers.data(
                name='scores',
                shape=[1, 21, 10],
                append_batch_size=False,
                dtype='float32')
            anchors = layers.data(
                name='anchors',
                shape=[21, 4],
                append_batch_size=False,
                dtype='float32')
            im_info = layers.data(
                name="im_info",
                shape=[1, 3],
                append_batch_size=False,
                dtype='float32')
            nmsed_outs = layers.retinanet_detection_output(
                bboxes=[bboxes, bboxes],
                scores=[scores, scores],
                anchors=[anchors, anchors],
                im_info=im_info,
                score_threshold=0.05,
                nms_top_k=1000,
                keep_top_k=100,
                nms_threshold=0.3,
                nms_eta=1.)
            return (nmsed_outs)

    def test_warpctc_with_padding(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            input_length = layers.data(
                name='logits_length', shape=[11], dtype='int64')
            label_length = layers.data(
                name='labels_length', shape=[12], dtype='int64')
            label = layers.data(name='label', shape=[12, 1], dtype='int32')
            predict = layers.data(
                name='predict', shape=[4, 4, 8], dtype='float32')
            output = layers.warpctc(
                input=predict,
                label=label,
                input_length=input_length,
                label_length=label_length)
            return (output)

    def test_edit_distance(self):
        with self.static_graph():
            predict = layers.data(
                name='predict', shape=[-1, 1], dtype='int64', lod_level=1)
            label = layers.data(
                name='label', shape=[-1, 1], dtype='int64', lod_level=1)
            evaluator = fluid.evaluator.EditDistance(predict, label)
            return evaluator.metrics

    def test_basic_gru(self):
        input_size = 128
        hidden_size = 256
        with self.static_graph():
            input = fluid.data(
                name="input", shape=[None, None, input_size], dtype='float32')
            pre_hidden = fluid.data(
                name="pre_hidden", shape=[None, hidden_size], dtype='float32')
            sequence_length = fluid.data(
                name="sequence_length", shape=[None], dtype='int32')

            for bidirectional in [True, False]:
                for batch_first in [True, False]:
                    rnn_out, last_hidden = fluid.contrib.layers.basic_gru(
                        input,
                        pre_hidden,
                        hidden_size=256,
                        num_layers=2,
                        sequence_length=sequence_length,
                        dropout_prob=0.5,
                        bidirectional=bidirectional,
                        batch_first=batch_first)


class TestMetricsDetectionMap(unittest.TestCase):
    def test_detection_map(self):
        program = fluid.Program()
        with program_guard(program):
            detect_res = fluid.layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = fluid.layers.data(
                name='label',
                shape=[10, 1],
                append_batch_size=False,
                dtype='float32')
            box = fluid.layers.data(
                name='bbox',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            map_eval = fluid.metrics.DetectionMAP(
                detect_res, label, box, class_num=21)
            cur_map, accm_map = map_eval.get_map_var()
            self.assertIsNotNone(cur_map)
            self.assertIsNotNone(accm_map)
        print(str(program))


if __name__ == '__main__':
    unittest.main()
