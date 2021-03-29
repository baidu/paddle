# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
import numpy as np
import unittest


class TestSoftmaxWithXe(unittest.TestCase):
    def setUp(self):
        self.initParameter()
        self.m, self.n = np.random.random_integers(
            low=100, high=2000, size=[2]).astype('int64')
        self.executed_api()

    def initParameter(self):
        self.dtype = 'float32'
        self.soft_label = False

    def executed_api(self):
        self.function = fluid.layers.softmax_with_cross_entropy

    def softmax_with_xe(self,
                        x,
                        y,
                        place,
                        inplace=True,
                        numeric_stable_mode=True):
        m, n = x.shape
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(fluid.Scope()):
                x_d = fluid.layers.data(
                    name='x',
                    shape=[m, n],
                    dtype=self.dtype,
                    append_batch_size=False)
                y_d = fluid.layers.data(
                    name='y',
                    shape=[m, 1] if not self.soft_label else [m, n],
                    dtype='int64' if not self.soft_label else self.dtype,
                    append_batch_size=False)
                z_d, s_d = self.function(
                    x_d,
                    y_d,
                    soft_label=self.soft_label,
                    return_softmax=True,
                    numeric_stable_mode=numeric_stable_mode)

                exe = fluid.Executor(place)

                exe.run(fluid.default_startup_program())

                build_strategy = fluid.BuildStrategy()
                build_strategy.enable_inplace = inplace
                prog = fluid.CompiledProgram(fluid.default_main_program(
                )).with_data_parallel(
                    build_strategy=build_strategy, places=place)

                fetch_list = [z_d.name, s_d.name]

                print('Inplace is {}'.format("ON" if inplace else "OFF"))

                z, s = exe.run(prog,
                               feed={x_d.name: x,
                                     y_d.name: y},
                               fetch_list=fetch_list)
                return z, s

    def main_with_place(self, place):
        x = np.random.random(size=[self.m, self.n]).astype(self.dtype)
        x_range = [(-30, 30), (10, 20), (-1, 1), (2, 3), (0, 0.3), (-200, -100)]

        for a, b in x_range:
            x = ((b - a) * x + a).astype(self.dtype)
            if not self.soft_label:
                y = np.random.random_integers(
                    size=[self.m, 1], low=0, high=self.n - 1).astype('int64')
            else:
                y = np.random.random(size=[self.m, self.n]).astype(self.dtype)
                norm_y = np.broadcast_to(
                    np.reshape(
                        np.sum(y, axis=1), [-1, 1]), y.shape)
                y = y / norm_y

            z1, s1 = self.softmax_with_xe(
                x, y, place, inplace=False, numeric_stable_mode=False)
            z2, s2 = self.softmax_with_xe(
                x, y, place, inplace=True, numeric_stable_mode=False)

            self.assertTrue((z1 == z2).all())
            self.assertTrue((s1 == s2).all())

            z1, s1 = self.softmax_with_xe(
                x, y, place, inplace=False, numeric_stable_mode=True)
            z2, s2 = self.softmax_with_xe(
                x, y, place, inplace=True, numeric_stable_mode=True)
            self.assertTrue((z1 == z2).all())
            self.assertTrue((s1 == s2).all())

    def test_main(self):
        paddle.enable_static()
        self.main_with_place(fluid.CPUPlace())
        if fluid.core.is_compiled_with_cuda():
            self.main_with_place(fluid.CUDAPlace(0))


class TestSoftmaxWithXe1(TestSoftmaxWithXe):
    def initParameter(self):
        self.dtype = 'float32'
        self.soft_label = True


class TestSoftmaxWithXe2(TestSoftmaxWithXe):
    def initParameter(self):
        self.dtype = 'float64'
        self.soft_label = False


class TestSoftmaxWithXe3(TestSoftmaxWithXe):
    def initParameter(self):
        self.dtype = 'float64'
        self.soft_label = True


class TestDygraphInplaceApiInStaticMode(TestSoftmaxWithXe):
    def executed_api(self):
        self.function = paddle.nn.functional.softmax_with_cross_entropy_


class TestDygraphInplaceApi(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.return_softmax = False

    def executed_non_inplace_api(self, data, label, return_softmax):
        return paddle.nn.functional.softmax_with_cross_entropy(
            logits=data, label=label, return_softmax=return_softmax)

    def executed_inplace_api(self, data, label, return_softmax):
        return paddle.nn.functional.softmax_with_cross_entropy_(
            logits=data, label=label, return_softmax=return_softmax)

    def test_dygraph_mode(self):
        paddle.disable_static()
        data = np.random.rand(128).astype("float32")
        label = np.random.rand(1).astype("int64")
        data = paddle.to_tensor(data)
        label = paddle.to_tensor(label)
        out_non_inplace = self.executed_non_inplace_api(data, label,
                                                        self.return_softmax)
        out_inplace = self.executed_inplace_api(data, label,
                                                self.return_softmax)
        if not self.return_softmax:
            self.assertTrue((out_non_inplace.numpy() == out_inplace.numpy()
                             ).all())
        else:
            self.assertTrue((out_non_inplace[0].numpy() == out_inplace[0].numpy(
            )).all())
            self.assertTrue((out_non_inplace[1].numpy() == out_inplace[1].numpy(
            )).all())
        paddle.enable_static()


class TestDygraphInplaceApiReturnSoftmax(TestDygraphInplaceApi):
    def init_data(self):
        self.return_softmax = True


if __name__ == '__main__':
    unittest.main()
