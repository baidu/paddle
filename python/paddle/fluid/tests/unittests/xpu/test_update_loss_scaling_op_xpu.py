#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import sys
sys.path.append("..")
import numpy as np
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.contrib.mixed_precision.amp_nn as amp_nn

paddle.enable_static()


class TestUpdateLossScalingOp(XPUOpTest):
    def setUp(self):
        self.op_type = "update_loss_scaling"
        self.init()
        found_inf = np.array([False], dtype=np.bool)
        x = np.random.random((1024, 1024)).astype(self.dtype)

        self.inputs = {
            'X': [('x0', x)],
            'FoundInfinite': found_inf,
            'PrevLossScaling': self.prev_loss_scaling,
            'InGoodSteps': self.num_good_steps,
            'InBadSteps': self.num_bad_steps
        }

        self.outputs = {
            'Out': [('out0', x)],
            'LossScaling': self.prev_loss_scaling * self.incr_ratio,
            'OutGoodSteps': self.zero_steps,
            'OutBadSteps': self.zero_steps
        }

    def init(self):
        self.incr_ratio = 2.0
        self.decr_ratio = 0.8
        self.dtype = np.float32
        self.prev_loss_scaling = np.array([2048]).astype(self.dtype)
        self.num_good_steps = np.array([999], dtype=np.int32)
        self.num_bad_steps = np.array([1], dtype=np.int32)
        self.zero_steps = np.array([0], dtype=np.int32)
        self.attrs = {
            'incr_every_n_steps': 1000,
            'decr_every_n_nan_or_inf': 2,
            'incr_ratio': self.incr_ratio,
            'decr_ratio': self.decr_ratio,
        }

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, no_check_set=['Out'])


class TestUpdateLossScalingOpBad(TestUpdateLossScalingOp):
    def setUp(self):
        self.op_type = "update_loss_scaling"
        self.init()
        found_inf = np.array([True], dtype=np.bool)
        x = np.random.random((1024, 1024)).astype(self.dtype)
        i = np.random.randint(0, 1024, 1)
        j = np.random.randint(0, 1024, 1)
        x[i[0]][j[0]] = np.inf

        self.inputs = {
            'X': [('x0', x)],
            'FoundInfinite': found_inf,
            'PrevLossScaling': self.prev_loss_scaling,
            'InGoodSteps': self.num_good_steps,
            'InBadSteps': self.num_bad_steps
        }

        self.outputs = {
            'Out': [('out0', np.zeros_like(x))],
            'LossScaling': self.prev_loss_scaling * self.decr_ratio,
            'OutGoodSteps': self.zero_steps,
            'OutBadSteps': self.zero_steps
        }

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)
        #self.check_output()


class TestUpdateLossScalingLayer(unittest.TestCase):
    def loss_scaling_check(self, scope=fluid.Scope()):
        a = fluid.data(name="a", shape=[1024, 1024], dtype='float32')
        b = fluid.data(name="b", shape=[512, 128], dtype='float32')
        x = [a, b]
        found_inf = fluid.data(name="found_inf", shape=[1], dtype='bool')
        prev_loss_scaling = fluid.data(
            name="prev_loss_scaling", shape=[1], dtype='float32')
        num_good_steps = fluid.data(
            name="num_good_steps", shape=[1], dtype='int32')
        num_bad_steps = fluid.data(
            name="num_bad_steps", shape=[1], dtype='int32')

        a_v = np.random.random([1024, 1024]).astype('float32')
        b_v = np.random.random([512, 128]).astype('float32')
        found_inf_v = np.array([False]).astype('bool')
        prev_loss_scaling_v = np.array([2048]).astype('float32')
        num_good_steps_v = np.array([999], dtype=np.int32)
        num_bad_steps_v = np.array([1], dtype=np.int32)

        incr_every_n_steps = 1000
        decr_every_n_nan_or_inf = 2
        incr_ratio = 2
        decr_ratio = 0.8

        result = amp_nn.update_loss_scaling(
            x,
            found_inf,
            prev_loss_scaling,
            num_good_steps,
            num_bad_steps,
            incr_every_n_steps,
            decr_every_n_nan_or_inf,
            incr_ratio,
            decr_ratio,
            name="update_loss_scaling")

        place = fluid.XPUPlace(0)
        exe = fluid.Executor(place)
        with fluid.scope_guard(scope):
            exe.run(fluid.default_startup_program())
            result_v = exe.run(feed={
                'a': a_v,
                'b': b_v,
                'found_inf': found_inf_v,
                'prev_loss_scaling': prev_loss_scaling_v,
                'num_good_steps': num_good_steps_v,
                'num_bad_steps': num_bad_steps_v
            },
                               fetch_list=[
                                   result, x, found_inf, prev_loss_scaling,
                                   num_good_steps, num_bad_steps
                               ])
        assert np.array_equal(result_v[0], a_v)
        assert np.array_equal(result_v[1], b_v)
        assert np.array_equal(result_v[0], result_v[2])
        assert np.array_equal(result_v[1], result_v[3])
        assert np.array_equal(result_v[4], found_inf_v)
        assert np.array_equal(result_v[5], prev_loss_scaling_v * incr_ratio)
        assert np.array_equal(result_v[6], np.zeros_like(num_good_steps_v))
        assert np.array_equal(result_v[7], np.zeros_like(num_bad_steps_v))

    def loss_scaling_check_inf(self, use_cuda=True, scope=fluid.Scope()):
        a = fluid.data(name="a", shape=[1024, 1024], dtype='float32')
        b = fluid.data(name="b", shape=[512, 128], dtype='float32')
        x = [a, b]
        found_inf = fluid.data(name="found_inf", shape=[1], dtype='bool')
        prev_loss_scaling = fluid.data(
            name="prev_loss_scaling", shape=[1], dtype='float32')
        num_good_steps = fluid.data(
            name="num_good_steps", shape=[1], dtype='int32')
        num_bad_steps = fluid.data(
            name="num_bad_steps", shape=[1], dtype='int32')

        a_v = np.random.random([1024, 1024]).astype('float32')
        b_v = np.random.random([512, 128]).astype('float32')
        i = np.random.randint(0, 1024, 1)
        j = np.random.randint(0, 1024, 1)
        a_v[i[0]][j[0]] = np.inf
        found_inf_v = np.array([True]).astype('bool')
        prev_loss_scaling_v = np.array([2048]).astype('float32')
        num_good_steps_v = np.array([999], dtype=np.int32)
        num_bad_steps_v = np.array([1], dtype=np.int32)

        incr_every_n_steps = 1000
        decr_every_n_nan_or_inf = 2
        incr_ratio = 2
        decr_ratio = 0.8

        result = amp_nn.update_loss_scaling(
            x,
            found_inf,
            prev_loss_scaling,
            num_good_steps,
            num_bad_steps,
            incr_every_n_steps,
            decr_every_n_nan_or_inf,
            incr_ratio,
            decr_ratio,
            name="update_loss_scaling")

        place = fluid.XPUPlace(0)
        exe = fluid.Executor(place)
        with fluid.scope_guard(scope):
            exe.run(fluid.default_startup_program())
            result_v = exe.run(feed={
                'a': a_v,
                'b': b_v,
                'found_inf': found_inf_v,
                'prev_loss_scaling': prev_loss_scaling_v,
                'num_good_steps': num_good_steps_v,
                'num_bad_steps': num_bad_steps_v
            },
                               fetch_list=[
                                   result, x, found_inf, prev_loss_scaling,
                                   num_good_steps, num_bad_steps
                               ])
        assert np.array_equal(result_v[0], np.zeros_like(a_v))
        assert np.array_equal(result_v[1], np.zeros_like(b_v))
        assert np.array_equal(result_v[2], np.zeros_like(a_v))
        assert np.array_equal(result_v[3], np.zeros_like(b_v))
        assert np.array_equal(result_v[4], found_inf_v)
        assert np.array_equal(result_v[5], prev_loss_scaling_v * decr_ratio)
        assert np.array_equal(result_v[6], np.zeros_like(num_good_steps_v))
        assert np.array_equal(result_v[7], np.zeros_like(num_bad_steps_v))

    def test_loss_scaling(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                self.loss_scaling_check()

    def test_loss_scaling_inf(self):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                self.loss_scaling_check_inf()


if __name__ == '__main__':
    unittest.main()
