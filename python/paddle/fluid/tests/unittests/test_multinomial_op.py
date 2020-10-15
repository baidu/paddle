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
import paddle
import paddle.fluid as fluid
from op_test import OpTest
import numpy as np


class TestMultinomialOp(OpTest):
    def setUp(self):
        self.op_type = "multinomial"
        self.init_data()
        self.inputs = {"X": self.input_np}

    def init_data(self):
        # input probability is a vector, and replacement is True
        self.input_np = np.random.rand(4)
        self.outputs = {"Out": np.zeros(100000).astype("int64")}
        self.attrs = {"num_samples": 100000, "replacement": True}

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def sample_output(self, out):
        # count numbers of different categories
        sample_prob = np.unique(out, return_counts=True)[1].astype("float32")
        sample_prob /= sample_prob.sum()
        return sample_prob

    def verify_output(self, outs):
        # normalize the input to get the probability
        prob = self.input_np / self.input_np.sum(axis=-1, keepdims=True)
        sample_prob = self.sample_output(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))


class TestMultinomialOp2(TestMultinomialOp):
    def init_data(self):
        # input probability is a matrix
        self.input_np = np.random.rand(3, 4)
        self.outputs = {"Out": np.zeros((3, 100000)).astype("int64")}
        self.attrs = {"num_samples": 100000, "replacement": True}

    def sample_output(self, out):
        out_list = np.split(out, 3, axis=0)
        count_array = [0] * 3
        for i in range(3):
            count_array[i] = np.unique(
                out_list[i], return_counts=True)[1].astype("float32")
        sample_prob = np.stack(count_array, axis=0)
        sample_prob /= sample_prob.sum(axis=-1, keepdims=True)
        return sample_prob


class TestMultinomialOp3(TestMultinomialOp):
    def init_data(self):
        # replacement is False. number of samples must be less than number of categories.
        self.input_np = np.random.rand(1000)
        self.outputs = {"Out": np.zeros(100).astype("int64")}
        self.attrs = {"num_samples": 100, "replacement": False}

    def verify_output(self, outs):
        out = np.array(outs[0])
        unique_out = np.unique(out)
        self.assertEqual(
            len(unique_out), 100,
            "replacement is False. categories can't be sampled repeatedly")


class TestMultinomialApi(unittest.TestCase):
    def test_dygraph(self):
        # input probability is a vector, and replacement is True
        paddle.disable_static()
        x = paddle.rand([4])
        out = paddle.multinomial(x, num_samples=100000, replacement=True)
        x_numpy = x.numpy()
        paddle.enable_static()

        sample_prob = np.unique(
            out.numpy(), return_counts=True)[1].astype("float32")
        sample_prob /= sample_prob.sum()

        prob = x_numpy / x_numpy.sum(axis=-1, keepdims=True)
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))

    def test_dygraph2(self):
        # input probability is a matrix, and replacement is True
        paddle.disable_static()
        x = paddle.rand([3, 4])
        out = paddle.multinomial(x, num_samples=100000, replacement=True)
        x_numpy = x.numpy()

        out_list = np.split(out.numpy(), 3, axis=0)
        count_array = [0] * 3
        for i in range(3):
            count_array[i] = np.unique(
                out_list[i], return_counts=True)[1].astype("float32")
        sample_prob = np.stack(count_array, axis=0)
        sample_prob /= sample_prob.sum(axis=-1, keepdims=True)

        prob = x_numpy / x_numpy.sum(axis=-1, keepdims=True)
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))
        paddle.enable_static()

    def test_dygraph3(self):
        # replacement is False. number of samples must be less than number of categories.
        paddle.disable_static()
        x = paddle.rand([1000])
        out = paddle.multinomial(x, num_samples=100, replacement=False)
        x_numpy = x.numpy()

        unique_out = np.unique(out.numpy())
        self.assertEqual(
            len(unique_out), 100,
            "replacement is False. categories can't be sampled repeatedly")
        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            x = fluid.data('x', shape=[4], dtype='float32')
            out = paddle.multinomial(x, num_samples=100000, replacement=True)

            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

        exe.run(startup_program)
        x_np = np.random.rand(4).astype('float32')
        out = exe.run(train_program, feed={'x': x_np}, fetch_list=[out])

        sample_prob = np.unique(out, return_counts=True)[1].astype("float32")
        sample_prob /= sample_prob.sum()

        prob = x_np / x_np.sum(axis=-1, keepdims=True)
        self.assertTrue(
            np.allclose(
                sample_prob, prob, rtol=0, atol=0.01),
            "sample_prob: " + str(sample_prob) + "\nprob: " + str(prob))


class TestMultinomialAlias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        x = paddle.rand([4])
        paddle.multinomial(x, num_samples=10, replacement=True)
        paddle.tensor.multinomial(x, num_samples=10, replacement=True)
        paddle.tensor.random.multinomial(x, num_samples=10, replacement=True)


if __name__ == "__main__":
    unittest.main()
