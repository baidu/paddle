# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import unittest
from paddle import nn


class Net(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2)

    @paddle.jit.to_static
    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class TestLstm(unittest.TestCase):
    def run_lstm(self, to_static):
        paddle.jit.ProgramTranslator().enable(to_static)

        paddle.disable_static()
        paddle.static.default_main_program().random_seed = 1001
        paddle.static.default_startup_program().random_seed = 1001

        net = Net(12, 2)
        x = paddle.zeros((2, 10, 12))
        y = net(paddle.to_tensor(x))
        return y.numpy()

    def test_lstm_to_static(self):
        dygraph_out = self.run_lstm(to_static=False)
        static_out = self.run_lstm(to_static=True)
        self.assertTrue(
            np.allclose(dygraph_out, static_out),
            msg='dygraph_out is {}\n static_out is \n{}'.format(dygraph_out,
                                                                static_out))


if __name__ == "__main__":
    unittest.main()
