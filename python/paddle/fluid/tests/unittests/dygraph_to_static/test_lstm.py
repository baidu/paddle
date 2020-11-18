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


class LSTMLayer(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(LSTMLayer, self).__init__()
        self.cell = nn.LSTM(
            in_channels, hidden_size, direction='bidirectional', num_layers=2)

    def forward(self, x):
        x, _ = self.cell(x)
        return x


class Net(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(Net, self).__init__()
        self.lstm = LSTMLayer(in_channels, hidden_size)

    def forward(self, x):
        x = self.lstm(x)
        return x


class TestLstm(unittest.TestCase):
    def run_lstm(self, to_static):
        paddle.jit.ProgramTranslator().enable(to_static)

        paddle.disable_static()
        paddle.static.default_main_program().random_seed = 1001
        paddle.static.default_startup_program().random_seed = 1001

        net = Net(12, 2)
        net = paddle.jit.to_static(net)
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

    def test_save_in_eval(self):
        paddle.jit.ProgramTranslator().enable(True)
        net = Net(12, 2)
        x = paddle.randn((2, 10, 12))
        dygraph_out = net(x)
        # switch eval mode firstly
        net.eval()

        net = paddle.jit.to_static(
            net, input_spec=[paddle.static.InputSpec(shape=[-1, 10, 12])])
        paddle.jit.save(net, 'simple_lstm')
        # load saved model
        load_net = paddle.jit.load('simple_lstm')

        static_out = load_net(x)
        self.assertTrue(
            np.allclose(dygraph_out.numpy(), static_out.numpy()),
            msg='dygraph_out is {}\n static_out is \n{}'.format(dygraph_out,
                                                                static_out))
        # switch back into train mode.
        net.train()
        train_out = net(x)
        self.assertTrue(
            np.allclose(dygraph_out.numpy(), train_out.numpy()),
            msg='dygraph_out is {}\n static_out is \n{}'.format(dygraph_out,
                                                                train_out))


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(10, 12)
        self.dropout = nn.Dropout(0.5)

    @paddle.jit.to_static
    def forward(self, x):
        y = self.fc(x)
        y = self.dropout(y)
        return y


class TestSaveInEvalMode(unittest.TestCase):
    def test_save_in_eval(self):
        x = paddle.randn((2, 10))
        paddle.jit.ProgramTranslator().enable(True)
        net = LinearNet()
        # switch eval mode firstly
        eval_out = net(x)
        net.eval()
        # save directly
        net = paddle.jit.to_static(
            net, input_spec=[paddle.static.InputSpec(shape=[-1, 10])])
        paddle.jit.save(net, 'linear_net')
        # load saved model
        load_net = paddle.jit.load('linear_net')

        eval_out = net(x)

        infer_out = load_net(x)
        self.assertTrue(
            np.allclose(eval_out.numpy(), infer_out.numpy()),
            msg='eval_out is {}\n infer_out is \n{}'.format(eval_out,
                                                            infer_out))


class TestEvalAfterSave(unittest.TestCase):
    def test_eval_after_save(self):
        x = paddle.randn((2, 10, 12)).astype('float32')
        net = Net(12, 2)
        dy_out = net(x)
        # save model
        paddle.jit.save(net, 'jit.save/lstm', input_spec=[x])
        load_net = paddle.jit.load('jit.save/lstm')
        load_out = load_net(x)
        self.assertTrue(np.allclose(dy_out.numpy(), load_out.numpy()))
        # eval
        net.eval()
        out = net(x)
        self.assertTrue(np.allclose(dy_out.numpy(), out.numpy()))


if __name__ == "__main__":
    unittest.main()
