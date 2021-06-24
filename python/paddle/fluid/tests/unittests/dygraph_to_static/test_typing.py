# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
from typing import Tuple, List, Dict, TypeVar


class BaseLayer(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super(BaseLayer, self).__init__()
        self._linear = paddle.nn.Linear(in_size, out_size)

    def build(self, x):
        out1 = self._linear(x)
        out2 = paddle.mean(out1)
        return out1, out2


class LinearNetWithTuple(BaseLayer):
    def __init__(self, in_size, out_size):
        super(LinearNetWithTuple, self).__init__(in_size, out_size)

    def forward(self, x) -> Tuple[paddle.Tensor, str]:
        out1, out2 = self.build(x)
        return (out2, 'str')


class LinearNetWithTuple2(BaseLayer):
    def __init__(self, in_size, out_size):
        super(LinearNetWithTuple2, self).__init__(in_size, out_size)

    def forward(self, x) -> Tuple[paddle.Tensor, np.array]:
        out1, out2 = self.build(x)
        return (out2, np.ones([4, 16]))


class LinearNetWithList(BaseLayer):
    def __init__(self, in_size, out_size):
        super(LinearNetWithList, self).__init__(in_size, out_size)

    def forward(self, x) -> List[paddle.Tensor]:
        out1, out2 = self.build(x)
        return [out2]


class LinearNetWithDict(BaseLayer):
    def __init__(self, in_size, out_size):
        super(LinearNetWithDict, self).__init__(in_size, out_size)

    def forward(self, x) -> Dict[str, paddle.Tensor]:
        out1, out2 = self.build(x)
        return {'out': out2}


class TestTyping(unittest.TestCase):
    def setUp(self):
        self.in_num = 16
        self.out_num = 16
        self.x = paddle.randn([4, 16])
        self.spec = [paddle.static.InputSpec(shape=[None, 16], dtype='float32')]

    def build_net(self):
        return LinearNetWithTuple(self.in_num, self.out_num)

    def save_and_load(self, suffix=''):
        path = './layer_typing_' + suffix
        paddle.jit.save(self.net, path, input_spec=self.spec)
        return paddle.jit.load(path)

    def run_dy(self):
        out, _ = self.net(self.x)
        return out

    def test_type(self):
        self.net = self.build_net()
        out = self.run_dy()
        load_net = self.save_and_load('tuple')
        load_out = load_net(self.x)
        self.assertTrue(np.allclose(out, load_out))


class TestTypingTuple(TestTyping):
    def build_net(self):
        return LinearNetWithTuple2(self.in_num, self.out_num)

    def run_dy(self):
        out, np_data = self.net(self.x)
        self.assertTrue(np.equal(np_data, np.ones_like(np_data)).all())
        return out


class TestTypingList(TestTyping):
    def build_net(self):
        return LinearNetWithList(self.in_num, self.out_num)

    def run_dy(self):
        out = self.net(self.x)[0]
        return out


class TestTypingDict(TestTyping):
    def build_net(self):
        return LinearNetWithDict(self.in_num, self.out_num)

    def run_dy(self):
        out = self.net(self.x)['out']
        return out


if __name__ == '__main__':
    unittest.main()
