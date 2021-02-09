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
import os
import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import time


class LayerWithLargeParameters(nn.Layer):
    def __init__(self):
        super(LayerWithLargeParameters, self).__init__()
        outs = 2**8
        self._l = nn.Linear(10, outs)
        self._l2 = nn.Linear(outs, outs)
        self._l3 = nn.Linear(101, outs)

    def forward(self, x):
        y = self._l(x)
        return y


class TestSaveLoad(unittest.TestCase):
    def test_save_load(self):
        # paddle.disable_static()
        # create network
        # time.sleep(15)
        layer = LayerWithLargeParameters()
        save_dict = layer.state_dict()
        save_dict['abs'] = 666
        save_dict['lr'] = 1.0
        save_dict[666] = 'zxc'

        path = os.path.join("test_paddle_save_load_cpp_py",
                            "layer_cpp_py_layer.pdparams_z")
        paddle.save(save_dict, path)
        dict_load = paddle.load(path)

        for key, val in save_dict.items():
            if isinstance(val, paddle.Tensor):
                self.assertTrue(
                    np.array_equal(dict_load[key], save_dict[key].numpy()))
            else:
                self.assertTrue(val == dict_load[key])

        for key, val in save_dict.items():
            paddle.save(val, path)
            t = paddle.load(path)
            if isinstance(val, paddle.Tensor):
                self.assertTrue(np.array_equal(t, val.numpy()))
            else:
                self.assertTrue(val == t)


if __name__ == '__main__':
    unittest.main()
