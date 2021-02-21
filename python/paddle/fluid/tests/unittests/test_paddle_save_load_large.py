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

from __future__ import print_function

import unittest
import numpy as np
import os
import sys
import paddle
import paddle.nn as nn

BATCH_SIZE = 16
BATCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

LARGE_PARAM = 2**26


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)


class LayerWithLargeParameters(paddle.nn.Layer):
    def __init__(self):
        super(LayerWithLargeParameters, self).__init__()
        self._l = paddle.nn.Linear(10, LARGE_PARAM)

    def forward(self, x):
        y = self._l(x)
        return y


class TestSaveLoadLargeParameters(unittest.TestCase):
    def test_large_parameters_paddle_save(self):
        # enable dygraph mode
        paddle.disable_static()
        # create network
        layer = LayerWithLargeParameters()
        save_dict = layer.state_dict()

        path = os.path.join("test_paddle_save_load_large_param_save",
                            "layer.pdparams")
        paddle.save(layer.state_dict(), path)
        dict_load = paddle.load(path)
        # compare results before and after saving
        for key, value in save_dict.items():
            self.assertTrue(np.array_equal(dict_load[key], value.numpy()))


class TestSaveLoadPickle(unittest.TestCase):
    def test_pickle_protocol(self):
        # create network
        layer = LinearNet()
        save_dict = layer.state_dict()

        path = os.path.join("test_paddle_save_load_pickle_protocol",
                            "layer.pdparams")
        protocols = [2, ]
        if sys.version_info.major >= 3 and sys.version_info.minor >= 4:
            protocols += [3, 4]
        for protocol in protocols:
            paddle.save(layer.state_dict(), path, protocol=protocol)
            dict_load = paddle.load(path)
            # compare results before and after saving
            for key, value in save_dict.items():
                self.assertTrue(np.array_equal(dict_load[key], value.numpy()))

    def test_save_large_var(self):
        # create network
        layer = LayerWithLargeParameters()
        save_dict = layer.state_dict()
        path = os.path.join("test_paddle_save_load_pickle_protocol", "var_")

        for key, var in layer.state_dict().items():
            path_var = path + key + '.pdparams'
            paddle.save(var, path_var)
            var_load = paddle.load(path_var)
            # compare results before and after saving
            self.assertTrue(np.array_equal(var_load, var.numpy()))


if __name__ == '__main__':
    unittest.main()
