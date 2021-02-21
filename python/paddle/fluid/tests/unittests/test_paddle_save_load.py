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

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4
SEED = 10

IMAGE_SIZE = 784
CLASS_NUM = 10

LARGE_PARAM = 2**26


def random_batch_reader():
    def _get_random_inputs_and_labels():
        np.random.seed(SEED)
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (
            BATCH_SIZE,
            1, )).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_inputs_and_labels()
            batch_image = paddle.to_tensor(batch_image)
            batch_label = paddle.to_tensor(batch_label)
            yield batch_image, batch_label

    return __reader__


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)


def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()

        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def build_and_train_model(self):
        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()

        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

        # create data loader
        # TODO: using new DataLoader cause unknown Timeout on windows, replace it
        loader = random_batch_reader()

        # train
        train(layer, loader, loss_fn, adam)

        return layer, adam

    def check_load_state_dict(self, orig_dict, load_dict):
        for var_name, value in orig_dict.items():
            self.assertTrue(np.array_equal(value.numpy(), load_dict[var_name]))

    def test_save_load(self):
        layer, opt = self.build_and_train_model()

        # save
        layer_save_path = "test_paddle_save_load.linear.pdparams"
        opt_save_path = "test_paddle_save_load.linear.pdopt"
        layer_state_dict = layer.state_dict()
        opt_state_dict = opt.state_dict()

        paddle.save(layer_state_dict, layer_save_path)
        paddle.save(opt_state_dict, opt_save_path)

        # load
        load_layer_state_dict = paddle.load(layer_save_path)
        load_opt_state_dict = paddle.load(opt_save_path)

        self.check_load_state_dict(layer_state_dict, load_layer_state_dict)
        self.check_load_state_dict(opt_state_dict, load_opt_state_dict)

        # test save load in static mode
        paddle.enable_static()
        static_save_path = "static_mode_test/test_paddle_save_load.linear.pdparams"
        paddle.save(layer_state_dict, static_save_path)
        load_static_state_dict = paddle.load(static_save_path)
        self.check_load_state_dict(layer_state_dict, load_static_state_dict)

        # error test cases, some tests relay base test above
        # 1. test save obj not dict error
        test_list = [1, 2, 3]
        with self.assertRaises(NotImplementedError):
            paddle.save(test_list, "not_dict_error_path")

        # 2. test save path format error
        with self.assertRaises(ValueError):
            paddle.save(layer_state_dict, "test_paddle_save_load.linear.model/")

        # 3. test load path not exist error
        with self.assertRaises(ValueError):
            paddle.load("test_paddle_save_load.linear.params")

        # 4. test load old save path error
        with self.assertRaises(ValueError):
            paddle.load("test_paddle_save_load.linear")


if __name__ == '__main__':
    unittest.main()
