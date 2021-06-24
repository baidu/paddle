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

import sys
import unittest
import time
import random
import tempfile
import shutil
import numpy as np

import paddle
from paddle import Model
from paddle.static import InputSpec
from paddle.vision.models import LeNet
from paddle.hapi.callbacks import config_callbacks
import paddle.vision.transforms as T
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn.layer.loss import CrossEntropyLoss


class MnistDataset(MNIST):
    def __len__(self):
        return 512


class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def test_visualdl_callback(self):
        # visualdl not support python2
        if sys.version_info < (3, ):
            return

        inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
        labels = [InputSpec([None, 1], 'int64', 'label')]

        transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
        train_dataset = MnistDataset(mode='train', transform=transform)
        eval_dataset = MnistDataset(mode='test', transform=transform)

        net = paddle.vision.models.LeNet()
        model = paddle.Model(net, inputs, labels)

        optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
        model.prepare(
            optimizer=optim,
            loss=paddle.nn.CrossEntropyLoss(),
            metrics=paddle.metric.Accuracy())

        callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir')
        model.fit(train_dataset,
                  eval_dataset,
                  batch_size=64,
                  callbacks=callback)


if __name__ == '__main__':
    unittest.main()
