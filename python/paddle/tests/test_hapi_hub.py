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

from __future__ import division
from __future__ import print_function

import unittest
import os

import paddle
from paddle.hapi import hub

import numpy as np


class TestHub(unittest.TestCase):
    def setUp(self, ):
        self.local_repo = os.path.dirname(os.path.abspath(__file__))
        # self.github_repo = 'lyuwenyu/paddlehub_demo'

    def testLoad(self, ):
        model = hub.load(
            self.local_repo, model='MM', source='local', out_channels=8)

        data = paddle.rand((1, 3, 100, 100))
        out = model(data)

        np.testing.assert_equal(out.shape, [1, 8, 50, 50])

    def testHelp(self, ):
        docs = hub.help(
            self.local_repo,
            model='MM',
            source='local', )
        print(docs)

    def testList(self, ):
        models = hub.list(
            self.local_repo,
            source='local',
            force_reload=False, )

        print(models)


if __name__ == '__main__':
    unittest.main()
