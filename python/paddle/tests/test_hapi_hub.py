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
        self.github_repo = 'lyuwenyu/paddlehub_demo:main'

    def testLoad(self, ):
        model = hub.load(
            self.local_repo, model='MM', source='local', out_channels=8)

        data = paddle.rand((1, 3, 100, 100))
        out = model(data)
        np.testing.assert_equal(out.shape, [1, 8, 50, 50])

        model = hub.load(
            self.github_repo, model='MM', source='github', force_reload=True)

        model = hub.load(
            self.github_repo,
            model='MM',
            source='github',
            force_reload=False,
            pretrained=False)

        model = hub.load(
            self.github_repo.split(':')[0],
            model='MM',
            source='github',
            force_reload=False,
            pretrained=False)

        model = hub.load(
            self.github_repo,
            model='MM',
            source='github',
            force_reload=False,
            pretrained=True,
            out_channels=8)

        data = paddle.ones((1, 3, 2, 2))
        out = model(data)

        gt = np.array([
            1.53965068, 0., 0., 1.39455748, 0.72066200, 0.19773030, 2.09201908,
            0.37345418
        ])
        np.testing.assert_equal(out.shape, [1, 8, 1, 1])
        np.testing.assert_almost_equal(
            out.numpy(), gt.reshape(1, 8, 1, 1), decimal=5)

    def testHelp(self, ):
        docs1 = hub.help(
            self.local_repo,
            model='MM',
            source='local', )

        docs2 = hub.help(
            self.github_repo, model='MM', source='github', force_reload=False)

        assert docs1 == docs2 == 'This is a test demo for paddle hub\n    ', ''

    def testList(self, ):
        models1 = hub.list(
            self.local_repo,
            source='local',
            force_reload=False, )

        models2 = hub.list(
            self.github_repo,
            source='github',
            force_reload=False, )

        assert models1 == models2 == ['MM'], ''

    def testExcept(self, ):
        with self.assertRaises(ValueError):
            _ = hub.help(
                self.github_repo,
                model='MM',
                source='github-test',
                force_reload=False)

        with self.assertRaises(ValueError):
            _ = hub.load(
                self.github_repo,
                model='MM',
                source='github-test',
                force_reload=False)

        with self.assertRaises(ValueError):
            _ = hub.list(
                self.github_repo, source='github-test', force_reload=False)

        with self.assertRaises(ValueError):
            _ = hub.load(
                self.local_repo, model=123, source='local', force_reload=False)

        with self.assertRaises(RuntimeError):
            _ = hub.load(
                self.local_repo,
                model='123',
                source='local',
                force_reload=False)

    def testGithubclone(self, ):
        repo = 'lyuwenyu/paddlehub_demo:main'
        source = 'github'

        _ = hub.list(repo, source=source, force_reload=True, use_git=True)

        _ = hub.load(repo, 'MM', source=source, force_reload=True, use_git=True)

        _ = hub.help(
            repo, 'MM', source=source, force_reload=False, use_git=True)

        hub._git_archive_link('lyuwenyuL', 'paddlehub_test', 'master', 'gitee')
        hub._git_archive_link('lyuwenyu', 'paddlehub_demo', 'main', 'github')


if __name__ == '__main__':
    unittest.main()
