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

import unittest

from paddle.utils.download import get_weights_path_from_url
from paddle.utils.download import get_path_from_url
from paddle.utils.download import git_clone_from_url


class TestDownload(unittest.TestCase):
    def download(self, url, md5sum):
        get_weights_path_from_url(url, md5sum)

    def test_download_model(self):
        url = 'https://paddle-hapi.bj.bcebos.com/unittest/single_file.pdparams'
        md5sum = 'd41d8cd98f00b204e9800998ecf8427e'
        self.download(url, md5sum)

    def test_exist_download(self):
        url = 'https://paddle-hapi.bj.bcebos.com/unittest/single_file.pdparams'
        md5sum = 'd41d8cd98f00b204e9800998ecf8427e'
        self.download(url, md5sum)

    def test_download_without_md5sum(self):
        url = 'https://paddle-hapi.bj.bcebos.com/unittest/single_file.pdparams'
        self.download(url, None)

    def test_download_errors(self):
        with self.assertRaises(RuntimeError):
            url = 'https://paddle-hapi.bj.bcebos.com/unittest/single_file.pdparams'
            md5sum = '8ff74f291f72533f2a7956a4eftttttt'
            self.download(url, md5sum)

        with self.assertRaises(RuntimeError):
            url = 'https://paddle-hapi.bj.bcebos.com/models/mobilenet_v2_x1.0t.pdparams'
            self.download(url, None)

    def test_download_and_uncompress(self):
        urls = [
            "https://paddle-hapi.bj.bcebos.com/unittest/files.tar",
            "https://paddle-hapi.bj.bcebos.com/unittest/files.zip",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_dir.tar",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_dir.zip",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_file.tar",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_file.zip",
        ]
        for url in urls:
            self.download(url, None)

    def test_get_path_from_url(self):
        urls = [
            "https://paddle-hapi.bj.bcebos.com/unittest/files.tar",
            "https://paddle-hapi.bj.bcebos.com/unittest/files.zip",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_dir.tar",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_dir.zip",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_file.tar",
            "https://paddle-hapi.bj.bcebos.com/unittest/single_file.zip",
        ]
        for url in urls:
            get_path_from_url(url, root_dir='./test')


class TestWget(unittest.TestCase):
    def setUp(self, ):
        self.giturl = 'https://github.com/lyuwenyu/paddlehub_demo/archive/main.zip'

    def test_wget(self, ):
        import sys
        if sys.platform == 'linux':
            get_path_from_url(
                self.giturl,
                './test',
                check_exist=False,
                decompress=False,
                use_wget=True, )


class TestGitclone(unittest.TestCase):
    def test_git_clone_from_url(self, ):
        giturl = 'https://github.com/lyuwenyu/paddlehub_demo.git'
        branch = 'main'
        repo_dir = './test/lyuwenyu_paddlehub_demo_main'

        git_clone_from_url(giturl, repo_dir, branch=None, check_exist=False)
        git_clone_from_url(giturl, repo_dir, branch=branch, check_exist=True)
        git_clone_from_url(
            giturl, repo_dir + 'x', branch=branch, check_exist=False)

    def test_errors(self, ):
        with self.assertRaises(RuntimeError):
            git_clone_from_url('xx', 'xx', branch=None, check_exist=False)


if __name__ == '__main__':
    unittest.main()
