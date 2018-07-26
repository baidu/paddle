# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import paddle.dataset.common
import unittest
import tempfile
import glob


class TestCommon(unittest.TestCase):
    def test_md5file(self):
        _, temp_path = tempfile.mkstemp()
        with open(temp_path, 'w') as f:
            f.write("Hello\n")
        self.assertEqual('09f7e02f1290be211da707a266f153b3',
                         paddle.dataset.common.md5file(temp_path))

    def test_download(self):
        yi_avatar = 'https://avatars0.githubusercontent.com/u/1548775?v=3&s=460'
        self.assertEqual(
            paddle.dataset.common.DATA_HOME + '/test/1548775?v=3&s=460',
            paddle.dataset.common.download(yi_avatar, 'test',
                                           'f75287202d6622414c706c36c16f8e0d'))

    def test_split(self):
        def test_reader():
            def reader():
                for x in range(10):
                    yield x

            return reader

        _, temp_path = tempfile.mkstemp()
        paddle.dataset.common.split(
            test_reader(), 4, suffix=temp_path + '/test-%05d.pickle')
        files = glob.glob(temp_path + '/test-%05d.pickle')
        self.assertEqual(len(files), 3)

    def test_cluster_file_reader(self):
        _, temp_path = tempfile.mkstemp()
        for x in range(5):
            with open(temp_path + '/%05d.test' % x) as f:
                f.write('%d\n' % x)
        reader = paddle.dataset.common.cluster_files_reader(
            temp_path + '/*.test', 5, 0)
        for idx, e in enumerate(reader()):
            self.assertEqual(e, str("0"))

    def test_convert(self):
        record_num = 10
        num_shards = 4

        def test_reader():
            def reader():
                for x in range(record_num):
                    yield x

            return reader

        path = tempfile.mkdtemp()
        paddle.dataset.common.convert(path,
                                      test_reader(), num_shards,
                                      'random_images')

        files = glob.glob(path + '/random_images-*')
        self.assertEqual(len(files), num_shards)

        recs = []
        for i in range(0, num_shards):
            n = "%s/random_images-%05d-of-%05d" % (path, i, num_shards - 1)
            r = recordio.reader(n)
            while True:
                d = r.read()
                if d is None:
                    break
                recs.append(d)

        recs.sort()
        self.assertEqual(total, record_num)


if __name__ == '__main__':
    unittest.main()
