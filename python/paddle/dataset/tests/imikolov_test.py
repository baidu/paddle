#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.dataset.imikolov
import unittest

WORD_DICT = paddle.dataset.imikolov.build_dict()


class TestMikolov(unittest.TestCase):
    def check_reader(self, reader, n):
        for l in reader():
            self.assertEqual(len(l), n)

    def test_train(self):
        n = 5
        self.check_reader(paddle.dataset.imikolov.train(WORD_DICT, n), n)

        first_line = 'aer banknote berlitz calloway centrust cluett fromstein '\
            'gitano guterman hydro-quebec ipo kia memotec mlx nahb punts '\
            'rake regatta rubens sim snack-food ssangyong swapo wachter'
        first_line = [
            WORD_DICT.get(ch, WORD_DICT['<unk>'])
            for ch in first_line.split(' ')
        ]
        for l in paddle.dataset.imikolov.train(
                WORD_DICT, n=-1,
                data_type=paddle.dataset.imikolov.DataType.SEQ)():
            read_line = l[0][1:]
            break
        self.assertEqual(first_line, read_line)

    def test_test(self):
        n = 5
        self.check_reader(paddle.dataset.imikolov.test(WORD_DICT, n), n)

        first_line = 'consumers may want to move their telephones a little '\
                'closer to the tv set'
        first_line = [
            WORD_DICT.get(ch, WORD_DICT['<unk>'])
            for ch in first_line.split(' ')
        ]
        for l in paddle.dataset.imikolov.test(
                WORD_DICT, n=-1,
                data_type=paddle.dataset.imikolov.DataType.SEQ)():
            read_line = l[0][1:]
            break
        self.assertEqual(first_line, read_line)

    def test_total(self):
        _, idx = list(zip(*list(WORD_DICT.items())))
        self.assertEqual(sorted(idx)[-1], len(WORD_DICT) - 1)


if __name__ == '__main__':
    unittest.main()
