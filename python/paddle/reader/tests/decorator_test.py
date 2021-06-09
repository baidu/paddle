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

import sys
import time
import unittest
import functools

import paddle.reader

__all__ = []


def reader_creator_10(dur):
    def reader():
        for i in range(10):
            # this invocation helps testing paddle.reader.buffer
            time.sleep(dur)
            yield i

    return reader


class TestMap(unittest.TestCase):
    def test_map(self):
        d = {"h": 0, "i": 1}

        def tokenize(x):
            return d[x]

        def read():
            yield "h"
            yield "i"

        r = paddle.reader.map_readers(tokenize, read)
        for i, e in enumerate(r()):
            self.assertEqual(e, i)


class TestShuffle(unittest.TestCase):
    def test_shuffle(self):
        case = [(0, True), (1, True), (10, False), (100, False)]
        a = reader_creator_10(0)
        for size, checkEq in case:
            s = paddle.reader.shuffle(a, size)
            total = 0
            for idx, e in enumerate(s()):
                if checkEq:
                    self.assertEqual(idx, e)
                total += 1
            self.assertEqual(total, 10)


class TestXmap(unittest.TestCase):
    def test_xmap(self):
        def mapper(x):
            return (x + 1)

        orders = (True, False)
        thread_nums = (1, 2, 4, 8, 16)
        buffered_size = (1, 2, 4, 8, 16)
        for order in orders:
            for tNum in thread_nums:
                for size in buffered_size:
                    reader = paddle.reader.xmap_readers(mapper,
                                                        reader_creator_10(0),
                                                        tNum, size, order)
                    for n in range(3):
                        result = []
                        for i in reader():
                            result.append(i)
                        if not order:
                            result.sort()
                        for idx, e in enumerate(result):
                            self.assertEqual(e, mapper(idx))


if __name__ == '__main__':
    unittest.main()
