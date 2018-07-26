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

import time
import unittest

import paddle.reader


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


class TestBuffered(unittest.TestCase):
    def test_read(self):
        for size in range(20):
            b = paddle.reader.buffered(reader_creator_10(0), size)
            c = 0
            for i in b():
                self.assertEqual(i, c)
                c += 1
            self.assertEqual(c, 10)

    def test_buffering(self):
        # read have 30ms delay.
        b = paddle.reader.buffered(reader_creator_10(0.03), 10)
        last_time = time.time()
        for idx, i in enumerate(b()):
            elapsed_time = time.time() - last_time
            if i == 0:
                time.sleep(0.3)
            else:
                # read time should be short, meaning already buffered.
                self.assertLess(elapsed_time, 0.05)
            last_time = time.time()


class TestCompose(unittest.TestCase):
    def test_compse(self):
        reader = paddle.reader.compose(
            reader_creator_10(0), reader_creator_10(0))
        for idx, e in enumerate(reader()):
            self.assertEqual(e, (idx, idx))

    def test_compose_not_aligned(self):
        total = 0
        reader = paddle.reader.compose(
            paddle.reader.chain(reader_creator_10(0), reader_creator_10(0)),
            reader_creator_10(0))
        with self.assertRaises(paddle.reader.ComposeNotAligned):
            for e in reader():
                total += 1
        # expecting 10, not 20
        self.assertEqual(total, 10)

    def test_compose_not_aligned_no_check(self):
        total = 0
        reader = paddle.reader.compose(
            paddle.reader.chain(reader_creator_10(0), reader_creator_10(0)),
            reader_creator_10(0),
            check_alignment=False)
        for e in reader():
            total += 1
        # expecting 10, not 20
        self.assertEqual(total, 10)


class TestChain(unittest.TestCase):
    def test_chain(self):
        c = paddle.reader.chain(reader_creator_10(0), reader_creator_10(0))
        idx = 0
        for e in c():
            self.assertEqual(e, idx % 10)
            idx += 1
        self.assertEqual(idx, 20)


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


class TestPipeReader(unittest.TestCase):
    def test_pipe_reader(self):
        def example_reader(myfiles):
            for f in myfiles:
                pr = paddle.reader.PipeReader("cat %s" % f, bufsize=128)
                for l in pr.get_line():
                    yield l

        import tempfile

        records = [str(i) for i in range(5)]
        temp = tempfile.NamedTemporaryFile()
        try:
            with open(temp.name, 'w') as f:
                for r in records:
                    f.write('%s\n' % r)

            result = []
            for r in example_reader([temp.name]):
                result.append(r)

            for idx, e in enumerate(records):
                self.assertEqual(e, result[idx])
        finally:
            # delete the temporary file
            temp.close()


if __name__ == '__main__':
    unittest.main()
