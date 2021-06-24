# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
from paddle.reader import multiprocess_reader
import unittest
import numpy as np
import six
import sys


class ReaderException(Exception):
    pass


class TestMultiprocessReaderExceptionWithQueueSuccess(unittest.TestCase):
    def setUp(self):
        self.use_pipe = False
        self.raise_exception = False

    def places(self):
        if fluid.is_compiled_with_cuda():
            return [fluid.CPUPlace(), fluid.CUDAPlace(0)]
        else:
            return [fluid.CPUPlace()]

    def main_impl(self, place, iterable):
        sample_num = 40
        batch_size = 4

        def fake_reader():
            def __impl__():
                for _ in range(sample_num):
                    if not self.raise_exception:
                        yield list(
                            np.random.uniform(
                                low=-1, high=1, size=[10])),
                    else:
                        raise ValueError()

            return __impl__

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            image = fluid.data(name='image', dtype='float32', shape=[None, 10])
            reader = fluid.io.DataLoader.from_generator(
                feed_list=[image], capacity=2, iterable=iterable)

            image_p_1 = image + 1

            decorated_reader = multiprocess_reader(
                [fake_reader(), fake_reader()], use_pipe=self.use_pipe)

            if isinstance(place, fluid.CUDAPlace):
                reader.set_sample_generator(
                    decorated_reader,
                    batch_size=batch_size,
                    places=fluid.cuda_places(0))
            else:
                reader.set_sample_generator(
                    decorated_reader,
                    batch_size=batch_size,
                    places=fluid.cpu_places(1))

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            batch_num = int(sample_num * 2 / batch_size)

            if iterable:
                for _ in range(3):
                    num = 0
                    try:
                        for data in reader():
                            exe.run(feed=data, fetch_list=[image_p_1])
                            num += 1
                        self.assertEqual(num, batch_num)
                    except SystemError as ex:
                        self.assertEqual(num, 0)
                        raise ReaderException()
            else:
                for _ in range(3):
                    num = 0
                    reader.start()
                    try:
                        while True:
                            exe.run(fetch_list=[image_p_1])
                            num += 1
                    except fluid.core.EOFException:
                        reader.reset()
                        self.assertFalse(self.raise_exception)
                        self.assertEqual(num, batch_num)
                    except SystemError as ex:
                        self.assertTrue(self.raise_exception)
                        self.assertEqual(num, 0)
                        raise ReaderException()

    def test_main(self):
        for p in self.places():
            for iterable in [False, True]:
                try:
                    with fluid.scope_guard(fluid.Scope()):
                        self.main_impl(p, iterable)

                    self.assertTrue(not self.raise_exception)
                except ReaderException:
                    self.assertTrue(self.raise_exception)


class TestMultiprocessReaderExceptionWithQueueFailed(
        TestMultiprocessReaderExceptionWithQueueSuccess):
    def setUp(self):
        self.use_pipe = False
        self.raise_exception = True


class TestMultiprocessReaderExceptionWithPipeSuccess(
        TestMultiprocessReaderExceptionWithQueueSuccess):
    def setUp(self):
        self.use_pipe = True
        self.raise_exception = False


class TestMultiprocessReaderExceptionWithPipeFailed(
        TestMultiprocessReaderExceptionWithQueueSuccess):
    def setUp(self):
        self.use_pipe = True
        self.raise_exception = True


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
