#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import unittest
import paddle.distributed.fleet as fleet
import os
import sys
import platform


class MyMultiSlotDataGenerator(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", [1, 2, 3, 4]), ("label", [0])

        return data_iter


class MyMultiSlotStringDataGenerator(fleet.MultiSlotStringDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", ["1", "2", "3", "4"]), ("label", ["0"])

        return data_iter


class MyMultiSlotDataGenerator_error(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield "words"

        return data_iter


class MyMultiSlotDataGenerator_error_2(fleet.MultiSlotStringDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield "words"

        return data_iter


class MyMultiSlotDataGenerator_error_3(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield (1, ["1", "2", "3", "4"]), (2, ["0"])

        return data_iter


class MyMultiSlotDataGenerator_error_4(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", "1"), ("label", "0")

        return data_iter


class MyMultiSlotDataGenerator_error_5(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(40):
                if i == 1:
                    yield None
                yield ("words", []), ("label", [])

        return data_iter


class TestMultiSlotDataGenerator(unittest.TestCase):
    def test_MultiSlotDataGenerator_basic(self):
        my_ms_dg = MyMultiSlotDataGenerator()
        my_ms_dg.set_batch(1)
        my_ms_dg.run_from_memory()


class TestMultiSlotStringDataGenerator(unittest.TestCase):
    def test_MyMultiSlotStringDataGenerator_basic(self):
        my_ms_dg = MyMultiSlotStringDataGenerator()
        my_ms_dg.set_batch(1)
        my_ms_dg.run_from_memory()


class TestMultiSlotStringDataGenerator_2(unittest.TestCase):
    def test_MyMultiSlotStringDataGenerator_stdin(self):
        plats = platform.platform()
        if 'Linux' not in plats:
            print("skip pipecommand UT on MacOS/Win")
            return
        with open("test_queue_dataset_run_a.txt", "w") as f:
            data = "2 1 2\n"
            data += "2 6 2\n"
            data += "2 5 2\n"
            data += "2 7 2\n"
            f.write(data)

        tmp = os.popen(
            "cat test_queue_dataset_run_a.txt | python my_data_generator.py"
        ).readlines()
        expected_res = [
            '1 2 1 1 1 2\n', '1 2 1 6 1 2\n', '1 2 1 5 1 2\n', '1 2 1 7 1 2\n'
        ]
        self.assertEqual(tmp, expected_res)
        os.remove("./test_queue_dataset_run_a.txt")


class TestMultiSlotDataGenerator_error(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_2(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_2()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_3(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_3()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_4(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_4()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


class TestMultiSlotDataGenerator_error_5(unittest.TestCase):
    def test_MultiSlotDataGenerator_error(self):
        with self.assertRaises(ValueError):
            my_ms_dg = MyMultiSlotDataGenerator_error_5()
            my_ms_dg.set_batch(1)
            my_ms_dg.run_from_memory()


if __name__ == '__main__':
    unittest.main()
