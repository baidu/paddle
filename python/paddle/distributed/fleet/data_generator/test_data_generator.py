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
import paddle.distributed.fleet as fleet


class SyntheticData(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(10000):
                yield ("words", [1, 2, 3, 4]), ("label", [0])

        return data_iter


class SyntheticStringData(fleet.MultiSlotStringDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            for i in range(10000):
                yield [("words", ["1", "2", "3", "4"]), ("label", ["0"])]

        return data_iter


sd = SyntheticData()
sd.run_from_memory()

sd2 = SyntheticStringData()
sd2.run_from_memory()
