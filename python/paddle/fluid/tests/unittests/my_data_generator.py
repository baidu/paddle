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

import sys
import os
import paddle
import re
import collections
import time
import paddle.distributed.fleet as fleet


class MyDataset(fleet.MultiSlotDataGenerator):
    def generate_sample(self, line):
        def data_iter():
            elements = line.strip().split()[0:]
            output = [("show", [int(elements[0])]),
                      ("click", [int(elements[1])]),
                      ("slot1", [int(elements[2])])]
            yield output

        return data_iter


if __name__ == "__main__":
    d = MyDataset()
    d.run_from_stdin()
