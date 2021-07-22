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
# limitations under the License.

# from paddle.inbubate.hapi.text import Stack, Pad, Tuple, BertTokenizer
# from paddle.inbubate.hapi.text import BertForSequenceClassification

from paddle.incubate.hapi.text.data_utils import Stack, Pad, Tuple
import unittest


class TestBatchify(unittest.TestCase):
    def test_stack(self):
        a = [1, 2, 3, 4]
        b = [5, 6, 7, 8]
        c = [9, 1, 2, 3]
        data = [a, b, c]
        stack_data = Stack()(data)
        assert stack_data.shape == (3, 4)
        assert stack_data[1][3] == 8 and stack_data[2][1] == 1

    def test_pad(self):
        a = [1, 2, 3, 4]
        d = [1, 2]
        e = [3, 5, 6]
        pad_data = Pad(pad_val=0)([a, d, e])
        assert pad_data.shape == (3, 4)

    def test_tuple(self):
        tuple_fn = Tuple(Pad(), Stack())
        a = ([1, 2, 3, 4], 3)
        b = ([2, 3], 2)
        tuple_data = tuple_fn([a, b])
        assert tuple_data[0].shape == (2, 4) and tuple_data[1].shape == (2, )


if __name__ == '__main__':
    unittest.main()
