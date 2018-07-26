# /usr/bin/env python
# -*- coding:utf-8 -*-

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

import unittest
import nltk
import paddle.dataset.sentiment as st
from nltk.corpus import movie_reviews


class TestSentimentMethods(unittest.TestCase):
    def test_get_word_dict(self):
        word_dict = st.get_word_dict()[0:10]
        test_word_list = [(',', 0), ('the', 1), ('.', 2), ('a', 3), ('and', 4),
                          ('of', 5), ('to', 6), ("'", 7), ('is', 8), ('in', 9)]
        for idx, each in enumerate(word_dict):
            self.assertEqual(each, test_word_list[idx])
        self.assertTrue("/root/.cache/paddle/dataset" in nltk.data.path)

    def test_sort_files(self):
        last_label = ''
        for sample_file in st.sort_files():
            current_label = sample_file.split("/")[0]
            self.assertNotEqual(current_label, last_label)
            last_label = current_label

    def test_data_set(self):
        data_set = st.load_sentiment_data()
        last_label = -1
        for each in st.test():
            self.assertNotEqual(each[1], last_label)
            last_label = each[1]
        self.assertEqual(len(data_set), st.NUM_TOTAL_INSTANCES)
        self.assertEqual(len(list(st.train())), st.NUM_TRAINING_INSTANCES)
        self.assertEqual(
            len(list(st.test())),
            (st.NUM_TOTAL_INSTANCES - st.NUM_TRAINING_INSTANCES))


if __name__ == '__main__':
    unittest.main()
