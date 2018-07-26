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
import paddle.fluid as fluid
import paddle.v2 as paddle


def load_vocab(filename):
    """
    load vocabulary
    """
    vocab = {}
    with open(filename) as f:
        wid = 0
        for line in f:
            vocab[line.strip()] = wid
            wid += 1
    return vocab


# load word dict with paddle inner function
if len(sys.argv) == 1:
    word_dict = paddle.dataset.imdb.word_dict()
else:
    word_dict = load_vocab(sys.argv[1])
    word_dict["<unk>"] = len(word_dict)
print("Dict dim = ", len(word_dict))

# input text data
data = fluid.layers.data(name="words", shape=[1], dtype="int64", lod_level=1)

# label data
label = fluid.layers.data(name="label", shape=[1], dtype="int64")
# like placeholder
feeder = fluid.DataFeeder(feed_list=[data, label], place=fluid.CPUPlace())

# train data set
BATCH_SIZE = 128
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.imdb.train(word_dict), buf_size=25000),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)

fluid.recordio_writer.convert_reader_to_recordio_file(
    "train.recordio", feeder=feeder, reader_creator=train_reader)
fluid.recordio_writer.convert_reader_to_recordio_file(
    "test.recordio", feeder=feeder, reader_creator=test_reader)
