#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
import math
import sys
from functools import partial

PASS_NUM = 100
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32

word_dict = paddle.dataset.imikolov.build_dict()
dict_size = len(word_dict)


def inference_program(is_sparse):
    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')

    embed_first = fluid.layers.embedding(
        input=first_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_second = fluid.layers.embedding(
        input=second_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_third = fluid.layers.embedding(
        input=third_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')
    embed_forth = fluid.layers.embedding(
        input=forth_word,
        size=[dict_size, EMBED_SIZE],
        dtype='float32',
        is_sparse=is_sparse,
        param_attr='shared_w')

    concat_embed = fluid.layers.concat(
        input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
    hidden1 = fluid.layers.fc(input=concat_embed,
                              size=HIDDEN_SIZE,
                              act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden1, size=dict_size, act='softmax')
    return predict_word


def train_program(is_sparse):
    # The declaration of 'next_word' must be after the invoking of inference_program,
    # or the data input order of train program would be [next_word, firstw, secondw,
    # thirdw, forthw], which is not correct.
    predict_word = inference_program(is_sparse)
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.SGD(learning_rate=0.001)


def train(use_cuda, train_program, params_dirname):
    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            outs = trainer.test(
                reader=test_reader,
                feed_order=['firstw', 'secondw', 'thirdw', 'forthw', 'nextw'])
            avg_cost = outs[0]
            print(("loss= ", avg_cost))

            if avg_cost < 10.0:
                trainer.save_params(params_dirname)
                trainer.stop()

            if math.isnan(avg_cost):
                sys.exit("got NaN loss, training failed.")

    trainer = fluid.Trainer(
        train_func=train_program, optimizer_func=optimizer_func, place=place)

    trainer.train(
        reader=train_reader,
        num_epochs=1,
        event_handler=event_handler,
        feed_order=['firstw', 'secondw', 'thirdw', 'forthw', 'nextw'])


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    # Setup inputs by creating 4 LoDTensors representing 4 words. Here each word 
    # is simply an index to look up for the corresponding word vector and hence 
    # the shape of word (base_shape) should be [1]. The recursive_sequence_lengths, 
    # which is length-based level of detail (lod) of each LoDTensor, should be [[1]] 
    # meaning there is only one level of detail and there is only one sequence of 
    # one word on this level.
    # Note that recursive_sequence_lengths should be a list of lists.
    recursive_seq_lens = [[1]]
    base_shape = [1]
    # The range of random integers is [low, high]
    first_word = fluid.create_random_int_lodtensor(
        recursive_seq_lens, base_shape, place, low=0, high=dict_size - 1)
    second_word = fluid.create_random_int_lodtensor(
        recursive_seq_lens, base_shape, place, low=0, high=dict_size - 1)
    third_word = fluid.create_random_int_lodtensor(
        recursive_seq_lens, base_shape, place, low=0, high=dict_size - 1)
    fourth_word = fluid.create_random_int_lodtensor(
        recursive_seq_lens, base_shape, place, low=0, high=dict_size - 1)

    result = inferencer.infer(
        {
            'firstw': first_word,
            'secondw': second_word,
            'thirdw': third_word,
            'forthw': fourth_word
        },
        return_numpy=False)
    print((np.array(result[0])))


def main(use_cuda, is_sparse):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    params_dirname = "word2vec.inference.model"

    train(
        use_cuda=use_cuda,
        train_program=partial(train_program, is_sparse),
        params_dirname=params_dirname)

    infer(
        use_cuda=use_cuda,
        inference_program=partial(inference_program, is_sparse),
        params_dirname=params_dirname)


if __name__ == '__main__':
    for use_cuda in (False, True):
        for is_sparse in (False, True):
            main(use_cuda=use_cuda, is_sparse=is_sparse)
