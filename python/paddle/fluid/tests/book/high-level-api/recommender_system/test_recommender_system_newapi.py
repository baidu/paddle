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

import math
import sys
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.nets as nets

IS_SPARSE = True
USE_GPU = False
BATCH_SIZE = 256


def get_usr_combined_features():
    # FIXME(dzh) : old API integer_value(10) may have range check.
    # currently we don't have user configurated check.

    USR_DICT_SIZE = paddle.dataset.movielens.max_user_id() + 1

    uid = layers.data(name='user_id', shape=[1], dtype='int64')

    usr_emb = layers.embedding(
        input=uid,
        dtype='float32',
        size=[USR_DICT_SIZE, 32],
        param_attr='user_table',
        is_sparse=IS_SPARSE)

    usr_fc = layers.fc(input=usr_emb, size=32)

    USR_GENDER_DICT_SIZE = 2

    usr_gender_id = layers.data(name='gender_id', shape=[1], dtype='int64')

    usr_gender_emb = layers.embedding(
        input=usr_gender_id,
        size=[USR_GENDER_DICT_SIZE, 16],
        param_attr='gender_table',
        is_sparse=IS_SPARSE)

    usr_gender_fc = layers.fc(input=usr_gender_emb, size=16)

    USR_AGE_DICT_SIZE = len(paddle.dataset.movielens.age_table)
    usr_age_id = layers.data(name='age_id', shape=[1], dtype="int64")

    usr_age_emb = layers.embedding(
        input=usr_age_id,
        size=[USR_AGE_DICT_SIZE, 16],
        is_sparse=IS_SPARSE,
        param_attr='age_table')

    usr_age_fc = layers.fc(input=usr_age_emb, size=16)

    USR_JOB_DICT_SIZE = paddle.dataset.movielens.max_job_id() + 1
    usr_job_id = layers.data(name='job_id', shape=[1], dtype="int64")

    usr_job_emb = layers.embedding(
        input=usr_job_id,
        size=[USR_JOB_DICT_SIZE, 16],
        param_attr='job_table',
        is_sparse=IS_SPARSE)

    usr_job_fc = layers.fc(input=usr_job_emb, size=16)

    concat_embed = layers.concat(
        input=[usr_fc, usr_gender_fc, usr_age_fc, usr_job_fc], axis=1)

    usr_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

    return usr_combined_features


def get_mov_combined_features():

    MOV_DICT_SIZE = paddle.dataset.movielens.max_movie_id() + 1

    mov_id = layers.data(name='movie_id', shape=[1], dtype='int64')

    mov_emb = layers.embedding(
        input=mov_id,
        dtype='float32',
        size=[MOV_DICT_SIZE, 32],
        param_attr='movie_table',
        is_sparse=IS_SPARSE)

    mov_fc = layers.fc(input=mov_emb, size=32)

    CATEGORY_DICT_SIZE = len(paddle.dataset.movielens.movie_categories())

    category_id = layers.data(
        name='category_id', shape=[1], dtype='int64', lod_level=1)

    mov_categories_emb = layers.embedding(
        input=category_id, size=[CATEGORY_DICT_SIZE, 32], is_sparse=IS_SPARSE)

    mov_categories_hidden = layers.sequence_pool(
        input=mov_categories_emb, pool_type="sum")

    MOV_TITLE_DICT_SIZE = len(paddle.dataset.movielens.get_movie_title_dict())

    mov_title_id = layers.data(
        name='movie_title', shape=[1], dtype='int64', lod_level=1)

    mov_title_emb = layers.embedding(
        input=mov_title_id, size=[MOV_TITLE_DICT_SIZE, 32], is_sparse=IS_SPARSE)

    mov_title_conv = nets.sequence_conv_pool(
        input=mov_title_emb,
        num_filters=32,
        filter_size=3,
        act="tanh",
        pool_type="sum")

    concat_embed = layers.concat(
        input=[mov_fc, mov_categories_hidden, mov_title_conv], axis=1)

    # FIXME(dzh) : need tanh operator
    mov_combined_features = layers.fc(input=concat_embed, size=200, act="tanh")

    return mov_combined_features


def inference_program():
    usr_combined_features = get_usr_combined_features()
    mov_combined_features = get_mov_combined_features()

    inference = layers.cos_sim(X=usr_combined_features, Y=mov_combined_features)
    scale_infer = layers.scale(x=inference, scale=5.0)

    return scale_infer


def train_program():

    scale_infer = inference_program()

    label = layers.data(name='score', shape=[1], dtype='float32')
    square_cost = layers.square_error_cost(input=scale_infer, label=label)
    avg_cost = layers.mean(square_cost)

    return [avg_cost, scale_infer]


def optimizer_func():
    return fluid.optimizer.SGD(learning_rate=0.2)


def train(use_cuda, train_program, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_func)

    feed_order = [
        'user_id', 'gender_id', 'age_id', 'job_id', 'movie_id', 'category_id',
        'movie_title', 'score'
    ]

    def event_handler(event):
        if isinstance(event, fluid.EndStepEvent):
            test_reader = paddle.batch(
                paddle.dataset.movielens.test(), batch_size=BATCH_SIZE)
            avg_cost_set = trainer.test(
                reader=test_reader, feed_order=feed_order)

            # get avg cost
            avg_cost = np.array(avg_cost_set).mean()

            print(("avg_cost: %s" % avg_cost))

            if float(avg_cost) < 4:  # Smaller value to increase CI speed
                trainer.save_params(params_dirname)
                trainer.stop()
            else:
                print(
                    ('BatchID {0}, Test Loss {1:0.2}'.format(event.epoch + 1,
                                                             float(avg_cost))))
                if math.isnan(float(avg_cost)):
                    sys.exit("got NaN loss, training failed.")

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.movielens.train(), buf_size=8192),
        batch_size=BATCH_SIZE)

    trainer.train(
        num_epochs=1,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=feed_order)


def infer(use_cuda, inference_program, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    inferencer = fluid.Inferencer(
        inference_program, param_path=params_dirname, place=place)

    # Use the first data from paddle.dataset.movielens.test() as input.
    # Use create_lod_tensor(data, recursive_sequence_lengths, place) API 
    # to generate LoD Tensor where `data` is a list of sequences of index 
    # numbers, `recursive_sequence_lengths` is the length-based level of detail 
    # (lod) info associated with `data`.
    # For example, data = [[10, 2, 3], [2, 3]] means that it contains
    # two sequences of indexes, of length 3 and 2, respectively.
    # Correspondingly, recursive_sequence_lengths = [[3, 2]] contains one 
    # level of detail info, indicating that `data` consists of two sequences 
    # of length 3 and 2, respectively. 
    user_id = fluid.create_lod_tensor([[1]], [[1]], place)
    gender_id = fluid.create_lod_tensor([[1]], [[1]], place)
    age_id = fluid.create_lod_tensor([[0]], [[1]], place)
    job_id = fluid.create_lod_tensor([[10]], [[1]], place)
    movie_id = fluid.create_lod_tensor([[783]], [[1]], place)
    category_id = fluid.create_lod_tensor([[10, 8, 9]], [[3]], place)
    movie_title = fluid.create_lod_tensor([[1069, 4140, 2923, 710, 988]], [[5]],
                                          place)

    results = inferencer.infer(
        {
            'user_id': user_id,
            'gender_id': gender_id,
            'age_id': age_id,
            'job_id': job_id,
            'movie_id': movie_id,
            'category_id': category_id,
            'movie_title': movie_title
        },
        return_numpy=False)

    print(("infer results: ", np.array(results[0])))


def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return
    params_dirname = "recommender_system.inference.model"
    train(
        use_cuda=use_cuda,
        train_program=train_program,
        params_dirname=params_dirname)
    infer(
        use_cuda=use_cuda,
        inference_program=inference_program,
        params_dirname=params_dirname)


if __name__ == '__main__':
    main(USE_GPU)
