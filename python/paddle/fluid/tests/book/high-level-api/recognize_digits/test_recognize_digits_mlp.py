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

import argparse
import paddle.fluid as fluid
import paddle
import sys
import numpy
import unittest
import math
import sys
import os

BATCH_SIZE = 64


def inference_program():
    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')

    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    return prediction


def train_program():
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    predict = inference_program()
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, acc]


def optimizer_func():
    return fluid.optimizer.Adam(learning_rate=0.001)


def train(use_cuda, train_program, params_dirname):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    trainer = fluid.Trainer(
        train_func=train_program, place=place, optimizer_func=optimizer_func)

    def event_handler(event):
        if isinstance(event, fluid.EndEpochEvent):
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
            avg_cost, acc = trainer.test(
                reader=test_reader, feed_order=['img', 'label'])

            print(("avg_cost: %s" % avg_cost))
            print(("acc     : %s" % acc))

            if acc > 0.2:  # Smaller value to increase CI speed
                trainer.save_params(params_dirname)
            else:
                print(('BatchID {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(
                    event.epoch + 1, avg_cost, acc)))
                if math.isnan(avg_cost):
                    sys.exit("got NaN loss, training failed.")

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    trainer.train(
        num_epochs=1,
        event_handler=event_handler,
        reader=train_reader,
        feed_order=['img', 'label'])


def infer(use_cuda, inference_program, params_dirname=None):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    inferencer = fluid.Inferencer(
        infer_func=inference_program, param_path=params_dirname, place=place)

    batch_size = 1
    tensor_img = numpy.random.uniform(-1.0, 1.0,
                                      [batch_size, 1, 28, 28]).astype("float32")

    results = inferencer.infer({'img': tensor_img})

    print(("infer results: ", results[0]))


def main(use_cuda):
    params_dirname = "recognize_digits_mlp.inference.model"

    # call train() with is_local argument to run distributed train
    train(
        use_cuda=use_cuda,
        train_program=train_program,
        params_dirname=params_dirname)
    infer(
        use_cuda=use_cuda,
        inference_program=inference_program,
        params_dirname=params_dirname)


if __name__ == '__main__':
    # for use_cuda in (False, True):
    main(use_cuda=False)
