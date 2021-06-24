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

from __future__ import print_function

import paddle
import paddle.fluid as fluid
import paddle.static.amp as amp

import contextlib
import numpy
import unittest
import math
import sys
import os

paddle.enable_static()


def train(use_cuda, save_dirname, is_local, use_bf16, pure_bf16):
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    if use_bf16:
        if not pure_bf16:
            with amp.bf16.bf16_guard():
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)
        else:
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            with amp.bf16.bf16_guard():
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)
    else:
        y_predict = fluid.layers.fc(input=x, size=1, act=None)
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)

    if use_bf16:
        sgd_optimizer = amp.bf16.decorate_bf16(
            sgd_optimizer,
            amp_lists=amp.bf16.AutoMixedPrecisionListsBF16(),
            use_bf16_guard=False,
            use_pure_bf16=pure_bf16)
    sgd_optimizer.minimize(
        avg_cost, startup_program=fluid.default_startup_program())

    BATCH_SIZE = 20

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    def train_loop(main_program):
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe.run(fluid.default_startup_program())
        test_prog = main_program.clone(for_test=True)
        if pure_bf16:
            sgd_optimizer.amp_init(
                exe.place, test_program=test_prog, use_bf16_test=True)

        PASS_NUM = 100
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_loss_value, = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost])
                if avg_loss_value[0] < 10.0 or pure_bf16:
                    if save_dirname is not None:
                        paddle.static.save_inference_model(save_dirname, [x],
                                                           [y_predict], exe)
                    return
                if math.isnan(float(avg_loss_value)):
                    sys.exit("got NaN loss, training failed.")
        raise AssertionError("Fit a line cost is too large, {0:2.2}".format(
            avg_loss_value[0]))

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        t = fluid.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def infer(use_cuda, save_dirname=None, use_bf16=False):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use paddle.static.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = paddle.static.load_inference_model(save_dirname, exe)

        # The input's dimension should be 2-D and the second dim is 13
        # The input data should be >= 0
        batch_size = 10

        test_reader = paddle.batch(
            paddle.dataset.uci_housing.test(), batch_size=batch_size)

        test_data = next(test_reader())
        test_feat = numpy.array(
            [data[0] for data in test_data]).astype("float32")
        test_label = numpy.array(
            [data[1] for data in test_data]).astype("float32")

        assert feed_target_names[0] == 'x'
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: numpy.array(test_feat)},
                          fetch_list=fetch_targets)
        print("infer shape: ", results[0].shape)
        print("infer results: ", results[0])
        print("ground truth: ", test_label)


def main(use_cuda, is_local=True, use_bf16=False, pure_bf16=False):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    if use_bf16 and not fluid.core.is_compiled_with_mkldnn():
        return

    # Directory for saving the trained model
    save_dirname = "fit_a_line.inference.model"

    train(use_cuda, save_dirname, is_local, use_bf16, pure_bf16)
    infer(use_cuda, save_dirname, use_bf16)


class TestFitALineBase(unittest.TestCase):
    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


class TestFitALine(TestFitALineBase):
    def test_cpu(self):
        with self.program_scope_guard():
            main(use_cuda=False)

    def test_cuda(self):
        with self.program_scope_guard():
            main(use_cuda=True)


@unittest.skipIf(not fluid.core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestFitALineBF16(TestFitALineBase):
    def test_bf16(self):
        with self.program_scope_guard():
            main(use_cuda=False, use_bf16=True)

    def test_pure_bf16(self):
        with self.program_scope_guard():
            main(use_cuda=False, use_bf16=True, pure_bf16=True)


if __name__ == '__main__':
    unittest.main()
