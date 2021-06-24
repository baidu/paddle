# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import six
import unittest
import time
import math
import multiprocessing
import numpy as np

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler

# open eager delete mode
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ['FLAGS_fast_eager_deletion_mode'] = 'true'
os.environ['CPU_NUM'] = '2'


class BuildIrMemOptBase(unittest.TestCase):
    def setup_reader(self):
        self.batch_size = 32
        self.word_dict = paddle.dataset.imdb.word_dict()
        self.train_reader = paddle.batch(
            paddle.dataset.imdb.train(self.word_dict),
            batch_size=self.batch_size)

    def check_network_convergence(self,
                                  network,
                                  use_cuda=True,
                                  use_ir_memory_optimize=True,
                                  enable_inplace=True,
                                  iter=5):
        if use_cuda and not core.is_compiled_with_cuda():
            print('Skip use_cuda=True because Paddle is not compiled with cuda')
            return

        if os.name == 'nt':
            print(
                'Skip use_parallel_executor=True because Paddle comes without parallel support on windows'
            )
            return
        fluid.default_startup_program().random_seed = 100
        fluid.default_main_program().random_seed = 100

        data = fluid.layers.data(
            name="words", shape=[1], dtype="int64", lod_level=1)

        label = fluid.layers.data(name="label", shape=[1], dtype="int64")

        cost = network(data, label, len(self.word_dict))
        optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        optimizer.minimize(cost)
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = enable_inplace
        build_strategy.memory_optimize = use_ir_memory_optimize

        # execution
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
        reader = feeder.decorate_reader(self.train_reader, multi_devices=True)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        train_cp = compiler.CompiledProgram(fluid.default_main_program())
        train_cp = train_cp.with_data_parallel(
            loss_name=cost.name, build_strategy=build_strategy)
        fetch_list = [cost.name]

        begin = time.time()
        first_loss, last_loss = None, None
        step_id = 0
        custom_iter = getattr(self, "iter", None)
        if not custom_iter == None:
            iter = custom_iter
        for data in reader():
            ret = exe.run(train_cp, feed=data, fetch_list=fetch_list)
            print(ret)
            step_id += 1
            if step_id == 1:
                first_loss = ret[0]
            if step_id == iter:
                last_loss = ret[0]
                break
        end = time.time()

        print("%.4f Instance per second" % (
            (self.batch_size * iter) / (end - begin)))

        print(first_loss, last_loss)
        avg_last_loss_val = np.array(last_loss).mean()
        avg_first_loss_val = np.array(first_loss).mean()
        if math.isnan(float(avg_last_loss_val)) or math.isnan(
                float(avg_first_loss_val)):
            sys.exit("got NaN loss, training failed.")

        return first_loss, last_loss


class TestIrMemOptBase(BuildIrMemOptBase):
    def setUp(self):
        self.network = None

    def test_network(self):
        if self.network is None or not core.is_compiled_with_cuda():
            return

        self.setup_reader()

        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(core.Scope()):
                baseline_first_loss, baseline_last_loss = self.check_network_convergence(
                    self.network)

                cur_first_loss, cur_last_loss = self.check_network_convergence(
                    self.network)

                self.assertAlmostEquals(
                    np.mean(baseline_last_loss),
                    np.mean(cur_last_loss),
                    delta=1e-6)
                self.assertAlmostEquals(
                    np.mean(baseline_first_loss),
                    np.mean(cur_first_loss),
                    delta=1e-6)
