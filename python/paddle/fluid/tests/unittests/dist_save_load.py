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

import os
import sys
import signal
import subprocess
import argparse
import time
import math
import random
from multiprocessing import Process
from functools import reduce

import numpy as np
import pickle
import unittest
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import io

from test_dist_base import TestDistRunnerBase, runtime_main, RUN_STEP
from dist_simnet_bow import TestDistSimnetBow2x2, DATA_URL, DATA_MD5


class TestDistSaveLoad2x2(TestDistSimnetBow2x2):
    def _load_persistable_vars(self, executor, dirname, program):
        def _is_checkpoint_var(var):
            """
            the checkpoint will not save or load all the variables.
            var type is FEED_MINIBATCH/FETCH_LIST/RAW or var name ends with @GRAD are discarded.

            : param var(Variable)
            """
            if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
                    var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
                    var.desc.type() == core.VarDesc.VarType.RAW:
                return False
            # @GRAD are named for gradient variables, checkpoint will not save it.
            if "@GRAD" in var.name:
                return False
            # .trainer_ are named for distribute train variables, checkpoint will not save it.
            if ".trainer_" in var.name:
                return False

            # .block is named for distribute train variables, checkpoint will not save it.
            if ".block" in var.name:
                return False

            if "tmp_" in var.name:
                return False

            return var.persistable

        io.load_vars(
            executor,
            dirname=dirname,
            main_program=program,
            predicate=_is_checkpoint_var,
            filename=None)

    def run_pserver(self, args):
        self.get_model(batch_size=2)
        # NOTE: pserver should not call memory optimize
        t = self.get_transpiler(args.trainer_id,
                                fluid.default_main_program(), args.endpoints,
                                args.trainers, args.sync_mode, False,
                                args.current_endpoint)
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        startup_prog = t.get_startup_program(args.current_endpoint,
                                             pserver_prog)

        need_load = bool(int(os.getenv("LOAD", "0")))
        model_dir = os.getenv("MODEL_DIR", "")

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)

        if need_load and model_dir:
            fluid.io.load_persistables(exe, model_dir, pserver_prog)

        exe.run(pserver_prog)

    def run_trainer(self, args):
        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
            self.get_model(batch_size=2)

        if args.update_method == "pserver":
            t = self.get_transpiler(args.trainer_id,
                                    fluid.default_main_program(),
                                    args.endpoints, args.trainers,
                                    args.sync_mode)

            trainer_prog = t.get_trainer_program()
        else:
            trainer_prog = fluid.default_main_program()

        if args.use_cuda:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()

        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())

        strategy = fluid.ExecutionStrategy()
        strategy.num_threads = 1

        build_stra = fluid.BuildStrategy()

        if args.use_reduce:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        else:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce

        exe = fluid.ParallelExecutor(
            args.use_cuda,
            loss_name=avg_cost.name,
            exec_strategy=strategy,
            build_strategy=build_stra)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method == "pserver" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        need_save = bool(int(os.getenv("SAVE", "0")))
        model_dir = os.getenv("MODEL_DIR", "")
        save_mode = os.getenv("SAVE_MODE", "")

        if save_mode == "LOCAL":
            if need_save:
                for _ in six.moves.xrange(RUN_STEP):
                    loss, = exe.run(fetch_list=[avg_cost.name],
                                    feed=feeder.feed(get_data()))
                if need_save and model_dir:
                    io.save_persistables(startup_exe, model_dir, trainer_prog)

            var = np.array(fluid.global_scope().find_var('__fc_b__').get_tensor(
            ))
            sys.stdout.buffer.write(pickle.dumps(np.ravel(var).tolist()))

        elif save_mode == "DIST":
            skip_steps = int(os.getenv("SKIP_STEPS"))
            loss = None
            if need_save:
                for idx in six.moves.xrange(8):
                    loss, = exe.run(fetch_list=[avg_cost.name],
                                    feed=feeder.feed(get_data()))
                    if need_save and model_dir and idx == skip_steps and args.trainer_id == 0:
                        io.save_persistables(startup_exe, model_dir,
                                             trainer_prog)
            else:
                for idx in six.moves.xrange(8):
                    data = get_data()
                    if idx <= skip_steps:
                        continue
                    loss, = exe.run(fetch_list=[avg_cost.name],
                                    feed=feeder.feed(data))
            sys.stdout.buffer.write(pickle.dumps(loss.tolist()))
        else:
            raise Exception("save_mode must be LOCAL or DIST")


if __name__ == "__main__":
    paddle.dataset.common.download(DATA_URL, 'simnet', DATA_MD5, "train")
    runtime_main(TestDistSaveLoad2x2)
