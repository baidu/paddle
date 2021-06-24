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
import time

import ast
import unittest
import os
import sys
import signal
import subprocess
import six
import argparse
import pickle
import random
import numpy as np
import time

import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.parallel import DataParallel

from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
import paddle.fluid.incubate.fleet.base.role_maker as role_maker

RUN_STEP = 5
DEFAULT_BATCH_SIZE = 2
DIST_UT_PORT = 0


def print_to_out(out_losses):
    sys.stdout.buffer.write(pickle.dumps(out_losses))


def print_to_err(class_name, log_str):
    localtime = time.asctime(time.localtime(time.time()))
    print_str = localtime + "\t" + class_name + "\t" + log_str
    sys.stderr.buffer.write(pickle.dumps(print_str))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class TestDistRunnerBase(object):
    def get_model(self,
                  batch_size=DEFAULT_BATCH_SIZE,
                  lr=0.1,
                  single_device=False,
                  use_dgc=False):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    @staticmethod
    def get_transpiler(trainer_id,
                       main_program,
                       pserver_endpoints,
                       trainers,
                       sync_mode,
                       dc_asgd=False,
                       current_endpoint=None,
                       nccl_comm_num=1,
                       hogwild_mode=False):
        # NOTE: import fluid until runtime, or else forking processes will cause error.
        config = fluid.DistributeTranspilerConfig()
        config.enable_dc_asgd = dc_asgd
        config.sync_mode = sync_mode
        config.runtime_split_send_recv = hogwild_mode

        if nccl_comm_num > 1:
            config.nccl_comm_num = nccl_comm_num
        # config.runtime_split_send_recv = True
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id=trainer_id,
            program=main_program,
            pservers=pserver_endpoints,
            trainers=trainers,
            sync_mode=sync_mode,
            current_endpoint=current_endpoint)
        return t

    def run_pserver(self, args):
        self.lr = args.lr
        self.get_model(batch_size=args.batch_size)
        # NOTE: pserver should not call memory optimize

        t = self.get_transpiler(
            trainer_id=args.trainer_id,
            main_program=fluid.default_main_program(),
            pserver_endpoints=args.endpoints,
            trainers=args.trainers,
            sync_mode=args.sync_mode,
            dc_asgd=args.dc_asgd,
            hogwild_mode=args.hogwild)
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        startup_prog = t.get_startup_program(args.current_endpoint,
                                             pserver_prog)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        print_to_err(type(self).__name__, "run pserver startup program done.")
        exe.run(pserver_prog)
        print_to_err(type(self).__name__, "run pserver main program done.")

    def run_pipeline_trainer(self, args):
        self.lr = args.lr

        dist_strategy = DistributedStrategy()
        test_program, avg_cost, train_reader, test_reader, batch_acc, predict, data_loader = \
            self.get_model(batch_size=args.batch_size, dist_strategy=dist_strategy)

        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        eprint(type(self).__name__, "device_id: %d." % device_id)
        place = fluid.CUDAPlace(device_id)

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        eprint(type(self).__name__, "run worker startup program done.")

        data_loader.set_sample_list_generator(train_reader, place)
        data_loader.start()
        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []
        for i in six.moves.xrange(RUN_STEP):
            loss = exe.run(fluid.default_main_program(), fetch_list=[avg_cost])
            loss = loss[0] if loss else None
            out_losses.append(loss)
            print_to_err(type(self).__name__, "run step %d finished" % i)
        print_to_err(type(self).__name__, "trainer run finished")

        sys.stdout.buffer.write(pickle.dumps(out_losses))

        if args.save_model:
            model_save_dir = "/tmp"
            if fleet.worker_index() == 0:
                model_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_persistables")
                model_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_persistables")
                infer_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_infer")
                infer_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_infer")
            else:
                model_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_persistables_2")
                model_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_persistables_2")
                infer_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_infer_2")
                infer_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_infer_2")
            fluid.io.save_persistables(exe, model_save_dir_fluid,
                                       fleet._origin_program)
            fleet.save_persistables(executor=exe, dirname=model_save_dir_fleet)
            feeded_var_names = [var.name for var in feed_var_list]
            fluid.io.save_inference_model(infer_save_dir_fluid,
                                          feeded_var_names, [avg_cost], exe,
                                          fleet._origin_program)
            fleet.save_inference_model(exe, infer_save_dir_fleet,
                                       feeded_var_names, [avg_cost])

    def run_use_fleet_api_20_trainer(self, args):
        """
        1. remove codes for DistributedStrategy and leave the DistributedStrategy part to get_model()
        2. to run with fleet 2.0 api, set flags _use_fleet_api and _use_fleet_api_20 to True
        3. for now, not support test for model save
        """
        assert args.update_method == "nccl2" or "bkcl"

        self.lr = args.lr
        print_to_err("use_fleet 2.0", "fleet.node_num:")

        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
            self.get_model(batch_size=args.batch_size)

        if fluid.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(device_id)
        elif fluid.core.is_compiled_with_xpu():
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = fluid.XPUPlace(device_id)
        else:
            raise ValueError(
                "fleet dygraph api must in paddlepaddle-xpu or paddlepaddle-gpu."
            )

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        eprint(type(self).__name__, "run worker startup program done.")

        feed_var_list = [
            var
            for var in fluid.default_main_program().global_block().vars.values()
            if var.is_data
        ]

        eprint("feed_var_list:", feed_var_list)

        if feed_var_list[0].name == 'label':
            feed_var_list = feed_var_list[::-1]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []
        for i in six.moves.xrange(RUN_STEP):
            loss, = exe.run(fluid.default_main_program(),
                            fetch_list=[avg_cost.name],
                            feed=feeder.feed(get_data()))
            out_losses.append(loss[0])
            print_to_err(type(self).__name__, "run step %d finished" % i)
        print_to_err(type(self).__name__, "trainer run finished")
        print_to_err(type(self).__name__, "dist losses: {}".format(out_losses))

        sys.stdout.buffer.write(pickle.dumps(out_losses))

    def run_use_fleet_api_trainer(self, args):
        assert args.update_method == "nccl2" or "bkcl"

        self.lr = args.lr

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1

        dist_strategy = DistributedStrategy()
        dist_strategy.exec_strategy = exec_strategy
        dist_strategy.fuse_memory_size = 1  # MB
        dist_strategy.fuse_laryer_size = 1
        if args.use_local_sgd:
            dist_strategy.use_local_sgd = True
        if args.ut4grad_allreduce:
            dist_strategy._ut4grad_allreduce = True
        if args.sync_batch_norm:
            dist_strategy.sync_batch_norm = True

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print_to_err("use_fleet", "fleet.node_num:")
        # "fleet.node_id:", fleet.node_id(),
        # "fleet.trainer_num:", fleet.worker_num())

        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
            self.get_model(batch_size=args.batch_size, dist_strategy=dist_strategy)

        trainer_prog = fleet._origin_program
        dist_prog = fleet.main_program

        if fluid.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(device_id)
        elif fluid.core.is_compiled_with_xpu():
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = fluid.XPUPlace(device_id)
        else:
            raise ValueError(
                "fleet dygraph api must in paddlepaddle-xpu or paddlepaddle-gpu."
            )

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        eprint(type(self).__name__, "run worker startup program done.")

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        eprint("feed_var_list:", feed_var_list)

        # tmp add this code to pass python35 gcc8 CI
        # Fixme(gongweibao, wangxi), need fix fleet api program order
        if feed_var_list[0].name == 'label':
            feed_var_list = feed_var_list[::-1]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []
        for i in six.moves.xrange(RUN_STEP):
            loss, = exe.run(dist_prog,
                            fetch_list=[avg_cost.name],
                            feed=feeder.feed(get_data()))
            out_losses.append(loss[0])
            print_to_err(type(self).__name__, "run step %d finished" % i)
        print_to_err(type(self).__name__, "trainer run finished")

        sys.stdout.buffer.write(pickle.dumps(out_losses))

        if args.save_model:
            model_save_dir = "/tmp"
            if fleet.worker_index() == 0:
                model_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_persistables")
                model_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_persistables")
                infer_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_infer")
                infer_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_infer")
            else:
                model_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_persistables_2")
                model_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_persistables_2")
                infer_save_dir_fluid = os.path.join(model_save_dir,
                                                    "fluid_infer_2")
                infer_save_dir_fleet = os.path.join(model_save_dir,
                                                    "fleet_infer_2")
            fluid.io.save_persistables(exe, model_save_dir_fluid,
                                       fleet._origin_program)
            fleet.save_persistables(executor=exe, dirname=model_save_dir_fleet)
            feeded_var_names = [var.name for var in feed_var_list]
            fluid.io.save_inference_model(infer_save_dir_fluid,
                                          feeded_var_names, [avg_cost], exe,
                                          fleet._origin_program)
            fleet.save_inference_model(exe, infer_save_dir_fleet,
                                       feeded_var_names, [avg_cost])

    def run_trainer(self, args):
        self.lr = args.lr
        if args.nccl2_reduce_layer_local_run:
            test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
                self.get_model(batch_size=args.batch_size, single_device=True)
        elif args.use_dgc:
            test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
                self.get_model(batch_size=args.batch_size, use_dgc=args.use_dgc)
        else:
            test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
                self.get_model(batch_size=args.batch_size)

        if args.update_method == "pserver":
            print_to_err(
                type(self).__name__,
                "begin to run transpile on trainer with pserver mode")
            t = self.get_transpiler(
                trainer_id=args.trainer_id,
                main_program=fluid.default_main_program(),
                pserver_endpoints=args.endpoints,
                trainers=args.trainers,
                sync_mode=args.sync_mode,
                dc_asgd=args.dc_asgd,
                hogwild_mode=args.hogwild)

            trainer_prog = t.get_trainer_program()
            print_to_err(
                type(self).__name__,
                "get trainer program done with pserver mode.")
        elif args.update_method == "nccl2" or args.update_method == "nccl2_reduce_layer":
            # transpile for nccl2
            config = fluid.DistributeTranspilerConfig()
            config.mode = "nccl2"
            config.nccl_comm_num = args.nccl_comm_num
            if args.use_hallreduce:
                config.use_hierarchical_allreduce = True
                config.hierarchical_allreduce_inter_nranks = args.hallreduce_inter_nranks
            print_to_err(
                type(self).__name__,
                "begin to run transpile on trainer with nccl2 mode")
            nccl2_t = fluid.DistributeTranspiler(config=config)
            nccl2_t.transpile(
                args.trainer_id,
                program=fluid.default_main_program(),
                startup_program=fluid.default_startup_program(),
                trainers=args.endpoints,
                current_endpoint=args.current_endpoint)
            print_to_err(
                type(self).__name__,
                "get trainer program done. with nccl2 mode")
            trainer_prog = fluid.default_main_program()
        else:
            print_to_err(
                type(self).__name__,
                "do nothing about main program, just use it")
            trainer_prog = fluid.default_main_program()
            print_to_err(type(self).__name__, "use main program done.")

        # FIXME(gongwb):wait pserver initialization.
        time.sleep(1)

        if args.use_cuda:
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(device_id)
        else:
            place = fluid.CPUPlace()

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        print_to_err(type(self).__name__, "run worker startup program done.")

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1

        build_stra = fluid.BuildStrategy()
        # FIXME force disable enable_inplace and memory_optimize
        build_stra.enable_inplace = False
        build_stra.memory_optimize = False

        if args.fuse_all_reduce is not None:
            sys.stderr.write('fuse_all_reduce={}'.format(args.fuse_all_reduce))
            build_stra.fuse_all_reduce_ops = args.fuse_all_reduce

        if args.hogwild:
            build_stra.async_mode = True

        if args.enable_backward_deps:
            build_stra.enable_backward_optimizer_op_deps = True

        if args.use_reduce:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        else:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce

        pass_builder = None
        if args.batch_merge_repeat > 1:
            pass_builder = build_stra._finalize_strategy_and_create_passes()
            mypass = pass_builder.insert_pass(0, "multi_batch_merge_pass")
            mypass.set("num_repeats", args.batch_merge_repeat)

        if args.update_method == "nccl2" or args.update_method == "nccl2_reduce_layer":
            build_stra.num_trainers = len(args.endpoints.split(","))
            build_stra.trainer_id = args.trainer_id
        else:
            # case args.update_method == "nccl2_reduce_layer":
            build_stra.num_trainers = 1
            build_stra.trainer_id = 0

        print_to_err(type(self).__name__, "begin to compile with data parallel")
        binary = compiler.CompiledProgram(trainer_prog).with_data_parallel(
            loss_name=avg_cost.name,
            build_strategy=build_stra,
            exec_strategy=exec_strategy)
        print_to_err(type(self).__name__, "program compiled with data parallel")

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        print_to_err(type(self).__name__, "begin to train on trainer")
        out_losses = []
        for i in six.moves.xrange(RUN_STEP):
            loss, = exe.run(binary,
                            fetch_list=[avg_cost.name],
                            feed=feeder.feed(get_data()))
            out_losses.append(loss[0])
            print_to_err(type(self).__name__, "run step %d finished" % i)
        print_to_err(type(self).__name__, "trainer run finished")

        print_to_out(out_losses)


class TestParallelDyGraphRunnerBase(object):
    def get_model(self):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    def run_one_loop(self, model, opt, data):
        raise NotImplementedError(
            "train_one_loop should be implemented by the child classes.")

    def _get_data(self, batch, args):
        if args.update_method != "local":
            new_batch = []
            for offset, item in enumerate(batch):
                if offset % 2 == args.trainer_id:
                    new_batch.append(item)
            return new_batch
        else:
            return batch

    def run_trainer(self, args):

        seed = 90
        if fluid.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(device_id)
        elif fluid.core.is_compiled_with_xpu():
            device_id = int(os.getenv("FLAGS_selected_xpus", "0"))
            place = fluid.XPUPlace(device_id)
        else:
            assert ("Only support CUDAPlace or XPUPlace for now.")

        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            np.random.seed(seed)
            import random
            random.seed(seed)
            model, train_reader, opt = self.get_model()
            nranks = len(args.endpoints.split(",")) if args.endpoints else 1

            #if args.update_method == "nccl2":
            if args.update_method == "nccl2" or args.update_method == "bkcl":
                strategy = dygraph.parallel.ParallelStrategy()
                strategy.nranks = nranks
                strategy.local_rank = args.trainer_id
                strategy.trainer_endpoints = args.endpoints.split(",")
                strategy.current_endpoint = args.current_endpoint
                print_to_err(
                    type(self).__name__,
                    "begin to prepare context in dygraph with nccl2")
                dygraph.parallel.prepare_context(strategy)
                if not args.find_unused_parameters:
                    model = dygraph.parallel.DataParallel(
                        model, strategy, find_unused_parameters=False)
                else:
                    model = dygraph.parallel.DataParallel(
                        model, strategy, find_unused_parameters=True)
                print_to_err(type(self).__name__, "model built in dygraph")
            out_losses = []
            print_to_err(type(self).__name__, "begin to run dygraph training")
            for step_id, data in enumerate(train_reader()):
                data = self._get_data(data, args)
                if step_id == RUN_STEP:
                    break
                loss = self.run_one_loop(model, opt, data)
                if step_id % 10 == 0:
                    print_to_err(
                        type(self).__name__,
                        "loss at step %d: %f" % (step_id, loss.numpy()))
                out_losses.append(loss.numpy())

                loss.backward()

                opt.minimize(loss)
                if not args.accumulate_gradient:
                    model.clear_gradients()
        print_to_out(out_losses)

    def run_trainer_with_spawn(self, args):
        # 1. enable dygraph
        paddle.disable_static()

        # 2. init seed
        seed = 90
        paddle.static.default_startup_program().random_seed = seed
        paddle.static.default_main_program().random_seed = seed
        np.random.seed(seed)
        random.seed(seed)
        # get trainer id
        args.trainer_id = paddle.distributed.get_rank()

        # 3. init parallel env
        if args.update_method == "nccl2":
            paddle.distributed.init_parallel_env()

        # 4. train model
        model, train_reader, opt = self.get_model()
        if args.update_method == "nccl2":
            if args.find_unused_parameters:
                model = paddle.DataParallel(model, find_unused_parameters=True)
            else:
                model = paddle.DataParallel(model, find_unused_parameters=False)

        out_losses = []
        for step_id, data in enumerate(train_reader()):
            data = self._get_data(data, args)
            if step_id == RUN_STEP:
                break
            loss = self.run_one_loop(model, opt, data)
            out_losses.append(loss.numpy())

            loss.backward()

            opt.minimize(loss)
            model.clear_gradients()
        return out_losses

    def run_use_fleet_api_trainer(self, args):
        import paddle.distributed.fleet as fleet
        import paddle.distributed.fleet.base.role_maker as role_maker
        # 1. enable dygraph
        paddle.disable_static()

        # 2. init seed
        seed = 90
        paddle.static.default_startup_program().random_seed = seed
        paddle.static.default_main_program().random_seed = seed
        np.random.seed(seed)
        random.seed(seed)
        # get trainer id
        args.trainer_id = paddle.distributed.get_rank()

        # set strategy
        strategy = fleet.DistributedStrategy()
        if args.find_unused_parameters:
            strategy.find_unused_parameters = True

        # 3. init parallel env
        if args.update_method == "nccl2" or "bkcl":
            fleet.init(is_collective=True, strategy=strategy)

        # 4. train model
        model, train_reader, opt = self.get_model()
        if args.update_method == "nccl2" or "bkcl":
            opt = fleet.distributed_optimizer(opt)
            model = fleet.distributed_model(model)

        out_losses = []
        for step_id, data in enumerate(train_reader()):
            data = self._get_data(data, args)
            if step_id == RUN_STEP:
                break
            loss = self.run_one_loop(model, opt, data)
            out_losses.append(loss.numpy())

            loss.backward()

            opt.step()
            if not args.accumulate_gradient:
                opt.clear_grad()
        print_to_out(out_losses)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description='Run dist test.')
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--update_method',
        type=str,
        default="local",
        choices=["pserver", "nccl2", "bkcl", "local", "nccl2_reduce_layer"])
    parser.add_argument('--trainer_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument('--nccl_comm_num', type=int, required=False, default=1)
    parser.add_argument('--enable_backward_deps', action='store_true')
    parser.add_argument('--use_hallreduce', action='store_true')
    parser.add_argument('--use_pipeline', action='store_true')
    parser.add_argument('--use_fleet_api', action='store_true')
    parser.add_argument('--use_fleet_api_20', action='store_true')
    parser.add_argument('--use_local_sgd', action='store_true')
    parser.add_argument('--ut4grad_allreduce', action='store_true')
    parser.add_argument(
        '--hallreduce_inter_nranks', type=int, required=False, default=2)
    parser.add_argument(
        '--current_endpoint', type=str, required=False, default="")
    parser.add_argument('--sync_mode', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_xpu', action='store_true')
    parser.add_argument('--use_dgc', action='store_true')
    parser.add_argument('--accumulate_gradient', action='store_true')
    parser.add_argument('--find_unused_parameters', action='store_true')
    parser.add_argument('--use_reduce', action='store_true')
    parser.add_argument('--dc_asgd', action='store_true')
    parser.add_argument('--hogwild', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument(
        '--use_reader_alloc', action='store_true', required=False)
    parser.add_argument('--batch_size', required=False, type=int, default=2)
    parser.add_argument('--lr', required=False, type=float, default=0.001)
    parser.add_argument(
        '--batch_merge_repeat', required=False, type=int, default=1)
    parser.add_argument(
        '--nccl2_reduce_layer_local_run',
        required=False,
        type=bool,
        default=False)
    parser.add_argument('--sync_batch_norm', action='store_true')
    parser.add_argument(
        '--fuse_all_reduce',
        required=False,
        type=ast.literal_eval,
        default=None)

    args = parser.parse_args()

    model = test_class()
    if args.role == "pserver" and args.update_method == "pserver":
        model.run_pserver(args)
    elif args.use_fleet_api:
        model.run_use_fleet_api_trainer(args)
    elif args.use_fleet_api_20:
        model.run_use_fleet_api_20_trainer(args)
    elif args.use_pipeline:
        model.run_pipeline_trainer(args)
    else:
        model.run_trainer(args)


import paddle.compat as cpt
import socket
from contextlib import closing


class TestDistBase(unittest.TestCase):
    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def _after_setup_config(self):
        if self._enforce_place == "CPU":
            self.__use_cuda = False
            self.__use_xpu = False
            self._use_dgc = False
        elif self._enforce_place == "GPU":
            self.__use_cuda = True
            self.__use_xpu = False
        elif self._enforce_place == "XPU":
            self.__use_cuda = False
            self.__use_xpu = True
            self._use_dgc = False
        else:
            if fluid.core.is_compiled_with_cuda():
                self.__use_cuda = True
            else:
                self.__use_cuda = False
                self._use_dgc = False

        if self._use_reduce:
            assert not self._use_dgc

    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()
        self._python_interp = sys.executable
        self._sync_mode = True
        self._hogwild_mode = False
        self._enforce_place = None
        self._use_reduce = False
        self._dc_asgd = False  # must use with async mode
        self._use_reader_alloc = True
        self._nccl2_mode = False
        self._bkcl_mode = False
        self._pipeline_mode = False
        self._mp_mode = False
        # FIXME(typhoonzero): I added this stupid argument to enable
        # testing allreduce layers, which users can call layers.allreduce
        # to accumulate tensors at anywhere. Find a better way to do this
        # test, reduce check this argument everywhere.
        self._nccl2_reduce_layer = False
        self._lr = 0.001
        self._use_dgc = False
        self._dygraph = False
        self._nccl_comm_num = 1
        self._enable_backward_deps = False
        self._use_fleet_api = False
        self._use_fleet_api_20 = False
        self._use_local_sgd = False
        self._ut4grad_allreduce = False
        self._use_hallreduce = False
        self._save_model = False
        self._fuse_all_reduce = None
        self._accumulate_gradient = False
        self._find_unused_parameters = False
        self._setup_config()

        global DIST_UT_PORT
        if DIST_UT_PORT == 0 and os.getenv("PADDLE_DIST_UT_PORT"):
            DIST_UT_PORT = int(os.getenv("PADDLE_DIST_UT_PORT"))

        if DIST_UT_PORT == 0:
            self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                self._find_free_port(), self._find_free_port())
        else:
            print("set begin_port:", DIST_UT_PORT)
            self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                DIST_UT_PORT, DIST_UT_PORT + 1)
            DIST_UT_PORT += 2

        self._after_setup_config()

    def _find_free_port(self):
        def __free_port():
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                print_to_err(
                    type(self).__name__, "socket name: %s" % s.getsockname()[1])
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def start_pserver(self,
                      model_file,
                      check_error_log,
                      required_envs,
                      log_name=""):
        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        ps_cmd = "%s"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            required_envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            ps_cmd += " -m coverage run --branch -p"

        ps_cmd += " %s --role pserver --endpoints %s --trainer_id 0 --current_endpoint %s --trainers %d --update_method pserver"

        ps0_cmd = ps_cmd % \
                  (self._python_interp, model_file, self._ps_endpoints, ps0_ep,
                   self._trainers)
        ps1_cmd = ps_cmd % \
                  (self._python_interp, model_file, self._ps_endpoints, ps1_ep,
                   self._trainers)

        if self._sync_mode:
            ps0_cmd += " --sync_mode"
            ps1_cmd += " --sync_mode"

        print(ps0_cmd)
        print(ps1_cmd)
        ps0_pipe = open(log_name + "_ps0_err.log", "wb")
        ps1_pipe = open(log_name + "_ps1_err.log", "wb")

        print_to_err(type(self).__name__, "going to start pserver process 0")
        ps0_proc = subprocess.Popen(
            ps0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps0_pipe,
            env=required_envs)
        print_to_err(type(self).__name__, "going to start pserver process 1")
        ps1_proc = subprocess.Popen(
            ps1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps1_pipe,
            env=required_envs)

        return ps0_proc, ps1_proc, ps0_pipe, ps1_pipe

    def _run_local(self,
                   model,
                   envs,
                   check_error_log=False,
                   batch_size=DEFAULT_BATCH_SIZE,
                   batch_merge_repeat=1,
                   log_name="",
                   devices="0"):

        cmd = self._python_interp

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            cmd += " -m coverage run --branch -p"

        cmd += " %s --role trainer --update_method local --lr %f" % (model,
                                                                     self._lr)

        if batch_size != DEFAULT_BATCH_SIZE:
            cmd += " --batch_size %d" % batch_size
        if batch_merge_repeat > 1:
            cmd += " --batch_merge_repeat %d" % batch_merge_repeat
        if self._nccl2_reduce_layer:
            cmd += " --nccl2_reduce_layer_local_run 1"

        if self.__use_cuda:
            cmd += " --use_cuda"
            env_local = {
                "CUDA_VISIBLE_DEVICES": devices,
                "PADDLE_TRAINERS_NUM": "1",
                "PADDLE_TRAINER_ID": "0"
            }
        elif self.__use_xpu:
            cmd += " --use_xpu"
            env_local = {
                "FLAGS_selected_xpus": devices,
                "PADDLE_TRAINERS_NUM": "1",
                "PADDLE_TRAINER_ID": "0"
            }
        else:
            env_local = {'CPU_NUM': '1'}

        # not use dgc in single card
        if len(devices) > 1 and self._use_dgc:
            cmd += " --use_dgc"

        if self._accumulate_gradient:
            cmd += " --accumulate_gradient"

        if self._find_unused_parameters:
            cmd += " --find_unused_parameters"

        env_local.update(envs)
        print("local_cmd: {}, env: {}".format(cmd, env_local))

        if check_error_log:
            err_log = open(log_name + "_local.log", "wb")
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=err_log,
                env=env_local)
        else:
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_local)

        local_out, local_err = local_proc.communicate()

        if check_error_log:
            err_log.close()

        sys.stderr.write('local_stderr: %s\n' % local_err)
        sys.stderr.write('local_stdout: %s\n' % pickle.loads(local_out))

        return pickle.loads(local_out)

    def _run_cluster(self, model, envs, check_error_log, log_name):
        # Run dist train to compare with local results
        ps0, ps1, ps0_pipe, ps1_pipe = self.start_pserver(
            model, check_error_log, envs, log_name=log_name)

        ps0_ep, ps1_ep = self._ps_endpoints.split(",")

        tr_cmd = "%s"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            tr_cmd += " -m coverage run --branch -p"

        tr_cmd += " %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --trainers %d --update_method pserver --lr %f"

        tr0_cmd = tr_cmd % \
                  (self._python_interp, model, self._ps_endpoints,
                   0, ps0_ep, self._trainers, self._lr)
        tr1_cmd = tr_cmd % \
                  (self._python_interp, model, self._ps_endpoints,
                   1, ps1_ep, self._trainers, self._lr)

        if self._sync_mode:
            tr0_cmd += " --sync_mode"
            tr1_cmd += " --sync_mode"
        if self._hogwild_mode:
            tr0_cmd += " --hogwild"
            tr1_cmd += " --hogwild"
        if self._use_reduce:
            tr0_cmd += " --use_reduce"
            tr1_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr0_cmd += " --use_reader_alloc"
            tr1_cmd += " --use_reader_alloc"
        if self.__use_cuda:
            tr0_cmd += " --use_cuda"
            tr1_cmd += " --use_cuda"
            env0 = {"CUDA_VISIBLE_DEVICES": "0"}
            env1 = {"CUDA_VISIBLE_DEVICES": "1"}
        else:
            env0 = {'CPU_NUM': '1'}
            env1 = {'CPU_NUM': '1'}

        env0.update(envs)
        env1.update(envs)

        print("tr0_cmd: {}, env: {}".format(tr0_cmd, env0))
        print("tr1_cmd: {}, env: {}".format(tr1_cmd, env1))
        tr0_pipe = open(log_name + "_tr0_err.log", "wb")
        tr1_pipe = open(log_name + "_tr1_err.log", "wb")

        print_to_err(type(self).__name__, "going to start trainer process 0")
        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=env0)
        print_to_err(type(self).__name__, "going to start trainer process 1")
        tr1_proc = subprocess.Popen(
            tr1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=env1)

        # Wait until trainer process terminate
        while True:
            stat0 = tr0_proc.poll()
            time.sleep(0.1)
            if stat0 is not None:
                break
        while True:
            stat1 = tr1_proc.poll()
            time.sleep(0.1)
            if stat1 is not None:
                break

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()

        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        ps0_pipe.close()
        ps1_pipe.close()

        ps0.terminate()
        ps1.terminate()

        return pickle.loads(tr0_out), pickle.loads(tr1_out)

    def _get_nccl2_trainer_cmd(self, model, ep, update_method, trainer_id,
                               trainer_num):
        env = {}
        tr_cmd = "%s -u"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            tr_cmd += " -m coverage run --branch -p"

        tr_cmd += " %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --update_method %s --lr %f"

        tr_cmd = tr_cmd % \
                 (self._python_interp, model, self._ps_endpoints,
                  trainer_id, ep, update_method, self._lr)

        if self._use_reduce:
            tr_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr_cmd += " --use_reader_alloc"
        if self._save_model:
            tr_cmd += " --save_model"
        if self.__use_cuda:
            tr_cmd += " --use_cuda"
            env.update({
                "FLAGS_selected_gpus": "{}".format(0),
                "CUDA_VISIBLE_DEVICES": "{}".format(trainer_id),
                "PADDLE_TRAINERS_NUM": "{}".format(trainer_num),
                "PADDLE_TRAINER_ID": "{}".format(trainer_id),
                "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
                "PADDLE_CURRENT_ENDPOINT": ep,
            })
        # TODO(liuyuhui):XPU_VISIBLE_DEVICES is not working right now,
        # will update it after Badiu Kunlun partners' support.
        elif self.__use_xpu:
            tr_cmd += " --use_xpu"
            env.update({
                "FLAGS_selected_xpus": "{}".format(trainer_id),
                #"XPU_VISIBLE_DEVICES": "{}".format(trainer_id + 1),
                "PADDLE_TRAINERS_NUM": "{}".format(trainer_num),
                "PADDLE_TRAINER_ID": "{}".format(trainer_id),
                "PADDLE_TRAINER_ENDPOINTS": self._ps_endpoints,
                "PADDLE_CURRENT_ENDPOINT": ep,
                "GLOG_v": "2",
            })
        else:
            env.update({'CPU_NUM': '1'})

        if self._use_dgc:
            tr_cmd += " --use_dgc"

        if self._accumulate_gradient:
            tr_cmd += " --accumulate_gradient"

        if self._find_unused_parameters:
            tr_cmd += " --find_unused_parameters"

        if self._pipeline_mode:
            tr_cmd += " --use_pipeline"
        if self._mp_mode:
            env = {"FLAGS_selected_gpus": "{}".format(trainer_id)}

        if self._nccl_comm_num > 1:
            tr_cmd += " --nccl_comm_num {}".format(self._nccl_comm_num)

        if self._use_hallreduce:
            tr_cmd += " --use_hallreduce --hallreduce_inter_nranks 2"

        if self._enable_backward_deps:
            tr_cmd += " --enable_backward_deps"

        if self._fuse_all_reduce is not None:
            tr_cmd += " --fuse_all_reduce {}".format(self._fuse_all_reduce)

        if self._use_fleet_api:
            tr_cmd += " --use_fleet_api_20" if self._use_fleet_api_20 else " --use_fleet_api"
            if self._use_local_sgd:
                tr_cmd += " --use_local_sgd"
            if self._ut4grad_allreduce:
                tr_cmd += " --ut4grad_allreduce"
            if hasattr(self, '_sync_batch_norm') and self._sync_batch_norm:
                tr_cmd += " --sync_batch_norm"

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            env['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')

        return tr_cmd, env

    def _run_cluster_nccl2(self, model, envs, update_method, check_error_log,
                           log_name):
        if self._use_hallreduce:
            self._ps_endpoints = ""

            global DIST_UT_PORT
            if DIST_UT_PORT == 0:
                # NOTE(wangxi). hallreduce test must use 4cards after nccl>=2.7
                for i in range(0, 4):
                    self._ps_endpoints += "127.0.0.1:%s," % (
                        self._find_free_port())
            else:
                for i in range(0, 4):
                    self._ps_endpoints += "127.0.0.1:%s," % (DIST_UT_PORT + i)
                DIST_UT_PORT += 4
            self._ps_endpoints = self._ps_endpoints[:-1]

        # NOTE: we reuse ps_endpoints as nccl2 worker endpoints
        worker_endpoints = self._ps_endpoints.split(",")

        trainer_num = len(worker_endpoints)

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_nccl2_trainer_cmd(
                model, worker_endpoints[i], update_method, i, trainer_num)
            tr_env.update(envs)
            print("use_hallreduce:{} tr_cmd:{}, env: {}".format(
                self._use_hallreduce, tr_cmd, tr_env))

            tr_pipe = open(log_name + "_tr{}_err.log".format(i), "wb")

            print_to_err(
                type(self).__name__,
                "going to start process {} with nccl2".format(i))
            tr_proc = subprocess.Popen(
                tr_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=tr_env)

            procs.append(tr_proc)
            pipes.append(tr_pipe)

        outs = []
        for i in range(0, trainer_num):
            tr_out, tr_err = procs[i].communicate()
            outs.append(tr_out)
            pipes[i].close()
            sys.stderr.write('trainer {} stderr: {}\n'.format(i, tr_err))

        if check_error_log:
            print("outs[0]:", outs[0])
            print("outs[1]:", outs[1])

        return pickle.loads(outs[0]), pickle.loads(outs[1])

    def _run_pipeline(self, model, envs, check_error_log, log_name):
        # NOTE: we reuse ps_endpoints as nccl2 worker endpoints
        worker_endpoints = self._ps_endpoints.split(",")
        update_method = "nccl2"

        trainer_num = len(worker_endpoints)

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_nccl2_trainer_cmd(
                model, worker_endpoints[i], update_method, i, trainer_num)
            tr_env.update(envs)
            tr_env['CUDA_VISIBLE_DEVICES'] = "0,1"
            tr_env['NCCL_SHM_DISABLE'] = '1'
            tr_env['FLAGS_selected_gpus'] = str(i)
            tr_env['FLAGS_cudnn_deterministic'] = '0'
            print("tr_cmd:{}, env: {}".format(tr_cmd, tr_env))

            tr_pipe = open("/tmp/" + "tr{}_err.log".format(i), "wb")

            print_to_err(
                type(self).__name__,
                "going to start process {} with nccl2".format(i))
            tr_proc = subprocess.Popen(
                tr_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=tr_env)

            procs.append(tr_proc)
            pipes.append(tr_pipe)

        outs = []
        for i in range(0, trainer_num):
            tr_out, tr_err = procs[i].communicate()
            outs.append(tr_out)
            pipes[i].close()
            sys.stderr.write('trainer {} stderr: {}\n'.format(i, tr_err))

        if check_error_log:
            print("outs[0]:", outs[0])
            print("outs[1]:", outs[1])
        return pickle.loads(outs[0]), pickle.loads(outs[1])

    def _get_required_envs(self, check_error_log=False, need_envs={}):
        # TODO(typhoonzero): should auto adapt GPU count on the machine.
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_rpc_deadline": "30000",  # 5sec to fail fast
            "FLAGS_rpc_retry_bind_port": "50",
            "FLAGS_cudnn_deterministic": "1",
            "FLAGS_rpc_disable_reuse_port": "1",
            "http_proxy": "",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "1"
        }

        if check_error_log:
            required_envs["GLOG_vmodule"] = \
                "fused_all_reduce_op_handle=10,all_reduce_op_handle=10,alloc_continuous_space_op=10,fuse_all_reduce_op_pass=10," \
                "alloc_continuous_space_for_grad_pass=10,fast_threaded_ssa_graph_executor=10,executor=10,operator=10," \
                "sparse_all_reduce_op_handle=10,gen_nccl_id_op=10,gen_nccl_id_op_help=10,nccl_helper=10,grpc_client=10," \
                "grpc_server=10,request_handler_impl=10"
            required_envs["GLOG_logtostderr"] = "1"

        required_envs.update(need_envs)
        return required_envs

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={},
                         log_name=""):

        required_envs = self._get_required_envs(check_error_log, need_envs)

        local_losses \
            = self._run_local(model_file, required_envs,
                              check_error_log, log_name=log_name)

        if self._nccl2_mode:
            if self._nccl2_reduce_layer:
                tr0_losses, tr1_losses = self._run_cluster_nccl2(
                    model_file,
                    required_envs,
                    update_method="nccl2_reduce_layer",
                    check_error_log=check_error_log,
                    log_name=log_name)
            else:
                tr0_losses, tr1_losses = self._run_cluster_nccl2(
                    model_file,
                    required_envs,
                    update_method='nccl2',
                    check_error_log=check_error_log,
                    log_name=log_name)
        elif self._bkcl_mode:
            tr0_losses, tr1_losses = self._run_cluster_nccl2(
                model_file,
                required_envs,
                update_method='bkcl',
                check_error_log=check_error_log,
                log_name=log_name)

        elif self._pipeline_mode:
            tr0_losses, tr1_losses = self._run_pipeline(
                model_file, required_envs, check_error_log, log_name=log_name)
        else:
            tr0_losses, tr1_losses = self._run_cluster(
                model_file, required_envs, check_error_log, log_name=log_name)

        for step_id in range(RUN_STEP):
            local_loss = local_losses[step_id]
            tr0_loss = tr0_losses[step_id]
            tr1_loss = tr1_losses[step_id]
            if self._pipeline_mode:
                dist_loss = np.array([tr1_loss])
            else:
                dist_loss = (np.array([tr0_loss]) + np.array([tr1_loss])) / 2
            print("=======", local_loss, ":", dist_loss[0], "=======")
            self.assertAlmostEqual(local_loss, dist_loss[0], delta=delta)

    def check_with_place_multi_cards(self,
                                     model_file,
                                     delta=1e-3,
                                     check_error_log=False,
                                     need_envs={},
                                     log_name=""):

        # need open p2p or shm otherwise multi cards mode will hang
        need_envs.update({"NCCL_P2P_DISABLE": "0", "NCCL_SHM_DISABLE": "0"})

        required_envs = self._get_required_envs(check_error_log, need_envs)

        if self._use_dgc:
            multi_cards_losses = self._run_local(
                model_file,
                required_envs,
                check_error_log,
                log_name=log_name + "_dgc_2cards",
                devices="0,1")

            self._use_dgc = False
            base_losses = self._run_local(
                model_file,
                required_envs,
                check_error_log,
                log_name=log_name + "_base_2cards",
                devices="0,1")

            self._use_dgc = True

            for step_id in range(RUN_STEP):
                base_loss = base_losses[step_id]
                multi_cards_loss = multi_cards_losses[step_id]
                print("=======", base_loss, ":", multi_cards_loss, "=======")
                self.assertAlmostEqual(base_loss, multi_cards_loss, delta=delta)
