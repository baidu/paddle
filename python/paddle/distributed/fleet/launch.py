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
"""
fleetrun is a module that spawns multiple distributed
process on each training node for gpu training and cpu training.
Usage:
    In both of single node training or multiple node training, this module
launch a process on each of the given gpu card or cpu machine.
    GPU training:
    1. for single node training with all visible gpu cards:
       fleetrun your_training_py (arg1 arg2 and all others)
    2. for single node training with [0,4) cards
       fleetrun --gpus="0,1,2,3" your_training_py (arg1 arg2 and all others)
    3. for multiple node training such as two node:192.168.0.16, 192.168.0.17
        on 192.168.0.16:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --ips="192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    CPU training:
    1. for single node training with multi servers and workers:
        fleetrun --server_num=2 --worker_num=2 your_training_py (arg1 arg2 and all others)
    2. for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers.
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6171" \
                --workers="192.168.0.16,192.168.0.17,192.168.0.16,192.168.0.17" \
                your_training_py (arg1 arg2 and all others)
    3. use gloo backend for multiple node training such as two node:192.168.0.16, 192.168.0.17 \
        with 2 servers and 4 workers. (workers should set port)
        on 192.168.0.16:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
        on 192.168.0.17:
            fleetrun --servers="192.168.0.16:6170,192.168.0.17:6170" \
                --workers="192.168.0.16:6171,192.168.0.17:6171,192.168.0.16:6172,192.168.0.17:6172" \
                your_training_py (arg1 arg2 and all others)
"""

from __future__ import print_function

import shutil
import sys
import tempfile
from sys import version
import subprocess
import os
import time
import six
import copy
from argparse import ArgumentParser, REMAINDER
import paddle
import paddle.fluid as fluid

from paddle.distributed.fleet.launch_utils import *
import paddle.distributed.fleet.cloud_utils as cloud_utils


def _print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def _parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description='''start paddle training using multi-process mode.
see: http://www.paddlepaddle.org/documentation/docs/zh/1.6/user_guides/howto/training/cluster_howto.html#permalink-8--nccl2-
''')
    base_group = parser.add_argument_group("Base Parameters")

    base_group.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="The path for each process's log.If it's not set, the log will printed to default pipe."
    )

    base_group.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="It's for gpu training and the training process will run on the gpus,"
        "each process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training."
    )

    base_group.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script")

    base_group.add_argument('training_script_args', nargs=REMAINDER)

    # Optional arguments for the launch helper
    # for collective
    collective_group = parser.add_argument_group("Collective Parameters")
    collective_group.add_argument(
        "--ips",
        type=str,
        default="127.0.0.1",
        help="Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..")

    ps_group = parser.add_argument_group("Parameter-Server Parameters")
    # for parameter server
    ps_group.add_argument(
        "--servers", type=str, default="", help="User defined servers ip:port")
    ps_group.add_argument(
        "--workers", type=str, default="", help="User defined workers ip:port")
    ps_group.add_argument(
        "--heter_workers",
        type=str,
        default="",
        help="User defined heter workers ip:port")

    ps_group.add_argument("--worker_num", type=int, help="number of workers")
    ps_group.add_argument("--server_num", type=int, help="number of servers")
    ps_group.add_argument(
        "--heter_worker_num", type=int, help="number of heter_workers")
    ps_group.add_argument("--http_port", type=int, help="Gloo http Port")

    return parser.parse_args()


def get_cluster_from_args(args, gpus):
    node_ips = [x.strip() for x in args.ips.split(',')]
    if len(node_ips) == 1:
        node_ip = node_ips[0]
    else:
        _, node_ip = get_host_name_ip()

    # node_ip = args.node_ip
    assert node_ip in node_ips, "Can't find your local ip {%s} in node_ips: {%s}" \
        % (node_ip, node_ips)
    node_rank = node_ips.index(node_ip)

    logger.debug("parsed from args: node_ips:{} node_ip:{} node_rank:{}".format(
        node_ips, node_ip, node_rank))

    free_ports = None
    if not cloud_utils.use_paddlecloud() and len(
            node_ips) <= 1 and os.environ.get('FLAGS_START_PORT') is None:
        free_ports = find_free_ports(len(gpus))
        if free_ports is not None:
            free_ports = list(free_ports)
    else:
        start_port = 6070
        if os.environ.get('FLAGS_START_PORT') is not None:
            start_port = int(os.environ.get('FLAGS_START_PORT'))

        free_ports = [x for x in range(start_port, start_port + len(gpus))]

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append(["%s:%d" % (ip, port) for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, gpus)


def launch_collective(args):
    # parse arguments, used for cloud-single-machine and local
    gpus = get_gpus(args.gpus)
    trainers_num = cloud_utils.get_trainers_num()
    logger.debug("parsed from args trainerss_num:{} gpus:{}".format(
        trainers_num, gpus))

    cluster = None
    pod = None

    start_port = 6170
    if os.environ.get('FLAGS_START_PORT') is not None:
        start_port = os.environ.get('FLAGS_START_PORT')
    if cloud_utils.use_paddlecloud() and trainers_num != 1:
        cluster, pod = cloud_utils.get_cloud_cluster(args.ips, gpus, start_port)
        logger.debug("get cluster from cloud:{}".format(cluster))
    else:
        # trainers_num = 1 or not use paddlecloud ips="a,b"
        cluster, pod = get_cluster_from_args(args, gpus)
        logger.debug("get cluster from args:{}".format(cluster))

    global_envs = copy.copy(os.environ.copy())
    gloo_rendezvous_dir = tempfile.mkdtemp()
    # add gloo env
    global_envs["PADDLE_WITH_GLOO"] = "1"
    global_envs["PADDLE_GLOO_RENDEZVOUS"] = "3"
    global_envs["PADDLE_GLOO_FS_PATH"] = gloo_rendezvous_dir

    procs = start_local_trainers(
        cluster,
        pod,
        training_script=args.training_script,
        training_script_args=args.training_script_args,
        log_dir=args.log_dir,
        envs=global_envs)

    while True:
        alive = watch_local_trainers(procs, cluster.trainers_nranks())

        if not alive:
            logger.info("Local processes completed.")
            logger.debug("POD info:{}".format(pod))
            break

        time.sleep(3)

    if os.path.exists(gloo_rendezvous_dir):
        shutil.rmtree(gloo_rendezvous_dir)


def launch_ps(args, distribute_mode):
    cloud_flag = cloud_utils.use_paddlecloud()

    # for ps-cpu on paddlecloud
    if cloud_flag and distribute_mode == DistributeMode.PS:
        direct_start(args)
        return
    elif cloud_flag and distribute_mode == DistributeMode.PS_HETER:
        cloud_ps_heter_env_set(args)
        args.workers = os.getenv("PADDLE_TRAINER_ENDPOINTS")
        args.servers = os.getenv("PADDLE_PSERVERS_IP_PORT_LIST")
        args.heter_workers = os.getenv("PADDLE_HETER_TRAINER_IP_PORT_LIST")

    ps_launcher = ParameterServerLauncher(args, distribute_mode)
    ps_launcher.start_ps()
    return


def which_distributed_mode(args):
    ps_args = [
        '--worker_num', '--server_num', '--heter_worker_num', '--servers',
        '--workers', '--heter_workers', '--http_port'
    ]
    collective_args = ['--ips']

    ps_heter_args = ["--heter_worker_num", "--heter_workers"]

    has_ps_args = [
        ps_arg for ps_arg in ps_args if ps_arg in " ".join(sys.argv[1:-1])
    ]
    has_collective_args = [
        co_arg for co_arg in collective_args
        if co_arg in " ".join(sys.argv[1:-1])
    ]

    if len(has_ps_args) > 1 and len(has_collective_args) > 1:
        raise ValueError(
            "Only one mode(Collective or Parameter-Server) can be selected at the same time, but more than one configuration was received."
        )

    if fluid.core.is_compiled_with_cuda():
        cuda_device_num = fluid.core.get_cuda_device_count()
    else:
        cuda_device_num = 0

    if len(has_ps_args) > 0:
        logger.info(
            "Run parameter-sever mode. pserver arguments:{}, cuda count:{}".
            format(has_ps_args, cuda_device_num))
        has_ps_heter_args = list(set(has_ps_args) & set(ps_heter_args))
        if len(has_ps_heter_args) > 0:
            return DistributeMode.PS_HETER
        else:
            return DistributeMode.PS
    elif len(has_collective_args) > 0:
        logger.info("Run collective gpu mode. gpu arguments:{}, cuda count:{}".
                    format(has_collective_args, cuda_device_num))
        return DistributeMode.COLLECTIVE
    else:
        if not fluid.core.is_compiled_with_cuda():
            logger.warning(
                "Not found distinct arguments and not compiled with cuda. Default use ps mode"
            )
            return DistributeMode.PS
        else:
            logger.warning(
                "Not found distinct arguments and compiled with cuda. Default use collective mode"
            )
            return DistributeMode.COLLECTIVE


def launch():
    args = _parse_args()
    logger = get_logger()
    _print_arguments(args)

    distribute_mode = which_distributed_mode(args)
    if distribute_mode == DistributeMode.COLLECTIVE:
        launch_collective(args)
    else:
        launch_ps(args, distribute_mode)


if __name__ == "__main__":
    launch()
