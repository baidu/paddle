# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import argparse
import os
import sys
import signal
import time
import socket
from contextlib import closing
from six import string_types
import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.fluid.unique_name as nameGen
from paddle.fluid import core
import paddle.distributed.fleet as fleet
from paddle.fluid.incubate.fleet.base import role_maker
import unittest
from multiprocessing import Process
import paddle.fluid.layers as layers
from functools import reduce
from test_collective_api_base import TestCollectiveAPIRunnerBase, runtime_main

paddle.enable_static()


class TestParallelEmbeddingAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank):
        with fluid.program_guard(main_prog, startup_program):
            fleet.init(is_collective=True)
            np.random.seed(2020)
            # (num_embeddings, embedding_dim) = (12, 8)
            size = (12, 8)
            np_array = np.random.rand(size[0], size[1])
            paddle.seed(2020)
            data_in = paddle.randint(0, size[0], shape=(10, 4))

            data = paddle.static.data(
                name='tindata', shape=[10, 1000], dtype="float32")
            per_part_size = size[0] // 2
            if rank == 0:
                param_attr = paddle.fluid.ParamAttr(
                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                        np_array[0:per_part_size, :]), )
            else:
                param_attr = paddle.fluid.ParamAttr(
                    initializer=paddle.fluid.initializer.NumpyArrayInitializer(
                        np_array[per_part_size:size[0], :]), )

            emb_out = paddle.distributed.split(
                data_in,
                size,
                operation="embedding",
                num_partitions=2,
                weight_attr=param_attr)

            return [data_in, emb_out]


if __name__ == "__main__":
    runtime_main(TestParallelEmbeddingAPI, "parallel_embedding")
