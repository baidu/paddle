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

import paddle.fluid as fluid
from utils import gen_data
from nets import mlp
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker

input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
input_y = fluid.layers.cast(input_y, dtype="float32")

with fluid.device_guard("gpu"):
    input_y = fluid.layers.cast(input_y, dtype="int64")
    cost = mlp(input_x, input_y)

optimizer = fluid.optimizer.Adagrad(learning_rate=0.01)

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)

optimizer = fleet.distributed_optimizer(optimizer)
optimizer.minimize(cost)

if fleet.is_server():
    fleet.init_server()
    fleet.run_server()
elif fleet.is_worker():
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fleet.startup_program)
    step = 1001
    for i in range(step):
        cost_val = exe.run(program=fleet.main_program,
                           feed=gen_data(),
                           fetch_list=[cost.name])
        print("worker_index: %d, step%d cost = %f" %
              (fleet.worker_index(), i, cost_val[0]))
