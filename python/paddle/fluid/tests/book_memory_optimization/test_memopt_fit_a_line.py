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

import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.device import get_places

# need to fix random seed and training data to compare the loss
# value accurately calculated by the default and the memory optimization
# version.
fluid.default_startup_program().random_seed = 111

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

device_type = 'CPU'
use_nccl = False
place = fluid.CPUPlace()
if fluid.core.is_compiled_with_cuda():
    device_type = 'CUDA'
    use_nccl = False
    place = fluid.CUDAPlace(0)

places = get_places(device_count=0, device_type=device_type)
pd = fluid.layers.ParallelDo(places, use_nccl=use_nccl)
with pd.do():
    x_ = pd.read_input(x)
    y_ = pd.read_input(y)
    y_predict = fluid.layers.fc(input=x_, size=1, act=None)
    cost = fluid.layers.square_error_cost(input=y_predict, label=y_)
    avg_cost = fluid.layers.mean(x=cost)
    pd.write_output(avg_cost)

cost = pd()
avg_cost = fluid.layers.mean(x=cost)
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

fluid.memory_optimize(fluid.default_main_program(), print_log=True)
# fluid.release_memory(fluid.default_main_program())

BATCH_SIZE = 200

# fix the order of training data
train_reader = paddle.batch(
    paddle.dataset.uci_housing.train(), batch_size=BATCH_SIZE, drop_last=False)

# train_reader = paddle.batch(
#     paddle.reader.shuffle(
#         paddle.dataset.uci_housing.train(), buf_size=500),
#     batch_size=BATCH_SIZE)

feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

PASS_NUM = 100
for pass_id in range(PASS_NUM):
    for data in train_reader():
        avg_loss_value, = exe.run(fluid.default_main_program(),
                                  feed=feeder.feed(data),
                                  fetch_list=[avg_cost])

        if avg_loss_value[0] < 10.0:
            exit(0)  # if avg cost less than 10.0, we think our code is good.
        print((avg_loss_value[0]))
        if math.isnan(float(avg_loss_value)):
            sys.exit("got NaN loss, training failed.")
exit(1)
