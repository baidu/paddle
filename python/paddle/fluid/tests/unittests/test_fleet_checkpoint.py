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

import unittest
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.collective import CollectiveOptimizer, fleet, TrainStatus
import os


class FleetTest(unittest.TestCase):
    def test_check_point(self):
        dir_path = "./my_paddle_model"
        file_name = "persistables"

        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:6070"

        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

        image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        feeder = fluid.DataFeeder(
            feed_list=[image, label], place=fluid.CPUPlace())
        predict = fluid.layers.fc(input=image, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=predict, label=label)
        avg_loss = fluid.layers.mean(loss)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)

        dist_optimizer = fleet.distributed_optimizer(optimizer)
        dist_optimizer.minimize(avg_loss)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        status = TrainStatus(2)
        fleet.save_check_point(exe, dir_path, train_status=status)
        n1 = fleet._get_last_checkpoint_no(dir_path)

        status2 = fleet.load_check_point(exe, dir_path)
        assert status2 == status, "Checkpoint error!"

        fleet.save_check_point(exe, dir_path, train_status=status)
        n2 = fleet._get_last_checkpoint_no(dir_path)
        assert n2 == n1 + 1, "checkpoint increment error:{} {}".format(n1, n2)

        fleet.clean_redundant_check_points(dir_path)


if __name__ == '__main__':
    unittest.main()
