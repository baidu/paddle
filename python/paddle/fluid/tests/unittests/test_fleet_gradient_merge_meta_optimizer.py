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
import paddle
import os
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base.role_maker as role_maker

from fleet_meta_optimizer_base import TestFleetMetaOptimizer

paddle.enable_static()


class TestFleetGradientMergeMetaOptimizer(TestFleetMetaOptimizer):
    def test_gradient_merge_optimizer(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'gradient_merge')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        vars = [x.name for x in train_prog.list_vars()]
        with open("main_program", 'w') as f:
            f.write(str(train_prog))

        self.assertIn('@GradientMerge', ''.join(vars))

    def test_recom_gm_optimizer(self):
        train_prog, startup_prog = paddle.fluid.Program(), paddle.fluid.Program(
        )
        avg_cost, strategy = self.net(train_prog, startup_prog)
        self.set_strategy(strategy, 'gradient_merge')
        self.set_strategy(strategy, 'recompute')
        self.optimizer(avg_cost, strategy, train_prog, startup_prog)

        vars = [x.name for x in train_prog.list_vars()]
        self.assertIn('@GradientMerge', ''.join(vars))
        self.assertIn('subprog', ''.join(vars))


if __name__ == "__main__":
    unittest.main()
