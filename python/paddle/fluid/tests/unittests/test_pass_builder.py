# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from simple_nets import simple_fc_net
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler
import numpy as np
import unittest
import os
import sys
import math
import tempfile


class TestPassBuilder(unittest.TestCase):
    def check_network_convergence(self, use_cuda, build_strategy=None):
        os.environ['CPU_NUM'] = str(4)
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            loss = simple_fc_net()
            test_program = main.clone(for_test=True)

            opt = fluid.optimizer.SGD(learning_rate=0.001)
            opt.minimize(loss)

            batch_size = 32
            image = np.random.normal(size=(batch_size, 784)).astype('float32')
            label = np.random.randint(0, 10, (batch_size, 1), dtype="int64")

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup)
            feed_dict = {'image': image, 'label': label}

            train_cp = compiler.CompiledProgram(main).with_data_parallel(
                loss_name=loss.name, build_strategy=build_strategy)
            test_cp = compiler.CompiledProgram(test_program).with_data_parallel(
                loss_name=loss.name,
                build_strategy=build_strategy,
                share_vars_from=train_cp)

            for i in range(5):
                _ = exe.run(train_cp, fetch_list=[loss.name], feed=feed_dict)
                test_loss, = exe.run(test_cp,
                                     fetch_list=[loss.name],
                                     feed=feed_dict)
                train_loss = exe.run(train_cp,
                                     fetch_list=[loss.name],
                                     feed=feed_dict)

                avg_test_loss_val = np.array(test_loss).mean()
                if math.isnan(float(avg_test_loss_val)):
                    sys.exit("got NaN loss, testing failed.")

                avg_train_loss_val = np.array(train_loss).mean()
                if math.isnan(float(avg_train_loss_val)):
                    sys.exit("got NaN loss, training failed.")

                self.assertTrue(
                    np.allclose(
                        train_loss, test_loss, atol=1e-8),
                    "Train loss: " + str(train_loss) + "\n Test loss:" +
                    str(test_loss))

    def test_parallel_testing_with_new_strategy(self):
        build_strategy = fluid.BuildStrategy()
        self.assertFalse(build_strategy.fuse_elewise_add_act_ops)
        build_strategy.fuse_elewise_add_act_ops = True
        #FIXME: currently fuse_elewise_add_act_ops not compatible with below options
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False
        pass_builder = build_strategy._finalize_strategy_and_create_passes()
        self.assertTrue("fuse_elewise_add_act_pass" in
                        [p.type() for p in pass_builder.all_passes()])

        origin_len = len(pass_builder.all_passes())

        viz_pass = pass_builder.append_pass("graph_viz_pass")
        self.assertEqual(origin_len + 1, len(pass_builder.all_passes()))

        pass_builder.insert_pass(
            len(pass_builder.all_passes()), "graph_viz_pass")
        self.assertEqual(origin_len + 2, len(pass_builder.all_passes()))

        pass_builder.remove_pass(len(pass_builder.all_passes()) - 1)
        self.assertEqual(origin_len + 1, len(pass_builder.all_passes()))
        with tempfile.TemporaryDirectory(prefix="dot_path_") as tmpdir:
            graph_viz_path = os.path.join(tmpdir, 'test_viz_pass.dot')
            viz_pass.set("graph_viz_path", graph_viz_path)

            self.check_network_convergence(
                use_cuda=core.is_compiled_with_cuda(),
                build_strategy=build_strategy)
            try:
                os.stat(graph_viz_path)
            except os.error:
                self.assertFalse(True)


if __name__ == '__main__':
    unittest.main()
