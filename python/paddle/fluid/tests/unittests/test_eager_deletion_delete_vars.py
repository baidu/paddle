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
import numpy as np
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['CPU_NUM'] = '4'

import paddle.fluid as fluid
import six
import unittest
import multiprocessing

fluid.core._set_eager_deletion_mode(0.0, 1.0, True)


def simple_fc_net():
    image = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = image
    for _ in range(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
    optimizer.minimize(loss)
    return image, label, loss


def get_persistables_and_non_persistables(prog, fetch_list):
    num_block = prog.num_blocks
    persitables = set()
    non_persistables = set()
    for bid in six.moves.range(num_block):
        block = prog.block(bid)
        for _, var in block.vars.items():
            if var.persistable or var.name in fetch_list:
                persitables.add(var.name)
            else:
                non_persistables.add(var.name)

    return persitables, non_persistables


class TestExecutor(unittest.TestCase):
    def test_executor_main(self):
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for p in places:
            self.place = p
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                with fluid.scope_guard(fluid.Scope()):
                    with fluid.unique_name.guard():
                        self.executor_main()

        for p in places:
            self.place = p
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                with fluid.scope_guard(fluid.Scope()):
                    with fluid.unique_name.guard():
                        self.pe_main()

    def prepare_feed(self, image, label, dev_cnt=1):
        batch_size = 32 * dev_cnt
        image_shape = (batch_size, ) + tuple(image.shape[1:])
        label_shape = (batch_size, ) + tuple(label.shape[1:])

        image_np = np.random.random(size=image_shape).astype('float32')
        label_np = np.random.random_integers(
            low=0, high=9, size=label_shape).astype('int64')

        return image_np, label_np

    def assertScopeVar(self, scope, persitables, non_persistables):
        outline_p_vars = []
        for name in persitables:
            var = scope.find_var(name)
            self.assertTrue(var is not None)
            t = var.get_tensor()
            if not t._is_initialized():
                outline_p_vars.append(name)

        outline_np_vars = []
        for name in non_persistables:
            var = scope.find_var(name)
            self.assertTrue(var is not None)
            t = var.get_tensor()
            if t._is_initialized():
                outline_np_vars.append(name)

        print('Non-alive persistable vars {} in {}'.format(outline_p_vars,
                                                           persitables))
        print('Alive non-persistable vars {} in {}'.format(outline_np_vars,
                                                           non_persistables))
        self.assertEqual(len(outline_p_vars), 0)
        self.assertEqual(len(outline_np_vars), 0)

    def executor_main(self):
        image, label, loss = simple_fc_net()
        loss.persistable = False
        persistables, non_persistables = get_persistables_and_non_persistables(
            fluid.default_main_program(), [loss.name])
        print('Non-persistable var number {}'.format(len(non_persistables)))
        print(non_persistables)

        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())

        p = fluid.core.Place()
        p.set_place(self.place)
        exe = fluid.core.Executor(p)

        for _ in six.moves.range(10):
            image_np, label_np = self.prepare_feed(image, label)
            fluid.global_scope().var(image.name).get_tensor().set(image_np,
                                                                  self.place)
            fluid.global_scope().var(label.name).get_tensor().set(label_np,
                                                                  self.place)
            # exe.run would not create local scope
            # so that we can detect whether gc clears temporary variables
            exe.run(fluid.default_main_program().desc,
                    fluid.global_scope(), 0, False, True, [loss.name])
            self.assertScopeVar(fluid.global_scope(), persistables,
                                non_persistables)

    def pe_main(self):
        image, label, loss = simple_fc_net()
        loss.persistable = False
        persistables, non_persistables = get_persistables_and_non_persistables(
            fluid.default_main_program(), [loss.name])

        exe = fluid.Executor(self.place)
        exe.run(fluid.default_startup_program())

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_iteration_per_drop_scope = 100

        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False

        prog = fluid.CompiledProgram(fluid.default_main_program(
        )).with_data_parallel(
            loss_name=loss.name, exec_strategy=exec_strategy)

        dev_cnt = fluid.core.get_cuda_device_count() if isinstance(self.place, fluid.CUDAPlace)    \
            else int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

        for idx in six.moves.range(10):
            image_np, label_np = self.prepare_feed(image, label, dev_cnt)
            feed = {image.name: image_np, label.name: label_np}

            exe.run(program=prog, feed=feed, fetch_list=[loss])

            local_scopes = prog._local_scopes
            for scope in local_scopes:
                kids = scope._kids()
                self.assertTrue(len(kids) == 1)
                self.assertScopeVar(kids[0], persistables, non_persistables)


if __name__ == '__main__':
    unittest.main()
