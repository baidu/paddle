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

from simple_nets import simple_fc_net, fc_with_batchnorm, init_data, bow_net
from fake_reader import fake_imdb_reader
from parallel_executor_test_base import TestParallelExecutorBase, DeviceType
from functools import partial
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import unittest
import os


class TestFuseOptimizationOps(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def _get_feed_dict(self):
        img, label = init_data()
        return {"image": img, "label": label}

    def _compare_fused_optimizer_ops(self,
                                     model,
                                     use_device,
                                     feed_dict=None,
                                     get_data_from_feeder=None,
                                     optimizer=fluid.optimizer.Adam):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        not_fuse_op_first_loss, not_fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict=feed_dict,
            get_data_from_feeder=get_data_from_feeder,
            use_device=use_device,
            fuse_all_optimizer_ops=False,
            optimizer=optimizer)
        fuse_op_first_loss, fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict=feed_dict,
            get_data_from_feeder=get_data_from_feeder,
            use_device=use_device,
            fuse_all_optimizer_ops=True,
            optimizer=optimizer)

        delta = 1e-4 if core.is_compiled_with_rocm() else 1e-6
        for loss in zip(not_fuse_op_first_loss, fuse_op_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta)
        for loss in zip(not_fuse_op_last_loss, fuse_op_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta)

    def _decorate_compare_fused_optimizer_ops(self, model, use_device,
                                              optimizer):
        self._compare_fused_optimizer_ops(
            model,
            use_device,
            feed_dict=self._get_feed_dict(),
            optimizer=optimizer)


class TestFuseAdamOps(TestFuseOptimizationOps):
    def optimizer(self, learning_rate=1e-4):
        return fluid.optimizer.Adam(learning_rate=learning_rate)

    def test_batchnorm_fc_with_fuse_op(self):
        self._decorate_compare_fused_optimizer_ops(
            fc_with_batchnorm, DeviceType.CUDA, optimizer=self.optimizer)
        self._decorate_compare_fused_optimizer_ops(
            fc_with_batchnorm, DeviceType.CPU, optimizer=self.optimizer)


class TestFuseSGDOps(TestFuseAdamOps):
    def optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.SGD(learning_rate=learning_rate)


class TestFuseMomentumOps(TestFuseAdamOps):
    def optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.Momentum(
            learning_rate=learning_rate, momentum=0.1)


class TestSpareFuseAdamOps(TestFuseOptimizationOps):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)
        cls.word_dict_len = 5147
        batch_size = 64
        reader = fake_imdb_reader(cls.word_dict_len, batch_size * 100)
        reader = paddle.batch(reader, batch_size=batch_size)()
        cls.train_data = next(reader)

    def _get_data_from_feeder(self):
        place = fluid.CPUPlace()
        feeder = fluid.DataFeeder(feed_list=["words", "label"], place=place)
        return feeder.feed(self.train_data)

    def _decorate_compare_fused_optimizer_ops(self, model, use_device,
                                              optimizer):
        self._compare_fused_optimizer_ops(
            model,
            use_device,
            get_data_from_feeder=self._get_data_from_feeder,
            optimizer=optimizer)

    def optimizer(self, learning_rate=1e-4):
        return fluid.optimizer.Adam(learning_rate=learning_rate)

    def test_simple_bow_net_with_fuse_op(self):
        model = partial(bow_net, dict_dim=self.word_dict_len, is_sparse=True)
        self._decorate_compare_fused_optimizer_ops(
            model, DeviceType.CUDA, optimizer=self.optimizer)
        self._decorate_compare_fused_optimizer_ops(
            model, DeviceType.CPU, optimizer=self.optimizer)


class TestSpareFuseSGDOps(TestSpareFuseAdamOps):
    def optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.SGD(learning_rate=learning_rate)


class TestSpareFuseMomentumOps(TestSpareFuseAdamOps):
    def optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.Momentum(
            learning_rate=learning_rate, momentum=0.1)


class TestPassConflictBase(TestFuseAdamOps):
    def _compare_fused_optimizer_ops(self,
                                     model,
                                     use_device,
                                     feed_dict=None,
                                     get_data_from_feeder=None,
                                     optimizer=fluid.optimizer.Adam):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        self.check_pass_conflict(
            model,
            feed_dict=feed_dict,
            get_data_from_feeder=get_data_from_feeder,
            use_device=use_device,
            fuse_all_optimizer_ops=True,
            optimizer=optimizer,
            enable_sequential_execution=True)


class TestFuseAdamOpsPassConflict(TestPassConflictBase):
    def optimizer(self, learning_rate=1e-4):
        return fluid.optimizer.Adam(learning_rate=learning_rate)

    def test_batchnorm_fc_with_fuse_op(self):
        self._decorate_compare_fused_optimizer_ops(
            fc_with_batchnorm, DeviceType.CPU, optimizer=self.optimizer)
        self._decorate_compare_fused_optimizer_ops(
            fc_with_batchnorm, DeviceType.CUDA, optimizer=self.optimizer)


class TestFuseSGDOpsPassConflict(TestFuseAdamOpsPassConflict):
    def optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.SGD(learning_rate=learning_rate)


class TestFuseMomentumOpsPassConflict(TestFuseAdamOpsPassConflict):
    def optimizer(self, learning_rate=1e-3):
        return fluid.optimizer.Momentum(
            learning_rate=learning_rate, momentum=0.1)


if __name__ == '__main__':
    unittest.main()
