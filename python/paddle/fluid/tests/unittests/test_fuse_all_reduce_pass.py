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
import paddle.fluid as fluid
import paddle.fluid.core as core
from functools import partial
import paddle
import unittest
import os

paddle.enable_static()


class TestFuseAllReduceOpsBase(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def compare_fuse_all_reduce_ops(self,
                                    model,
                                    use_device,
                                    init_feed_dict=None,
                                    get_data_from_feeder=None,
                                    optimizer=None,
                                    fuse_all_optimizer_ops=False):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return
        if use_device == DeviceType.XPU and not core.is_compiled_with_xpu():
            return

        feed_dict_data = None
        if init_feed_dict is not None:
            img, label = init_feed_dict()
            feed_dict_data = {"image": img, "label": label}

        not_fuse_op_first_loss, not_fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict=feed_dict_data,
            get_data_from_feeder=get_data_from_feeder,
            use_device=use_device,
            fuse_all_reduce_ops=False,
            fuse_all_optimizer_ops=fuse_all_optimizer_ops,
            optimizer=optimizer)
        fuse_op_first_loss, fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict=feed_dict_data,
            get_data_from_feeder=get_data_from_feeder,
            use_device=use_device,
            fuse_all_reduce_ops=True,
            fuse_all_optimizer_ops=fuse_all_optimizer_ops,
            optimizer=optimizer)

        delta = 1e-5 if core.is_compiled_with_rocm() else 1e-6
        for loss in zip(not_fuse_op_first_loss, fuse_op_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta)
        for loss in zip(not_fuse_op_last_loss, fuse_op_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=delta)

    def optimizer(self, learning_rate=1e-3):
        optimizer = fluid.optimizer.SGD(
            learning_rate=learning_rate,
            regularization=fluid.regularizer.L2Decay(1e-3))
        return optimizer


class TestFuseAllReduceOps(TestFuseAllReduceOpsBase):
    def _decorate_compare_fused_all_reduce(self, model, use_device):
        self.compare_fuse_all_reduce_ops(
            model,
            use_device,
            init_feed_dict=init_data,
            optimizer=self.optimizer,
            fuse_all_optimizer_ops=True)

    def test_simple_fc_with_fuse_all_reduce(self):
        self._decorate_compare_fused_all_reduce(simple_fc_net, DeviceType.CUDA)
        self._decorate_compare_fused_all_reduce(simple_fc_net, DeviceType.XPU)
        self._decorate_compare_fused_all_reduce(simple_fc_net, DeviceType.CPU)

    def test_batchnorm_fc_with_fuse_all_reduce(self):
        self._decorate_compare_fused_all_reduce(fc_with_batchnorm,
                                                DeviceType.CUDA)
        # TODO(wangxi): xpu batch_norm op only support dim = 4
        # self._decorate_compare_fused_all_reduce(fc_with_batchnorm,
        #                                         DeviceType.XPU)
        self._decorate_compare_fused_all_reduce(fc_with_batchnorm,
                                                DeviceType.CPU)


class TestFuseAllReduceOpsAndOptiOps(TestFuseAllReduceOps):
    def _decorate_compare_fused_all_reduce(self, model, use_device):
        self.compare_fuse_all_reduce_ops(
            model,
            use_device,
            init_feed_dict=init_data,
            optimizer=self.optimizer,
            fuse_all_optimizer_ops=True)


class TestFuseAllReduceOpsWithSparseGrad(TestFuseAllReduceOpsBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)
        cls.word_dict_len = 5147
        batch_size = 64
        reader = fake_imdb_reader(cls.word_dict_len, batch_size * 100)
        reader = paddle.batch(reader, batch_size=batch_size)()
        cls.train_data = next(reader)

    def get_data_from_feeder(self):
        place = fluid.CPUPlace()
        feeder = fluid.DataFeeder(feed_list=["words", "label"], place=place)
        return feeder.feed(self.train_data)

    def _decorate_compare_fused_all_reduce(self, model, use_device):
        self.compare_fuse_all_reduce_ops(
            model,
            use_device,
            get_data_from_feeder=self.get_data_from_feeder,
            optimizer=self.optimizer)

    def test_simple_bow_net_with_fuse_all_reduce(self):
        model = partial(bow_net, dict_dim=self.word_dict_len, is_sparse=True)
        self._decorate_compare_fused_all_reduce(model, DeviceType.CUDA)
        # TODO(wangxi): xpu sum op only support LodTensor for now
        # self._decorate_compare_fused_all_reduce(model, DeviceType.XPU)
        self._decorate_compare_fused_all_reduce(model, DeviceType.CPU)


if __name__ == '__main__':
    unittest.main()
