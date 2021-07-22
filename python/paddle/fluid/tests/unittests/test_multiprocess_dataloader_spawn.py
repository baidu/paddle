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

from __future__ import division

import os
import sys
import six
import time
import unittest
import multiprocessing
import numpy as np

import paddle.fluid as fluid
from paddle.io import Dataset, BatchSampler, DataLoader
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.dygraph.base import to_variable

from test_multiprocess_dataloader_static import RandomDataset, RandomBatchedDataset, prepare_places
from test_multiprocess_dataloader_static import EPOCH_NUM, BATCH_SIZE, IMAGE_SIZE, SAMPLE_NUM, CLASS_NUM


class SimpleFCNet(fluid.dygraph.Layer):
    def __init__(self):
        super(SimpleFCNet, self).__init__()

        param_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.8))
        bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=0.5))
        self._fcs = []
        in_channel = IMAGE_SIZE
        for hidden_size in [10, 20, 30]:
            self._fcs.append(
                Linear(
                    in_channel,
                    hidden_size,
                    act='tanh',
                    param_attr=param_attr,
                    bias_attr=bias_attr))
            in_channel = hidden_size
        self._fcs.append(
            Linear(
                in_channel,
                CLASS_NUM,
                act='softmax',
                param_attr=param_attr,
                bias_attr=bias_attr))

    def forward(self, image):
        out = image
        for fc in self._fcs:
            out = fc(out)
        return out


@unittest.skipIf(
    sys.version_info < (3, 4),
    "multiprocessing_context can only be set when Python version >= 3.4")
class TestDygraphDataLoaderSpawnStr(unittest.TestCase):
    def run_main(self, num_workers, places):
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1
        with fluid.dygraph.guard(places[0]):
            fc_net = SimpleFCNet()
            optimizer = fluid.optimizer.Adam(parameter_list=fc_net.parameters())

            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            multiprocessing_context = None if num_workers == 0 else 'spawn'
            dataloader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                drop_last=True,
                multiprocessing_context=multiprocessing_context)
            assert len(dataloader) == int(SAMPLE_NUM / BATCH_SIZE)

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in six.moves.range(EPOCH_NUM):
                step = 0
                for image, label in dataloader():
                    out = fc_net(image)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.reduce_mean(loss)
                    avg_loss.backward()
                    optimizer.minimize(avg_loss)
                    fc_net.clear_gradients()

                    loss_list.append(np.mean(avg_loss.numpy()))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list)
        }
        print("time cost", ret['time'], 'step_list', ret['step'])
        return ret

    def test_main(self):
        # dynamic graph do not run with_data_parallel
        for p in prepare_places(False):
            results = []
            for num_workers in [0, 2]:
                print(self.__class__.__name__, p, num_workers)
                sys.stdout.flush()
                ret = self.run_main(num_workers=num_workers, places=p)
                results.append(ret)
            diff = np.max(
                np.abs(results[0]['loss'] - results[1]['loss']) /
                np.abs(results[0]['loss']))
            self.assertLess(diff, 1e-2)


@unittest.skipIf(
    sys.version_info < (3, 4),
    "multiprocessing_context can only be set when Python version >= 3.4")
class TestDygraphDataLoaderSpawnContext(TestDygraphDataLoaderSpawnStr):
    def run_main(self, num_workers, places):
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1
        with fluid.dygraph.guard(places[0]):
            fc_net = SimpleFCNet()
            optimizer = fluid.optimizer.Adam(parameter_list=fc_net.parameters())

            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            multiprocessing_context = None if num_workers == 0 else \
                                        multiprocessing.get_context('spawn')
            dataloader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                drop_last=True,
                multiprocessing_context=multiprocessing_context)
            assert len(dataloader) == int(SAMPLE_NUM / BATCH_SIZE)

            step_list = []
            loss_list = []
            start_t = time.time()
            for _ in six.moves.range(EPOCH_NUM):
                step = 0
                for image, label in dataloader():
                    out = fc_net(image)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.reduce_mean(loss)
                    avg_loss.backward()
                    optimizer.minimize(avg_loss)
                    fc_net.clear_gradients()

                    loss_list.append(np.mean(avg_loss.numpy()))
                    step += 1
                step_list.append(step)

        end_t = time.time()
        ret = {
            "time": end_t - start_t,
            "step": step_list,
            "loss": np.array(loss_list)
        }
        print("time cost", ret['time'], 'step_list', ret['step'])
        return ret


@unittest.skipIf(
    sys.version_info < (3, 4),
    "multiprocessing_context can only be set when Python version >= 3.4")
class TestDygraphDataLoaderSpawnException(unittest.TestCase):
    def test_main(self):
        with fluid.dygraph.guard(fluid.cpu_places()[0]):
            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)

            # multiprocessing_context string wrong
            try:
                dataloader = DataLoader(
                    dataset,
                    num_workers=1,
                    batch_size=BATCH_SIZE,
                    drop_last=True,
                    multiprocessing_context='test')
                self.assertTrue(False)
            except ValueError:
                pass

            # num_workers = 0
            try:
                dataloader = DataLoader(
                    dataset,
                    num_workers=0,
                    batch_size=BATCH_SIZE,
                    drop_last=True,
                    multiprocessing_context='spawn')
                self.assertTrue(False)
            except ValueError:
                pass

            # multiprocessing_context not str
            # & not multiprocessing.context.BaseContext
            try:
                dataloader = DataLoader(
                    dataset,
                    num_workers=1,
                    batch_size=BATCH_SIZE,
                    drop_last=True,
                    multiprocessing_context=dataset)
                self.assertTrue(False)
            except TypeError:
                pass


if __name__ == '__main__':
    unittest.main()
