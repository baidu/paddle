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
from paddle.io import IterableDataset, BatchSampler, DataLoader, get_worker_info

EPOCH_NUM = 2
BATCH_SIZE = 8
IMAGE_SIZE = 32
SAMPLE_NUM = 80
CLASS_NUM = 10


class RandomDataset(IterableDataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = sample_num
        self.class_num = class_num

    def __iter__(self):
        for i in range(self.sample_num):
            np.random.seed(i)
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, self.class_num - 1,
                                      (1, )).astype('int64')
            yield image, label


def simple_fc_net_static():
    startup_prog = fluid.Program()
    main_prog = fluid.Program()
    startup_prog.random_seed = 1
    main_prog.random_seed = 1

    with fluid.unique_name.guard():
        with fluid.program_guard(main_prog, startup_prog):
            image = fluid.data(
                name='image', shape=[None, IMAGE_SIZE], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            hidden = image
            param_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.8))
            bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5))
            for hidden_size in [10, 20, 30]:
                hidden = fluid.layers.fc(hidden,
                                         size=hidden_size,
                                         act='tanh',
                                         param_attr=param_attr,
                                         bias_attr=bias_attr)

            predict_label = fluid.layers.fc(hidden,
                                            size=CLASS_NUM,
                                            act='softmax',
                                            param_attr=param_attr,
                                            bias_attr=bias_attr)
            loss = fluid.layers.reduce_mean(
                fluid.layers.cross_entropy(
                    input=predict_label, label=label))

            optimizer = fluid.optimizer.Adam()
            optimizer.minimize(loss)
    return startup_prog, main_prog, image, label, loss


def prepare_places(with_data_parallel, with_cpu=False, with_gpu=True):
    places = []
    if with_cpu:
        places.append([fluid.CPUPlace()])
        if with_data_parallel:
            places.append([fluid.CPUPlace()] * 2)

    if with_gpu and fluid.core.is_compiled_with_cuda():
        tmp = fluid.cuda_places()[:2]
        assert len(tmp) > 0, "no gpu detected"
        if with_data_parallel and len(tmp) > 1:
            places.append(tmp)
        places.append([tmp[0]])
    return places


class TestStaticDataLoader(unittest.TestCase):
    def run_main(self, num_workers, places, persistent_workers):
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            startup_prog, main_prog, image, label, loss = simple_fc_net_static()

            dataset = RandomDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                feed_list=[image, label],
                places=places,
                num_workers=num_workers,
                batch_size=BATCH_SIZE,
                return_list=False,
                drop_last=True,
                persistent_workers=persistent_workers)
            # assert len(dataloader) == int(SAMPLE_NUM / BATCH_SIZE)

            exe = fluid.Executor(place=places[0])
            exe.run(startup_prog)

            prog = fluid.CompiledProgram(main_prog)
            if len(places) > 1:
                prog = prog.with_data_parallel(
                    loss_name=loss.name, places=places)

            step_list = []
            loss_list = []
            start_t = time.time()
            for i in six.moves.range(EPOCH_NUM):
                step = 0
                for d in dataloader:
                    assert len(d) == len(places), "{} != {}".format(
                        len(d), len(places))
                    for i, item in enumerate(d):
                        image = item['image']
                        label = item['label']
                        assert image.shape() == [BATCH_SIZE, IMAGE_SIZE]
                        assert label.shape() == [BATCH_SIZE, 1]
                        assert image._place()._equals(places[i])
                        assert label._place()._equals(places[i])
                    L, = exe.run(program=prog,
                                 feed=d,
                                 fetch_list=[loss],
                                 use_program_cache=True)
                    loss_list.append(np.mean(L))
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
        for p in prepare_places(True):
            for persistent_workers in [False, True]:
                results = []
                for num_workers in [0, 2]:
                    print(self.__class__.__name__, p, num_workers,
                          persistent_workers)
                    sys.stdout.flush()
                    ret = self.run_main(
                        num_workers=num_workers,
                        places=p,
                        persistent_workers=persistent_workers)
                    results.append(ret)
                assert results[0]['loss'].shape[0] * 2 == results[1][
                    'loss'].shape[0]


class RandomBatchedDataset(IterableDataset):
    def __init__(self, sample_num, class_num):
        self.sample_num = sample_num // BATCH_SIZE
        self.class_num = class_num

    def __iter__(self):
        for i in range(self.sample_num):
            np.random.seed(i)
            images = []
            labels = []
            for _ in range(BATCH_SIZE):
                image = np.random.random([IMAGE_SIZE]).astype('float32')
                label = np.random.randint(0, self.class_num - 1,
                                          (1, )).astype('int64')
                images.append(image)
                labels.append(label)
            yield np.stack(images, axis=0), np.stack(labels, axis=0)


class TestStaticDataLoaderWithBatchedDataset(TestStaticDataLoader):
    def run_main(self, num_workers, places, persistent_workers):
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            startup_prog, main_prog, image, label, loss = simple_fc_net_static()

            dataset = RandomBatchedDataset(SAMPLE_NUM, CLASS_NUM)
            dataloader = DataLoader(
                dataset,
                feed_list=[image, label],
                places=places,
                num_workers=num_workers,
                batch_size=None,
                return_list=False,
                drop_last=True,
                persistent_workers=persistent_workers)

            exe = fluid.Executor(place=places[0])
            exe.run(startup_prog)

            prog = fluid.CompiledProgram(main_prog)
            if len(places) > 1:
                prog = prog.with_data_parallel(
                    loss_name=loss.name, places=places)

            step_list = []
            loss_list = []
            start_t = time.time()
            for i in six.moves.range(EPOCH_NUM):
                step = 0
                for d in dataloader:
                    assert len(d) == len(places), "{} != {}".format(
                        len(d), len(places))
                    for i, item in enumerate(d):
                        image = item['image']
                        label = item['label']
                        assert image.shape() == [BATCH_SIZE, IMAGE_SIZE]
                        assert label.shape() == [BATCH_SIZE, 1]
                        assert image._place()._equals(places[i])
                        assert label._place()._equals(places[i])
                    L, = exe.run(program=prog,
                                 feed=d,
                                 fetch_list=[loss],
                                 use_program_cache=True)
                    loss_list.append(np.mean(L))
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


if __name__ == '__main__':
    unittest.main()
