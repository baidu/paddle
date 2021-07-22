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

from __future__ import print_function

import unittest

import contextlib
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import unique_name
from test_imperative_base import new_program_scope
from jit_load_rename_var import rename_var_with_generator

import paddle.fluid.transpiler.details.program_utils as pu

LOADED_VAR_SUFFIX = ".load_0"

paddle.enable_static()

def while_softmax_regression(img):
    def cond(i, times, pred):
        return i < times

    def body(i, times, pred):
        pred = fluid.layers.fc(input=pred, size=10, act='softmax')
        i = i + 1
        return [i, times, pred]

    i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
    times = fluid.layers.fill_constant(shape=[1], dtype='int64', value=5)
    pred = fluid.layers.fc(input=img, size=10, act='softmax')
    i, times, pred = fluid.layers.while_loop(
        cond=cond, body=body, loop_vars=[i, times, pred])
    return pred


class TestImperativeStaticModelRunnerWhile(unittest.TestCase):
    def setUp(self):
        self.seed = 90
        self.batch_size = 32
        self.batch_num = 50
        self.save_dirname = "while.inference.model"
        self.model_filename = None
        self.params_filename = None

    def _random_batch_reader(self):
        def _get_random_images_and_labels(image_shape, label_shape):
            image = np.random.random(size=image_shape).astype('float32')
            label = np.random.random(size=label_shape).astype('int64')
            return image, label

        def __reader__():
            for _ in range(self.batch_num):
                batch_image, batch_label = _get_random_images_and_labels(
                    [self.batch_size, 784], [self.batch_size, 1])
                yield batch_image, batch_label

        return __reader__

    def train_and_save_model(self):
        startup_program = fluid.default_startup_program()
        main_program = fluid.default_main_program()

        img = fluid.data(name='img', shape=[None, 784], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')

        pred = while_softmax_regression(img)

        loss = fluid.layers.cross_entropy(input=pred, label=label)
        avg_loss = fluid.layers.mean(loss)

        optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        optimizer.minimize(avg_loss)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

        exe = fluid.Executor(place)
        exe.run(startup_program)

        loader = fluid.io.DataLoader.from_generator(
            feed_list=[img, label], capacity=5, iterable=True)
        loader.set_batch_generator(self._random_batch_reader(), places=place)

        for data in loader():
            exe.run(main_program, feed=data, fetch_list=[avg_loss])

        fluid.io.save_inference_model(
            self.save_dirname, ["img"], [pred],
            exe,
            model_filename=self.model_filename,
            params_filename=self.params_filename)

    def load_and_train_dygraph(self):
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            np.random.seed(self.seed)
            fluid.set_flags({'FLAGS_sort_sum_gradient': True})

            while_net = fluid.dygraph.static_runner.StaticModelRunner(
                self.save_dirname)

            dy_param_init_value = {}
            for param in while_net.parameters():
                dy_param_init_value[param.name] = param.numpy()

            sgd = fluid.optimizer.SGD(learning_rate=0.001,
                                      parameter_list=while_net.parameters())

            train_loader = fluid.io.DataLoader.from_generator(capacity=10)
            train_loader.set_batch_generator(
                self._random_batch_reader(), places=place)

            while_net.train()

            for data in train_loader():
                img = data[0]
                label = data[1]
                label.stop_gradient = True

                cost = while_net(img)

                loss = fluid.layers.cross_entropy(cost, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                sgd.minimize(avg_loss)
                while_net.clear_gradients()

            dy_out = avg_loss.numpy()
            dy_param_value = {}
            for param in while_net.parameters():
                dy_param_value[param.name] = param.numpy()

        return dy_out, dy_param_init_value, dy_param_value

    def load_and_train_static(self):
        with new_program_scope():
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            np.random.seed(self.seed)

            img = fluid.data(name='img', shape=[None, 784], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            pred = while_softmax_regression(img)

            loss = fluid.layers.cross_entropy(input=pred, label=label)
            avg_loss = fluid.layers.mean(loss)

            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            optimizer.minimize(avg_loss)

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            fluid.io.load_params(
                exe,
                self.save_dirname,
                main_program=fluid.default_main_program(),
                filename=self.params_filename)

            static_param_init_value = {}
            static_param_name_list = []
            for param in fluid.default_main_program().all_parameters():
                static_param_name_list.append(param.name)
                static_param_init_value[param.name] = fluid.executor._fetch_var(
                    param.name)

            loader = fluid.io.DataLoader.from_generator(
                feed_list=[img, label], capacity=5, iterable=True)
            loader.set_batch_generator(
                self._random_batch_reader(), places=place)

            for data in loader():
                fetch_list = [avg_loss.name]
                fetch_list.extend(static_param_name_list)

                out = exe.run(fluid.default_main_program(),
                              feed=data,
                              fetch_list=[avg_loss])

            static_param_value = {}
            static_out = out[0]
            for i in range(1, len(out)):
                static_param_value[static_param_name_list[i - 1]] = out[i]

        return static_out, static_param_init_value, static_param_value

    def test_while_no_params_filename(self):
        # Phase 1. run and save static model
        self.train_and_save_model()

        # # Phase 2. load model & train dygraph
        with unique_name.guard():
            dy_out, dy_param_init_value, dy_param_value = \
            self.load_and_train_dygraph()

        with unique_name.guard():
            static_out, static_param_init_value, static_param_value = \
                self.load_and_train_static()

        # Phase 3. compare
        with unique_name.guard():
            dict_old_new_init = rename_var_with_generator(
                static_param_init_value.keys())
        for key, value in six.iteritems(static_param_init_value):
            key = dict_old_new_init[key]
            self.assertTrue(np.array_equal(value, dy_param_init_value[key]))

        self.assertTrue(np.allclose(static_out, dy_out))

        for key, value in six.iteritems(static_param_value):
            key += LOADED_VAR_SUFFIX
            self.assertTrue(np.allclose(value, dy_param_value[key], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
