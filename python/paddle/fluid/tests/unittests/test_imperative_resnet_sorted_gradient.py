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

import unittest
import numpy as np
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope
from test_imperative_resnet import ResNet

batch_size = 8
train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": batch_size,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    },
    "batch_size": batch_size,
    "lr": 0.1,
    "total_images": 1281164,
}


def optimizer_setting(params, parameter_list=None):
    ls = params["learning_strategy"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        if fluid.in_dygraph_mode():
            optimizer = fluid.optimizer.SGD(learning_rate=0.01,
                                            parameter_list=parameter_list)
        else:
            optimizer = fluid.optimizer.SGD(learning_rate=0.01)
        # TODO(minqiyang): Add learning rate scheduler support to dygraph mode
        #  optimizer = fluid.optimizer.Momentum(
        #  learning_rate=params["lr"],
        #  learning_rate=fluid.layers.piecewise_decay(
        #  boundaries=bd, values=lr),
        #  momentum=0.9,
        #  regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer


class TestDygraphResnetSortGradient(unittest.TestCase):
    def test_resnet_sort_gradient_float32(self):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 10
        with fluid.dygraph.guard():
            fluid.set_flags({'FLAGS_sort_sum_gradient': True})
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            resnet = ResNet()
            optimizer = optimizer_setting(
                train_parameters, parameter_list=resnet.parameters())
            np.random.seed(seed)
            import random
            random.seed = seed
            train_reader = paddle.batch(
                paddle.dataset.flowers.train(use_xmap=False),
                batch_size=batch_size)

            dy_param_init_value = {}
            for param in resnet.parameters() + resnet.buffers():
                dy_param_init_value[param.name] = param.numpy()

            for batch_id, data in enumerate(train_reader()):
                if batch_id >= batch_num:
                    break

                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    batch_size, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)
                label.stop_gradient = True

                out = resnet(img)
                loss = fluid.layers.cross_entropy(input=out, label=label)
                avg_loss = fluid.layers.mean(x=loss)

                dy_out = avg_loss.numpy()

                if batch_id == 0:
                    for param in resnet.parameters() + resnet.buffers():
                        if param.name not in dy_param_init_value:
                            dy_param_init_value[param.name] = param.numpy()

                avg_loss.backward()

                dy_grad_value = {}
                for param in resnet.parameters():
                    if param.trainable:
                        np_array = np.array(param._grad_ivar().value()
                                            .get_tensor())
                        dy_grad_value[param.name + core.grad_var_suffix(
                        )] = np_array

                optimizer.minimize(avg_loss)
                resnet.clear_gradients()

                dy_param_value = {}
                for param in resnet.parameters() + resnet.buffers():
                    dy_param_value[param.name] = param.numpy()

        with new_program_scope():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))

            resnet = ResNet()
            optimizer = optimizer_setting(train_parameters)

            np.random.seed(seed)
            import random
            random.seed = seed
            train_reader = paddle.batch(
                paddle.dataset.flowers.train(use_xmap=False),
                batch_size=batch_size)

            img = fluid.layers.data(
                name='pixel', shape=[3, 224, 224], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            out = resnet(img)
            loss = fluid.layers.cross_entropy(input=out, label=label)
            avg_loss = fluid.layers.mean(x=loss)
            optimizer.minimize(avg_loss)

            # initialize params and fetch them
            static_param_init_value = {}
            static_param_name_list = []
            static_grad_name_list = []
            for param in resnet.parameters():
                static_param_name_list.append(param.name)
            for param in resnet.parameters():
                if param.trainable:
                    static_grad_name_list.append(param.name +
                                                 core.grad_var_suffix())

            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            for batch_id, data in enumerate(train_reader()):
                if batch_id >= batch_num:
                    break

                static_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    [batch_size, 1])

                fetch_list = [avg_loss.name]
                fetch_list.extend(static_param_name_list)
                fetch_list.extend(static_grad_name_list)
                out = exe.run(fluid.default_main_program(),
                              feed={"pixel": static_x_data,
                                    "label": y_data},
                              fetch_list=fetch_list)

                static_param_value = {}
                static_grad_value = {}
                static_out = out[0]
                param_start_pos = 1
                grad_start_pos = len(static_param_name_list) + param_start_pos
                for i in range(param_start_pos,
                               len(static_param_name_list) + param_start_pos):
                    static_param_value[static_param_name_list[
                        i - param_start_pos]] = out[i]
                for i in range(grad_start_pos,
                               len(static_grad_name_list) + grad_start_pos):
                    static_grad_value[static_grad_name_list[
                        i - grad_start_pos]] = out[i]

        self.assertTrue(np.allclose(static_out, dy_out))

        self.assertEqual(len(dy_param_init_value), len(static_param_init_value))

        for key, value in six.iteritems(static_param_init_value):
            self.assertTrue(np.allclose(value, dy_param_init_value[key]))
            self.assertTrue(np.isfinite(value.all()))
            self.assertFalse(np.isnan(value.any()))

        self.assertEqual(len(dy_grad_value), len(static_grad_value))
        for key, value in six.iteritems(static_grad_value):
            self.assertTrue(np.allclose(value, dy_grad_value[key]))
            self.assertTrue(np.isfinite(value.all()))
            self.assertFalse(np.isnan(value.any()))

        self.assertEqual(len(dy_param_value), len(static_param_value))
        for key, value in six.iteritems(static_param_value):
            self.assertTrue(np.allclose(value, dy_param_value[key]))
            self.assertTrue(np.isfinite(value.all()))
            self.assertFalse(np.isnan(value.any()))


if __name__ == '__main__':
    unittest.main()
