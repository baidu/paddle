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
import numpy as np
import os
import sys
import six

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.fluid as fluid
from paddle.fluid.optimizer import Adam
import paddle.fluid.framework as framework
from test_imperative_base import new_program_scope

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4
SEED = 10

IMAGE_SIZE = 784
CLASS_NUM = 10

if six.PY2:
    LARGE_PARAM = 2**20
else:
    LARGE_PARAM = 2**26


def random_batch_reader():
    def _get_random_inputs_and_labels():
        np.random.seed(SEED)
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (
            BATCH_SIZE,
            1, )).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_inputs_and_labels()
            batch_image = paddle.to_tensor(batch_image)
            batch_label = paddle.to_tensor(batch_label)
            yield batch_image, batch_label

    return __reader__


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)


class LayerWithLargeParameters(paddle.nn.Layer):
    def __init__(self):
        super(LayerWithLargeParameters, self).__init__()
        self._l = paddle.nn.Linear(10, LARGE_PARAM)

    def forward(self, x):
        y = self._l(x)
        return y


def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()


class TestSaveLoadLargeParameters(unittest.TestCase):
    def setUp(self):
        pass

    def test_large_parameters_paddle_save(self):
        # enable dygraph mode
        paddle.disable_static()
        # create network
        layer = LayerWithLargeParameters()
        save_dict = layer.state_dict()

        path = os.path.join("test_paddle_save_load_large_param_save",
                            "layer.pdparams")
        paddle.save(layer.state_dict(), path)
        dict_load = paddle.load(path)
        # compare results before and after saving
        for key, value in save_dict.items():
            self.assertTrue(np.array_equal(dict_load[key], value.numpy()))


class TestSaveLoadPickle(unittest.TestCase):
    def test_pickle_protocol(self):
        # enable dygraph mode
        paddle.disable_static()
        # create network
        layer = LinearNet()
        save_dict = layer.state_dict()

        path = os.path.join("test_paddle_save_load_pickle_protocol",
                            "layer.pdparams")

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 2.0)

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 1)

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 5)

        protocols = [2, ]
        if sys.version_info.major >= 3 and sys.version_info.minor >= 4:
            protocols += [3, 4]
        for protocol in protocols:
            paddle.save(save_dict, path, protocol)
            dict_load = paddle.load(path)
            # compare results before and after saving
            for key, value in save_dict.items():
                self.assertTrue(np.array_equal(dict_load[key], value.numpy()))


class TestSaveLoadAny(unittest.TestCase):
    def set_zero(self, prog, place):
        for var in prog.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                ten = fluid.global_scope().find_var(var.name).get_tensor()
                ten.set(np.zeros_like(np.array(ten)), place)

                new_t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                self.assertTrue(np.sum(np.abs(new_t)) == 0)

    def test_save_load_var_list_dygraph(self):
        # enable dygraph mode
        paddle.disable_static()
        layer = LinearNet()
        var_list = [v for k, v in layer.state_dict().items()]

        path = os.path.join("test_paddle_save_load_varlist", "layer.pdparams")

        protocols = [2, ]
        if sys.version_info.major >= 3 and sys.version_info.minor >= 4:
            protocols += [3, 4]
        for protocol in protocols:
            paddle.save(var_list, path, protocol)
            var_list_load = paddle.load(path)
            # compare results before and after saving
            for i, value in enumerate(var_list):
                self.assertTrue(np.array_equal(var_list_load[i], value.numpy()))

        paddle.save(var_list, path)
        var_list_load = paddle.load(path, return_tensor=True)
        # compare results before and after saving
        for i, value in enumerate(var_list):
            self.assertTrue(
                np.array_equal(var_list_load[i].numpy(), value.numpy()))

    def replace_static_save(self, program, model_path, pickle_protocol=2):
        with self.assertRaises(TypeError):
            program.state_dict(1)

        with self.assertRaises(ValueError):
            program.state_dict('x')

        state_dict_param = program.state_dict('param')
        paddle.save(state_dict_param, model_path + '.pdparams')

        state_dict_opt = program.state_dict('opt')
        paddle.save(state_dict_opt, model_path + '.pdopt')

        # paddle.save(program, model_path + ".pdmodel")

    def replace_static_load(self, program, model_path):
        with self.assertRaises(TypeError):
            program.set_state_dict(1)

        state_dict_param = paddle.load(model_path + '.pdparams')

        # UserWarning: Skip loading for fake_var_name.@@. Can not find Variable 'fake_var_name.@@' in the program.
        state_dict_param['fake_var_name.@@'] = np.random.randn(1, 2)

        program.set_state_dict(state_dict_param)

        state_dict_opt = paddle.load(model_path + '.pdopt')
        program.set_state_dict(state_dict_opt)

    def replace_save_vars(self, program, dirname):
        def predicate(var):
            return var.persistable

        vars_name = [var.name for var in filter(predicate, program.list_vars())]

        with self.assertRaises(TypeError):
            program.get_var(1)

        with self.assertRaises(ValueError):
            paddle.save(
                1, os.path.join(dirname, 'test_var'), use_binary_format=True)

        for name in vars_name:
            var = program.get_var(name)
            paddle.save(
                var, os.path.join(dirname, name), use_binary_format=True)

    def replace_load_vars(self, program, dirname):
        def predicate(var):
            return var.persistable

        var_list = list(filter(predicate, program.list_vars()))

        temp_tensor = np.zeros(var_list[0].shape, dtype='float32')

        with self.assertRaises(TypeError):
            program.set_var(var_list[0].name, temp_tensor, 6.66)

        with self.assertRaises(TypeError):
            program.set_var(1, temp_tensor)

        with self.assertRaises(TypeError):
            program.set_var(var_list[0].name, 6.66)

        # set non-existent variable 
        with self.assertRaises(ValueError):
            program.set_var('non-existent_@@@_...', temp_tensor)

        # mismatched shape
        with self.assertRaises(ValueError):
            fake_tensor = np.zeros((3, 2, 1, 2, 3), dtype='float32')
            program.set_var(var_list[0].name, fake_tensor)

        for var in var_list:
            var_load = paddle.load(os.path.join(dirname, var.name))
            # set var_load to scope
            program.set_var(var.name, var_load)

    def test_replace_static_save_load(self):
        # enable static mode
        paddle.enable_static()

        with new_program_scope():
            # create network
            x = paddle.static.data(name="x", shape=[None, 10], dtype='float32')
            z = paddle.static.nn.fc(x, 10)
            z = paddle.static.nn.fc(z, 10, bias_attr=False)
            loss = fluid.layers.reduce_mean(z)
            opt = Adam(learning_rate=1e-3)
            opt.minimize(loss)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            fake_inputs = np.random.randn(10, 10).astype('float32')
            exe.run(prog, feed={'x': fake_inputs}, fetch_list=[loss])

            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    base_map[var.name] = t

            path = os.path.join("test_replace_static_save_load", "model")

            # replace paddle.static.save/load
            self.replace_static_save(prog, path)
            # set var to zero
            self.set_zero(prog, place)

            self.replace_static_load(prog, path)

            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

            # test for save/load_vars

            path_vars = 'test_replace_save_load_vars_binary/model'
            self.replace_save_vars(prog, path_vars)
            # set var to zero
            self.set_zero(prog, place)
            self.replace_load_vars(prog, path_vars)
            for var in prog.list_vars():
                if var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]

                    self.assertTrue(np.array_equal(new_t, base_t))

            # test for return tensor
            path_vars = 'test_replace_save_load_return_tensor_static/model'
            for var in prog.list_vars():
                if var.persistable:
                    paddle.save(var, os.path.join(path_vars, var.name))
            # set var to zero
            self.set_zero(prog, place)
            for var in prog.list_vars():
                if var.persistable:
                    tensor = paddle.load(
                        os.path.join(path_vars, var.name), return_tensor=True)
                    prog.set_var(var.name, tensor)
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

    def test_paddle_save_load_v2(self):
        # enable dygraph mode
        paddle.disable_static()
        layer = LinearNet()
        state_dict = layer.state_dict()
        path = 'paddle_save_load_v2/model.pdparams'

        # paddle.save
        with self.assertRaises(TypeError):
            paddle.save(state_dict, path, use_binary_format=0)

        with self.assertRaises(NotImplementedError):
            paddle.save(state_dict, path, use_binary_format=True)


# paddle.load


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()

        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def build_and_train_model(self):
        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()

        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

        # create data loader
        # TODO: using new DataLoader cause unknown Timeout on windows, replace it
        loader = random_batch_reader()

        # train
        train(layer, loader, loss_fn, adam)

        return layer, adam

    def check_load_state_dict(self, orig_dict, load_dict):
        for var_name, value in orig_dict.items():
            self.assertTrue(np.array_equal(value.numpy(), load_dict[var_name]))

    def test_save_load(self):
        layer, opt = self.build_and_train_model()

        # save
        layer_save_path = "test_paddle_save_load.linear.pdparams"
        opt_save_path = "test_paddle_save_load.linear.pdopt"
        layer_state_dict = layer.state_dict()
        opt_state_dict = opt.state_dict()

        paddle.save(layer_state_dict, layer_save_path)
        paddle.save(opt_state_dict, opt_save_path)

        # load
        load_layer_state_dict = paddle.load(layer_save_path)
        load_opt_state_dict = paddle.load(opt_save_path)

        self.check_load_state_dict(layer_state_dict, load_layer_state_dict)
        self.check_load_state_dict(opt_state_dict, load_opt_state_dict)

        # test save load in static mode
        paddle.enable_static()
        static_save_path = "static_mode_test/test_paddle_save_load.linear.pdparams"
        paddle.save(layer_state_dict, static_save_path)
        load_static_state_dict = paddle.load(static_save_path)
        self.check_load_state_dict(layer_state_dict, load_static_state_dict)

        # error test cases, some tests relay base test above
        # 1. test save obj not dict error
        # test_list = [1, 2, 3]
        # with self.assertRaises(NotImplementedError):
        #     paddle.save(test_list, "not_dict_error_path")

        # 2. test save path format error
        with self.assertRaises(ValueError):
            paddle.save(layer_state_dict, "test_paddle_save_load.linear.model/")

        # 3. test load path not exist error
        with self.assertRaises(ValueError):
            paddle.load("test_paddle_save_load.linear.params")

        # 4. test load old save path error
        with self.assertRaises(ValueError):
            paddle.load("test_paddle_save_load.linear")


if __name__ == '__main__':
    unittest.main()
