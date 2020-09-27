#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import warnings

import paddle
import paddle.fluid.executor as executor
import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.io import save_inference_model, load_inference_model, save_persistables
from paddle.fluid.transpiler import memory_optimize


class TestBook(unittest.TestCase):
    class InferModel(object):
        def __init__(self, list):
            self.program = list[0]
            self.feed_var_names = list[1]
            self.fetch_vars = list[2]

    def test_fit_line_inference_model(self):
        MODEL_DIR = "./tmp/inference_model"
        UNI_MODEL_DIR = "./tmp/inference_model1"

        init_program = Program()
        program = Program()

        with program_guard(program, init_program):
            x = layers.data(name='x', shape=[2], dtype='float32')
            y = layers.data(name='y', shape=[1], dtype='float32')

            y_predict = layers.fc(input=x, size=1, act=None)

            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)

            sgd_optimizer = optimizer.SGDOptimizer(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost, init_program)

        place = core.CPUPlace()
        exe = executor.Executor(place)

        exe.run(init_program, feed={}, fetch_list=[])

        for i in six.moves.xrange(100):
            tensor_x = np.array(
                [[1, 1], [1, 2], [3, 4], [5, 2]]).astype("float32")
            tensor_y = np.array([[-2], [-3], [-7], [-7]]).astype("float32")

            exe.run(program,
                    feed={'x': tensor_x,
                          'y': tensor_y},
                    fetch_list=[avg_cost])

        # Separated model and unified model
        save_inference_model(MODEL_DIR, ["x", "y"], [avg_cost], exe, program)
        save_inference_model(UNI_MODEL_DIR, ["x", "y"], [avg_cost], exe,
                             program, 'model', 'params')
        main_program = program.clone()._prune_with_input(
            feeded_var_names=["x", "y"], targets=[avg_cost])
        params_str = save_persistables(exe, None, main_program, None)

        expected = exe.run(program,
                           feed={'x': tensor_x,
                                 'y': tensor_y},
                           fetch_list=[avg_cost])[0]

        six.moves.reload_module(executor)  # reload to build a new scope

        model_0 = self.InferModel(load_inference_model(MODEL_DIR, exe))
        with open(os.path.join(UNI_MODEL_DIR, 'model'), "rb") as f:
            model_str = f.read()
        model_1 = self.InferModel(
            load_inference_model(None, exe, model_str, params_str))

        for model in [model_0, model_1]:
            outs = exe.run(model.program,
                           feed={
                               model.feed_var_names[0]: tensor_x,
                               model.feed_var_names[1]: tensor_y
                           },
                           fetch_list=model.fetch_vars)
            actual = outs[0]

            self.assertEqual(model.feed_var_names, ["x", "y"])
            self.assertEqual(len(model.fetch_vars), 1)
            print("fetch %s" % str(model.fetch_vars[0]))
            self.assertEqual(expected, actual)

        self.assertRaises(ValueError, fluid.io.load_inference_model, None, exe,
                          model_str, None)


class TestSaveInferenceModel(unittest.TestCase):
    def test_save_inference_model(self):
        MODEL_DIR = "./tmp/inference_model2"
        init_program = Program()
        program = Program()

        # fake program without feed/fetch
        with program_guard(program, init_program):
            x = layers.data(name='x', shape=[2], dtype='float32')
            y = layers.data(name='y', shape=[1], dtype='float32')

            y_predict = layers.fc(input=x, size=1, act=None)

            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])

        save_inference_model(MODEL_DIR, ["x", "y"], [avg_cost], exe, program)

    def test_save_inference_model_with_auc(self):
        MODEL_DIR = "./tmp/inference_model4"
        init_program = Program()
        program = Program()

        # fake program without feed/fetch
        with program_guard(program, init_program):
            x = layers.data(name='x', shape=[2], dtype='float32')
            y = layers.data(name='y', shape=[1], dtype='int32')
            predict = fluid.layers.fc(input=x, size=2, act='softmax')
            acc = fluid.layers.accuracy(input=predict, label=y)
            auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict,
                                                                  label=y)
            cost = fluid.layers.cross_entropy(input=predict, label=y)
            avg_cost = fluid.layers.mean(x=cost)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            save_inference_model(MODEL_DIR, ["x", "y"], [avg_cost], exe,
                                 program)
            expected_warn = "please ensure that you have set the auc states to zeros before saving inference model"
            self.assertTrue(len(w) > 0)
            self.assertTrue(expected_warn == str(w[0].message))


class TestInstance(unittest.TestCase):
    def test_save_inference_model(self):
        MODEL_DIR = "./tmp/inference_model3"
        init_program = Program()
        program = Program()

        # fake program without feed/fetch
        with program_guard(program, init_program):
            x = layers.data(name='x', shape=[2], dtype='float32')
            y = layers.data(name='y', shape=[1], dtype='float32')

            y_predict = layers.fc(input=x, size=1, act=None)

            cost = layers.square_error_cost(input=y_predict, label=y)
            avg_cost = layers.mean(cost)

        place = core.CPUPlace()
        exe = executor.Executor(place)
        exe.run(init_program, feed={}, fetch_list=[])

        # will print warning message

        cp_prog = CompiledProgram(program).with_data_parallel(
            loss_name=avg_cost.name)

        save_inference_model(MODEL_DIR, ["x", "y"], [avg_cost], exe, cp_prog)
        self.assertRaises(TypeError, save_inference_model,
                          [MODEL_DIR, ["x", "y"], [avg_cost], [], cp_prog])


class TestLoadInferenceModelError(unittest.TestCase):
    def test_load_model_not_exist(self):
        place = core.CPUPlace()
        exe = executor.Executor(place)
        self.assertRaises(ValueError, load_inference_model,
                          './test_not_exist_dir', exe)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
