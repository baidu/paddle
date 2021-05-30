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
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from op_test import OpTest
import paddle


class TestSGDOp(OpTest):
    def setUp(self):
        self.op_type = "sgd"
        self.conf()
        w = np.random.random((self.h, self.w)).astype("float32")
        g = np.random.random((self.h, self.w)).astype("float32")
        lr = np.array([0.1]).astype("float32")

        self.inputs = {'Param': w, 'Grad': g, 'LearningRate': lr}
        self.outputs = {'ParamOut': w - lr * g}

    def conf(self):
        self.h = 102
        self.w = 105

    def test_check_output(self):
        self.check_output()


class TestSGDOpCase8X(TestSGDOp):
    def conf(self):
        self.h = 10
        self.w = 64


class TestSparseSGDOp(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable   
        height = 10
        rows = [0, 4, 7]
        self.conf()

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, self.row_numel), 5.0).astype("float32")
        param.set(param_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate')
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(param)

        # rows[0] = 0, 5.0 - 2.0 * 2.0
        self.assertAlmostEqual(1.0, result_array[rows[0], 0])
        # rows[0] = 0, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[0], 2])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[1, 0])
        # rows[1] = 4, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[1], 10])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[5, 8])
        # rows[2] = 7, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[2], 1])
        # rows[2] = 7, 5.0 - 2.0 * 4.0
        self.assertAlmostEqual(-3.0, result_array[rows[2], 8])

    def test_sparse_sgd(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)

    def conf(self):
        self.row_numel = 12


class TestSparseSGDOpCase8X(TestSparseSGDOp):
    def conf(self):
        self.row_numel = 16


class TestSGDOpOptimizeSelectedRows(unittest.TestCase):
    def check_with_place(self, place):
        scope = core.Scope()

        row_width = 12
        # create and initialize Grad Variable
        grad_height = 10
        grad_rows = [0, 4, 7]

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(grad_height)
        grad_selected_rows.set_rows(grad_rows)
        grad_array = np.ones((len(grad_rows), row_width)).astype("float32")
        grad_array[0, 0] = 2.0
        grad_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(grad_array, place)

        # create and initialize Param Variable
        # create and initialize W Variable
        param_rows = [0, 1, 2, 3, 4, 5, 6, 7]

        # init Param
        w_selected_rows = scope.var('Param').get_selected_rows()
        w_selected_rows.set_height(len(param_rows))
        w_selected_rows.set_rows(param_rows)
        w_selected_rows.sync_index()
        w_array = np.ones((len(param_rows), row_width)).astype("float32")
        for i in range(len(param_rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        w_before_optimize = np.array(w_tensor)

        # create and initialize LeraningRate Variable
        lr_value = 0.1
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), lr_value).astype("float32")
        lr.set(lr_array, place)

        # optimize with Python
        w_after_optimize = np.copy(w_before_optimize)
        for index, id in enumerate(grad_rows):
            w_after_optimize[id] = w_before_optimize[
                id] - lr_value * grad_array[index]

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate')
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(w_tensor)
        assert (result_array == w_after_optimize).all()

    def test_sparse_parameter_sgd(self):
        places = [core.CPUPlace()]
        # do not support GPU kernel currently
        for place in places:
            self.check_with_place(place)


class TestSGDOpWithLargeInput(unittest.TestCase):
    def runTest(self):
        data = fluid.layers.fill_constant(shape=[1], value=128, dtype='int64')
        label = fluid.layers.fill_constant(
            shape=[1, 150], value=0.5, dtype='float32')
        emb = fluid.embedding(input=data, size=(10000000, 150), dtype='float32')
        out = fluid.layers.l2_normalize(x=emb, axis=-1)

        cost = fluid.layers.square_error_cost(input=out, label=label)
        avg_cost = fluid.layers.mean(cost)
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        compiled_prog = fluid.compiler.CompiledProgram(
            fluid.default_main_program())
        result = exe.run(compiled_prog, fetch_list=[avg_cost])


class TestSGDV2(unittest.TestCase):
    def test_sgd_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = paddle.to_tensor(value)
        linear = paddle.nn.Linear(13, 5)
        # This can be any optimizer supported by dygraph.
        adam = paddle.optimizer.SGD(learning_rate=0.01,
                                    parameters=linear.parameters(),
                                    weight_decay=0.01)
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_sgd(self):
        place = fluid.CPUPlace()
        main = fluid.Program()
        with fluid.program_guard(main):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)

            rms_optimizer = paddle.optimizer.SGD(learning_rate=0.1)
            rms_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = paddle.batch(
                paddle.dataset.uci_housing.train(), batch_size=1)
            feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    def test_raise_error(self):
        self.assertRaises(ValueError, paddle.optimizer.SGD, learning_rate=None)


class TestSparseSGDOpWithMultiPrecision(unittest.TestCase):
    def setUp(self):
        pass

    def check_with_place(self, place):
        scope = core.Scope()

        # create and initialize Grad Variable   
        height = 10
        rows = [0, 4, 7]
        self.conf()

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), self.row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.full((height, self.row_numel), 5.0).astype("float32")
        param.set(param_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.full((1), 2.0).astype("float32")
        lr.set(lr_array, place)

        # create and run sgd operator
        sgd_op = Operator(
            "sgd",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            LearningRate='LearningRate',
            multi_precision=True,
            rescale_grad=1.0)
        sgd_op.run(scope, place)

        # get and compare result
        result_array = np.array(param)

        # rows[0] = 0, 5.0 - 2.0 * 2.0
        self.assertAlmostEqual(1.0, result_array[rows[0], 0])
        # rows[0] = 0, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[0], 2])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[1, 0])
        # rows[1] = 4, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[1], 10])
        # 5.0 - 2.0 * 0.0
        self.assertAlmostEqual(5.0, result_array[5, 8])
        # rows[2] = 7, 5.0 - 2.0 * 1.0
        self.assertAlmostEqual(3.0, result_array[rows[2], 1])
        # rows[2] = 7, 5.0 - 2.0 * 4.0
        self.assertAlmostEqual(-3.0, result_array[rows[2], 8])

    def test_sparse_sgd(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)

    def conf(self):
        self.row_numel = 12


if __name__ == "__main__":
    unittest.main()
