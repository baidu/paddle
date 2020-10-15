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
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid import Program, program_guard
import paddle.fluid.initializer as I
import math
from op_test import OpTest, skip_check_grad_ci

paddle.enable_static()
np.random.seed(100)


def find_latest_set(num):
    return 1 + int(math.floor(math.log(num, 2)))


class CodeTable(object):
    def __init__(self, num_classes, code):
        self.c = num_classes + code

    def cal_index(self, bit):
        return (self.c >> (bit + 1)) - 1

    def get_length(self):
        return find_latest_set(self.c) - 1

    def cal_bit(self, bit):
        return self.c & (1 << bit)


class CodeTableWithCustomTree(object):
    def __init__(self, path_table, path_code, index):
        self.ptable_ = path_table
        self.pcode_ = path_code
        self.index_ = index

    def cal_index(self, bit):
        return self.ptable_[self.index_][bit]

    def get_length(self):
        length = 0
        for ele in self.ptable_[self.index_]:  # find the first -1 to stop trace
            if ele >= 0:
                length = length + 1
            else:
                return length
        return length

    def cal_bit(self, bit):
        return self.pcode_[self.index_][bit]


def hsigmoid(x, w, label, bias, num_classes):
    batch_size = x.shape[0]
    code_length = find_latest_set(num_classes - 1)
    code_table = [0 for _ in range(code_length)]
    pre_output = np.zeros((batch_size, code_length)).astype('float64')
    pre_sum = np.zeros((batch_size, 1)).astype('float64')
    out = np.zeros((batch_size, 1)).astype('float64')
    for i in range(batch_size):
        code_table = CodeTable(num_classes, label[i])
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += bias[idx][0]
    for i in range(batch_size):
        code_table = CodeTable(num_classes, label[i])
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += np.dot(w[idx], x[i])
    # clip[-40.0, 40.0]
    pre_output = np.clip(pre_output, -40.0, 40.0)
    # out(i, 0) = \sum_j  bit(i, j) * preout(i, j)
    for i in range(batch_size):
        code_table = CodeTable(num_classes, label[i])
        length = code_table.get_length()
        sum = 0.0
        for j in range(length):
            if code_table.cal_bit(j):
                sum += pre_output[i][j]
        out[i] = -1.0 * sum
    # soft relu
    pre_output = np.log(1 + np.exp(pre_output))
    pre_sum = pre_output.sum(1).reshape((batch_size, 1))
    out += pre_sum
    return pre_output, out


def hsigmoid_grad(x, w, label, bias, num_classes):
    batch_size = x.shape[0]
    dx = np.zeros(x.shape).astype('float64')
    dw = np.zeros(w.shape).astype('float64')
    db = np.zeros(bias.shape).astype('float64')
    for i in range(batch_size):
        code_table = CodeTable(num_classes, label[i])
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            t = 1 / (1 + np.exp(-(np.dot(w[idx], x[i]) + bias[idx])))
            dx[i] = dx[i] + t * w[idx]
            dw[idx] += t * x[i]
            db[idx] += t
            if code_table.cal_bit(j):
                dx[i] = dx[i] - w[idx]
                dw[idx] -= x[i]
                db[idx] -= 1
    dx /= batch_size
    dw /= batch_size
    db /= batch_size
    return [dx, dw, db]


def hsigmoidWithCustomTree(x, w, path_table, path_code, label, bias,
                           num_classes):
    batch_size = x.shape[0]
    code_length = len(path_table[0])
    code_table = [0 for _ in range(code_length)]
    # init pre_out with shape [N, code_length]
    pre_output = np.zeros((batch_size, code_length)).astype('float64')
    pre_sum = np.zeros((batch_size, 1)).astype('float64')
    out = np.zeros((batch_size, 1)).astype('float64')
    if isinstance(bias, np.ndarray):
        for i in range(batch_size):
            code_table = CodeTableWithCustomTree(path_table, path_code, i)
            length = code_table.get_length()
            for j in range(length):
                idx = code_table.cal_index(j)
                pre_output[i][j] += bias[idx][0]
    for i in range(batch_size):
        code_table = CodeTableWithCustomTree(path_table, path_code, i)
        length = code_table.get_length()
        for j in range(length):
            idx = code_table.cal_index(j)
            pre_output[i][j] += np.dot(w[idx], x[i])
    # clip[-40.0, 40.0]
    pre_output = np.clip(pre_output, -40.0, 40.0)
    # out(i, 0) = \sum_j  bit(i, j) * preout(i, j)
    for i in range(batch_size):
        code_table = CodeTableWithCustomTree(path_table, path_code, i)
        length = code_table.get_length()
        sum = 0.0
        for j in range(length):
            if code_table.cal_bit(j):
                sum += pre_output[i][j]
        out[i] = -1.0 * sum
    # soft relu
    pre_output = np.log(1 + np.exp(pre_output))
    pre_sum = pre_output.sum(1).reshape((batch_size, 1))
    out += pre_sum
    return pre_output, out


class TestHSigmoidOp(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 101
        feature_size = 5
        batch_size = 20
        x = np.random.uniform(-1, 1,
                              (batch_size, feature_size)).astype('float64')
        w = np.random.uniform(-1, 1,
                              (num_classes - 1, feature_size)).astype('float64')
        label = np.random.randint(0, num_classes,
                                  (batch_size, 1)).astype('int64')
        bias = np.random.uniform(-1, 1, (num_classes - 1, 1)).astype('float64')
        self.attrs = {'num_classes': num_classes, 'is_sparse': False}
        self.inputs = {'X': x, 'W': w, 'Label': label, 'Bias': bias}
        pre_output, out = hsigmoid(x, w, label, bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}
        self.user_grads = hsigmoid_grad(x, w, label, bias, num_classes)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['X', 'W', 'Bias'], ['Out'], user_defined_grads=self.user_grads)


@skip_check_grad_ci(
    reason="For 'TestHSigmoidOpSparse', check_grad is is separately calculated by 'TestHSigmoidOpWithSparseGrad'."
)
class TestHSigmoidOpSparse(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 6  #using 1,2,3,4,5,6 to build a huffman tree and select 1,2,5,6 as sample
        feature_size = 8
        batch_size = 4
        x = np.random.random((batch_size, feature_size))
        w = np.random.random((num_classes - 1, feature_size))
        label = np.array([0, 1, 4, 5]).astype('int64')
        path_table = np.array([
            (0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (0, 1, 4, -1, -1), (0, 2, -1,
                                                                       -1, -1)
        ]).astype(
            'int64')  #np.array to store 1,2,5,6s' non-leaf path(root -> leaf)
        path_code = np.array(
            [(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (1, 0, 0, -1, -1),
             (0, 1, -1, -1, -1)]).astype('int64')  #np.array to store 
        bias = np.random.random((num_classes - 1, 1))
        self.attrs = {'num_classes': num_classes, 'is_sparse': True}
        self.inputs = {
            'X': x,
            'W': w,
            'PathTable': path_table,
            'PathCode': path_code,
            'Label': label,
            'Bias': bias
        }
        pre_output, out = hsigmoidWithCustomTree(x, w, path_table, path_code,
                                                 label, bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()


class TestHSigmoidOpWithSparseGrad(unittest.TestCase):
    def hs_net_conf(self, is_sparse):
        input_word = fluid.layers.data(name="x", shape=[1], dtype='int64')
        path_table = fluid.layers.data(
            name='path_table', shape=[3], dtype='int64')
        path_code = fluid.layers.data(
            name='path_code', shape=[3], dtype='int64')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        data_list = [input_word, path_table, path_code, label]

        emb = fluid.layers.embedding(
            input=input_word,
            is_sparse=is_sparse,
            size=[3, 3],
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(3))))

        cost = fluid.layers.hsigmoid(
            input=emb,
            label=label,
            bias_attr=True,
            num_classes=3,
            path_table=path_table,
            path_code=path_code,
            is_custom=True,
            is_sparse=is_sparse)

        avg_cost = fluid.layers.reduce_mean(cost)

        return avg_cost, data_list

    def training_test(self, is_sparse):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            paddle.manual_seed(1)
            start_up = fluid.default_startup_program()
            x = np.arange(6).reshape(6)
            path_table = np.array([(1, 2, -1), (1, 2, -1)]).astype('int64')
            path_code = np.array([(1, 0, -1), (0, 0, -1)]).astype('int64')
            label = np.array([1, 4]).astype('int64')

            loss, data_list = self.hs_net_conf(is_sparse)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)

            main_program = fluid.default_main_program()
            place = fluid.CPUPlace()
            feeder = fluid.DataFeeder(feed_list=data_list, place=place)
            exe = fluid.Executor(place)

            exe.run(start_up)
            result = list()
            for i in range(10):
                data = [([[x[i % 2]]], [list(path_table[i % 2])],
                         [list(path_code[i % 2])], [label[i % 2]])]

                loss_val = exe.run(main_program,
                                   feed=feeder.feed(data),
                                   fetch_list=[loss])
                result.append(loss_val)
        return result

    def test_hs_grad_with_sparse(self):
        dense_result = self.training_test(is_sparse=False)
        sparse_result = self.training_test(is_sparse=True)
        assert (dense_result == sparse_result)


@skip_check_grad_ci(
    reason="[skip shape check] The huffman tree is structed separately. It will be complicated if use large shape."
)
class TestHSigmoidOpWithCostumTree(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 6  #using 1,2,3,4,5,6 to build a huffman tree and select 1,2,5,6 as sample
        feature_size = 8
        batch_size = 4
        x = np.random.uniform(-1, 1, (batch_size, feature_size))
        w = np.random.uniform(-1, 1, (num_classes - 1, feature_size))
        label = np.array([0, 1, 4, 5]).astype('int64')
        path_table = np.array([
            (0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (0, 1, 4, -1, -1), (0, 2, -1,
                                                                       -1, -1)
        ]).astype(
            'int64')  #np.array to store 1,2,5,6s' non-leaf path(root -> leaf)
        path_code = np.array(
            [(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (1, 0, 0, -1, -1),
             (0, 1, -1, -1, -1)]).astype('int64')  #np.array to store 
        bias = np.random.random((num_classes - 1, 1))
        self.attrs = {'num_classes': num_classes, 'is_sparse': False}
        self.inputs = {
            'X': x,
            'W': w,
            'PathTable': path_table,
            'PathCode': path_code,
            'Label': label,
            'Bias': bias
        }
        pre_output, out = hsigmoidWithCustomTree(x, w, path_table, path_code,
                                                 label, bias, num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Bias', 'X', 'W'], ['Out'], no_grad_set=set('Label'))


@skip_check_grad_ci(
    reason="[skip shape check] The huffman tree is structed separately. It will be complicated if use large shape."
)
class TestHSigmoidOpWithCostumTreeWithoutBias(OpTest):
    def setUp(self):
        self.op_type = "hierarchical_sigmoid"
        num_classes = 6  #using 1,2,3,4,5,6 to build a huffman tree and select 1,2,5,6 as sample
        feature_size = 8
        batch_size = 4
        x = np.random.uniform(-1, 1, (batch_size, feature_size))
        w = np.random.uniform(-1, 1, (num_classes - 1, feature_size))
        label = np.array([0, 1, 4, 5]).astype('int64')
        path_table = np.array([
            (0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (0, 1, 4, -1, -1), (0, 2, -1,
                                                                       -1, -1)
        ]).astype(
            'int64')  #np.array to store 1,2,5,6s' non-leaf path(root -> leaf)
        path_code = np.array(
            [(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (1, 0, 0, -1, -1),
             (0, 1, -1, -1, -1)]).astype('int64')  #np.array to store 
        # bias = np.random.random((num_classes - 1, 1)).astype("float32")
        self.attrs = {'num_classes': num_classes, 'is_sparse': False}
        self.inputs = {
            'X': x,
            'W': w,
            'PathTable': path_table,
            'PathCode': path_code,
            'Label': label,
        }
        pre_output, out = hsigmoidWithCustomTree(
            x=x,
            w=w,
            path_table=path_table,
            path_code=path_code,
            label=label,
            bias=None,
            num_classes=num_classes)
        self.outputs = {'PreOut': pre_output, 'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'W'], ['Out'], no_grad_set=set('Label'))


class TestHSigmoidLossAPI(unittest.TestCase):
    # test paddle.nn.functional.hsigmoid_loss, paddle.nn.HSigmoidLoss
    def setUp(self):
        self.dtype = 'float32'
        self.batch_size = 4
        self.feature_size = 6
        self.num_classes = 8
        self.is_custom = False
        self.place = paddle.CPUPlace()

        paddle.set_default_dtype(self.dtype)

        self.x_np = np.random.uniform(
            -1, 1, [self.batch_size, self.feature_size]).astype(self.dtype)
        self.labels_np = np.random.randint(
            self.num_classes, size=(self.batch_size, 1), dtype='int64')
        self.weight_np = np.random.uniform(
            -1, 1, [self.num_classes - 1, self.feature_size]).astype(self.dtype)
        self.bias_np = np.random.uniform(-1, 1, (
            self.num_classes - 1, )).astype(self.dtype)
        self.path_table_np = None
        self.path_code_np = None
        _, self.out_np = hsigmoid(self.x_np, self.weight_np, self.labels_np,
                                  self.bias_np, self.num_classes)
        self.set_attrs()

        if self.is_custom:
            _, self.out_np = hsigmoidWithCustomTree(
                self.x_np, self.weight_np, self.path_table_np,
                self.path_code_np, self.labels_np,
                self.bias_np.reshape(-1, 1), self.num_classes)

    def set_attrs(self):
        pass

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        labels = paddle.to_tensor(self.labels_np)
        weight = paddle.to_tensor(self.weight_np)
        bias = paddle.to_tensor(self.bias_np)
        path_table = None
        path_code = None
        if self.is_custom:
            path_table = paddle.to_tensor(self.path_table_np)
            path_code = paddle.to_tensor(self.path_code_np)
        out1 = F.hsigmoid_loss(x, labels, self.num_classes, weight, bias,
                               path_table, path_code)

        weight_attr = I.NumpyArrayInitializer(self.weight_np)
        bias_attr = I.NumpyArrayInitializer(self.bias_np)
        m = paddle.nn.HSigmoidLoss(self.feature_size, self.num_classes,
                                   weight_attr, bias_attr, self.is_custom)
        out2 = m(x, labels, path_table, path_code)

        for out in [out1, out2]:
            self.assertTrue(np.allclose(self.out_np, out.numpy()))
        paddle.enable_static()

    def test_static_api(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_program):
            x = paddle.static.data('x', [-1, self.feature_size])
            labels = paddle.static.data('labels', [-1, 1], 'int64')
            weight = paddle.static.data('weight', [-1, self.feature_size])
            bias = paddle.static.data('bias', [-1, ])
            path_table = None
            path_code = None
            if self.is_custom:
                path_table = paddle.static.data('path_table', [-1, -1], 'int64')
                path_code = paddle.static.data('path_code', [-1, -1], 'int64')
            out1 = F.hsigmoid_loss(x, labels, self.num_classes, weight, bias,
                                   path_table, path_code)

            weight_attr = paddle.framework.ParamAttr(
                initializer=I.NumpyArrayInitializer(self.weight_np))
            bias_attr = paddle.framework.ParamAttr(
                initializer=I.NumpyArrayInitializer(self.bias_np))
            m = paddle.nn.HSigmoidLoss(self.feature_size, self.num_classes,
                                       weight_attr, bias_attr, self.is_custom)
            out2 = m(x, labels, path_table, path_code)

            exe = paddle.static.Executor(self.place)
            exe.run(startup_program)
            feed_dict = {
                'x': self.x_np,
                'labels': self.labels_np,
                'weight': self.weight_np,
                'bias': self.bias_np
            }
            if self.is_custom:
                feed_dict["path_code"] = self.path_code_np
                feed_dict["path_table"] = self.path_table_np
            ret1, ret2 = exe.run(train_program,
                                 feed=feed_dict,
                                 fetch_list=[out1, out2])

            for ret in [ret1, ret2]:
                self.assertTrue(np.allclose(self.out_np, ret))

    def test_fluid_api(self):
        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            x = fluid.data('x', [-1, self.feature_size])
            labels = fluid.data('labels', [-1, 1], 'int64')
            path_table = None
            path_code = None
            if self.is_custom:
                path_table = fluid.data('path_table', [-1, -1], 'int64')
                path_code = fluid.data('path_code', [-1, -1], 'int64')
            weight_attr = I.NumpyArrayInitializer(self.weight_np)
            bias_attr = I.NumpyArrayInitializer(self.bias_np)
            out = fluid.layers.hsigmoid(x, labels, self.num_classes,
                                        weight_attr, bias_attr, 'out',
                                        path_table, path_code, self.is_custom)

            exe = fluid.Executor(self.place)
            exe.run(startup_program)
            feed_dict = {'x': self.x_np, 'labels': self.labels_np}
            if self.is_custom:
                feed_dict["path_code"] = self.path_code_np
                feed_dict["path_table"] = self.path_table_np
            ret, = exe.run(train_program, feed=feed_dict, fetch_list=[out])

            self.assertTrue(np.allclose(ret, self.out_np))

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            # test paddle.nn.HSigmoidLoss
            self.assertRaises(ValueError, paddle.nn.HSigmoidLoss, 6, 1)

            # test paddle.nn.functional.hsigmoid_loss
            x = paddle.static.data('x', [4, 6])
            label = paddle.static.data('label', [4, 1], 'int64')
            weight = paddle.static.data('weight', [7, 6])
            bias = paddle.static.data('bias', [7])

            x_int32 = paddle.static.data('x_int32', [4, 6], 'int32')
            self.assertRaises(TypeError, F.hsigmoid_loss, x_int32, label, 8,
                              weight)

            label_float32 = paddle.static.data('label_float32', [4, 1],
                                               'float32')
            self.assertRaises(TypeError, F.hsigmoid_loss, x, label_float32, 8,
                              weight)

            weight_int32 = paddle.static.data('weight_int32', [7, 6], 'int32')
            self.assertRaises(TypeError, F.hsigmoid_loss, x, label, 8,
                              weight_int32)

            bias_int32 = paddle.static.data('bias_int32', [7], 'int32')
            self.assertRaises(
                TypeError,
                F.hsigmoid_loss,
                x,
                label,
                8,
                weight,
                bias=bias_int32)

            path_table_int32 = paddle.static.data('path_table_int32', [7],
                                                  'int32')
            self.assertRaises(
                TypeError,
                F.hsigmoid_loss,
                x,
                label,
                8,
                weight,
                path_table=path_table_int32)

            path_code_int32 = paddle.static.data('path_code_int32', [7],
                                                 'int32')
            self.assertRaises(
                TypeError,
                F.hsigmoid_loss,
                x,
                label,
                8,
                weight,
                path_code=path_code_int32)

        # test paddle.fluid.layers.hsigmoid
        with program_guard(Program()):
            label = fluid.data('label', [4, 1], 'int64')
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.hsigmoid, 1, label, 2)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[4, 3], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.hsigmoid, x_int32, label,
                              2)
            # support the input dtype is float32
            x_fp32 = fluid.data(name='x_fp32', shape=[4, 3], dtype='float32')
            fluid.layers.hsigmoid(x_fp32, label, 2)

            # The label type must be Variable.
            self.assertRaises(TypeError, fluid.layers.hsigmoid, x_fp32, 1, 2)
            # The label dtype must be int64.
            label_int32 = fluid.data('label_int32', [4, 1], 'int32')
            self.assertRaises(TypeError, fluid.layers.hsigmoid, x_fp32,
                              label_int32, 2)


class TestHSigmoidLossAPICustom(TestHSigmoidLossAPI):
    def set_attrs(self):
        self.is_custom = True
        self.path_table_np = np.array([(0, 2, -1, -1, -1), (0, 1, 3, -1, -1), (
            0, 1, 4, -1, -1), (0, 2, -1, -1, -1)]).astype(np.int64)
        self.path_code_np = np.array([(0, 0, -1, -1, -1), (1, 1, 1, -1, -1), (
            1, 0, 0, -1, -1), (0, 1, -1, -1, -1)]).astype(np.int64)

    def test_errors(self):
        pass


if __name__ == '__main__':
    unittest.main()
