# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np


class BackwardNet(object):
    """
    Abstract Base Class.
    All Net inherited this Class should implement two functions:
        build_model: build net to test the logic of backward
        init_data: fake input data to test all programs.
    """

    def __init__(self):
        self.stop_gradient_grad_vars = set()
        self.no_grad_vars = set()
        self.params_names = set()
        self.op_path = []

    def build_model(self):
        """
        Build net to test the logic of backward.
        :return: loss
        """
        raise NotImplementedError

    def init_data(self):
        """
        Fake input data to test all programs.
        :return: dict, {'var_name': var_data}
        """
        raise NotImplementedError


class TestBackward(unittest.TestCase):
    """
    All related TestClass should inherit this class,
    and only implement test_backward function.
    """

    def _check_all(self, net):
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = net.build_model()
            self._check_backward(loss, main)

            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss)
            exe.run(startup)
            exe.run(feed=net.init_data())

    def _check_backward(self, loss, main_program):
        global_block_idx = self.global_block_idx
        params_grads = self._check_params_grad(loss)
        # 1.1 get_stop_gradients
        no_grad_dict = self._check_stop_gradient(main_program)
        # 1.2 find_op_path
        op_path, block_no_grad_set = self._check_op_path(
            main_program.block(global_block_idx), [loss], [], no_grad_dict)
        # 1.3 _find_no_grad_vars
        no_grad_vars = self._check_find_no_grad_vars(
            main_program.block(global_block_idx), op_path, [loss],
            block_no_grad_set)
        # update no_grad_dict
        block_no_grad_set.update(no_grad_vars)
        no_grad_dict[global_block_idx].update(
            list(map(fluid.backward._append_grad_suffix_, block_no_grad_set)))

    def _check_params_grad(self, loss, parameter_list=None, no_grad_set=None):
        params_grads = fluid.backward.append_backward(loss, parameter_list,
                                                      no_grad_set)
        params_names = set(
            [param_var.name for (param_var, grad_var) in params_grads])
        self.assertSetEqual(params_names, self.net.params_names)

        return params_grads

    def _check_stop_gradient(self, program):
        no_grad_dict = fluid.backward._get_stop_gradients_(program)
        if no_grad_dict is not None and isinstance(no_grad_dict, dict):
            self.assertSetEqual(no_grad_dict[self.global_block_idx],
                                self.net.stop_gradient_grad_vars)

        return no_grad_dict

    def _check_op_path(self, root_block, outputs, inputs=[], no_grad_dict=None):
        if no_grad_dict is None or not isinstance(no_grad_dict, dict):
            block_no_grad_set = None
        else:
            block_no_grad_set = set(
                map(fluid.backward._strip_grad_suffix_, no_grad_dict[
                    self.global_block_idx]))
        op_path = fluid.backward._find_op_path_(root_block, outputs, inputs,
                                                block_no_grad_set)
        op_types = [op.type for op in op_path]
        self.assertListEqual(op_types, self.net.op_path)

        return op_path, block_no_grad_set

    def _check_find_no_grad_vars(self, root_block, op_path, targets,
                                 block_no_grad_set):
        no_grad_vars = fluid.backward._find_no_grad_vars(
            root_block, op_path, targets, block_no_grad_set)
        self.assertSetEqual(no_grad_vars, self.net.no_grad_vars)

        return no_grad_vars

    def _check_error_param_list(self, net, parameter_list):
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = net.build_model()
            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss, parameter_list=parameter_list)
            exe.run(startup)
            exe.run(feed=net.init_data())

    def _check_error_no_grad_set(self, net, no_grad_set):
        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        startup = fluid.Program()

        with fluid.program_guard(main, startup):
            loss = net.build_model()
            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss, no_grad_set=no_grad_set)
            exe.run(startup)
            exe.run(feed=net.init_data())


class SimpleNet(BackwardNet):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.stop_gradient_grad_vars = set([
            u'x_no_grad@GRAD', u'x2_no_grad@GRAD', u'x3_no_grad@GRAD',
            u'label_no_grad@GRAD'
        ])
        self.no_grad_vars = set()
        self.params_names = set([u'w2v', u'fc_predict.b_0', u'fc_w'])
        self.op_path = [
            u'lookup_table_v2',
            u'lookup_table_v2',  # embedding
            u'elementwise_add',  # merge
            u'mul',
            u'elementwise_add',
            u'softmax',  # fc
            u'elementwise_sub',
            u'square',
            u'mean'
        ]  # loss
        self.shape = [16, 50]

    def init_data(self):
        assert len(self.shape) == 2
        x = np.random.randint(0, 90, self.shape).astype('int64')
        x2 = np.random.randint(0, 90, self.shape).astype('int64')
        x3 = np.random.randint(0, 90, self.shape).astype('int64')
        label = np.random.random([self.shape[0], 1]).astype('float32')
        return {
            'x_no_grad': x,
            'x2_no_grad': x2,
            'x3_no_grad': x3,
            'label_no_grad': label
        }

    def build_model(self):
        # stop_gradient = True in input
        x = fluid.data(name='x_no_grad', shape=self.shape, dtype='int64')
        x2 = fluid.data(name='x2_no_grad', shape=self.shape, dtype='int64')
        x3 = fluid.data(name='x3_no_grad', shape=self.shape, dtype='int64')
        label = fluid.data(
            name='label_no_grad', shape=[self.shape[0], 1], dtype='float32')
        # shared layer, the grad of 'w2v' will be summed and renamed.
        # To test  _addup_repetitive_outputs_
        x_emb = fluid.embedding(
            x, size=[100, 64], param_attr=fluid.ParamAttr(name='w2v'))
        x2_emb = fluid.embedding(
            x2, size=[100, 64], param_attr=fluid.ParamAttr(name='w2v'))
        x3_emb = fluid.embedding(
            x3, size=[100, 64], param_attr=fluid.ParamAttr(name='w2v'))
        # merge layers
        x_merge = fluid.layers.elementwise_add(x_emb, x2_emb, name='x_add_x2')
        x2_merge = fluid.layers.elementwise_add(
            x2_emb, x3_emb, name='x2_add_x3')
        # shared fc_w
        predict = fluid.layers.fc(input=x_merge,
                                  size=1,
                                  act='softmax',
                                  param_attr=fluid.ParamAttr(name='fc_w'),
                                  name='fc_predict')
        # useless layer for calculating loss
        fc_no_use = fluid.layers.fc(input=x2_merge,
                                    size=1,
                                    act='sigmoid',
                                    param_attr=fluid.ParamAttr(name='fc_w'),
                                    name='fc_no_use')
        # loss
        cost = fluid.layers.square_error_cost(input=predict, label=label)
        loss = fluid.layers.mean(cost, name='mean_loss')

        return loss


class TestSimpleNet(TestBackward):
    def test_backward(self):
        """
        Instantiate each NetClass to test backward.
        """
        self.global_block_idx = 0
        self.net = SimpleNet()
        self._check_all(self.net)


class TestGradientsError(unittest.TestCase):
    def test_error(self):
        x = fluid.data(name='x', shape=[None, 2, 8, 8], dtype='float32')
        x.stop_gradient = False
        conv = fluid.layers.conv2d(x, 4, 1, bias_attr=False)
        y = fluid.layers.relu(conv)

        with self.assertRaises(TypeError):
            x_grad = fluid.gradients(y.name, x)

        with self.assertRaises(TypeError):
            x_grad = fluid.gradients(y, x.name)

        with self.assertRaises(TypeError):
            x_grad = fluid.gradients([y], [x], target_gradients=x.name)

        with self.assertRaises(TypeError):
            x_grad = fluid.gradients([y], x, no_grad_set=conv)


class TestSimpleNetWithErrorParamList(TestBackward):
    def test_parameter_list_type_error(self):
        self.global_block_idx = 0
        self.net = SimpleNet()
        # The type of parameter_list argument must be list or tuple
        with self.assertRaises(TypeError):
            self._check_error_param_list(self.net, "test")
        # The type of parameter_list's member must be Variable or str
        test = fluid.data(name='test', shape=[None, 90], dtype='float32')
        with self.assertRaises(TypeError):
            self._check_error_param_list(self.net, [test, "test", 3])


class TestSimpleNetWithErrorNoGradSet(TestBackward):
    def test_no_grad_set_type_error(self):
        self.global_block_idx = 0
        self.net = SimpleNet()
        # The type of no_grad_set argument must be set or list or tuple
        with self.assertRaises(TypeError):
            self._check_error_no_grad_set(self.net, "test")
        # The type of no_grad_set's member must be Variable or str
        test = fluid.data(name='test', shape=[None, 90], dtype='float32')
        with self.assertRaises(TypeError):
            self._check_error_no_grad_set(self.net, [test, "test", 3])


class TestAppendBackwardWithError(unittest.TestCase):
    def build_net(self):
        x = fluid.data(name='x', shape=[None, 13], dtype='int64')
        y = fluid.data(name='y', shape=[None, 1], dtype='float32')
        x_emb = fluid.embedding(x, size=[100, 256])
        y_predict = fluid.layers.fc(input=x_emb, size=1, name='my_fc')
        loss = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_loss = fluid.layers.mean(loss)
        param_names = [
            param.name
            for param in fluid.default_main_program().block(0).all_parameters()
        ]

        return avg_loss, param_names

    def setUp(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            self.avg_loss, self.param_names = self.build_net()

    def test_loss_type_error(self):
        with self.assertRaises(TypeError):
            fluid.backward.append_backward(loss=self.avg_loss.name)

    def test_parameter_list_type_error(self):
        with self.assertRaises(TypeError):
            self.param_names[0] = np.random.random([10])
            fluid.backward.append_backward(
                loss=self.avg_loss, parameter_list=self.param_names)

    def test_callback_type_error(self):
        with self.assertRaises(TypeError):

            def callback(block, context):
                return

            fluid.backward.append_backward(
                loss=self.avg_loss, callbacks=callback)


# TODO(Aurelius84): add conditional network test
class ConditionalNet(BackwardNet):
    def __init__(self):
        super(ConditionalNet, self).__init__()


if __name__ == '__main__':
    unittest.main()
