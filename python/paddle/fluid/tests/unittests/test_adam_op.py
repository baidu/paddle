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
from op_test import OpTest
from paddle.fluid import core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle


class TestAdamOp1(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.004
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        self.attrs = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, self.attrs)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOp2(OpTest):
    def setUp(self):
        '''Test Adam Op with supplied attributes
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        attributes = {'epsilon': epsilon, 'beta1': beta1, 'beta2': beta2}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOpMultipleSteps(OpTest):
    def setUp(self):
        '''Test Adam Operator with supplied attributes
        '''
        self.op_type = "adam"
        self.num_steps = 10

        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")

        learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        epsilon = 1e-8
        self.beta1_pow = self.beta1**10
        self.beta2_pow = self.beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([self.beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([self.beta2_pow]).astype("float32")
        }

        self.attrs = {
            'epsilon': epsilon,
            'beta1': self.beta1,
            'beta2': self.beta2
        }

    def test_check_output(self):
        for _ in range(self.num_steps):
            param_out, moment1_out, \
                moment2_out = adam_step(self.inputs, self.attrs)

            beta1_pow_out = self.inputs['Beta1Pow'] * self.beta1
            beta2_pow_out = self.inputs['Beta2Pow'] * self.beta2
            self.outputs = {
                'Moment1Out': moment1_out,
                'Moment2Out': moment2_out,
                'ParamOut': param_out,
                'Beta1PowOut': beta1_pow_out,
                'Beta2PowOut': beta2_pow_out
            }

            # Verify output for this step
            self.check_output()

            # Output of this step becomes input for next step
            self.inputs['Param'] = param_out
            self.inputs['Moment1'] = moment1_out
            self.inputs['Moment2'] = moment2_out

            # Update powers of Beta1 and Beta2 for next time step
            self.inputs['Beta1Pow'] = beta1_pow_out
            self.inputs['Beta2Pow'] = beta2_pow_out

            # Randomize gradient for next step
            self.inputs['Grad'] = np.random.uniform(
                -1, 1, (102, 105)).astype("float32")


def adam_step(inputs, attributes):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    epsilon = attributes['epsilon']

    if 'beta1' in attributes:
        beta1 = attributes['beta1']
    else:
        beta1 = inputs['Beta1Tensor'][0]
    if 'beta2' in attributes:
        beta2 = attributes['beta2']
    else:
        beta2 = inputs['Beta2Tensor'][0]

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)
    lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
    param_out = param - lr_t * (moment1_out / (np.sqrt(moment2_out) + epsilon))
    return param_out, moment1_out, moment2_out


def adam_step_sparse(inputs, attributes, height, rows, row_numel, np_grad,
                     lazy_mode):
    '''
    Simulate one step of the adam optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    # grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']

    moment1_out = np.zeros(shape=[height, row_numel])
    moment2_out = np.zeros(shape=[height, row_numel])
    param_out = np.zeros(shape=[height, row_numel])

    def update_row(row_id, update_value):
        moment1_out[row_id] = beta1 * moment1[row_id] + (1 - beta1
                                                         ) * update_value
        moment2_out[row_id] = beta2 * moment2[row_id] + (
            1 - beta2) * np.square(update_value)
        lr_t = lr * np.sqrt(1 - beta2_pow) / (1 - beta1_pow)
        param_out[row_id] = param[row_id] - lr_t * (moment1_out[row_id] / (
            np.sqrt(moment2_out[row_id]) + epsilon))

    if lazy_mode:
        for idx, row_id in enumerate(rows):
            update_row(row_id, np_grad[idx])
    else:
        for row_id in range(param_out.shape[0]):
            update_value = np.zeros(np_grad[0].shape).astype("float32")
            if row_id in rows:
                update_value = np_grad[rows.index(row_id)]
            update_row(row_id, update_value)

    return param_out, moment1_out, moment2_out


class TestSparseAdamOp(unittest.TestCase):
    def setup(self, scope, place, lazy_mode):
        beta1 = 0.78
        beta2 = 0.836
        epsilon = 1e-4
        beta1_pow = np.array([beta1**10]).astype("float32")
        beta2_pow = np.array([beta2**10]).astype("float32")

        height = 10
        rows = [0, 4, 7]
        self.rows = rows
        row_numel = 12
        self.row_numel = row_numel
        self.dense_inputs = {
            "Param": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment1": np.full((height, row_numel), 5.0).astype("float32"),
            "Moment2": np.full((height, row_numel), 5.0).astype("float32"),
            'Beta1Pow': beta1_pow,
            'Beta2Pow': beta2_pow,
            "LearningRate": np.full((1), 2.0).astype("float32")
        }
        self.init_output = np.full((height, row_numel), 0.0).astype("float32")
        self.attrs = {
            'epsilon': epsilon,
            'beta1': beta1,
            'beta2': beta2,
            'min_row_size_to_use_multithread': 2
        }

        grad_selected_rows = scope.var('Grad').get_selected_rows()
        grad_selected_rows.set_height(height)
        grad_selected_rows.set_rows(rows)
        np_array = np.ones((len(rows), row_numel)).astype("float32")
        np_array[0, 0] = 2.0
        np_array[2, 8] = 4.0

        grad_tensor = grad_selected_rows.get_tensor()
        grad_tensor.set(np_array, place)

        self.sparse_inputs = ["Grad"]

        param_out, mom1, mom2 = adam_step_sparse(self.dense_inputs, self.attrs,
                                                 height, rows, row_numel,
                                                 np_array, lazy_mode)
        self.outputs = {
            "ParamOut": param_out,
            "Moment1Out": mom1,
            "Moment2Out": mom2,
            'Beta1PowOut': beta1_pow * beta1,
            'Beta2PowOut': beta2_pow * beta2
        }

    def check_with_place(self, place, lazy_mode):
        scope = core.Scope()
        self.setup(scope, place, lazy_mode)

        op_args = dict()
        op_args['lazy_mode'] = lazy_mode
        for key, np_array in self.dense_inputs.items():
            var = scope.var(key).get_tensor()
            var.set(np_array, place)
            op_args[key] = key
        for s in self.sparse_inputs:
            op_args[s] = s
        for s in self.outputs:
            var = scope.var(s).get_tensor()
            var.set(self.init_output, place)
            op_args[s] = s
        for k in self.attrs:
            op_args[k] = self.attrs[k]

        # create and run sgd operator
        adam_op = Operator("adam", **op_args)
        adam_op.run(scope, place)

        for key, np_array in self.outputs.items():
            out_var = scope.var(key).get_tensor()
            actual = np.array(out_var)
            actual = actual.reshape([actual.size])
            np_array = np_array.reshape([np_array.size])

            for i in range(np_array.size):
                self.assertLess((actual[i] - np_array[i]), 0.00001)

    def test_sparse_adam(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            for lazy_mode in (True, False):
                self.check_with_place(place, lazy_mode)


class TestAdamOpBetaVariable(OpTest):
    def setUp(self):
        '''Test Adam Op with beta as Variable
        '''
        self.op_type = "adam"
        param = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        grad = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        moment1 = np.random.uniform(-1, 1, (102, 105)).astype("float32")
        # The second moment is positive
        moment2 = np.random.random((102, 105)).astype("float32")
        beta1 = 0.85
        beta2 = 0.95

        learning_rate = 0.001
        epsilon = 1e-8
        beta1_pow = beta1**10
        beta2_pow = beta2**10

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32"),
            "Beta1Tensor": np.array([beta1]).astype("float32"),
            "Beta2Tensor": np.array([beta2]).astype("float32"),
        }

        attributes = {'epsilon': epsilon}

        param_out, moment1_out, \
            moment2_out = adam_step(self.inputs, attributes)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': np.array([beta1_pow]).astype("float32") * beta1,
            'Beta2PowOut': np.array([beta2_pow]).astype("float32") * beta2
        }

    def test_check_output(self):
        self.check_output()


class TestAdamOpV2(unittest.TestCase):
    def test_adam_op(self):
        place = fluid.CPUPlace()
        shape = [2, 3, 8, 8]
        exe = fluid.Executor(place)
        train_prog = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(train_prog, startup):
            with fluid.unique_name.guard():
                data = fluid.data(name="data", shape=shape)
                conv = fluid.layers.conv2d(data, 8, 3)
                loss = fluid.layers.reduce_mean(conv)

                beta1 = fluid.layers.create_global_var(
                    shape=[1], value=0.85, dtype='float32', persistable=True)
                beta2 = fluid.layers.create_global_var(
                    shape=[1], value=0.95, dtype='float32', persistable=True)
                betas = [beta1, beta2]
                opt = paddle.optimizer.Adam(
                    learning_rate=1e-5,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=0.01,
                    epsilon=1e-8)
                opt.minimize(loss)

        exe.run(startup)
        data_np = np.random.random(shape).astype('float32')
        rets = exe.run(train_prog, feed={"data": data_np}, fetch_list=[loss])
        assert rets[0] is not None

    def test_adam_op_dygraph(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = fluid.dygraph.to_variable(value)
        linear = fluid.Linear(13, 5, dtype="float32")

        adam = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=linear.parameters())
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_adam_op_with_state_dict(self):

        paddle.disable_static()
        emb = paddle.nn.Embedding(10, 10)

        adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
        state_dict = adam.state_dict()
        adam.set_state_dict(state_dict)

        #learning_rate is LRScheduler
        learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=0.1, T_max=10)
        adam = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            weight_decay=fluid.regularizer.L2Decay(0.001),
            parameters=emb.parameters())
        lr = adam.get_lr()
        state_dict = adam.state_dict()
        adam.set_state_dict(state_dict)

        #leanrning_rate is Tensor
        with self.assertRaises(TypeError):
            learning_rate = np.array([0.01]).astype("float32")
            learning_rate = paddle.to_tensor(learning_rate)
            adam = paddle.optimizer.Adam(
                learning_rate=learning_rate, parameters=emb.parameters())

        params = adam.get_opti_var_name_list()
        assert (params is not None)

    def test_adam_with_grad_clip(self):
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype("float32")
        a = fluid.dygraph.to_variable(value)
        linear = fluid.Linear(13, 5, dtype="float32")
        clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
        adam = paddle.optimizer.Adam(
            0.1, parameters=linear.parameters(), grad_clip=clip)
        out = linear(a)
        out.backward()
        adam.step()
        adam.clear_gradients()

    def test_adam_op_with_set_lr(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

        lr = 0.01
        adam.set_lr(lr)
        cur_lr = adam.get_lr()
        assert (lr == cur_lr)
        with self.assertRaises(TypeError):
            lr_var = paddle.fluid.layers.create_global_var(
                shape=[1], value=lr, dtype='float32')
            adam.set_lr(lr_var)

    def test_adam_op_invalid_input(self):
        paddle.disable_static()
        linear = paddle.nn.Linear(10, 10)
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adam(
                0.1, beta1=-1, parameters=linear.parameters())
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adam(
                0.1, beta2=-1, parameters=linear.parameters())
        with self.assertRaises(ValueError):
            adam = paddle.optimizer.Adam(
                0.1, epsilon=-1, parameters=linear.parameters())

    def test_adam_op_with_sparse_input_and_weight_decay(self):

        paddle.disable_static()
        x_data = np.arange(0, 10).reshape((10, 1)).astype(np.int64)
        x = paddle.to_tensor(x_data, stop_gradient=False)
        emb = paddle.nn.Embedding(10, 10, sparse=True)
        adam = paddle.optimizer.Adam(
            0.001, parameters=emb.parameters(), weight_decay=0.01)

        with self.assertRaises(RuntimeError):
            out = emb(x)
            out.backward()
            adam.step()


if __name__ == "__main__":
    unittest.main()
