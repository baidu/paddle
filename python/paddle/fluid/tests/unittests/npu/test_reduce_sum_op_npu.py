#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestReduceSum(OpTest):
    def setUp(self):
        np.random.seed(SEED)
        self.set_npu()
        self.init_dtype()
        self.place = paddle.NPUPlace(0)
        self.init_op_type()
        self.initTestCase()

        self.use_mkldnn = False
        self.attrs = {
            'dim': self.axis,
            'keep_dim': self.keep_dim,
            'reduce_all': self.reduce_all
        }
        self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
        if self.attrs['reduce_all']:
            self.outputs = {'Out': self.inputs['X'].sum()}
        else:
            self.outputs = {
                'Out': self.inputs['X'].sum(axis=self.axis,
                                            keepdims=self.attrs['keep_dim'])
            }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_op_type(self):
        self.op_type = "reduce_sum"
        self.use_mkldnn = False
        self.keep_dim = False
        self.reduce_all = False

    def initTestCase(self):
        self.shape = (5, 6)
        self.axis = (0, )

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    # TODO(ascendrc): Add grad test
    # def test_check_grad(self):
    #     if self.dtype == np.float16:
    #         return
    #     self.check_grad(['X'], 'Out')
    #


class TestReduceSum2(OpTest):
    def init_dtype(self):
        self.dtype = np.int32


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestReduceSumNet(unittest.TestCase):
    def set_reduce_sum_function(self, x):
        # keep_dim = False
        return paddle.fluid.layers.reduce_sum(x, dim=-1)
    
    """
    def _get_tensor(self, name, value, place=None):
        block = fluid.default_main_program().global_block()
        var = create_var(
                name="e_1",
                dtype='float32',
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
                stop_gradient=True)
        return var, tensor
    """

    def _touch_inf(self, place):
        zero = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0.0)
        return fluid.layers.elementwise_div(zero, zero)

    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        a_np = np.random.random(size=(2, 3, 4)).astype('float32')
        b_np = np.random.random(size=(2, 3, 4)).astype('float32')
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype='float32')
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype='float32')
            label = paddle.static.data(
                name="label", shape=[2, 1], dtype='int64')

            zero = self._touch_inf(place)
            a_1 = fluid.layers.fc(input=a, size=4, num_flatten_dims=2, act=None)
            b_1 = fluid.layers.fc(input=b, size=4, num_flatten_dims=2, act=None)
            z = paddle.add(a_1, b_1)
            z_1 = self.set_reduce_sum_function(z)

            prediction = fluid.layers.fc(input=z_1, size=2, act='softmax')

            cost = fluid.layers.cross_entropy(input=prediction, label=label)
            loss = fluid.layers.reduce_mean(cost)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)


        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            pred_res, loss_res, zero_res = exe.run(
                main_prog,
                feed={"a": a_np,
                      "b": b_np,
                      "label": label_np},
                fetch_list=[prediction, loss, zero])
            if epoch % 10 == 0:
                print("Epoch {} | Prediction[0]: {}, Loss: {}, zero:{}".format(
                    epoch, pred_res[0], loss_res, zero_res))

        return pred_res, loss_res

    def test_npu(self):
        cpu_pred, cpu_loss = self._test(False)
        npu_pred, npu_loss = self._test(True)

        self.assertTrue(np.allclose(npu_pred, cpu_pred))
        self.assertTrue(np.allclose(npu_loss, cpu_loss))


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestReduceSumNet2(TestReduceSumNet):
    def set_reduce_sum_function(self, x):
        # keep_dim = True
        return paddle.fluid.layers.reduce_sum(x, dim=-1, keep_dim=True)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestReduceSumNet3(TestReduceSumNet):
    def _test(self, run_npu=True):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 3, 4)).astype('float32')
        b_np = np.random.random(size=(2, 3, 4)).astype('float32')

        with paddle.static.program_guard(main_prog, startup_prog):
            a = paddle.static.data(name="a", shape=[2, 3, 4], dtype='float32')
            b = paddle.static.data(name="b", shape=[2, 3, 4], dtype='float32')

            z = paddle.add(a, b)
            loss = fluid.layers.reduce_sum(z)
            sgd = fluid.optimizer.SGD(learning_rate=0.01)
            sgd.minimize(loss)

        if run_npu:
            place = paddle.NPUPlace(0)
        else:
            place = paddle.CPUPlace()

        exe = paddle.static.Executor(place)
        exe.run(startup_prog)

        print("Start run on {}".format(place))
        for epoch in range(100):

            loss_res = exe.run(main_prog,
                               feed={"a": a_np,
                                     "b": b_np},
                               fetch_list=[loss])
            if epoch % 10 == 0:
                print("Epoch {} | Loss: {}".format(epoch, loss_res))

        return loss_res, loss_res


if __name__ == '__main__':
    unittest.main()
