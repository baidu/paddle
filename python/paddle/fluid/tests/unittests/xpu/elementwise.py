#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle
import paddle.fluid as fluid
paddle.enable_static()


class TestXPUElementwiseOpBase(object):
    def setUp(self, op_type):
        self.op_type = op_type
        self.attrs = {'use_xpu': True}
        self.is_common_broadcast = False
        self.is_x_size_less_than_y = False
        self.grad_implemented = False
        self.y_grad_implemented = True
        self.dtype = np.float32
        self.__class__.op_type = self.op_type
        self.__class__.use_xpu = True
        self.__class__.dtype = self.dtype

    def net(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.layers.data(
                name='X', shape=self.inputs['X'].shape, dtype=self.dtype)
            y = fluid.layers.data(
                name='Y', shape=self.inputs['Y'].shape, dtype=self.dtype)
            op = getattr(fluid.layers, self.op_type)
            z = op(x, y)
            exe = fluid.Executor(place)
            z_value = exe.run(feed=self.inputs, fetch_list=[z.name])

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            if not self.is_common_broadcast and not self.is_x_size_less_than_y:
                self.check_output_with_place(place, atol=1e-3)
            else:
                with self.assertRaises(BaseException):
                    self.net(place)

    def _check_grad_xpu_helper(self,
                               inputs_to_check,
                               output_names,
                               no_grad_set=None,
                               max_relative_error=0.05):
        if self.grad_implemented and not self.is_common_broadcast   \
          and not self.is_x_size_less_than_y:
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    inputs_to_check,
                    output_names,
                    no_grad_set=no_grad_set,
                    max_relative_error=max_relative_error)

    def test_check_grad_normal(self):
        self._check_grad_xpu_helper(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self._check_grad_xpu_helper(['Y'], 'Out', set("X"))

    def test_check_grad_ingore_y(self):
        if self.y_grad_implemented:
            self._check_grad_xpu_helper(['X'], 'Out', set("Y"))

    def init_axis(self):
        self.axis = -1

    def make_input(self, x_shape=[13, 17], y_shape=[13, 17]):
        self.inputs = {
            'X': np.random.uniform(0.1, 1, x_shape).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, y_shape).astype(self.dtype)
        }

    def reshape_input(self, x_shape=None, y_shape=None):
        if x_shape is None:
            x = self.inputs['X']
        else:
            x = self.inputs['X'].reshape(x_shape)
        if y_shape is None:
            y = self.inputs['Y']
        else:
            y = self.inputs['Y'].reshape(y_shape)
        return x, y

    def make_output(self, x_shape=None, y_shape=None):
        pass
