#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import skip_check_grad_ci
from paddle.fluid.tests.unittests.test_elementwise_mul_op import ElementwiseMulOp
from paddle import enable_static


class TestMKLDNNElementwiseMulOp(ElementwiseMulOp):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNElementwiseMulOp2(TestMKLDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.random((100, )).astype(self.dtype)
        self.y = np.random.random((100, )).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


class TestMKLDNNElementwiseMulOp3(TestMKLDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)


class TestMKLDNNElementwiseMulOp4(TestMKLDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 32]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [4, 32]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    # TODO(jczaja): Enable when grad is ready
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestMKLDNNElementwiseMulOp5(TestMKLDNNElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.uniform(1, 2, [2, 3, 4, 100]).astype(self.dtype)
        self.y = np.random.uniform(1, 2, [100]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    # TODO(jczaja): Enable when grad is ready
    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_y(self):
        pass

    def test_check_grad_ingore_x(self):
        pass


''' INT8 Tests '''


@skip_check_grad_ci(
    reason="oneDNN's int8 elementwise_ops don't implemend grad kernel.")
class TestInt8(ElementwiseMulOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self._cpu_only = True

    def init_dtype(self):
        self.dtype = np.int8

    def init_input_output(self):
        self.x = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.y = np.random.randint(0, 3, (12, 9)).astype("int8")
        self.out = np.multiply(self.x, self.y)

    def init_scales(self):
        self.attrs['Scale_x'] = 1.0
        self.attrs['Scale_y'] = 1.0
        self.attrs['Scale_out'] = 1.0

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.init_scales()
        self.check_output(check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestInt8Scales(TestInt8):
    def quantize(self, tensor, dt="int8"):
        max_int = 127.0 if dt == "int8" else 255.0
        scale = max_int / np.abs(np.amax(tensor))
        quantized = np.round(scale * tensor).astype(dt)
        return scale, quantized

    def init_input_output(self):
        self.x_f = np.random.random((100, )).astype("float")
        self.y_f = np.random.random((100, )).astype("float")
        self.out_f = np.multiply(self.x_f, self.y_f)

        self.scale_x, self.x = self.quantize(self.x_f)
        self.scale_y, self.y = self.quantize(self.y_f)
        self.scale_o, self.out = self.quantize(self.out_f)

    def init_scales(self):
        self.attrs['Scale_x'] = self.scale_x
        self.attrs['Scale_y'] = self.scale_y
        self.attrs['Scale_out'] = self.scale_o

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.init_scales()
        int_atol = 1  # different quantization techniques
        self.check_output(
            check_dygraph=(self.use_mkldnn == False), atol=int_atol)


class TestUint8Scales(TestInt8Scales):
    def init_input_output(self):
        self.x_f = np.random.random((100, )).astype("float")
        self.y_f = np.random.random((100, )).astype("float")
        self.out_f = np.multiply(self.x_f, self.y_f)

        self.scale_x, self.x = self.quantize(self.x_f, "uint8")
        self.scale_y, self.y = self.quantize(self.y_f, "uint8")
        self.scale_o, self.out = self.quantize(self.out_f, "uint8")

    def init_dtype(self):
        self.dtype = np.uint8


if __name__ == '__main__':
    enable_static()
    unittest.main()
