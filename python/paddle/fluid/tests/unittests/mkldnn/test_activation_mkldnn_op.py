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
from scipy.special import expit, erf
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
from paddle.fluid.tests.unittests.test_activation_op import TestActivation, TestRelu, TestTanh, TestSqrt, TestAbs, TestLeakyRelu, TestSwish, TestHardSwish, TestRelu6, TestSigmoid
from paddle.fluid.tests.unittests.test_gelu_op import gelu
from mkldnn_op_test import check_if_mkldnn_primitives_exist_in_bwd


class TestMKLDNNReluDim2(TestRelu):
    def setUp(self):
        super(TestMKLDNNReluDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNRelu6Dim2(TestRelu6):
    def setUp(self):
        super(TestMKLDNNRelu6Dim2, self).setUp()
        self.attrs.update({"use_mkldnn": True})

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNLeakyReluDim2(TestLeakyRelu):
    def setUp(self):
        super(TestMKLDNNLeakyReluDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNGeluDim2(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, False)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNGeluDim2Approx(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, True)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True, "approximate": True}


class TestMKLDNNTanhDim2(TestTanh):
    def setUp(self):
        super(TestMKLDNNTanhDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSqrtDim2(TestSqrt):
    def setUp(self):
        super(TestMKLDNNSqrtDim2, self).setUp()

        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNAbsDim2(TestAbs):
    def setUp(self):
        super(TestMKLDNNAbsDim2, self).setUp()
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSwishDim2(TestSwish):
    def setUp(self):
        super(TestMKLDNNSwishDim2, self).setUp()

        self.attrs["use_mkldnn"] = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNHardSwishDim2(TestHardSwish):
    def setUp(self):
        super(TestMKLDNNHardSwishDim2, self).setUp()

        self.attrs["use_mkldnn"] = True

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSigmoidDim2(TestSigmoid):
    def setUp(self):
        super(TestMKLDNNSigmoidDim2, self).setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNReluDim4(TestRelu):
    def setUp(self):
        super(TestMKLDNNReluDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNLeakyReluDim4(TestLeakyRelu):
    def setUp(self):
        super(TestMKLDNNLeakyReluDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0.02 * x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNGeluDim4(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(self.dtype)
        out = gelu(x, False)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNGeluDim4Approx(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(self.dtype)
        out = gelu(x, True)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True, "approximate": True}


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestMKLDNNGeluBf16Dim4(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.dtype = np.uint16

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(np.float32)
        out = convert_float_to_uint16(gelu(x, False))

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestMKLDNNGeluBf16Dim4Approx(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.dtype = np.uint16

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(np.float32)
        out = convert_float_to_uint16(gelu(x, True))

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True, "approximate": True}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass


class TestMKLDNNTanhDim4(TestTanh):
    def setUp(self):
        super(TestMKLDNNTanhDim4, self).setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.tanh(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNSqrtDim4(TestSqrt):
    def setUp(self):
        super(TestMKLDNNSqrtDim4, self).setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.sqrt(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNAbsDim4(TestAbs):
    def setUp(self):
        super(TestMKLDNNAbsDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        self.inputs = {'X': x}
        self.outputs = {'Out': np.abs(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSwishDim4(TestSwish):
    def setUp(self):
        super(TestMKLDNNSwishDim4, self).setUp()

        x = np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype(self.dtype)
        beta = 2.3
        out = x * expit(beta * x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True, "beta": beta}

    def init_dtype(self):
        self.dtype = np.float32


def ref_hardswish(x, threshold=6.0, scale=6.0, offset=3.0):
    return (x * np.minimum(np.maximum(x + offset, 0.), threshold) /
            scale).astype(x.dtype)


class TestMKLDNNHardSwishDim4(TestHardSwish):
    def setUp(self):
        super(TestMKLDNNHardSwishDim4, self).setUp()

        x = np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02

        out = ref_hardswish(x, threshold, scale, offset)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSigmoidDim4(TestSigmoid):
    def setUp(self):
        super(TestMKLDNNSigmoidDim4, self).setUp()

        x = np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}


# Check if primitives already exist in backward
class TestMKLDNNAbsPrimitivesAlreadyExist(unittest.TestCase):
    def setUp(self):
        super(TestMKLDNNAbsPrimitivesAlreadyExist, self).setUp()

        np.random.seed(123)
        self.op_type = 'abs'
        self.x = np.random.uniform(-1, 1, [2, 2]).astype(np.float32)
        self.out = np.abs(self.x)
        self.out_grad = np.random.random_sample(self.x.shape).astype(np.float32)
        self.x_grad = self.__abs_bwd(self.x, self.out_grad)

    # Abs grad calculation
    def __abs_bwd(self, x, out_grad):
        return out_grad * np.sign(x)

    def test_check(self):
        check_if_mkldnn_primitives_exist_in_bwd(
            self, self.op_type, self.x, self.out, self.out_grad, self.x_grad)


if __name__ == '__main__':
    unittest.main()
