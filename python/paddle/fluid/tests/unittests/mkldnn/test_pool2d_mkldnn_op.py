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
from paddle.fluid.tests.unittests.test_pool2d_op import TestPool2D_Op, TestCase1, TestCase2, TestCase3, TestCase4, TestCase5, avg_pool2D_forward_naive


def create_test_mkldnn_use_ceil_class(parent):
    class TestMKLDNNPool2DUseCeilCase(parent):
        def init_kernel_type(self):
            self.use_mkldnn = True

        def init_ceil_mode(self):
            self.ceil_mode = True

        def init_data_type(self):
            self.dtype = np.float32

    cls_name = "{0}_{1}".format(parent.__name__, "MKLDNNCeilModeCast")
    TestMKLDNNPool2DUseCeilCase.__name__ = cls_name
    globals()[cls_name] = TestMKLDNNPool2DUseCeilCase


create_test_mkldnn_use_ceil_class(TestPool2D_Op)
create_test_mkldnn_use_ceil_class(TestCase1)
create_test_mkldnn_use_ceil_class(TestCase2)


def create_test_mkldnn_class(parent):
    class TestMKLDNNCase(parent):
        def init_kernel_type(self):
            self.use_mkldnn = True

        def init_data_type(self):
            self.dtype = np.float32

    cls_name = "{0}_{1}".format(parent.__name__, "MKLDNNOp")
    TestMKLDNNCase.__name__ = cls_name
    globals()[cls_name] = TestMKLDNNCase


create_test_mkldnn_class(TestPool2D_Op)
create_test_mkldnn_class(TestCase1)
create_test_mkldnn_class(TestCase2)
create_test_mkldnn_class(TestCase3)
create_test_mkldnn_class(TestCase4)
create_test_mkldnn_class(TestCase5)


class TestAvgPoolAdaptive(TestPool2D_Op):
    def init_adaptive(self):
        self.adaptive = True

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_test_case(self):
        self.ksize = [1, 1]
        self.strides = [1, 1]

    def init_data_type(self):
        self.dtype = np.float32

    def init_global_pool(self):
        self.global_pool = False


class TestAvgPoolAdaptive2(TestAvgPoolAdaptive):
    def init_test_case(self):
        self.ksize = [2, 3]
        self.strides = [1, 1]

    def init_shape(self):
        self.shape = [2, 3, 6, 6]


class TestAsymPad(TestPool2D_Op):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [1, 0, 1, 0]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = False

    def init_shape(self):
        self.shape = [2, 3, 7, 7]

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_global_pool(self):
        self.global_pool = False

    def init_data_type(self):
        self.dtype = np.float32


class TestAsymPadCase1(TestAsymPad):
    def init_paddings(self):
        self.paddings = [1, 1, 0, 0]


class TestAsymPadCase2(TestAsymPad):
    def init_paddings(self):
        self.paddings = [1, 0, 1, 2]


class TestAsymPadCase3(TestAsymPad):
    def init_paddings(self):
        self.paddings = [1, 2, 1, 2]


class TestAsymPadCase4(TestAsymPad):
    def init_paddings(self):
        self.paddings = [1, 0, 1, 2]


class TestAsymPadCase5(TestAsymPad):
    def init_paddings(self):
        self.paddings = [2, 2, 1, 2]


class TestAsymPadMaxCase1(TestAsymPadCase1):
    def init_pool_type(self):
        self.pool_type = "max"


class TestAsymPadMaxCase2(TestAsymPadCase2):
    def init_pool_type(self):
        self.pool_type = "max"


class TestAsymPadMaxCase3(TestAsymPadCase3):
    def init_pool_type(self):
        self.pool_type = "max"


class TestAsymPadMaxCase4(TestAsymPadCase4):
    def init_pool_type(self):
        self.pool_type = "max"


class TestAsymPadMaxCase5(TestAsymPadCase5):
    def init_pool_type(self):
        self.pool_type = "max"


class TestAsymPadSame(TestAsymPad):
    def init_paddings(self):
        self.paddings = [0, 0]
        self.padding_algorithm = "SAME"


class TestAsymPadValid(TestAsymPad):
    def init_paddings(self):
        self.paddings = [0, 0, 0, 0]
        self.padding_algorithm = "VALID"


class TestAsymPadValidNHWC(TestAsymPadValid):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_shape(self):
        self.shape = [2, 7, 7, 3]


if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()
