# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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
import paddle
from paddle.fluid.contrib import sparsity
from paddle.fluid.tests.unittests.asp.asp_pruning_base import TestASPHelperPruningBase

paddle.enable_static()


class TestASPHelperPruning1D(TestASPHelperPruningBase):
    def test_1D_inference_pruning(self):
        self.run_inference_pruning_test(sparsity.MaskAlgo.MASK_1D,
                                        sparsity.CheckMethod.CHECK_1D)

    def test_1D_training_pruning(self):
        self.run_training_pruning_test(sparsity.MaskAlgo.MASK_1D,
                                       sparsity.CheckMethod.CHECK_1D)


if __name__ == '__main__':
    unittest.main()
