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
import paddle.fluid.framework as framework


class ConstantTest(unittest.TestCase):
    def test_const_value(self):
        self.assertEqual(framework.GRAD_VAR_SUFFIX, "@GRAD")
        self.assertEqual(framework.TEMP_VAR_NAME, "@TEMP@")
        self.assertEqual(framework.GRAD_VAR_SUFFIX, "@GRAD")
        self.assertEqual(framework.ZERO_VAR_SUFFIX, "@ZERO")
        self.assertEqual(1, 2)


if __name__ == '__main__':
    unittest.main()
