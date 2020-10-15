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
import paddle.fluid as fluid


class TestUniqueName(unittest.TestCase):
    def test_guard(self):
        with fluid.unique_name.guard():
            name_1 = fluid.unique_name.generate('')

        with fluid.unique_name.guard():
            name_2 = fluid.unique_name.generate('')

        self.assertEqual(name_1, name_2)

        with fluid.unique_name.guard("A"):
            name_1 = fluid.unique_name.generate('')

        with fluid.unique_name.guard('B'):
            name_2 = fluid.unique_name.generate('')

        self.assertNotEqual(name_1, name_2)

    def test_generate(self):
        with fluid.unique_name.guard():
            name1 = fluid.unique_name.generate('fc')
            name2 = fluid.unique_name.generate('fc')
            name3 = fluid.unique_name.generate('tmp')
            self.assertNotEqual(name1, name2)
            self.assertEqual(name1[-2:], name3[-2:])


class TestImperativeUniqueName(unittest.TestCase):
    def test_name_generator(self):
        with fluid.dygraph.guard():
            tracer = fluid.framework._dygraph_tracer()
            tmp_var_0 = tracer._generate_unique_name()
            self.assertEqual(tmp_var_0, "dygraph_tmp_0")

            tmp_var_1 = tracer._generate_unique_name("dygraph_tmp")
            self.assertEqual(tmp_var_1, "dygraph_tmp_1")


if __name__ == '__main__':
    unittest.main()
