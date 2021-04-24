# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest as unittest


class TestGetAndSetFlags(unittest.TestCase):
    def test_api(self):
        flags = {
            'FLAGS_eager_delete_tensor_gb': 1.0,
            'FLAGS_check_nan_inf': True
        }

        paddle.set_flags(flags)

        flags_list = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
        flag = 'FLAGS_eager_delete_tensor_gb'

        res_list = paddle.get_flags(flags_list)
        res = paddle.get_flags(flag)

        self.assertTrue(res_list['FLAGS_eager_delete_tensor_gb'], 1.0)
        self.assertTrue(res_list['FLAGS_check_nan_inf'], True)
        self.assertTrue(res['FLAGS_eager_delete_tensor_gb'], 1.0)


class TestGetAndSetFlagsErrors(unittest.TestCase):
    def test_errors(self):
        flags_list = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
        flag = 1
        flag_private = {'FLAGS_free_idle_chunk': True}

        # flags type of set_flags should be dict.
        def test_set_flags_input_type():
            paddle.set_flags(flags_list)

        self.assertRaises(TypeError, test_set_flags_input_type)

        # flags in set_flags should be public flags.
        def test_set_private_flag():

            paddle.set_flags(flag_private)

        self.assertRaises(ValueError, test_set_private_flag)

        # flags type of set_flags should be list, tuple or string
        def test_get_flags_input_type():
            paddle.get_flags(flag)

        self.assertRaises(TypeError, test_get_flags_input_type)

        # flags in get_flags should be public flags.
        def test_get_private_flag():
            paddle.get_flags('FLAGS_free_idle_chunk')

        self.assertRaises(ValueError, test_get_private_flag)


if __name__ == '__main__':
    unittest.main()
