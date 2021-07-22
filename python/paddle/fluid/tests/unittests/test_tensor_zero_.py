# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import unittest
import numpy as np
import six
import paddle


class TensorFill_Test(unittest.TestCase):
    def setUp(self):
        self.shape = [32, 32]

    def test_tensor_fill_true(self):
        typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
            places.append(fluid.CUDAPinnedPlace())

        for p in places:
            np_arr = np.reshape(
                np.array(six.moves.range(np.prod(self.shape))), self.shape)
            for dtype in typelist:
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                newtensor = tensor.clone()
                newtensor[...] = 0

                tensor.zero_()
                self.assertEqual(
                    (tensor.numpy() == newtensor.numpy()).all().item(), True)


if __name__ == '__main__':
    unittest.main()
