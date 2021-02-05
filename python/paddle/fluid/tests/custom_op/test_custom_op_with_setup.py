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

import os
import unittest
import numpy as np
from test_custom_op import CustomOpTest, load_so
from paddle.utils.cpp_extension.extension_utils import run_cmd
from paddle.fluid.layer_helper import LayerHelper


def compile_so():
    """
    Compile .so file by running setup.py config.
    """
    # build .so with setup.py
    file_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = 'cd {} && python setup_build.py build'.format(file_dir)
    run_cmd(cmd)


def relu3(x, name=None):
    helper = LayerHelper("relu3", **locals())
    out = helper.create_variable(
        type=x.type, name=name, dtype=x.dtype, persistable=False)
    helper.append_op(type="relu3", inputs={"X": x}, outputs={"Y": out})
    return out


class TestCompileMultiOp(unittest.TestCase):
    def test_relu3(self):
        raw_data = np.array([[-1, 1, 0], [1, -1, -1]]).astype('float32')
        x = paddle.to_tensor(raw_data, dtype='float32')
        # use custom api
        out = relu3(x)

        self.assertTrue(
            np.array_equal(out.numpy(),
                           np.array([[0, 1, 0], [1, 0, 0]]).astype('float32')))


if __name__ == '__main__':
    compile_so()
    load_so(so_name='librelu2_op_from_setup.so')
    unittest.main()
