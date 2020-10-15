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

from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
import os
from paddle.fluid.layer_helper import LayerHelper


def check():
    print("check: fluid.core.globals()['FLAGS_use_mkldnn']=",
          fluid.core.globals()["FLAGS_use_mkldnn"])
    print("check: fluid.get_flags('FLAGS_use_mkldnn')=",
          fluid.get_flags(['FLAGS_use_mkldnn']))
    print("check: DNNL_VERBOSE=", os.environ['DNNL_VERBOSE'])
    print("check: FLAGS_tracer_mkldnn_ops_on=",
          fluid.core.globals()['FLAGS_tracer_mkldnn_ops_on'])
    print("check: FLAGS_tracer_mkldnn_ops_off=",
          fluid.core.globals()['FLAGS_tracer_mkldnn_ops_off'])
    a_np = np.random.uniform(-2, 2, (10, 20, 30)).astype(np.float32)
    b_np = np.random.uniform(-5, 5, (10, 20, 30)).astype(np.float32)
    helper = LayerHelper(fluid.unique_name.generate(str("test")), act="relu")
    func = helper.append_activation
    with fluid.dygraph.guard(fluid.core.CPUPlace()):
        a = fluid.dygraph.to_variable(a_np)
        b = fluid.dygraph.to_variable(b_np)
        y = fluid.layers.elementwise_add(x=a, y=b)
        y = fluid.layers.matmul(x=y, y=b, transpose_y=True)
        res1 = func(y)

        np_res = np.add(a_np, b_np)
        np_res = np.matmul(np_res, np.transpose(b_np, (0, 2, 1)))
        np_res = np.maximum(np_res, 0)
    assert np.allclose(res1.numpy(), np_res, atol=1e-3)


if __name__ == '__main__':
    try:
        check()
    except Exception as e:
        print(e)
        print(type(e))
