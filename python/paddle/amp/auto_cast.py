#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.dygraph.amp import amp_guard

__all__ = []


def auto_cast(enable=True, custom_white_list=None, custom_black_list=None):
    """
    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.
    If enabled, the input data type (float32 or float16) of each operator is decided 
    by autocast algorithm for better performance. 
    
    Commonly, it is used together with `GradScaler` to achieve Auto-Mixed-Precision in 
    imperative mode.

    Args:
        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
        custom_white_list(set|list|tuple, optional): The custom white_list. It's the set of ops that support
             fp16 calculation and are considered numerically-safe and performance-critical. These ops 
             will be converted to fp16.
        custom_black_list(set|list|tuple, optional): The custom black_list. The set of ops that support fp16
             calculation and are considered numerically-dangerous and whose effects may also be 
             observed in downstream ops. These ops will not be converted to fp16.
        
    Examples:

     .. code-block:: python

        import paddle

        conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast():
            conv = conv2d(data)
            print(conv.dtype) # FP16

        with paddle.amp.auto_cast(enable=False):
            conv = conv2d(data)
            print(conv.dtype) # FP32

        with paddle.amp.auto_cast(custom_black_list={'conv2d'}):
            conv = conv2d(data)
            print(conv.dtype) # FP32

        a = paddle.rand([2,3])
        b = paddle.rand([2,3])
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
            c = a + b
            print(c.dtype) # FP16

    """
    return amp_guard(enable, custom_white_list, custom_black_list)
