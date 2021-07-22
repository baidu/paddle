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

import contextlib
import functools
import os
import paddle
import paddle.fluid.core as core
import re


# Get cuda runtime version from source file
def get_cuda_runtime_version():
    command = os.popen("nvcc --version").read()
    command = re.split(' ', command)
    tag = 0
    for _ in command:
        tag += 1
        if _ == 'release':
            break
    return int(float(command[tag].replace(',', '')) * 1000)


# Check if tf32 is supported on current hardware
# returns true if cuda >= 11 && arch >= ampere
def tf32_is_not_fp32():
    if not core.is_compiled_with_cuda():
        return False
    if core.get_cuda_device_count() == 0:
        return False
    if get_cuda_runtime_version() < 11000:
        return False
    return True


# test tf32 off-mode
@contextlib.contextmanager
def tf32_off():
    old_allow_tf32_matmul = core.get_cublas_switch()
    try:
        core.set_cublas_switch(False)
        yield
    finally:
        core.set_cublas_switch(old_allow_tf32_matmul)


# test tf32 on-mode
@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    old_allow_tf32_matmul = core.get_cublas_switch()
    old_precision = self.precision
    try:
        core.set_cublas_switch(True)
        self.precision = tf32_precision
        yield
    finally:
        core.set_cublas_switch(old_allow_tf32_matmul)
        self.precision = old_precision


def tf32_on_and_off(tf32_precision=1e-5):
    """
    Specify the precision manuelly and
    assertTrue() will compare with reduced precison
    when its tf32 on-mode,

    Parameters:
        tf32_precision(float): A lower precision

    Returns:
        None

    Examples:

        .. code-block:: python
            default_precision=1e-6

            @tf32_on_and_off(0.001)
            def test_dygraph():
                if core.is_compiled_with_cuda():
                    place = fluid.CUDAPlace(0)
                    with fluid.dygraph.guard(place):
                        input_array1 = np.random.rand(8,8).astype("float32")
                        input_array2 = np.random.rand(8,8).astype("float32")
                        data1 = fluid.dygraph.to_variable(input_array1)
                        data2 = fluid.dygraph.to_variable(input_array2)
                        out = paddle.matmul(data1, data2)
                        expected_result = np.matmul(input_array1, input_array2)

                    self.assertTrue(np.allclose(expected_result, out.numpy(), default_precision))
                else:
                    pass
    """

    def with_tf32_disabled(self, function_call):
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(func):
        @functools.wraps(func)
        def wrapped(self):
            if tf32_is_not_fp32():
                with_tf32_disabled(self, lambda: func(self))
                with_tf32_enabled(self, lambda: func(self))
            else:
                func(self)

        return wrapped

    return wrapper
