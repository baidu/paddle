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

import six
import sys
import paddle
import numpy as np
from .. import core
from ..multiprocess_utils import MP_STATUS_CHECK_INTERVAL
from ..framework import in_dygraph_mode, _set_expected_place

# NOTE: queue has a different name in python2 and python3
if six.PY2:
    import Queue as queue
else:
    import queue

try:
    from collections.abc import Sequence, Mapping
except:
    from collections import Sequence, Mapping


def pin_memory(data):
    if isinstance(data, np.ndarray):
        if in_dygraph_mode():
            return paddle.to_tensor(data, place=paddle.CUDAPinnedPlace())
        else:
            tensor = core.LoDTensor()
            tensor.set(data, core.CUDAPinnedPlace())
            return tensor
    if isinstance(data, paddle.Tensor):
        return data.pin_memory()
    if isinstance(data, paddle.fluid.LoDTensor):
        if in_dygraph_mode():
            # LoDTensor -> paddle.Tensor(VarBase)
            return core.VarBase(data)
        else:
            data_ = data.pin_memory()
            del data
            return data_
    if isinstance(data, Sequence):
        return [pin_memory(d) for d in data]
    if isinstance(data, Mapping):
        return {k: pin_memory(d) for k, d in data.items()}
    if isinstance(data, tuple) or hasattr(data, '_fields'):
        return type(data)(*(pin_memory(d) for d in data))
    else:
        return data


def _pin_memory_loop(in_queue, out_queue, done_event, legacy_expected_place):
    #NOTE(zhiqiu): Set the expected place for new thread as the same as father thread,
    # and it will call platform::SetDeviceId() in c++ internally.
    # If we do not set cudaDeviceId in new thread, the default cudaDeviceId will be 0,
    # Which may cost hundreds of MB of GPU memory on CUDAPlace(0) if calling some cuda 
    # APIs in this thread.
    _set_expected_place(legacy_expected_place)

    while not done_event.is_set():
        try:
            result = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue

        idx, data = result
        if not done_event.is_set():
            try:
                data = pin_memory(data)
            except Exception as e:
                six.reraise(*sys.exc_info())
            result = (idx, data)

        while not done_event.is_set():
            try:
                out_queue.put(result, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

        # delete result to save memory
        del result
