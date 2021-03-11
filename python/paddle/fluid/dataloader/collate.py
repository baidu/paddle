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

import paddle
import numpy as np
from ..framework import in_dygraph_mode
from .. import core, layers

try:
    from collections.abc import Sequence, Mapping
except:
    from collections import Sequence, Mapping


def _to_tensor(data):
    if in_dygraph_mode():
        return paddle.to_tensor(data, place=paddle.CPUPlace())
    else:
        tensor = core.LoDTensor()
        tensor.set(data, core.CPUPlace())
        return tensor


def default_collate_fn(batch):
    """
    Default batch collating function for :code:`fluid.io.DataLoader`,
    batch should be a list of samples, and each sample should be a list
    of fields as follows:
    
    [[filed1, filed2, ...], [filed1, filed2, ...], ...]
    
    This default collate function zipped each filed together and stack
    each filed as the batch field as follows:

    [batch_filed1, batch_filed2, ...]

    Args:  
        batch(list of list of numpy array|paddle.Tensor): the batch data, each fields
              should be a numpy array, each sample should be a list of
              fileds, and batch should be a list of sample.
    
    Returns:
        a list of numpy array|Paddle.Tensor: collated batch of input batch data,
            fields data type as same as fields in each sample.
    """
    sample = batch[0]
    if isinstance(sample, np.ndarray):
        batch = np.stack(batch, axis=0)
        # if CPU-only version, pin_memory will be disabled,
        # so we compate tensor here
        if not paddle.is_compiled_with_cuda():
            batch = _to_tensor(batch)
        return batch
    elif isinstance(sample, paddle.Tensor):
        return layers.stack(batch, axis=0)
    elif isinstance(sample, (int, float)):
        batch = np.array(batch)
        if not paddle.is_compiled_with_cuda():
            batch = _to_tensor(batch)
        return batch
    elif isinstance(sample, Mapping):
        return {
            key: default_collate_fn([d[key] for d in batch])
            for key in sample
        }
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch")
        return [default_collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))
    return outputs


def default_convert_fn(batch):
    if isinstance(batch, paddle.Tensor):
        return batch
    elif isinstance(batch, np.ndarray):
        if not paddle.is_compiled_with_cuda():
            batch = _to_tensor(batch)
        return batch
    elif isinstance(batch, Mapping):
        return {key: default_convert_fn(batch[key]) for key in batch}
    elif isinstance(batch, Sequence):
        return [default_convert_fn(d) for d in batch]
    else:
        return batch
