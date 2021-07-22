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

import numpy as np

__all__ = [
    'Stack',
    'Pad',
    'Tuple',
]


class Stack(object):
    """
    Stack the input data samples to construct the batch. The N input samples
    must have the same shape/length and will be stacked to construct a batch.

    Args:
        axis (int, optional): The axis in the result data along which the input
            data are stacked. Default: 0.
        dtype (str|numpy.dtype, optional): The value type of the output. If it
            is set to None, the input data type is used. Default: None.

    Example:
        .. code-block:: python

            from paddle.incubate.hapi.text.data_utils import Stack
            # Stack multiple lists
            a = [1, 2, 3, 4]
            b = [4, 5, 6, 8]
            c = [8, 9, 1, 2]
            Stack()([a, b, c])
            '''
            [[1 2 3 4]
             [4 5 6 8]
             [8 9 1 2]]
             '''
    """

    def __init__(self, axis=0, dtype=None):
        self._axis = axis
        self._dtype = dtype

    def __call__(self, data):
        """
        Batchify the input data by stacking.

        Args:
            data (list(numpy.ndarray)): The input data samples.
        Returns:
            numpy.ndarray: Stacked batch data.

        """
        data = np.stack(
            data,
            axis=self._axis).astype(self._dtype) if self._dtype else np.stack(
                data, axis=self._axis)
        return data


class Pad(object):
    """
    Return a callable that pads and stacks data.

    Args:
        pad_val (float|int, optional): The padding value. Default: 0.
        axis (int, optional): The axis to pad the arrays. The arrays will be
            padded to the largest dimension at axis. For example, 
            assume the input arrays have shape (10, 8, 5), (6, 8, 5), (3, 8, 5)
            and the axis is 0. Each input will be padded into 
            (10, 8, 5) and then stacked to form the final output, which has
            shape（3, 10, 8, 5). Default: 0.
        ret_length (bool|numpy.dtype, optional): If it is bool, indicate whether
            to return the valid length in the output, and the data type of
            returned length is int32 if True. If it is numpy.dtype, indicate the
            data type of returned length. Default: False.
        dtype (numpy.dtype, optional): The value type of the output. If it is
            set to None, the input data type is used. Default: None.

    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Pad
            # Inputs are multiple lists
            a = [1, 2, 3, 4]
            b = [4, 5, 6]
            c = [8, 2]
            Pad(pad_val=0)([a, b, c])
            '''
            [[1. 2. 3. 4.]
                [4. 5. 6. 0.]
                [8. 2. 0. 0.]]
            '''
     """

    def __init__(self, pad_val=0, axis=0, ret_length=None, dtype=None):
        self._pad_val = pad_val
        self._axis = axis
        self._ret_length = ret_length
        self._dtype = dtype

    def __call__(self, data):
        """
        Batchify the input data by padding The input can be list of numpy.ndarray. 

        The arrays will be padded to the largest dimension at axis and then
        stacked to form the final output.  In addition, the function will output
        the original dimensions at the axis if ret_length is not None.

        Args:
            data (list(numpy.ndarray)|list(list)): List of samples to pad and stack.

        Returns:
            numpy.ndarray|tuple: If `ret_length` is False, it is a numpy.ndarray \
                representing the padded batch data and the shape is (N, …). \
                Otherwise, it is a tuple, except for the padded batch data, the \
                tuple also includes a numpy.ndarray representing all samples' \
                original length shaped `(N,)`. 
        """
        arrs = [np.asarray(ele) for ele in data]
        original_length = [ele.shape[self._axis] for ele in arrs]
        max_size = max(original_length)
        ret_shape = list(arrs[0].shape)
        ret_shape[self._axis] = max_size
        ret_shape = (len(arrs), ) + tuple(ret_shape)
        ret = np.full(
            shape=ret_shape,
            fill_value=self._pad_val,
            dtype=arrs[0].dtype if self._dtype is None else self._dtype)
        for i, arr in enumerate(arrs):
            if arr.shape[self._axis] == max_size:
                ret[i] = arr
            else:
                slices = [slice(None) for _ in range(arr.ndim)]
                slices[self._axis] = slice(0, arr.shape[self._axis])
                if slices[self._axis].start != slices[self._axis].stop:
                    slices = [slice(i, i + 1)] + slices
                    ret[tuple(slices)] = arr
        if self._ret_length:
            return ret, np.asarray(
                original_length,
                dtype="int32") if self._ret_length == True else np.asarray(
                    original_length, self._ret_length)
        else:
            return ret


class Tuple(object):
    """
    Wrap multiple batchify functions together. The input functions will be applied
    to the corresponding input fields.
    
    Each sample should be a list or tuple containing multiple fields. The i'th
    batchify function stored in Tuple will be applied on the i'th field. 
    
    For example, when data sample is (nd_data, label), you can wrap two batchify
    functions using `Tuple(DataBatchify, LabelBatchify)` to batchify nd_data and
    label correspondingly.

    Args:
        fn (list|tuple|callable): The batchify functions to wrap.
        *args (tuple of callable): The additional batchify functions to wrap.

    Example:
        .. code-block:: python
            from paddle.incubate.hapi.text.data_utils import Tuple, Pad, Stack
            batchify_fn = Tuple(Pad(axis=0, pad_val=0), Stack())
    """

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn, ) + args
        for i, ele_fn in enumerate(self._fn):
            assert callable(
                ele_fn
            ), 'Batchify functions must be callable! type(fn[%d]) = %s' % (
                i, str(type(ele_fn)))

    def __call__(self, data):
        """
        Batchify data samples by applying each function on the corresponding data
        field, and each data field is produced by stacking the field data of samples.

        Args:
            data (list): The samples to batchfy. Each sample should contain N fields.

        Returns:
            tuple: A tuple composed of results from all including batchifying functions.
        """

        assert len(data[0]) == len(self._fn),\
            'The number of attributes in each data sample should contain' \
            ' {} elements'.format(len(self._fn))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            result = ele_fn([ele[i] for ele in data])
            if isinstance(result, (tuple, list)):
                ret.extend(result)
            else:
                ret.append(result)
        return tuple(ret)
