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

from . import core
import numpy as np

__all__ = ['create_lod_tensor', 'create_random_int_lodtensor']


def create_lod_tensor(data, recursive_seq_lens, place):
    """
    Create a lod tensor from a numpy array, a list, or an existing lod tensor.

    Create a lod tensor by doing the following:

    1. Check that the length-based level of detail (LoD) also known as
       recursive_sequence_lengths of the input is valid.

    2. Convert recursive_sequence_lengths to a offset-based LoD.

    3. Copy the data from a numpy array, a list or a existing lod tensor to
       CPU or GPU device (based on input place).

    4. Set the level of detail (LoD) using the offset-based LoD.

    Examples:

        Suppose we want LoDTensor to hold data for sequences of word, where each
        word is represented by an integer. If we want to create a LoDTensor to
        represent two sentences, one of 2 words, and one of 3 words.

        Then :code:`data` can be a numpy array of integers with shape (5, 1).
        :code:`recursive_seq_lens` will be [[2, 3]], indicating the length(# of words) in each
        sentence. This length-based :code:`recursive_seq_lens` [[2, 3]] will be converted to
        offset-based LoD [[0, 2, 5]] inside the function call.

    Please reference :ref:`api_guide_low_level_lod_tensor` for more details
    regarding LoD.

    Args:
        data(numpy.ndarray|list|LoDTensor): a numpy array or a LoDTensor or a
            list holding the data to be copied.
        recursive_seq_lens(list): a list of lists indicating the length-based level of detail
            info specified by the user.
        place(Place): CPU or GPU place indicating where the data in the new
            LoDTensor will be stored.

    Returns:
        A fluid LoDTensor object with tensor data and recursive_seq_lens info.
    """
    if isinstance(data, core.LoDTensor):
        return create_lod_tensor(np.array(data), recursive_seq_lens, place)
    elif isinstance(data, list):
        # When input data is a list, it only deal with the case where the base element
        # is an index of shape [1] and dtype int64 (e.g., word id). Hence, the generated
        # LoDTensor will be of shape [n, 1] and dtype int64, where `n` is the total number
        # of words or other indexes in the sequence.
        new_recursive_seq_lens = []
        for seq in data:
            new_recursive_seq_lens.append(len(seq))
        assert [
            new_recursive_seq_lens
        ] == recursive_seq_lens, "data and recursive_seq_lens do not match"
        flattened_data = np.concatenate(data, axis=0).astype("int64")
        flattened_data = flattened_data.reshape([len(flattened_data), 1])
        return create_lod_tensor(flattened_data, recursive_seq_lens, place)
    elif isinstance(data, np.ndarray):
        tensor = core.LoDTensor()
        tensor.set(data, place)
        tensor.set_recursive_sequence_lengths(recursive_seq_lens)
        assert tensor.has_valid_recursive_sequence_lengths(
        ), "the provided lod info is invalid"
        return tensor
    else:
        raise TypeError(
            "data should be either a LoDTensor, a Numpy array or a list")


def create_random_int_lodtensor(recursive_seq_lens, base_shape, place, low,
                                high):
    """
    Create a LoDTensor containing random integers.

    This function is frequently used in the book examples. So we revised it
    based on the new create_lod_tensor API and put it here in the lod_tensor
    module to simplify the code.

    The function does the following:

    1. Calculate the overall shape of the LoDTensor based on the length-based
       :code:`recursive_seq_lens` input and the shape of the basic element in
       :code:`base_shape`.

    2. Create a numpy array of this shape.

    3. Create the LoDTensor using create_lod_tensor API.

    Suppose we want LoDTensor to hold data for sequences of word, where each
    word is represented by an integer. If we want to create a LoDTensor to
    represent two sentences, one of 2 words, and one of 3 words. Then
    'base_shape' is [1], input length-based 'recursive_seq_lens' is [[2, 3]].
    Then the overall shape of the LoDTensor would be [5, 1], holding 5 words
    for two sentences.

    Args:
        recursive_seq_lens(list): a list of lists indicating the length-based
            level of detail info specified by the user.
        base_shape(list): the shape of the basic element to be held by the
            LoDTensor.
        place(Place): CPU or GPU place indicating where the data in the new
            LoDTensor will be stored.
        low(int): the lower bound of the random integers.
        high(int): the upper bound of the random integers.

    Returns:
        A fluid LoDTensor object with tensor data and recursive_seq_lens info.
    """
    assert isinstance(base_shape, list), "base_shape should be a list"
    # append the total number of basic elements to the front of its shape
    overall_shape = [sum(recursive_seq_lens[-1])] + base_shape
    # the range of integer data elements is [low, high]
    data = np.random.random_integers(low, high, overall_shape).astype("int64")
    return create_lod_tensor(data, recursive_seq_lens, place)
