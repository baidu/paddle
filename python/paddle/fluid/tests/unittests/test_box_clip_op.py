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

from __future__ import print_function

import unittest
import numpy as np
import sys
import math
from op_test import OpTest
import copy


def box_clip(input_box, im_shape, output_box):
    im_w = im_shape[1]
    im_h = im_shape[0]
    output_box[:, :, 0] = np.maximum(
        np.minimum(input_box[:, :, 0], im_w - 1), 0)
    output_box[:, :, 1] = np.maximum(
        np.minimum(input_box[:, :, 1], im_h - 1), 0)
    output_box[:, :, 2] = np.maximum(
        np.minimum(input_box[:, :, 2], im_w - 1), 0)
    output_box[:, :, 3] = np.maximum(
        np.minimum(input_box[:, :, 3], im_h - 1), 0)


def batch_box_clip(input_boxes, im_shape, lod):
    n = input_boxes.shape[0]
    m = input_boxes.shape[1]
    output_boxes = np.zeros((n, m, 4), dtype=np.float32)
    cur_offset = 0
    for i in range(len(lod)):
        box_clip(input_boxes[cur_offset:(cur_offset + lod[i]), :, :],
                 im_shape[i, :],
                 output_boxes[cur_offset:(cur_offset + lod[i]), :, :])
        cur_offset += lod[i]
    return output_boxes


class TestBoxClipOp(OpTest):
    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "box_clip"
        lod = [[1, 2, 3]]
        input_boxes = np.random.random((6, 10, 4)) * 5
        im_shape = np.array([[5, 8], [6, 6], [7, 5]])
        output_boxes = batch_box_clip(input_boxes, im_shape, lod[0])

        self.inputs = {
            'Input': (input_boxes.astype('float32'), lod),
            'ImShape': im_shape.astype('float32'),
        }
        self.outputs = {'Output': output_boxes}


class TestBoxClipWithRoisNumOp(TestBoxClipOp):
    def setUp(self):
        self.op_type = "box_clip"
        lod = [[1, 2, 3]]
        rois_num = lod[0]
        input_boxes = np.random.random((6, 10, 4)) * 5
        im_shape = np.array([[5, 8], [6, 6], [7, 5]])
        output_boxes = batch_box_clip(input_boxes, im_shape, lod[0])
        self.inputs = {
            'Input': (input_boxes.astype('float32')),
            'ImShape': im_shape.astype('float32'),
            'RoisNum': np.array(rois_num).astype('int32')
        }
        self.outputs = {'Output': output_boxes}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
