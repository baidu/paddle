#    Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import math
import sys
from op_test import OpTest


class TestROIAlignOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_roi_align()

        self.inputs = {
            'X': self.x,
            'ROIs': (self.rois[:, 1:5], self.rois_lod),
        }
        self.attrs = {
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
            'sampling_ratio': self.sampling_ratio,
            'aligned': self.aligned,
        }

        self.outputs = {'Out': self.out_data}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3
        self.height = 8
        self.width = 6

        # n, c, h, w
        self.x_dim = (self.batch_size, self.channels, self.height, self.width)

        self.spatial_scale = 1.0 / 2.0
        self.pooled_height = 2
        self.pooled_width = 2
        self.sampling_ratio = -1
        self.aligned = False

        self.x = np.random.random(self.x_dim).astype('float64')

    def pre_calc(self, x_i, roi_xmin, roi_ymin, roi_bin_grid_h, roi_bin_grid_w,
                 bin_size_h, bin_size_w):
        count = roi_bin_grid_h * roi_bin_grid_w
        bilinear_pos = np.zeros(
            [self.channels, self.pooled_height, self.pooled_width, count, 4],
            np.float64)
        bilinear_w = np.zeros(
            [self.pooled_height, self.pooled_width, count, 4], np.float64)
        for ph in range(self.pooled_width):
            for pw in range(self.pooled_height):
                c = 0
                for iy in range(roi_bin_grid_h):
                    y = roi_ymin + ph * bin_size_h + (iy + 0.5) * \
                        bin_size_h / roi_bin_grid_h
                    for ix in range(roi_bin_grid_w):
                        x = roi_xmin + pw * bin_size_w + (ix + 0.5) * \
                            bin_size_w / roi_bin_grid_w
                        if y < -1.0 or y > self.height or \
                               x < -1.0 or x > self.width:
                            continue
                        if y <= 0:
                            y = 0
                        if x <= 0:
                            x = 0
                        y_low = int(y)
                        x_low = int(x)
                        if y_low >= self.height - 1:
                            y = y_high = y_low = self.height - 1
                        else:
                            y_high = y_low + 1
                        if x_low >= self.width - 1:
                            x = x_high = x_low = self.width - 1
                        else:
                            x_high = x_low + 1
                        ly = y - y_low
                        lx = x - x_low
                        hy = 1 - ly
                        hx = 1 - lx
                        for ch in range(self.channels):
                            bilinear_pos[ch, ph, pw, c, 0] = x_i[ch, y_low,
                                                                 x_low]
                            bilinear_pos[ch, ph, pw, c, 1] = x_i[ch, y_low,
                                                                 x_high]
                            bilinear_pos[ch, ph, pw, c, 2] = x_i[ch, y_high,
                                                                 x_low]
                            bilinear_pos[ch, ph, pw, c, 3] = x_i[ch, y_high,
                                                                 x_high]
                        bilinear_w[ph, pw, c, 0] = hy * hx
                        bilinear_w[ph, pw, c, 1] = hy * lx
                        bilinear_w[ph, pw, c, 2] = ly * hx
                        bilinear_w[ph, pw, c, 3] = ly * lx
                        c = c + 1
        return bilinear_pos, bilinear_w

    def calc_roi_align(self):
        self.out_data = np.zeros(
            (self.rois_num, self.channels, self.pooled_height,
             self.pooled_width)).astype('float64')

        offset = 0.5 if self.aligned else 0.
        for i in range(self.rois_num):
            roi = self.rois[i]
            roi_batch_id = int(roi[0])
            x_i = self.x[roi_batch_id]
            roi_xmin = roi[1] * self.spatial_scale - offset
            roi_ymin = roi[2] * self.spatial_scale - offset
            roi_xmax = roi[3] * self.spatial_scale - offset
            roi_ymax = roi[4] * self.spatial_scale - offset

            roi_width = roi_xmax - roi_xmin
            roi_height = roi_ymax - roi_ymin
            if not self.aligned:
                roi_width = max(roi_width, 1)
                roi_height = max(roi_height, 1)

            bin_size_h = float(roi_height) / float(self.pooled_height)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            roi_bin_grid_h = self.sampling_ratio if self.sampling_ratio > 0 else \
                                 math.ceil(roi_height / self.pooled_height)
            roi_bin_grid_w = self.sampling_ratio if self.sampling_ratio > 0 else \
                                 math.ceil(roi_width / self.pooled_width)
            count = max(int(roi_bin_grid_h * roi_bin_grid_w), 1)
            pre_size = count * self.pooled_width * self.pooled_height
            bilinear_pos, bilinear_w = self.pre_calc(x_i, roi_xmin, roi_ymin,
                                                     int(roi_bin_grid_h),
                                                     int(roi_bin_grid_w),
                                                     bin_size_h, bin_size_w)
            for ch in range(self.channels):
                align_per_bin = (bilinear_pos[ch] * bilinear_w).sum(axis=-1)
                output_val = align_per_bin.mean(axis=-1)
                self.out_data[i, ch, :, :] = output_val

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.random_integers(
                    0, self.width // self.spatial_scale - self.pooled_width)
                y1 = np.random.random_integers(
                    0, self.height // self.spatial_scale - self.pooled_height)

                x2 = np.random.random_integers(x1 + self.pooled_width,
                                               self.width // self.spatial_scale)
                y2 = np.random.random_integers(
                    y1 + self.pooled_height, self.height // self.spatial_scale)

                roi = [bno, x1, y1, x2, y2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype("float64")

    def setUp(self):
        self.op_type = "roi_align"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestROIAlignInLodOp(TestROIAlignOp):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_roi_align()

        seq_len = self.rois_lod[0]

        self.inputs = {
            'X': self.x,
            'ROIs': (self.rois[:, 1:5], self.rois_lod),
            'RoisNum': np.asarray(seq_len).astype('int32')
        }

        self.attrs = {
            'spatial_scale': self.spatial_scale,
            'pooled_height': self.pooled_height,
            'pooled_width': self.pooled_width,
            'sampling_ratio': self.sampling_ratio,
            'aligned': self.aligned
        }

        self.outputs = {'Out': self.out_data}


class TestROIAlignOpWithAligned(TestROIAlignOp):
    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3
        self.height = 8
        self.width = 6

        # n, c, h, w
        self.x_dim = (self.batch_size, self.channels, self.height, self.width)

        self.spatial_scale = 1.0 / 2.0
        self.pooled_height = 2
        self.pooled_width = 2
        self.sampling_ratio = -1
        self.aligned = True

        self.x = np.random.random(self.x_dim).astype('float64')


if __name__ == '__main__':
    unittest.main()
