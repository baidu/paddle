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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import math


class Gsz:
    def __init__(self, h, w, gd, gh, gw, input_chans):
        self.h = h
        self.w = w
        self.gd = gd
        self.gh = gh
        self.gw = gw
        self.input_chans = input_chans


def diff_abs(x):
    eps = 1e-8
    return math.sqrt(x * x + eps)


def d_diff_abs(x):
    eps = 1e-8
    return x / math.sqrt(x * x + eps)


def weight_z(x):
    abx = diff_abs(x)
    return max(1.0 - abx, 0.0)


def d_weight_z(x):
    abx = diff_abs(x)
    if abx > 1.0:
        return 0.0
    else:
        return d_diff_abs(x)


def naive_bilateral_slice_forward(output, grid, guide, input, gsz, has_offset,
                                  total_count, output_chans):
    h = gsz.h
    w = gsz.w
    gd = gsz.gd
    gh = gsz.gh
    gw = gsz.gw
    input_chans = gsz.input_chans
    coeff_stride = input_chans
    grid_chans = input_chans * output_chans

    if has_offset:
        grid_chans += output_chans
        coeff_stride += 1

    for idx in range(total_count):
        x = idx % w
        y = idx // w % h
        out_c = (idx // (h * w)) % output_chans
        b = (idx // (output_chans * w * h))

        gx = (x + 0.5) * gw / (1.0 * w)
        gy = (y + 0.5) * gh / (1.0 * h)
        gz = guide[int(b), int(y), int(x)] * gd

        fx = int(np.floor(gx - 0.5))
        fy = int(np.floor(gy - 0.5))
        fz = int(np.floor(gz - 0.5))

        value = 0.0
        for in_c in range(0, coeff_stride):
            coeff_sample = 0.0

            for xx in range(fx, fx + 2):
                x_ = max(min(xx, gw - 1), 0)
                wx = max(1.0 - abs(xx + 0.5 - gx), 0.0)

                for yy in range(fy, fy + 2):
                    y_ = max(min(yy, gh - 1), 0)
                    wy = max(1.0 - abs(yy + 0.5 - gy), 0.0)

                    for zz in range(fz, fz + 2):
                        z_ = max(min(zz, gd - 1), 0)
                        wz = weight_z(zz + 0.5 - gz)
                        c_ = coeff_stride * out_c + in_c

                        coeff_sample += grid[int(b), int(c_), int(z_), int(y_),
                                             int(x_)] * wx * wy * wz

            if in_c < input_chans:
                value += coeff_sample * input[int(b), int(in_c), int(y), int(x)]
            else:
                value += coeff_sample
        output[int(b), int(out_c), int(y), int(x)] = value


def naive_bilateral_slice(x, guide, grid, has_offset):
    bs = x.shape[0]
    h = x.shape[2]
    w = x.shape[3]
    input_chans = x.shape[1]

    coeffs_chans = grid.shape[1]
    if has_offset:
        output_chans = coeffs_chans // (input_chans + 1)
    else:
        output_chans = coeffs_chans // input_chans

    output = np.zeros([bs, int(output_chans), h, w]).astype(x.dtype)

    gd = grid.shape[2]
    gh = grid.shape[3]
    gw = grid.shape[4]

    gsz = Gsz(h, w, gd, gh, gw, input_chans)
    total_count = bs * h * w * output.shape[1]
    naive_bilateral_slice_forward(output, grid, guide, x, gsz, has_offset,
                                  total_count, output.shape[1])
    return output


@unittest.skipIf(not paddle.fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestBilateralSliceOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'bilateral_slice'
        batch_size = 3
        h = 50
        w = 30
        c = 1
        gh = 5
        gw = 3
        gd = 2
        gc = 2
        x = np.random.rand(batch_size, c, h, w).astype(self.data_type)
        guide = np.random.rand(batch_size, h, w).astype(self.data_type)
        grid = np.random.rand(batch_size, gc, gd, gh, gw).astype(self.data_type)
        output_np = naive_bilateral_slice(x, guide, grid, self.has_offset)

        self.inputs = {'X': x, 'Grid': grid, 'Guide': guide}
        self.attrs = {'has_offset': self.has_offset, }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        place = paddle.fluid.CUDAPlace(0)
        self.check_output_with_place(place, atol=1e-5)
        self.check_output

    def test_check_grad(self):
        place = paddle.fluid.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out')

    def initTestCase(self):
        self.has_offset = False
        self.data_type = 'float64'


@unittest.skipIf(not paddle.fluid.is_compiled_with_cuda(),
                 'CPU testing is not supported')
class TestBilateralSliceOp1(TestBilateralSliceOp):
    def initTestCase(self):
        self.has_offset = True
        self.data_type = 'float32'


class TestBilateralSliceApi(unittest.TestCase):
    def test_api(self):
        x = paddle.fluid.data(
            name='x', shape=[None, 3, 25, 15], dtype='float32')
        guide = paddle.fluid.data(
            name='guide', shape=[None, 25, 15], dtype='float32')
        grid = paddle.fluid.data(
            name='grid', shape=[None, None, 8, 5, 3], dtype='float32')
        paddle.fluid.contrib.layers.bilateral_slice(x, guide, grid, False)

        if not paddle.fluid.is_compiled_with_cuda():
            return

        with paddle.fluid.dygraph.guard():
            x1 = paddle.rand([3, 1, 50, 30])
            guide1 = paddle.rand([3, 50, 30])
            grid1 = paddle.rand([3, 2, 2, 5, 3])

            paddle.fluid.contrib.bilateral_slice(x1, guide1, grid1, False)


if __name__ == "__main__":
    unittest.main()
