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

from __future__ import print_function
from __future__ import division

import unittest
import numpy as np

import paddle.fluid.core as core
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def adaptive_pool3d_forward(x,
                            output_size,
                            adaptive=True,
                            data_format='NCDHW',
                            pool_type='avg'):

    N = x.shape[0]
    C, D, H, W = [x.shape[1], x.shape[2], x.shape[3], x.shape[4]] \
        if data_format == 'NCDHW' else [x.shape[4], x.shape[1], x.shape[2],x.shape[3]]

    if (isinstance(output_size, int) or output_size == None):
        H_out = output_size
        W_out = output_size
        D_out = output_size
        output_size = [D_out, H_out, W_out]
    else:
        D_out, H_out, W_out = output_size

    if output_size[0] == None:
        output_size[0] = D
        D_out = D
    if output_size[1] == None:
        output_size[1] = H
        H_out = H
    if output_size[2] == None:
        output_size[2] = W
        W_out = W

    out = np.zeros((N, C, D_out, H_out, W_out)) if data_format=='NCDHW' \
        else np.zeros((N, D_out, H_out, W_out, C))
    for k in range(D_out):
        d_start = adaptive_start_index(k, D, output_size[0])
        d_end = adaptive_end_index(k, D, output_size[0])

        for i in range(H_out):
            h_start = adaptive_start_index(i, H, output_size[1])
            h_end = adaptive_end_index(i, H, output_size[1])

            for j in range(W_out):
                w_start = adaptive_start_index(j, W, output_size[2])
                w_end = adaptive_end_index(j, W, output_size[2])

                if data_format == 'NCDHW':
                    x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:
                                 w_end]
                    if pool_type == 'avg':
                        field_size = (d_end - d_start) * (h_end - h_start) * (
                            w_end - w_start)
                        out[:, :, k, i, j] = np.sum(x_masked,
                                                    axis=(2, 3, 4)) / field_size
                    elif pool_type == 'max':
                        out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                elif data_format == 'NDHWC':
                    x_masked = x[:, d_start:d_end, h_start:h_end, w_start:
                                 w_end, :]
                    if pool_type == 'avg':
                        field_size = (d_end - d_start) * (h_end - h_start) * (
                            w_end - w_start)
                        out[:, k, i, j, :] = np.sum(x_masked,
                                                    axis=(1, 2, 3)) / field_size
                    elif pool_type == 'max':
                        out[:, k, i, j, :] = np.max(x_masked, axis=(1, 2, 3))
    return out


class TestAdaptiveAvgPool3dAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[3, 3, 3], pool_type="avg")

        self.res_2_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=5, pool_type="avg")

        self.res_3_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[2, 3, 5], pool_type="avg")

        self.res_4_np = adaptive_pool3d_forward(
            x=self.x_np,
            output_size=[3, 3, 3],
            pool_type="avg",
            data_format="NDHWC")

        self.res_5_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[None, 3, None], pool_type="avg")

    def test_static_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(name="x", shape=[2, 3, 5, 7, 7], dtype="float32")

            out_1 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[3, 3, 3])

            out_2 = paddle.nn.functional.adaptive_avg_pool3d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[2, 3, 5])

            out_4 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[3, 3, 3], data_format="NDHWC")

            out_5 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[None, 3, None])

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_4, res_5] = exe.run(
                fluid.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_4, out_5])

            assert np.allclose(res_1, self.res_1_np)

            assert np.allclose(res_2, self.res_2_np)

            assert np.allclose(res_3, self.res_3_np)

            assert np.allclose(res_4, self.res_4_np)

            assert np.allclose(res_5, self.res_5_np)

    def test_dynamic_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
            x = paddle.to_tensor(self.x_np)

            out_1 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[3, 3, 3])

            out_2 = paddle.nn.functional.adaptive_avg_pool3d(x=x, output_size=5)

            out_3 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[2, 3, 5])

            out_4 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[3, 3, 3], data_format="NDHWC")

            out_5 = paddle.nn.functional.adaptive_avg_pool3d(
                x=x, output_size=[None, 3, None])

            out_6 = paddle.nn.functional.interpolate(
                x=x, mode="area", size=[2, 3, 5])

            assert np.allclose(out_1.numpy(), self.res_1_np)

            assert np.allclose(out_2.numpy(), self.res_2_np)

            assert np.allclose(out_3.numpy(), self.res_3_np)

            assert np.allclose(out_4.numpy(), self.res_4_np)

            assert np.allclose(out_5.numpy(), self.res_5_np)

            assert np.allclose(out_6.numpy(), self.res_3_np)


class TestAdaptiveAvgPool3dClassAPI(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.random([2, 3, 5, 7, 7]).astype("float32")
        self.res_1_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[3, 3, 3], pool_type="avg")

        self.res_2_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=5, pool_type="avg")

        self.res_3_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[2, 3, 5], pool_type="avg")

        self.res_4_np = adaptive_pool3d_forward(
            x=self.x_np,
            output_size=[3, 3, 3],
            pool_type="avg",
            data_format="NDHWC")

        self.res_5_np = adaptive_pool3d_forward(
            x=self.x_np, output_size=[None, 3, None], pool_type="avg")

    def test_static_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.enable_static()
            x = paddle.fluid.data(name="x", shape=[2, 3, 5, 7, 7], dtype="float32")

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[3, 3, 3])
            out_1 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(output_size=5)
            out_2 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[2, 3, 5])
            out_3 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[3, 3, 3], data_format="NDHWC")
            out_4 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[None, 3, None])
            out_5 = adaptive_avg_pool(x=x)

            exe = paddle.static.Executor(place=place)
            [res_1, res_2, res_3, res_4, res_5] = exe.run(
                fluid.default_main_program(),
                feed={"x": self.x_np},
                fetch_list=[out_1, out_2, out_3, out_4, out_5])

            assert np.allclose(res_1, self.res_1_np)

            assert np.allclose(res_2, self.res_2_np)

            assert np.allclose(res_3, self.res_3_np)

            assert np.allclose(res_4, self.res_4_np)

            assert np.allclose(res_5, self.res_5_np)

    def test_dynamic_graph(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            paddle.disable_static(place=place)
            x = paddle.to_tensor(self.x_np)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[3, 3, 3])
            out_1 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(output_size=5)
            out_2 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[2, 3, 5])
            out_3 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[3, 3, 3], data_format="NDHWC")
            out_4 = adaptive_avg_pool(x=x)

            adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3d(
                output_size=[None, 3, None])
            out_5 = adaptive_avg_pool(x=x)

            assert np.allclose(out_1.numpy(), self.res_1_np)

            assert np.allclose(out_2.numpy(), self.res_2_np)

            assert np.allclose(out_3.numpy(), self.res_3_np)

            assert np.allclose(out_4.numpy(), self.res_4_np)

            assert np.allclose(out_5.numpy(), self.res_5_np)


if __name__ == '__main__':
    unittest.main()
