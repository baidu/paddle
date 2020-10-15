# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import OpTest
import paddle
import paddle.nn.functional as F
import paddle.fluid.core as core
import paddle.fluid as fluid


def pixel_shuffle_np(x, up_factor, data_format="NCHW"):
    if data_format == "NCHW":
        n, c, h, w = x.shape
        new_shape = (n, c // (up_factor * up_factor), up_factor, up_factor, h,
                     w)
        # reshape to (num,output_channel,upscale_factor,upscale_factor,h,w)
        npresult = np.reshape(x, new_shape)
        # transpose to (num,output_channel,h,upscale_factor,w,upscale_factor)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, c // (up_factor * up_factor), h * up_factor, w * up_factor]
        npresult = np.reshape(npresult, oshape)
        return npresult
    else:
        n, h, w, c = x.shape
        new_shape = (n, h, w, c // (up_factor * up_factor), up_factor,
                     up_factor)
        # reshape to (num,h,w,output_channel,upscale_factor,upscale_factor)
        npresult = np.reshape(x, new_shape)
        # transpose to (num,h,upscale_factor,w,upscale_factor,output_channel)
        npresult = npresult.transpose(0, 1, 4, 2, 5, 3)
        oshape = [n, h * up_factor, w * up_factor, c // (up_factor * up_factor)]
        npresult = np.reshape(npresult, oshape)
        return npresult


class TestPixelShuffleOp(OpTest):
    def setUp(self):
        self.op_type = "pixel_shuffle"
        self.init_data_format()
        n, c, h, w = 2, 9, 4, 4

        if self.format == "NCHW":
            shape = [n, c, h, w]
        if self.format == "NHWC":
            shape = [n, h, w, c]

        up_factor = 3

        x = np.random.random(shape).astype("float64")
        npresult = pixel_shuffle_np(x, up_factor, self.format)

        self.inputs = {'X': x}
        self.outputs = {'Out': npresult}
        self.attrs = {'upscale_factor': up_factor, "data_format": self.format}

    def init_data_format(self):
        self.format = "NCHW"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestChannelLast(TestPixelShuffleOp):
    def init_data_format(self):
        self.format = "NHWC"


class TestPixelShuffleAPI(unittest.TestCase):
    def setUp(self):
        self.x_1_np = np.random.random([2, 9, 4, 4]).astype("float64")
        self.x_2_np = np.random.random([2, 4, 4, 9]).astype("float64")
        self.out_1_np = pixel_shuffle_np(self.x_1_np, 3)
        self.out_2_np = pixel_shuffle_np(self.x_2_np, 3, "NHWC")

    def test_static_graph_functional(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            x_1 = paddle.fluid.data(name="x", shape=[2, 9, 4, 4], dtype="float64")
            x_2 = paddle.fluid.data(name="x2", shape=[2, 4, 4, 9], dtype="float64")
            out_1 = F.pixel_shuffle(x_1, 3)
            out_2 = F.pixel_shuffle(x_2, 3, "NHWC")

            exe = paddle.static.Executor(place=place)
            res_1 = exe.run(fluid.default_main_program(),
                            feed={"x": self.x_1_np},
                            fetch_list=out_1,
                            use_prune=True)

            res_2 = exe.run(fluid.default_main_program(),
                            feed={"x2": self.x_2_np},
                            fetch_list=out_2,
                            use_prune=True)

            assert np.allclose(res_1, self.out_1_np)
            assert np.allclose(res_2, self.out_2_np)

    # same test between layer and functional in this op.
    def test_static_graph_layer(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.enable_static()
            x_1 = paddle.fluid.data(name="x", shape=[2, 9, 4, 4], dtype="float64")
            x_2 = paddle.fluid.data(name="x2", shape=[2, 4, 4, 9], dtype="float64")
            # init instance
            ps_1 = paddle.nn.PixelShuffle(3)
            ps_2 = paddle.nn.PixelShuffle(3, "NHWC")
            out_1 = ps_1(x_1)
            out_2 = ps_2(x_2)
            out_1_np = pixel_shuffle_np(self.x_1_np, 3)
            out_2_np = pixel_shuffle_np(self.x_2_np, 3, "NHWC")

            exe = paddle.static.Executor(place=place)
            res_1 = exe.run(fluid.default_main_program(),
                            feed={"x": self.x_1_np},
                            fetch_list=out_1,
                            use_prune=True)

            res_2 = exe.run(fluid.default_main_program(),
                            feed={"x2": self.x_2_np},
                            fetch_list=out_2,
                            use_prune=True)

            assert np.allclose(res_1, out_1_np)
            assert np.allclose(res_2, out_2_np)

    def run_dygraph(self, up_factor, data_format):

        n, c, h, w = 2, 9, 4, 4

        if data_format == "NCHW":
            shape = [n, c, h, w]
        if data_format == "NHWC":
            shape = [n, h, w, c]

        x = np.random.random(shape).astype("float64")

        npresult = pixel_shuffle_np(x, up_factor, data_format)

        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

            paddle.disable_static(place=place)

            pixel_shuffle = paddle.nn.PixelShuffle(
                up_factor, data_format=data_format)
            result = pixel_shuffle(paddle.to_tensor(x))

            self.assertTrue(np.allclose(result.numpy(), npresult))

            result_functional = F.pixel_shuffle(
                paddle.to_tensor(x), 3, data_format)
            self.assertTrue(np.allclose(result_functional.numpy(), npresult))

    def test_dygraph1(self):
        self.run_dygraph(3, "NCHW")

    def test_dygraph2(self):
        self.run_dygraph(3, "NHWC")


class TestPixelShuffleError(unittest.TestCase):
    def test_error_functional(self):
        def error_upscale_factor():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                pixel_shuffle = F.pixel_shuffle(paddle.to_tensor(x), 3.33)

        self.assertRaises(TypeError, error_upscale_factor)

        def error_data_format():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                pixel_shuffle = F.pixel_shuffle(paddle.to_tensor(x), 3, "WOW")

        self.assertRaises(ValueError, error_data_format)

    def test_error_layer(self):
        def error_upscale_factor_layer():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                ps = paddle.nn.PixelShuffle(3.33)

        self.assertRaises(TypeError, error_upscale_factor_layer)

        def error_data_format_layer():
            with paddle.fluid.dygraph.guard():
                x = np.random.random([2, 9, 4, 4]).astype("float64")
                ps = paddle.nn.PixelShuffle(3, "MEOW")

        self.assertRaises(ValueError, error_data_format_layer)


if __name__ == '__main__':
    unittest.main()
