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

from op_test import OpTest
import numpy as np
import unittest
import paddle.fluid as fluid
import paddle

paddle.enable_static()


def strided_slice_native_forward(input, axes, starts, ends, strides):
    dim = input.ndim
    start = []
    end = []
    stride = []
    for i in range(dim):
        start.append(0)
        end.append(input.shape[i])
        stride.append(1)

    for i in range(len(axes)):
        start[axes[i]] = starts[i]
        end[axes[i]] = ends[i]
        stride[axes[i]] = strides[i]

    result = {
        1: lambda input, start, end, stride: input[start[0]:end[0]:stride[0]],
        2: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1]],
        3: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2]],
        4: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3]],
        5: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3], start[4]:end[4]:stride[4]],
        6: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3], \
                start[4]:end[4]:stride[4], start[5]:end[5]:stride[5]]
    }[dim](input, start, end, stride)

    return result


class TestStrideSliceOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'strided_slice'
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(set(['Input']), 'Out')

    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [-4]
        self.ends = [-3]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp1(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp2(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [5]
        self.ends = [0]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp3(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100)
        self.axes = [0]
        self.starts = [-1]
        self.ends = [-3]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp4(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 10)
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp5(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 1, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp6(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5)
        self.axes = [0, 1, 2]
        self.starts = [1, -1, 0]
        self.ends = [2, -3, 3]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp7(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(5, 5, 5)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 2, 3]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOp8(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1)
        self.axes = [1]
        self.starts = [1]
        self.ends = [2]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOp9(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 100, 1)
        self.axes = [1]
        self.starts = [-1]
        self.ends = [-2]
        self.strides = [-1]
        self.infer_flags = [1]


class TestStrideSliceOp10(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(10, 10)
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]
        self.infer_flags = [1, 1]


class TestStrideSliceOp11(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4)
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp12(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5)
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOp13(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8)
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]


class TestStrideSliceOpBool(TestStrideSliceOp):
    def test_check_grad(self):
        pass


class TestStrideSliceOpBool1D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(100).astype("bool")
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]
        self.infer_flags = [1]


class TestStrideSliceOpBool2D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(10, 10).astype("bool")
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]
        self.infer_flags = [1, 1]


class TestStrideSliceOpBool3D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 10).astype("bool")
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]
        self.infer_flags = [1, 1, 1]


class TestStrideSliceOpBool4D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4).astype("bool")
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOpBool5D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5).astype("bool")
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]
        self.infer_flags = [1, 1, 1, 1]


class TestStrideSliceOpBool6D(TestStrideSliceOpBool):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8).astype("bool")
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]
        self.infer_flags = [1, 1, 1, 1, 1]


class TestStridedSliceOp_starts_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.config()

        starts_tensor = []
        for index, ele in enumerate(self.starts):
            starts_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {'Input': self.input, 'StartsTensorList': starts_tensor}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts_infer,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [1, -1, 1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

        self.starts_infer = [1, 10, 2]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestStridedSliceOp_ends_ListTensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.config()

        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs = {'Input': self.input, 'EndsTensorList': ends_tensor}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends_infer,
            'strides': self.strides,
            'infer_flags': self.infer_flags
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 0]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 2]
        self.infer_flags = [1, -1, 1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

        self.ends_infer = [3, 1, 4]

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestStridedSliceOp_starts_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.config()
        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype="int32")
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestStridedSliceOp_ends_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.config()
        self.inputs = {
            'Input': self.input,
            "EndsTensor": np.array(
                self.ends, dtype="int32")
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            #'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestStridedSliceOp_listTensor_Tensor(OpTest):
    def setUp(self):
        self.config()
        ends_tensor = []
        for index, ele in enumerate(self.ends):
            ends_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))
        self.op_type = "strided_slice"

        self.inputs = {
            'Input': self.input,
            "StartsTensor": np.array(
                self.starts, dtype="int32"),
            "EndsTensorList": ends_tensor
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            #'starts': self.starts,
            #'ends': self.ends,
            'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, 1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


class TestStridedSliceOp_strides_Tensor(OpTest):
    def setUp(self):
        self.op_type = "strided_slice"
        self.config()
        self.inputs = {
            'Input': self.input,
            "StridesTensor": np.array(
                self.strides, dtype="int32")
        }
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            #'strides': self.strides,
            'infer_flags': self.infer_flags,
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float64")
        self.starts = [1, -1, 2]
        self.ends = [2, 0, 4]
        self.axes = [0, 1, 2]
        self.strides = [1, -1, 1]
        self.infer_flags = [-1, -1, -1]
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Input'], 'Out', max_relative_error=0.006)


# Test python API
class TestStridedSliceAPI(unittest.TestCase):
    def test_1(self):
        input = np.random.random([3, 4, 5, 6]).astype("float64")
        minus_1 = fluid.layers.fill_constant([1], "int32", -1)
        minus_3 = fluid.layers.fill_constant([1], "int32", -3)
        starts = fluid.layers.data(
            name='starts', shape=[3], dtype='int32', append_batch_size=False)
        ends = fluid.layers.data(
            name='ends', shape=[3], dtype='int32', append_batch_size=False)
        strides = fluid.layers.data(
            name='strides', shape=[3], dtype='int32', append_batch_size=False)

        x = fluid.layers.data(
            name="x",
            shape=[3, 4, 5, 6],
            append_batch_size=False,
            dtype="float64")
        out_1 = fluid.layers.strided_slice(
            x,
            axes=[0, 1, 2],
            starts=[-3, 0, 2],
            ends=[3, 100, -1],
            strides=[1, 1, 1])
        out_2 = fluid.layers.strided_slice(
            x,
            axes=[0, 1, 3],
            starts=[minus_3, 0, 2],
            ends=[3, 100, -1],
            strides=[1, 1, 1])
        out_3 = fluid.layers.strided_slice(
            x,
            axes=[0, 1, 3],
            starts=[minus_3, 0, 2],
            ends=[3, 100, minus_1],
            strides=[1, 1, 1])
        out_4 = fluid.layers.strided_slice(
            x, axes=[0, 1, 2], starts=starts, ends=ends, strides=strides)

        out_5 = x[-3:3, 0:100:2, -1:2:-1]
        out_6 = x[minus_3:3:1, 0:100:2, :, minus_1:2:minus_1]
        out_7 = x[minus_1, 0:100:2, :, -1:2:-1]

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3, res_4, res_5, res_6, res_7 = exe.run(
            fluid.default_main_program(),
            feed={
                "x": input,
                'starts': np.array([-3, 0, 2]).astype("int32"),
                'ends': np.array([3, 2147483648, -1]).astype("int64"),
                'strides': np.array([1, 1, 1]).astype("int32")
            },
            fetch_list=[out_1, out_2, out_3, out_4, out_5, out_6, out_7])
        assert np.array_equal(res_1, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_2, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_3, input[-3:3, 0:100, :, 2:-1])
        assert np.array_equal(res_4, input[-3:3, 0:100, 2:-1, :])
        assert np.array_equal(res_5, input[-3:3, 0:100:2, -1:2:-1, :])
        assert np.array_equal(res_6, input[-3:3, 0:100:2, :, -1:2:-1])
        assert np.array_equal(res_7, input[-1, 0:100:2, :, -1:2:-1])

    def test_dygraph_op(self):
        x = paddle.zeros(shape=[3, 4, 5, 6], dtype="float32")
        axes = [1, 2, 3]
        starts = [-3, 0, 2]
        ends = [3, 2, 4]
        strides_1 = [1, 1, 1]
        sliced_1 = paddle.strided_slice(
            x, axes=axes, starts=starts, ends=ends, strides=strides_1)
        assert sliced_1.shape == (3, 2, 2, 2)

    @unittest.skipIf(not paddle.is_compiled_with_cuda(),
                     "Cannot use CUDAPinnedPlace in CPU only version")
    def test_cuda_pinned_place(self):
        with paddle.fluid.dygraph.guard():
            x = paddle.to_tensor(
                np.random.randn(2, 10), place=paddle.CUDAPinnedPlace())
            self.assertTrue(x.place.is_cuda_pinned_place())
            y = x[:, ::2]
            self.assertFalse(x.place.is_cuda_pinned_place())
            self.assertFalse(y.place.is_cuda_pinned_place())


if __name__ == "__main__":
    unittest.main()
