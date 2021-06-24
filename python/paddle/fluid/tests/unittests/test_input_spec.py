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

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.static import InputSpec
from paddle.fluid.framework import core, convert_np_dtype_to_dtype_
from paddle.fluid.dygraph.dygraph_to_static.utils import _compatible_non_tensor_spec


class TestInputSpec(unittest.TestCase):
    def test_default(self):
        tensor_spec = InputSpec([3, 4])
        self.assertEqual(tensor_spec.dtype,
                         convert_np_dtype_to_dtype_('float32'))
        self.assertEqual(tensor_spec.name, None)

    def test_from_tensor(self):
        x_bool = fluid.layers.fill_constant(shape=[1], dtype='bool', value=True)
        bool_spec = InputSpec.from_tensor(x_bool)
        self.assertEqual(bool_spec.dtype, x_bool.dtype)
        self.assertEqual(list(bool_spec.shape), list(x_bool.shape))
        self.assertEqual(bool_spec.name, x_bool.name)

        bool_spec2 = InputSpec.from_tensor(x_bool, name='bool_spec')
        self.assertEqual(bool_spec2.name, bool_spec2.name)

    def test_from_numpy(self):
        x_numpy = np.ones([10, 12])
        x_np_spec = InputSpec.from_numpy(x_numpy)
        self.assertEqual(x_np_spec.dtype,
                         convert_np_dtype_to_dtype_(x_numpy.dtype))
        self.assertEqual(x_np_spec.shape, x_numpy.shape)
        self.assertEqual(x_np_spec.name, None)

        x_numpy2 = np.array([1, 2, 3, 4]).astype('int64')
        x_np_spec2 = InputSpec.from_numpy(x_numpy2, name='x_np_int64')
        self.assertEqual(x_np_spec2.dtype,
                         convert_np_dtype_to_dtype_(x_numpy2.dtype))
        self.assertEqual(x_np_spec2.shape, x_numpy2.shape)
        self.assertEqual(x_np_spec2.name, 'x_np_int64')

    def test_shape_with_none(self):
        tensor_spec = InputSpec([None, 4, None], dtype='int8', name='x_spec')
        self.assertEqual(tensor_spec.dtype, convert_np_dtype_to_dtype_('int8'))
        self.assertEqual(tensor_spec.name, 'x_spec')
        self.assertEqual(tensor_spec.shape, (-1, 4, -1))

    def test_shape_raise_error(self):
        # 1. shape should only contain int and None.
        with self.assertRaises(ValueError):
            tensor_spec = InputSpec(['None', 4, None], dtype='int8')

        # 2. shape should be type `list` or `tuple`
        with self.assertRaises(TypeError):
            tensor_spec = InputSpec(4, dtype='int8')

        # 3. len(shape) should be greater than 0.
        with self.assertRaises(ValueError):
            tensor_spec = InputSpec([], dtype='int8')

    def test_batch_and_unbatch(self):
        tensor_spec = InputSpec([10])
        # insert batch_size
        batch_tensor_spec = tensor_spec.batch(16)
        self.assertEqual(batch_tensor_spec.shape, (16, 10))

        # unbatch
        unbatch_spec = batch_tensor_spec.unbatch()
        self.assertEqual(unbatch_spec.shape, (10, ))

        # 1. `unbatch` requires len(shape) > 1
        with self.assertRaises(ValueError):
            unbatch_spec.unbatch()

        # 2. `batch` requires len(batch_size) == 1
        with self.assertRaises(ValueError):
            tensor_spec.batch([16, 12])

        # 3. `batch` requires type(batch_size) == int
        with self.assertRaises(TypeError):
            tensor_spec.batch('16')

    def test_eq_and_hash(self):
        tensor_spec_1 = InputSpec([10, 16], dtype='float32')
        tensor_spec_2 = InputSpec([10, 16], dtype='float32')
        tensor_spec_3 = InputSpec([10, 16], dtype='float32', name='x')
        tensor_spec_4 = InputSpec([16], dtype='float32', name='x')

        # override ``__eq__`` according to [shape, dtype, name]
        self.assertTrue(tensor_spec_1 == tensor_spec_2)
        self.assertTrue(tensor_spec_1 != tensor_spec_3)  # different name
        self.assertTrue(tensor_spec_3 != tensor_spec_4)  # different shape

        # override ``__hash__``  according to [shape, dtype]
        self.assertTrue(hash(tensor_spec_1) == hash(tensor_spec_2))
        self.assertTrue(hash(tensor_spec_1) == hash(tensor_spec_3))
        self.assertTrue(hash(tensor_spec_3) != hash(tensor_spec_4))


class NetWithNonTensorSpec(paddle.nn.Layer):
    def __init__(self, in_num, out_num):
        super(NetWithNonTensorSpec, self).__init__()
        self.linear_1 = paddle.nn.Linear(in_num, out_num)
        self.bn_1 = paddle.nn.BatchNorm1D(out_num)

        self.linear_2 = paddle.nn.Linear(in_num, out_num)
        self.bn_2 = paddle.nn.BatchNorm1D(out_num)

        self.linear_3 = paddle.nn.Linear(in_num, out_num)
        self.bn_3 = paddle.nn.BatchNorm1D(out_num)

    def forward(self, x, bool_v=False, str_v="bn", int_v=1, list_v=None):
        x = self.linear_1(x)
        if 'bn' in str_v:
            x = self.bn_1(x)

        if bool_v:
            x = self.linear_2(x)
            x = self.bn_2(x)

        config = {"int_v": int_v, 'other_key': "value"}
        if list_v and list_v[-1] > 2:
            x = self.linear_3(x)
            x = self.another_func(x, config)

        out = paddle.mean(x)
        return out

    def another_func(self, x, config=None):
        # config is a dict actually
        use_bn = config['int_v'] > 0

        x = self.linear_1(x)
        if use_bn:
            x = self.bn_3(x)

        return x


class TestNetWithNonTensorSpec(unittest.TestCase):
    def setUp(self):
        self.in_num = 16
        self.out_num = 16
        self.x_spec = paddle.static.InputSpec([-1, 16], name='x')
        self.x = paddle.randn([4, 16])

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()

    def test_non_tensor_bool(self):
        specs = [self.x_spec, False]
        self.check_result(specs, 'bool')

    def test_non_tensor_str(self):
        specs = [self.x_spec, True, "xxx"]
        self.check_result(specs, 'str')

    def test_non_tensor_int(self):
        specs = [self.x_spec, True, "bn", 10]
        self.check_result(specs, 'int')

    def test_non_tensor_list(self):
        specs = [self.x_spec, False, "bn", -10, [4]]
        self.check_result(specs, 'list')

    def check_result(self, specs, path):
        path = './net_non_tensor_' + path

        net = NetWithNonTensorSpec(self.in_num, self.out_num)
        net.eval()
        # dygraph out
        dy_out = net(self.x, *specs[1:])

        # jit.save directly
        paddle.jit.save(net, path + '_direct', input_spec=specs)
        load_net = paddle.jit.load(path + '_direct')
        load_net.eval()
        pred_out = load_net(self.x)

        self.assertTrue(np.allclose(dy_out, pred_out))

        # @to_static by InputSpec
        net = paddle.jit.to_static(net, input_spec=specs)
        st_out = net(self.x, *specs[1:])

        self.assertTrue(np.allclose(dy_out, st_out))

        # jit.save and jit.load
        paddle.jit.save(net, path)
        load_net = paddle.jit.load(path)
        load_net.eval()
        load_out = load_net(self.x)

        self.assertTrue(np.allclose(st_out, load_out))

    def test_spec_compatible(self):
        net = NetWithNonTensorSpec(self.in_num, self.out_num)

        specs = [self.x_spec, False, "bn", -10]
        net = paddle.jit.to_static(net, input_spec=specs)
        net.eval()

        path = './net_twice'

        # NOTE: check input_specs_compatible
        new_specs = [self.x_spec, True, "bn", 10]
        with self.assertRaises(ValueError):
            paddle.jit.save(net, path, input_spec=new_specs)

        dy_out = net(self.x)

        paddle.jit.save(net, path, [self.x_spec, False, "bn"])
        load_net = paddle.jit.load(path)
        load_net.eval()
        pred_out = load_net(self.x)

        self.assertTrue(np.allclose(dy_out, pred_out))


class NetWithNonTensorSpecPrune(paddle.nn.Layer):
    def __init__(self, in_num, out_num):
        super(NetWithNonTensorSpecPrune, self).__init__()
        self.linear_1 = paddle.nn.Linear(in_num, out_num)
        self.bn_1 = paddle.nn.BatchNorm1D(out_num)

    def forward(self, x, y, use_bn=False):
        x = self.linear_1(x)
        if use_bn:
            x = self.bn_1(x)

        out = paddle.mean(x)

        if y is not None:
            loss = paddle.mean(y) + out

        return out, loss


class TestNetWithNonTensorSpecWithPrune(unittest.TestCase):
    def setUp(self):
        self.in_num = 16
        self.out_num = 16
        self.x_spec = paddle.static.InputSpec([-1, 16], name='x')
        self.y_spec = paddle.static.InputSpec([16], name='y')
        self.x = paddle.randn([4, 16])
        self.y = paddle.randn([16])

    @classmethod
    def setUpClass(cls):
        paddle.disable_static()

    def test_non_tensor_with_prune(self):
        specs = [self.x_spec, self.y_spec, True]
        path = './net_non_tensor_prune_'

        net = NetWithNonTensorSpecPrune(self.in_num, self.out_num)
        net.eval()
        # dygraph out
        dy_out, _ = net(self.x, self.y, *specs[2:])

        # jit.save directly
        paddle.jit.save(net, path + '_direct', input_spec=specs)
        load_net = paddle.jit.load(path + '_direct')
        load_net.eval()
        pred_out, _ = load_net(self.x, self.y)

        self.assertTrue(np.allclose(dy_out, pred_out))

        # @to_static by InputSpec
        net = paddle.jit.to_static(net, input_spec=specs)
        st_out, _ = net(self.x, self.y, *specs[2:])

        self.assertTrue(np.allclose(dy_out, st_out))

        # jit.save and jit.load with prune y and loss
        prune_specs = [self.x_spec, True]
        paddle.jit.save(net, path, prune_specs, output_spec=[st_out])
        load_net = paddle.jit.load(path)
        load_net.eval()
        load_out = load_net(self.x)  # no y and no loss

        self.assertTrue(np.allclose(st_out, load_out))


class UnHashableObject:
    def __init__(self, val):
        self.val = val

    def __hash__(self):
        raise TypeError("Unsupported to call hash()")


class TestCompatibleNonTensorSpec(unittest.TestCase):
    def test_case(self):
        self.assertTrue(_compatible_non_tensor_spec([1, 2, 3], [1, 2, 3]))
        self.assertFalse(_compatible_non_tensor_spec([1, 2, 3], [1, 2]))
        self.assertFalse(_compatible_non_tensor_spec([1, 2, 3], [1, 3, 2]))

        # not supported unhashable object.
        self.assertTrue(
            _compatible_non_tensor_spec(
                UnHashableObject(1), UnHashableObject(1)))


if __name__ == '__main__':
    unittest.main()
