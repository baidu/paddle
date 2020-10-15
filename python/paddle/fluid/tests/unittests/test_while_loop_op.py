# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.backward import append_backward

paddle.enable_static()


class TestApiWhileLoop(unittest.TestCase):
    def test_var_tuple(self):
        def cond(i):
            return layers.less_than(i, ten)

        def body(i):
            return layers.elementwise_add(x=i, y=one)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.fill_constant(shape=[1], dtype='int64', value=0)
            one = layers.fill_constant(shape=[1], dtype='int64', value=1)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            out = layers.while_loop(cond, body, (i, ))

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, fetch_list=out)
        self.assertTrue(
            np.allclose(np.asarray(res[0]), np.full((1), 10, np.int64)))

    def test_var_list(self):
        def cond(i, mem):
            return layers.less_than(i, ten)

        def body(i, mem):
            mem = layers.elementwise_add(x=mem, y=one)
            i = layers.increment(i)
            return [i, mem]

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.zeros(shape=[1], dtype='int64')
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            mem = fluid.data(name='mem', shape=[10], dtype='float32')
            one = layers.fill_constant(shape=[10], dtype='float32', value=1)
            out = layers.while_loop(cond, body, [i, mem])

            data = np.random.rand(10).astype('float32')
            data_one = np.ones(10).astype('float32')

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, feed={'mem': data}, fetch_list=out)
        for i in range(10):
            data = np.add(data, data_one)
        self.assertTrue(np.allclose(np.asarray(res[1]), data))

    def test_var_dict(self):
        def cond(i, ten, test_dict, test_list, test_list_dict):
            return layers.less_than(i, ten)

        def body(i, ten, test_dict, test_list, test_list_dict):
            test_dict["test_key"] = i
            test_dict["test_key"] += 1

            test_list[0] = fluid.layers.reshape(test_list[0], [2, -1]) + 1

            test_list_dict[0]["test_key"] += 1
            test_list_dict[0]["test_key"] = fluid.layers.relu(test_list_dict[0][
                "test_key"])

            i = layers.increment(i)
            return [i, ten, test_dict, test_list, test_list_dict]

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.zeros(shape=[1], dtype='int64')
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            test_data = layers.fill_constant(shape=[1], dtype='int64', value=0)

            test_dict = {"test_key": test_data}
            test_list = [
                layers.fill_constant(
                    shape=[1, 2], dtype='int64', value=0)
            ]
            test_list_dict = [{
                "test_key": layers.fill_constant(
                    shape=[1], dtype='float32', value=0)
            }]

            i, ten, test_dict, test_list, test_list_dict = layers.while_loop(
                cond, body, [i, ten, test_dict, test_list, test_list_dict])
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program,
                      fetch_list=[
                          test_dict["test_key"], test_list[0],
                          test_list_dict[0]["test_key"]
                      ])
        self.assertTrue(
            np.allclose(
                np.asarray(res[0]),
                np.full(
                    shape=(1), fill_value=10, dtype=np.int64)))
        self.assertTrue(
            np.allclose(
                np.asarray(res[1]),
                np.full(
                    shape=(2, 1), fill_value=10, dtype=np.int64)))
        self.assertTrue(
            np.allclose(
                np.asarray(res[2]),
                np.full(
                    shape=(1), fill_value=10, dtype=np.float32)))


class TestApiWhileLoop_Nested(unittest.TestCase):
    def test_nested_net(self):
        def external_cond(i, j, init, sums):
            return layers.less_than(i, loop_len1)

        def external_body(i, j, init, sums):
            def internal_cond(j, init, sums):
                return layers.less_than(j, loop_len2)

            def internal_body(j, init, sums):
                init = layers.elementwise_add(x=init, y=ones)
                sums = layers.elementwise_add(x=init, y=sums)
                j = layers.increment(j)
                return [j, init, sums]

            result = layers.while_loop(internal_cond, internal_body,
                                       [j, init, sums])
            j = result[0]
            init = result[1]
            sums = result[2]
            sums = layers.elementwise_add(x=init, y=sums)
            i = layers.increment(i)
            return [i, j, init, sums]

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            i = layers.zeros(shape=[1], dtype='int64')
            j = layers.zeros(shape=[1], dtype='int64')
            init = fluid.data(name='init', shape=[3, 3], dtype='float32')
            sums = fluid.data(name='sums', shape=[3, 3], dtype='float32')
            loop_len1 = layers.fill_constant(shape=[1], dtype='int64', value=2)
            loop_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)
            ones = layers.fill_constant(shape=[3, 3], dtype='float32', value=1)

            out = layers.while_loop(external_cond, external_body,
                                    [i, j, init, sums])

            data = np.random.rand(3, 3).astype('float32')
            data_sums = np.zeros([3, 3]).astype('float32')

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program,
                      feed={'init': data,
                            'sums': data_sums},
                      fetch_list=out)
        for i in range(3):
            data = np.add(data, 1)
            data_sums = np.add(data, data_sums)
        for j in range(2):
            data_sums = np.add(data, data_sums)
        self.assertTrue(np.allclose(np.asarray(res[3]), data_sums))


class TestApiWhileLoop_Backward(unittest.TestCase):
    def test_while_loop_backward(self):
        def cond(i, x):
            return layers.less_than(i, eleven)

        def body(i, x):
            x = layers.elementwise_mul(x=i, y=i)
            i = layers.increment(i)
            return [i, x]

        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            i = fluid.data(name='i', shape=[1], dtype='float32')
            i.stop_gradient = False
            eleven = layers.fill_constant(shape=[1], dtype='float32', value=11)
            one = layers.fill_constant(shape=[1], dtype='float32', value=1)
            x = fluid.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False

            out = layers.while_loop(cond, body, [i, x])
            mean = layers.mean(out[1])
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        feed_i = np.ones(1).astype('float32')
        feed_x = np.ones(1).astype('float32')
        data = np.asarray([100]).astype('float32')
        i_grad = np.asarray([110]).astype('float32')

        res = exe.run(main_program,
                      feed={'i': feed_i,
                            'x': feed_x},
                      fetch_list=[mean.name, i.grad_name])
        self.assertTrue(np.allclose(np.asarray(res[0]), data))
        self.assertTrue(
            np.allclose(np.asarray(res[1]), i_grad),
            msg=" \nres = \n{} \n\n ans = \n{}".format(res[1], i_grad))

    def test_while_loop_backward2(self):
        def cond(i, x):
            return i < 3

        def body(i, x):
            x = x * i
            i = i + 1
            return [i, x]

        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            i = fluid.data(name='i', shape=[1], dtype='float32')
            i.stop_gradient = False
            x = fluid.data(name='x', shape=[1], dtype='float32')
            x.stop_gradient = False

            out = layers.while_loop(cond, body, [i, x])
            mean = layers.mean(out[1])
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        feed_i = np.ones(1).astype('float32')
        feed_x = np.ones(1).astype('float32')
        data = np.asarray([2]).astype('float32')
        i_grad = np.asarray([3]).astype('float32')
        x_grad = np.asarray([2]).astype('float32')

        res = exe.run(main_program,
                      feed={'i': feed_i,
                            'x': feed_x},
                      fetch_list=[mean.name, i.grad_name, x.grad_name])
        self.assertTrue(np.allclose(np.asarray(res[0]), data))
        self.assertTrue(
            np.allclose(np.asarray(res[1]), i_grad),
            msg=" \nres = \n{} \n\n ans = \n{}".format(res[1], i_grad))
        self.assertTrue(
            np.allclose(np.asarray(res[2]), x_grad),
            msg=" \nres = \n{} \n\n ans = \n{}".format(res[2], x_grad))


class TestApiWhileLoop_NestedWithBackwardAndLoDTensorArray(unittest.TestCase):
    def test_nested_net_with_backward_and_lodtensor(self):
        def external_cond(i, j, x, mem_array):
            return layers.less_than(i, array_len)

        def external_body(i, j, x, mem_array):
            def internal_cond(j, x, mem_array):
                return layers.less_than(j, array_len2)

            def internal_body(j, x, mem_array):
                inner_data = layers.array_read(array=data_array, i=j)
                inner_prev = layers.array_read(array=mem_array, i=j)
                inner_sum_0 = layers.elementwise_add(x=inner_data, y=inner_prev)
                inner_sum_1 = layers.elementwise_add(x=x, y=inner_sum_0)
                j = layers.increment(x=j, in_place=True)
                layers.array_write(inner_sum_1, i=j, array=mem_array)
                return [j, x, mem_array]

            outer_data = layers.array_read(array=data_array, i=i)
            outer_prev = layers.array_read(array=mem_array, i=i)
            outer_sum_0 = layers.elementwise_add(x=outer_data, y=outer_prev)
            outer_sum_1 = layers.elementwise_add(x=x, y=outer_sum_0)
            i = layers.increment(x=i, in_place=True)
            layers.array_write(outer_sum_1, i=i, array=mem_array)
            j, x, mem_array = layers.while_loop(internal_cond, internal_body,
                                                [j, x, mem_array])
            return [i, j, x, mem_array]

        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            d0 = fluid.data(name='d0', shape=[10], dtype='float32')
            d1 = fluid.data(name='d1', shape=[10], dtype='float32')
            d2 = fluid.data(name='d2', shape=[10], dtype='float32')
            x = fluid.data(name='x', shape=[10], dtype='float32')
            x.stop_gradient = False
            i = layers.zeros(shape=[1], dtype='int64')
            i.stop_gradient = True
            init = layers.zeros(shape=[10], dtype='float32')
            mem_array = layers.array_write(x=init, i=i)
            data_array = layers.array_write(x=d0, i=i)
            i = layers.increment(i)
            layers.array_write(d1, i, array=data_array)
            i = layers.increment(i)
            layers.array_write(d2, i, array=data_array)
            i = layers.zeros(shape=[1], dtype='int64')
            i.stop_gradient = True
            array_len = layers.fill_constant(shape=[1], dtype='int64', value=1)
            j = layers.fill_constant(shape=[1], dtype='int64', value=1)
            j.stop_gradient = True
            array_len2 = layers.fill_constant(shape=[1], dtype='int64', value=3)

            out = layers.while_loop(external_cond, external_body,
                                    [i, j, x, mem_array])

            sum_result = layers.array_read(array=mem_array, i=j)
            mean = layers.mean(sum_result)
            append_backward(mean)

            place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
            ) else fluid.CPUPlace()
            exe = fluid.Executor(place)

            d = []
            for i in range(3):
                d.append(np.random.random(size=[10]).astype('float32'))
            feed_x = np.ones(10).astype('float32')
            data_sum = d[0] + d[1] + d[2] + 3 * feed_x
            x_grad = [0.3] * 10
            res = exe.run(
                main_program,
                feed={'d0': d[0],
                      'd1': d[1],
                      'd2': d[2],
                      'x': feed_x},
                fetch_list=[sum_result.name, x.grad_name])
            self.assertTrue(np.allclose(res[0], data_sum))
            self.assertTrue(np.allclose(res[1], x_grad))


class TestApiWhileLoopWithSwitchCase(unittest.TestCase):
    def test_with_switch_case(self):
        def cond(i):
            return layers.less_than(i, ten)

        def body(i):
            def fn_add_three():
                data_add_three = layers.elementwise_add(x=i, y=three)
                return data_add_three

            def fn_square():
                data_mul_data = layers.elementwise_mul(x=i, y=i)
                return data_mul_data

            def fn_add_one():
                data_add_one = layers.elementwise_add(x=i, y=one)
                return data_add_one

            return layers.switch_case(
                branch_index=i,
                branch_fns={2: fn_add_three,
                            5: fn_square},
                default=fn_add_one)

        main_program = Program()
        startup_program = Program()
        with fluid.program_guard(main_program, startup_program):
            i = layers.fill_constant(shape=[1], dtype='int64', value=1)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            three = layers.fill_constant(shape=[1], dtype='int64', value=3)
            one = layers.fill_constant(shape=[1], dtype='int64', value=1)
            out = layers.while_loop(cond, body, [i])

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        res = exe.run(main_program, fetch_list=out)

        data = np.asarray([25]).astype('int64')
        self.assertTrue(np.allclose(np.asarray(res[0]), data))


class TestApiWhileLoop_Error(unittest.TestCase):
    def test_error(self):
        def cond_returns_constant(i):
            return 1

        def cond_returns_not_bool_tensor(i):
            return layers.increment(i)

        def cond_returns_bool_tensor(i):
            return layers.less_than(i, ten)

        def cond_returns_2d_tensor(i):
            return layers.less_than(i, ten_2d)

        def cond_receives_two_args(i, ten):
            return layers.less_than(i, ten)

        def body(i):
            return layers.increment(i)

        def body_returns_error_length(i):
            i = layers.increment(i)
            return [i, i]

        def body_returns_error_type(i, ten):
            return layers.increment(i)

        def cond_returns_with_mutable_dict(i, test_dict):
            return i > 0

        def body_returns_with_mutable_dict(i, test_dict):
            test_dict['new_key'] = layers.fill_constant(
                shape=[1], dtype='int64', value=1)
            return layers.increment(i), test_dict

        def cond_returns_with_mutable_list(i, test_list):
            return i > 0

        def body_returns_with_mutable_list(i, test_list):
            test_list.append(
                layers.fill_constant(
                    shape=[1], dtype='int64', value=1))
            return layers.increment(i), test_list

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            data = layers.fill_constant(shape=[1], dtype='int64', value=1)
            data_1d = layers.fill_constant(shape=[1], dtype='int64', value=1)
            data_2d = layers.fill_constant(shape=[2, 2], dtype='int64', value=1)
            ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
            ten_2d = layers.fill_constant(shape=[2, 2], dtype='int64', value=10)

            # The type of `cond` in Op(while_loop) must be callable
            def type_error_cond():
                out = layers.while_loop(data, body, [data_1d])

            self.assertRaises(TypeError, type_error_cond)

            # The type of `body` in Op(while_loop) must be callable
            def type_error_body():
                out = layers.while_loop(cond_returns_bool_tensor, data,
                                        [data_1d])

            self.assertRaises(TypeError, type_error_body)

            # The type of `loop_vars` in Op(while_loop) must be list or tuple
            def type_error_loop_vars():
                out = layers.while_loop(cond_returns_bool_tensor, body, data_1d)

            self.assertRaises(TypeError, type_error_loop_vars)

            # The value of `loop_vars` is empty
            def value_error_loop_vars():
                out = layers.while_loop(cond_returns_bool_tensor, body, [])

            self.assertRaises(ValueError, value_error_loop_vars)

            # The type of `cond` returns in Op(while_loop) must be Variable
            def type_error_cond_returns_not_variable():
                out = layers.while_loop(cond_returns_constant, body, [data_1d])

            self.assertRaises(TypeError, type_error_cond_returns_not_variable)

            # The type of `cond` returns in Op(while_loop) must be a bollean variable
            def type_error_cond_returns_not_boolean():
                out = layers.while_loop(cond_returns_not_bool_tensor, body,
                                        [data_1d])

            self.assertRaises(TypeError, type_error_cond_returns_not_boolean)

            # The shape of `cond` returns in Op(while_loop) must be 1
            def type_error_shape_cond_returns_2d():
                out = layers.while_loop(cond_returns_2d_tensor, body, [data_2d])

            self.assertRaises(TypeError, type_error_shape_cond_returns_2d)

            # The length of `body` returns in Op(while_loop) must be same as `loop_vars`
            def value_error_body_returns_error_length():
                out = layers.while_loop(cond_returns_bool_tensor,
                                        body_returns_error_length, [data])

            self.assertRaises(ValueError, value_error_body_returns_error_length)

            # The type of `body` returns in Op(while_loop) must be same as `loop_vars`
            def value_error_body_returns_error_type():
                out = layers.while_loop(cond_receives_two_args,
                                        body_returns_error_type, [data, ten])

            self.assertRaises(ValueError, value_error_body_returns_error_type)

            # The length of `output_vars` with mutable value should keep same with `loop_vars`
            def value_error_body_returns_with_mutable_dict():
                test_dict = {
                    "int_constant": layers.fill_constant(
                        shape=[2, 2], dtype='int64', value=1)
                }
                out = layers.while_loop(cond_returns_with_mutable_dict,
                                        body_returns_with_mutable_dict,
                                        [data, test_dict])

            self.assertRaises(ValueError,
                              value_error_body_returns_with_mutable_dict)

            def value_error_body_returns_with_mutable_list():
                test_list = [
                    layers.fill_constant(
                        shape=[2, 2], dtype='int64', value=1)
                ]
                out = layers.while_loop(cond_returns_with_mutable_list,
                                        body_returns_with_mutable_list,
                                        [data, test_list])

            self.assertRaises(ValueError,
                              value_error_body_returns_with_mutable_list)


class TestApiWhileLoopSliceInBody(unittest.TestCase):
    def test_var_slice(self):
        def cond(z, i):
            return i + 1 <= x_shape[0]

        def body(z, i):
            z = z + x[i]
            i += 1
            return z, i

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[5], dtype='int32')
            z = fluid.layers.fill_constant([1], 'int32', 0)
            x_shape = fluid.layers.shape(x)
            i = fluid.layers.fill_constant([1], 'int32', 0)
            z, _ = fluid.layers.while_loop(cond, body, [z, i])

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        np_x = np.array([1, 2, 3, 4, 5], dtype='int32')
        res = exe.run(main_program, feed={'x': np_x}, fetch_list=[z])
        self.assertTrue(np.array_equal(res[0], [np.sum(np_x)]))


if __name__ == '__main__':
    unittest.main()
