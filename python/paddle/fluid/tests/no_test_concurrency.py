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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import framework, unique_name, layer_helper
from paddle.fluid.executor import Executor
from paddle.fluid.layers import fill_constant, assign, While, elementwise_add, Print


class TestRoutineOp(unittest.TestCase):
    def test_simple_routine(self):
        ch = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)

        # Create LOD_TENSOR<INT64> and put it into the scope.  This placeholder
        # variable will be filled in and returned by fluid.channel_recv
        result = self._create_tensor('return_value',
                                     core.VarDesc.VarType.LOD_TENSOR,
                                     core.VarDesc.VarType.INT64)

        with fluid.Go():
            input_value = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.FP64, value=1234)
            fluid.channel_send(ch, input_value)

        result, status = fluid.channel_recv(ch, result)
        fluid.channel_close(ch)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        outs = exe.run(fetch_list=[result])
        self.assertEqual(outs[0], 1234)

    def test_daisy_chain(self):
        '''
        Mimics classic Daisy-chain test:  https://talks.golang.org/2012/concurrency.slide#39
        '''
        n = 100

        leftmost = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
        left = leftmost

        # TODO(thuan): Use fluid.While() after scope capture is implemented.
        # https://github.com/PaddlePaddle/Paddle/issues/8502
        for i in range(n):
            right = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
            with fluid.Go():
                one_tensor = self._create_one_dim_tensor(1)
                result = self._create_tensor('return_value',
                                             core.VarDesc.VarType.LOD_TENSOR,
                                             core.VarDesc.VarType.INT64)

                result, status = fluid.channel_recv(right, result)
                one_added = fluid.layers.elementwise_add(x=one_tensor, y=result)
                fluid.channel_send(left, one_added)
            left = right

        # Trigger the channel propagation by sending a "1" to rightmost channel
        with fluid.Go():
            one_tensor = self._create_one_dim_tensor(1)
            fluid.channel_send(right, one_tensor)

        leftmost_result = self._create_tensor('return_value',
                                              core.VarDesc.VarType.LOD_TENSOR,
                                              core.VarDesc.VarType.INT64)
        leftmost_result, status = fluid.channel_recv(leftmost, leftmost_result)

        cpu = core.CPUPlace()
        exe = Executor(cpu)
        leftmost_data = exe.run(fetch_list=[leftmost_result])

        # The leftmost_data should be equal to the number of channels + 1
        self.assertEqual(leftmost_data[0][0], n + 1)

    def _create_one_dim_tensor(self, value):
        one_dim_tensor = fill_constant(shape=[1], dtype='int', value=value)
        one_dim_tensor.stop_gradient = True
        return one_dim_tensor

    def _create_tensor(self, name, type, dtype):
        return framework.default_main_program().current_block().create_var(
            name=unique_name.generate(name), type=type, dtype=dtype)

    def _create_persistable_tensor(self, name, type, dtype):
        return framework.default_main_program().current_block().create_var(
            name=unique_name.generate(name),
            type=type,
            dtype=dtype,
            persistable=True)

    def test_select(self):
        with framework.program_guard(framework.Program()):
            ch1 = fluid.make_channel(
                dtype=core.VarDesc.VarType.LOD_TENSOR, capacity=1)

            result1 = self._create_tensor('return_value',
                                          core.VarDesc.VarType.LOD_TENSOR,
                                          core.VarDesc.VarType.FP64)

            input_value = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.FP64, value=10)

            with fluid.Select() as select:
                with select.case(fluid.channel_send, ch1, input_value):
                    # Execute something.
                    pass

                with select.default():
                    pass

            # This should not block because we are using a buffered channel.
            result1, status = fluid.channel_recv(ch1, result1)
            fluid.channel_close(ch1)

            cpu = core.CPUPlace()
            exe = Executor(cpu)

            result = exe.run(fetch_list=[result1])
            self.assertEqual(result[0][0], 10)

    def test_fibonacci(self):
        """
        Mimics Fibonacci Go example: https://tour.golang.org/concurrency/5
        """
        with framework.program_guard(framework.Program()):
            quit_ch_input_var = self._create_persistable_tensor(
                'quit_ch_input', core.VarDesc.VarType.LOD_TENSOR,
                core.VarDesc.VarType.INT32)
            quit_ch_input = fill_constant(
                shape=[1],
                dtype=core.VarDesc.VarType.INT32,
                value=0,
                out=quit_ch_input_var)

            result = self._create_persistable_tensor(
                'result', core.VarDesc.VarType.LOD_TENSOR,
                core.VarDesc.VarType.INT32)
            fill_constant(
                shape=[1],
                dtype=core.VarDesc.VarType.INT32,
                value=0,
                out=result)

            x = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.INT32, value=0)
            y = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.INT32, value=1)

            while_cond = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.BOOL, value=True)

            while_false = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.BOOL, value=False)

            x_tmp = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.INT32, value=0)

            def fibonacci(channel, quit_channel):
                while_op = While(cond=while_cond)
                with while_op.block():
                    result2 = fill_constant(
                        shape=[1], dtype=core.VarDesc.VarType.INT32, value=0)

                    with fluid.Select() as select:
                        with select.case(
                                fluid.channel_send, channel, x, is_copy=True):
                            assign(input=x, output=x_tmp)
                            assign(input=y, output=x)
                            assign(elementwise_add(x=x_tmp, y=y), output=y)

                        with select.case(fluid.channel_recv, quit_channel,
                                         result2):
                            # Quit
                            helper = layer_helper.LayerHelper('assign')
                            helper.append_op(
                                type='assign',
                                inputs={'X': [while_false]},
                                outputs={'Out': [while_cond]})

            ch1 = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
            quit_ch = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)

            with fluid.Go():
                for i in range(10):
                    fluid.channel_recv(ch1, result)
                    Print(result)

                fluid.channel_send(quit_ch, quit_ch_input)

            fibonacci(ch1, quit_ch)

            fluid.channel_close(ch1)
            fluid.channel_close(quit_ch)

            cpu = core.CPUPlace()
            exe = Executor(cpu)

            exe_result = exe.run(fetch_list=[result])
            self.assertEqual(exe_result[0][0], 34)

    def test_ping_pong(self):
        """
        Mimics Ping Pong example: https://gobyexample.com/channel-directions
        """
        with framework.program_guard(framework.Program()):
            result = self._create_tensor('return_value',
                                         core.VarDesc.VarType.LOD_TENSOR,
                                         core.VarDesc.VarType.FP64)

            ping_result = self._create_tensor('ping_return_value',
                                              core.VarDesc.VarType.LOD_TENSOR,
                                              core.VarDesc.VarType.FP64)

            def ping(ch, message):
                fluid.channel_send(ch, message, is_copy=True)

            def pong(ch1, ch2):
                fluid.channel_recv(ch1, ping_result)
                fluid.channel_send(ch2, ping_result, is_copy=True)

            pings = fluid.make_channel(
                dtype=core.VarDesc.VarType.LOD_TENSOR, capacity=1)
            pongs = fluid.make_channel(
                dtype=core.VarDesc.VarType.LOD_TENSOR, capacity=1)

            msg = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.FP64, value=9)

            ping(pings, msg)
            pong(pings, pongs)

            fluid.channel_recv(pongs, result)

            fluid.channel_close(pings)
            fluid.channel_close(pongs)

            cpu = core.CPUPlace()
            exe = Executor(cpu)

            exe_result = exe.run(fetch_list=[result])
            self.assertEqual(exe_result[0][0], 9)


if __name__ == '__main__':
    unittest.main()
