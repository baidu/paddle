#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import six
import paddle
from paddle import fluid
from paddle import static

paddle.enable_static()


def program_to_IRGraph(program):
    graph = fluid.core.Graph(program.desc)
    ir_graph = fluid.framework.IrGraph(graph, for_test=False)
    return ir_graph


def IRGraph_to_program(ir_graph):
    return ir_graph.to_program()


class GraphToProgramPassTest(unittest.TestCase):
    def check_vars_equal(self, o_block, c_block):
        o_params = sorted(o_block.all_parameters(), key=lambda p: p.name)
        c_params = sorted(c_block.all_parameters(), key=lambda p: p.name)
        self.assertEqual(len(o_params), len(c_params))
        for p_idx in range(len(o_params)):
            self.assertEqual(o_params[p_idx].name, c_params[p_idx].name)

        o_vars = sorted(o_block.vars.values(), key=lambda v: v.name)
        c_vars = sorted(c_block.vars.values(), key=lambda v: v.name)
        self.assertEqual(len(o_vars), len(c_vars))
        for v_idx in range(len(o_vars)):
            self.assertEqual(o_vars[v_idx].name, c_vars[v_idx].name)

    def check_op_output_equal(self, o_op, c_op):
        self.assertEqual(len(o_op.output_names), len(c_op.output_names))
        for out_idx in range(len(o_op.output_names)):
            o_out = o_op.output_names[out_idx]
            c_out = c_op.output_names[out_idx]
            self.assertEqual(o_out, c_out)
            self.assertEqual(o_op.output(o_out), c_op.output(c_out))

    def check_op_input_equal(self, o_op, c_op):
        self.assertEqual(len(o_op.input_names), len(c_op.input_names))
        for in_idx in range(len(o_op.input_names)):
            o_in = o_op.input_names[in_idx]
            c_in = c_op.input_names[in_idx]
            self.assertEqual(o_in, c_in)
            self.assertEqual(o_op.input(o_in), c_op.input(c_in))

    def check_op_attrs_equal(self, o_op, c_op):
        o_attrs = sorted(o_op.attr_names)
        c_attrs = sorted(c_op.attr_names)
        self.assertEqual(len(o_attrs), len(c_attrs))
        for attr_idx in range(len(o_attrs)):
            o_attr = o_attrs[attr_idx]
            c_attr = c_attrs[attr_idx]
            self.assertEqual(o_attr, c_attr)
            self.assertEqual(
                o_op.desc.attr_type(o_attr), c_op.desc.attr_type(c_attr))


class SingleGraphToProgramPass(GraphToProgramPassTest):
    def setUp(self):
        self.origin_program = self.build_program()
        ir_graph = program_to_IRGraph(self.origin_program)
        self.converted_program = IRGraph_to_program(ir_graph)

    @staticmethod
    def build_program():
        program = static.Program()
        with static.program_guard(program):
            data = static.data(name='x', shape=[None, 13], dtype='float32')
            hidden = static.nn.fc(data, size=10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
        return program

    def test_check_parameter(self):
        origin_parameter = sorted(
            self.origin_program.all_parameters(), key=lambda p: p.name)
        converted_parameter = sorted(
            self.converted_program.all_parameters(), key=lambda p: p.name)

        self.assertEqual(len(origin_parameter), len(converted_parameter))

        for i in range(len(origin_parameter)):
            o_para = origin_parameter[i]
            c_para = converted_parameter[i]
            self.assertEqual(o_para.name, c_para.name)
            self.assertEqual(o_para.is_parameter, c_para.is_parameter)

    def test_check_stop_gradient(self):
        origin_vars = list(self.origin_program.list_vars())
        origin_vars = sorted(origin_vars, key=lambda v: v.name)

        converted_vars = list(self.converted_program.list_vars())
        converted_vars = sorted(converted_vars, key=lambda v: v.name)

        self.assertEqual(len(origin_vars), len(converted_vars))

        for i in range(len(origin_vars)):
            o_var = origin_vars[i]
            c_var = converted_vars[i]
            self.assertEqual(o_var.name, c_var.name)
            self.assertEqual(o_var.stop_gradient, c_var.stop_gradient)

    def test_check_ops(self):
        o_block = self.origin_program.global_block()
        c_block = self.converted_program.global_block()
        self.assertEqual(len(o_block.ops), len(c_block.ops))

        # ensure op ordering and content same
        for i in range(len(o_block.ops)):
            o_op = o_block.ops[i]
            c_op = c_block.ops[i]

            self.assertEqual(o_op.type, c_op.type)

            self.check_op_input_equal(o_op, c_op)
            self.check_op_output_equal(o_op, c_op)
            self.check_op_attrs_equal(o_op, c_op)


'''
#TODO(jiangcheng): Open after PR33949 and PR33949 merged
class MultiBlockGraphToProgramPass(GraphToProgramPassTest):
    def setUp(self):
        self.origin_program = self.build_program()
        ir_graph = program_to_IRGraph(self.origin_program)
        self.converted_program = IRGraph_to_program(ir_graph)

    @staticmethod
    def multiblock_model():
        data = static.data(name='t', shape=[None, 10], dtype='float32')
        a = static.data(name='a', shape=[10, 1], dtype='int64')
        b = static.data(name='b', shape=[10, 1], dtype='int64')

        cond = paddle.greater_than(a, b)
        ie = fluid.layers.IfElse(cond)
        with ie.true_block():
            hidden = paddle.nn.functional.relu(data)
            ie.output(hidden)
        with ie.false_block():
            hidden = paddle.nn.functional.softmax(data)
            ie.output(hidden)

        hidden = ie()
        return hidden[0]

    @staticmethod
    def build_program():
        program = static.Program()
        with static.program_guard(program):
            hidden = MultiBlockGraphToProgramPass.multiblock_model()
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
        return program

    def check_ops_equal(self, o_block, c_block):
        o_ops = o_block.ops
        c_ops = c_block.ops
        self.assertEqual(len(o_ops), len(c_ops))
        for op_idx in range(len(o_ops)):
            o_op = o_ops[op_idx]
            c_op = c_ops[op_idx]
            self.assertEqual(o_op.type, c_op.type)

            self.check_op_input_equal(o_op, c_op)
            self.check_op_output_equal(o_op, c_op)
            self.check_op_attrs_equal(o_op, c_op)

    def check_block_equal(self, o_block, c_block):
        self.check_vars_equal(o_block, c_block)
        self.check_ops_equal(o_block, c_block)

    def test_check_block(self):
        self.assertEqual(self.origin_program.num_blocks,
                         self.converted_program.num_blocks)

        for block_idx in range(self.origin_program.num_blocks):
            o_block = self.origin_program.block(block_idx)
            c_block = self.converted_program.block(block_idx)

            self.assertEqual(o_block.idx, c_block.idx)
            self.check_block_equal(o_block, c_block)
'''

if __name__ == "__main__":
    unittest.main()
