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

import unittest
import paddle
import paddle.fluid.core as core
import paddle.fluid.ir_pass as ir_pass


@ir_pass.RegisterPass
def test_generate_pass_v1():
    # pattern
    @ir_pass.DescribeFunction
    def pattern(x, w, b, out):
        mul = ir_pass.Op("matmul_v2")
        mul.SetInput(X=x, Y=w).SetOutput(out=ir_pass.Var("out"))
        mul.Attr("use_mkldnn").Set(False)
        mul.Attr("scale_y").Set([1.0])
        ewadd = ir_pass.Op("elementwise_add")
        ewadd.SetInput(X=mul.Output("out")[0], bias=b).SetOutput(out=out)
        return mul, ewadd

    # replace
    @ir_pass.DescribeFunction
    def replace(x, w, b, out):
        fc = ir_pass.Op("fc")
        fc.SetInput(Input=x, W=w, Bias=b).SetOutput(Out=out)
        # attr map
        fc.Attr("in_num_col_dims").Reuse("mul", "x_num_col_dims")
        fc.Attr("activation_type").Set("")
        return fc

    return [(pattern, replace)]


@ir_pass.RegisterPass
def test_generate_pass_v2():
    # pattern
    @ir_pass.APIFunction
    def pattern(x, y, z):
        add_out = paddle.add(x, y)
        return paddle.add(add_out, z)

    # replace
    @ir_pass.APIFunction
    def replace(x, y, z):
        return paddle.add_n([x, y, z])

    return [(pattern, replace)]


class FCFusePassTest(unittest.TestCase):
    @unittest.skipIf(True, "")
    def test_check_pass_v1(self):
        ir_pass.UsePass("test_generate_pass_v1")
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.static.data("data", [32, 32])
            weight = paddle.static.data("weight", [32, 32])
            bias = paddle.static.data("bias", [32, 32])
            relu_out = paddle.nn.functional.relu(data)
            mul_out = paddle.matmul(relu_out, weight)
            ewadd_out = paddle.add(mul_out, bias)
        graph = core.Graph(main_program.desc)
        before_nodes_num = len(graph.nodes())
        test_pass = core.get_pass("test_generate_pass_v1")
        test_pass.apply(graph)
        after_nodes_num = len(graph.nodes())
        self.assertEqual(before_nodes_num, after_nodes_num + 2)

    def test_check_pass_v2(self):
        ir_pass.UsePass("test_generate_pass_v2")
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data("x", [32, 32])
            y = paddle.static.data("y", [32, 32])
            z = paddle.static.data("z", [32, 32])
            add_out = paddle.add(x, y)
            paddle.add(add_out, z)
        graph = core.Graph(main_program.desc)
        before_nodes_num = len(graph.nodes())
        test_pass = core.get_pass("test_generate_pass_v2")
        test_pass.apply(graph)
        after_nodes_num = len(graph.nodes())
        self.assertEqual(before_nodes_num, after_nodes_num + 2)
