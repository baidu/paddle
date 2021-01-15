# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
import numpy as np

registerd_op = {
    "elementwise_add": "AddParser",
    "matmul": "MatMulParser",
    "mul": "MulParser",
    "relu": "ReluParser",
    "softmax_with_cross_entropy": "SoftmaxWithCrossEntropyParser",
    "shape": "ShapeParser",
    "fill_constant": "FillConstantParser",
    "reduce_sum": "ReduceSumParser",
    "reduce_sum_grad": "ReduceSumGradParser",
    "matmul_grad": "MatMulGradParser",
    "mul_grad": "MulGradParser",
    "relu_grad": "ReluGradParser",
    "softmax_with_cross_entropy_grad": "SoftmaxWithCrossEntropyGradParser",
    "truncated_gaussian_random": "TruncatedNormalParser",
    "sgd": "SGDParser"
}
global_cnt = -1
global_input_cnt = -1


class AscendHelper(object):
    def __init__(self):
        self.dtype2ge_map = {
            0: core.GEDataType.DT_BOOL,
            1: core.GEDataType.DT_INT16,
            2: core.GEDataType.DT_INT32,
            3: core.GEDataType.DT_INT64,
            4: core.GEDataType.DT_FLOAT16,
            5: core.GEDataType.DT_FLOAT,
            6: core.GEDataType.DT_DOUBLE
        }
        self.dtype2np_map = {
            0: "bool",
            1: "int16",
            2: "int32",
            3: "int64",
            4: "float16",
            5: "float32",
            6: "float64"
        }

    def dtype2ge(self, dtype):
        assert dtype in self.dtype2ge_map, "dtype[%d] is not supported %d" % (
            dtype)
        return self.dtype2ge_map[dtype]

    def dtype2np(self, index):
        assert index in self.dtype2np_map, "index[%d] is not supported %d" % (
            dtype)
        return self.dtype2np_map[index]


class AscendParserFactory(object):
    def __init__(self, graph, var2geop):
        self.graph = graph
        self.var2geop = var2geop

    def create_parse(self, parser_class):
        try:
            parser = globals()[parser_class](self.graph, self.var2geop)
            return parser
        except:
            raise ValueError("parser class %s does not exist" % parser_class)


class AscendParserBase(object):
    def __init__(self, graph, var2geop):
        self.graph = graph
        self.var2geop = var2geop
        self.op = None
        self.ascend_helper = AscendHelper()

    def _get_ge_input(self, input_var_name):
        assert input_var_name in self.var2geop, "var %s not created before" % (
            input_var_name)
        return self.var2geop[input_var_name]

    def update_output(self, geop_list, index_list):
        output_num = len(self.op.output_names)
        assert output_num == len(
            index_list
        ), "Parser[%s]'s output number[%d] is not equal to parameters number[%d]" % (
            self.parser_name, len(index_list), output_num)
        for output_id in range(output_num):
            arguments = self.op.output(self.op.output_names[output_id])
            print("%d argument:  %s" % (output_id, str(arguments)))
            if len(arguments) > 0:
                assert len(arguments) == len(
                    index_list[output_id]
                ), "Parser[%s]'s %dth argument number[%d] is not equal to paddle's number[%d]" % (
                    self.parser_name, output_id, len(index_list[output_id]),
                    len(arguments))
                for i in range(len(arguments)):
                    print("assgin index_list[%d][%d] to %s" %
                          (output_id, i, arguments[i]))
                    self.var2geop[arguments[i]] = geop_list[index_list[
                        output_id][i]]

        for geop in geop_list:
            self.graph.add_op(geop)

    def apply(self, op):
        self.op = op
        assert self.op.type == self.parser_name, "op [%s] != parser_name[%s]" % (
            self.op.type, self.parser_name)
        print("begin to parse op %s" % (self.parser_name))
        geop_list, index_list = self._apply()
        self.update_output(geop_list, index_list)

    def _mark_as_input(self, ge_tensor):
        global global_input_cnt
        global_input_cnt += 1
        self.var2geop["geinput." + str(global_input_cnt)] = ge_tensor

    def _accumulated_op_id(self):
        global global_cnt
        global_cnt += 1
        return "." + str(global_cnt)

    def _create_ge_tensor(self, shape, dtype, value):
        tensor_desc = core.GETensorDesc(
            core.GEShape(shape), core.GEFormat.FORMAT_ND,
            self.ascend_helper.dtype2ge(dtype))
        tensor = core.GETensor(tensor_desc)

        data = (value * np.ones((
            shape))).reshape(shape).astype(self.ascend_helper.dtype2np(dtype))
        buf = data.tobytes()
        data_8 = np.frombuffer(buf, dtype=np.uint8)
        tensor.set_data(data_8)
        return tensor


class AddParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(AddParser, self).__init__(graph, var2geop)
        self.parser_name = "elementwise_add"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])
        add = core.GEOperatorFactory.create_operator(
            "add" + self._accumulated_op_id(), "Add").set_input(
                "x1", x).set_input("x2", y)
        return [add], [[0]]


class ReduceSumParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        axes = self.op.attr("dim")
        keep_dims = self.op.attr("keep_dim")
        reduce_sum = core.GEOperatorFactory.create_operator(
            "reduce_sum" + self._accumulated_op_id(), "ReduceSumD").set_input(
                "x", x, 0).set_attr_vec_int32("axes", axes).set_attr_bool(
                    "keep_dims", keep_dims)
        return [reduce_sum], [[0]]


class ReduceSumGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReduceSumGradParser, self).__init__(graph, var2geop)
        self.parser_name = "reduce_sum_grad"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        input = self._get_ge_input(self.op.input_arg_names[1])

        shape_tensor = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", input,
                                                                    0)
        axis_const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", self._create_ge_tensor([1], 2, -1))
        self._mark_as_input(axis_const)

        broadcast = core.GEOperatorFactory.create_operator(
            "broadcast_to_d" + self._accumulated_op_id(),
            "BroadcastTo").set_input("x", x).set_input("shape", shape_tensor)
        # unsqueeze cannot get right result, but ExpandDims seems have the same functionality.
        reduce_sum_grad = core.GEOperatorFactory.create_operator(
            "expand" + self._accumulated_op_id(), "ExpandDims").set_input(
                "x", broadcast).set_input("axis", axis_const)
        return [shape_tensor, axis_const, broadcast, reduce_sum_grad], [[3]]


class MatMulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MatMulParser, self).__init__(graph, var2geop)
        self.parser_name = "matmul"

    def _apply(self):
        x1 = self._get_ge_input(self.op.input_arg_names[0])
        x2 = self._get_ge_input(self.op.input_arg_names[1])
        matmul = core.GEOperatorFactory.create_operator(
            "matmul" + self._accumulated_op_id(), "MatMul").set_input(
                "x1", x1).set_input("x2", x2)
        return [matmul], [[0]]


class MatMulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MatMulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "matmul_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        y = self._get_ge_input(self.op.input_arg_names[2])

        x_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(), "MatMul").set_input(
                "x1", out_grad).set_input("x2", y).set_attr_bool(
                    "transpose_x1", False).set_attr_bool("transpose_x2", True)
        y_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(), "MatMul").set_input(
                "x1", x).set_input("x2", out_grad).set_attr_bool(
                    "transpose_x1", True).set_attr_bool("transpose_x2", False)
        return [x_grad, y_grad], [[0], [1]]


class MulGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MulGradParser, self).__init__(graph, var2geop)
        self.parser_name = "mul_grad"

    def _apply(self):
        out_grad = self._get_ge_input(self.op.input_arg_names[0])
        x = self._get_ge_input(self.op.input_arg_names[1])
        y = self._get_ge_input(self.op.input_arg_names[2])

        x_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(), "MatMul").set_input(
                "x1", out_grad).set_input("x2", y).set_attr_bool(
                    "transpose_x1", False).set_attr_bool("transpose_x2", True)
        y_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(), "MatMul").set_input(
                "x1", x).set_input("x2", out_grad).set_attr_bool(
                    "transpose_x1", True).set_attr_bool("transpose_x2", False)

        return [x_grad, y_grad], [[0], [1]]


class MulParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(MulParser, self).__init__(graph, var2geop)
        self.parser_name = "mul"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        y = self._get_ge_input(self.op.input_arg_names[1])

        matmul = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "MatMul").set_input(
                "x1", x).set_input("x2", y)
        return [matmul], [[0]]


class ReluParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReluParser, self).__init__(graph, var2geop)
        self.parser_name = "relu"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        relu = core.GEOperatorFactory.create_operator(
            "relu" + self._accumulated_op_id(), "Relu").set_input("x", x)
        return [relu], [[0]]


class ReluGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ReluGradParser, self).__init__(graph, var2geop)
        self.parser_name = "relu_grad"

    def _apply(self):
        out = self._get_ge_input(self.op.input_arg_names[0])
        out_grad = self._get_ge_input(self.op.input_arg_names[1])
        relu_grad = core.GEOperatorFactory.create_operator(
            self.parser_name + self._accumulated_op_id(), "ReluGrad").set_input(
                "gradients", out_grad).set_input("features", out)
        return [relu_grad], [[0]]


class SoftmaxWithCrossEntropyParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxWithCrossEntropyParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_with_cross_entropy"

    def _apply(self):
        label = self._get_ge_input(self.op.input_arg_names[0])
        logits = self._get_ge_input(self.op.input_arg_names[1])

        cls_num = self.op.block.var(self.op.input_arg_names[1]).shape[1]
        softmax = core.GEOperatorFactory.create_operator(
            "softmax" + self._accumulated_op_id(), "SoftmaxV2").set_input(
                "x", logits)
        label = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", label).set_attr_int32("dst_type", 3)

        tensoron = self._create_ge_tensor([1], 5, 1)
        on_const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensoron)
        self._mark_as_input(on_const)
        tensoroff = self._create_ge_tensor([1], 5, 0)
        off_const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensoroff)
        self._mark_as_input(off_const)
        onehot = core.GEOperatorFactory.create_operator(
            "onehot" + self._accumulated_op_id(), "OneHotD").set_input(
                "x", label).set_input("on_value", on_const).set_input(
                    "off_value", off_const).set_attr_int32("depth", cls_num)
        squeeze = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "Squeeze").set_input("x", onehot)
        loss = core.GEOperatorFactory.create_operator(
            "loss" + self._accumulated_op_id(),
            "SoftmaxCrossEntropyWithLogits").set_input(
                "features", logits).set_input("labels", squeeze)

        return [label, softmax, on_const, off_const, onehot, squeeze,
                loss], [[6], [1]]


class SoftmaxWithCrossEntropyGradParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SoftmaxWithCrossEntropyGradParser, self).__init__(graph, var2geop)
        self.parser_name = "softmax_with_cross_entropy_grad"

    def _apply(self):
        label = self._get_ge_input(self.op.input_arg_names[0])
        loss_grad = self._get_ge_input(self.op.input_arg_names[1])
        softmax = self._get_ge_input(self.op.input_arg_names[2])
        cls_num = self.op.block.var(self.op.input_arg_names[2]).shape[1]

        tensoron = self._create_ge_tensor([1], 5, 1)
        on_const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensoron)
        self._mark_as_input(on_const)
        tensoroff = self._create_ge_tensor([1], 5, 0)
        off_const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensoroff)
        self._mark_as_input(off_const)
        label = core.GEOperatorFactory.create_operator(
            "cast" + self._accumulated_op_id(), "Cast").set_input(
                "x", label).set_attr_int32("dst_type", 3)
        onehot = core.GEOperatorFactory.create_operator(
            "onehot" + self._accumulated_op_id(), "OneHotD").set_input(
                "x", label).set_input("on_value", on_const).set_input(
                    "off_value", off_const).set_attr_int32("depth", cls_num)
        # the fuck onehot will add a demension, so must call squeeze afterward
        squeeze = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "Squeeze").set_input("x", onehot)
        sub = core.GEOperatorFactory.create_operator(
            "sub" + self._accumulated_op_id(), "Sub").set_input(
                "x1", softmax).set_input("x2", squeeze)
        grad = core.GEOperatorFactory.create_operator(
            "mul" + self._accumulated_op_id(), "Mul").set_input(
                "x1", loss_grad).set_input("x2", sub)
        return [on_const, off_const, label, onehot, squeeze, sub, grad], [[-1]]


class ShapeParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(ShapeParser, self).__init__(graph, var2geop)
        self.parser_name = "shape"

    def _apply(self):
        x = self._get_ge_input(self.op.input_arg_names[0])
        shape = core.GEOperatorFactory.create_operator(
            "shape" + self._accumulated_op_id(), "Shape").set_input("x", x)
        return [shape], [[0]]


class FillConstantParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(FillConstantParser, self).__init__(graph, var2geop)
        self.parser_name = "fill_constant"

    def _apply(self):
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")
        value = self.op.attr("value")
        print("shape: ", shape)
        print("dtype: ", dtype)
        print("value: ", value)
        tensor = self._create_ge_tensor(shape, dtype, value)
        const = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensor)
        self._mark_as_input(const)
        if self.op.block.var(self.op.output('Out')[0]).persistable:
            print("%s fill_constant" % (self.op.output('Out')[0]))
            var = core.GEOperatorFactory.create_operator(
                self.op.output('Out')[0], "Variable")
            var.update_output_desc("y",
                                   core.GETensorDesc(
                                       core.GEShape(shape),
                                       core.GEFormat.FORMAT_ND,
                                       core.GEDataType.DT_FLOAT))
            assign = core.GEOperatorFactory.create_operator(
                "assign" + self._accumulated_op_id(), "Assign").set_input(
                    "value", const).set_input("ref", var)
            return [const], [[0]]
        else:
            print(
                "self.op.output('Out')[0] is not persistable in fill_constant")
            return [const], [[0]]


class SGDParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(SGDParser, self).__init__(graph, var2geop)
        self.parser_name = "sgd"

    def _apply(self):
        grad = self._get_ge_input(self.op.input_arg_names[0])
        lr = self._get_ge_input(self.op.input_arg_names[1])
        param = self._get_ge_input(self.op.input_arg_names[2])
        sgd = core.GEOperatorFactory.create_operator(
            "momentum" + self._accumulated_op_id(),
            "ApplyGradientDescent").set_input("var", param).set_input(
                "alpha", lr).set_input("delta", grad)
        return [sgd], [[0]]


class TruncatedNormalParser(AscendParserBase):
    def __init__(self, graph, var2geop):
        super(TruncatedNormalParser, self).__init__(graph, var2geop)
        self.parser_name = "truncated_gaussian_random"

    def _apply(self):
        shape = self.op.attr("shape")
        dtype = self.op.attr("dtype")
        mean = self.op.attr("mean")
        std = self.op.attr("std")
        seed = self.op.attr("seed")
        tensor1 = self._create_ge_tensor([len(shape)], 2, shape)
        shape_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensor1)

        tensor2 = self._create_ge_tensor([1], dtype, mean)
        mean_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensor2)

        tensor3 = self._create_ge_tensor([1], dtype, std)
        std_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensor3)

        tensor4 = self._create_ge_tensor([1], dtype, mean - 2 * std)
        min_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensor4)

        tensor5 = self._create_ge_tensor([1], dtype, mean + 2 * std)
        max_tensor = core.GEOperatorFactory.create_operator(
            "const" + self._accumulated_op_id(), "Const").set_attr_tensor(
                "value", tensor5)

        self._mark_as_input(shape_tensor)
        self._mark_as_input(mean_tensor)
        self._mark_as_input(std_tensor)
        self._mark_as_input(min_tensor)
        self._mark_as_input(max_tensor)

        truncated_normal = core.GEOperatorFactory.create_operator(
            "truncated_normal" + self._accumulated_op_id(),
            "ParameterizedTruncatedNormal").set_input(
                "shape", shape_tensor).set_input(
                    "means", mean_tensor).set_input(
                        "stdevs", std_tensor).set_input(
                            "min", min_tensor).set_input(
                                "max", max_tensor).set_attr_int32("seed", 0)

        ## wirte the output of truncatedNormal from startup_program to main_program
        if self.op.block.var(self.op.output('Out')[0]).persistable:
            print("%s is Persistable in truncated_normal" %
                  (self.op.output('Out')[0]))
            #var = core.GEOperatorFactory.create_operator(self.op.output('Out')[0], "Variable").set_input("x", truncated_normal)
            var = core.GEOperatorFactory.create_operator(
                self.op.output('Out')[0], "Variable")
            var.update_output_desc("y",
                                   core.GETensorDesc(
                                       core.GEShape(shape),
                                       core.GEFormat.FORMAT_ND,
                                       core.GEDataType.DT_FLOAT))
            assign = core.GEOperatorFactory.create_operator(
                "assign" + self._accumulated_op_id(), "Assign").set_input(
                    "value", truncated_normal).set_input("ref", var)
            return [
                shape_tensor, mean_tensor, std_tensor, min_tensor, max_tensor,
                truncated_normal
            ], [[-1]]
        else:
            print(
                "self.op.output('Out')[0] is not persistable in truncated_noraml"
            )
            return [truncated_normal], [[0]]  #[assign]
