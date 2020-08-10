// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pybind/op_layer_generator.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {

namespace details {
// NOTE(zhiqiu): Commonly, the inputs in auto-generated OP function are
// determined by the OP`s proto automatically, i.e., all the inputs registered
// in OpMaker.
// However, some OPs have dispensable inputs, which means the input can
// be none for some conditions. It is discovered that most dispensable inputs
// is not used in imperative mode, so we drop those inputs when generating OP
// functions. While, for very few OPs, the dispensable inputs are used, we
// need to manually specify them in this map.
std::map<std::string, std::set<std::string>> op_ins_map = {
    {"layer_norm", {"X", "Scale", "Bias"}},
    {"instance_norm", {"X", "Scale", "Bias"}},
    {"gru_unit", {"Input", "HiddenPrev", "Weight", "Bias"}},
    {"label_smooth", {"X", "PriorDist"}},
    {"assign", {"X"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"X", "InScale", "InAccum", "InState"}},
};

// NOTE(zhiqiu): Like op_ins_map.
// Commonly, the outputs in auto-generated OP function are determined by the
// OP`s proto automatically, i.e., all the outputs registered in OpMaker.
// However, some OPs have dispensable outputs, which means the output can
// be none for some conditions. It is discovered that most dispensable outputs
// is not used in imperative mode, so we drop those outputs when generating OP
// functions. While, for very few OPs, the dispensable outputs are used, we
// need to manually specify them in this map.
std::map<std::string, std::set<std::string>> op_outs_map = {
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"batch_norm",
     {"Y", "MeanOut", "VarianceOut", "SavedMean", "SavedVariance",
      "ReserveSpace"}},
};

// NOTE(zhiqiu): Commonly, the outputs in auto-generated OP function are
// generated in C++ automatically.
// However, some OPs need to pass the outputs from Python instead of generating
// them in C++. There are mainly 2 reasons for that,
// (1) Optimizer OPs need to update the input param in-place, like sgd.
//     So they need to pass the output which is same as input param.
// (2) Very few python APIs has out in their arguments, like fill_constant.
//     So they need to pass the python output to C++.
//     Actually, this is not a good design, since it may break the SSA graph,
//     especially in declarative mode.
// For those OPs, we need to manually specify the outs need to pass in this map.
std::map<std::string, std::set<std::string>> op_passing_outs_map = {
    {"sgd", {"ParamOut"}},
    {"adam",
     {"ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut"}},
    {"momentum", {"ParamOut", "VelocityOut"}},
    {"batch_norm", {"MeanOut", "VarianceOut"}},
    {"accuracy", {"Correct", "Total"}},
    {"fill_constant", {"Out"}},
    {"matmul", {"Out"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"fake_quantize_dequantize_abs_max", {"Out", "OutScale"}},
    {"amp_check_finite_and_scale", {"Out", "FoundInfinite"}},
};

std::set<std::string> op_need_generate_layer = {
    "cwise_mul",
};

std::map<std::string, std::set<std::string>> op_layer_params_map = {
    {"cwise_mul", {"axis"}},
};

// clang-format off
const char* OUT_INITIALIZER_TEMPLATE =
    R"({"%s", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase(tracer->GenerateUniqueName()))}})";
const char* OUT_DUPLICABLE_INITIALIZER_TEMPLATE = R"({"%s", ConstructDuplicableOutput(%s)})";

const char* INPUT_INITIALIZER_TEMPLATE = R"({"%s", {%s}})";
const char* INPUT_LIST_INITIALIZER_TEMPLATE = R"({"%s", %s})";

const char* INPUT_INITIALIZER_TEMPLATE_WITH_NULL = R"(	
    if (%s != nullptr) {	
      ins["%s"] = {%s};	
    }	
)";

const char* INPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST = R"(	
    if (%s.size() != 0) {
      ins["%s"] = %s;	
    }	
)";

const char* OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL = R"(
    outs["%s"] = {%s};
)";

const char* OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST = R"(
    outs["%s"] = %s;
)";
// if inputs is list, no need {}
const char* ARG_OUT_NUM = R"(%sNum)";
const char* ARG_OUT_NUM_TYPE = R"(size_t )";

const char* VAR_TYPE = R"(std::shared_ptr<imperative::VarBase>)";
const char* VAR_LIST_TYPE = R"(std::vector<std::shared_ptr<imperative::VarBase>>)";
const char* ARG_TEMPLATE = R"(const %s& %s)";

const char* RETURN_TUPLE_TYPE = R"(std::tuple<%s>)";
const char* RETURN_TYPE = R"(%s)";
const char* RETURN_TUPLE_TEMPLATE = R"(std::make_tuple(%s))";
const char* RETURN_LIST_TEMPLATE = R"(outs["%s"])";
const char* RETURN_TEMPLATE = R"(outs["%s"][0])";

const char* FUNCTION_ARGS = R"(%s, const py::args& args)";
const char* FUNCTION_ARGS_NO_INPUT = R"(const py::args& args)";

const char* ATTR_TEMPLATE = R"("%s", %s)";
const char* MAP_ITEM_TEMPLATE = R"("%s": %s)";

const char* IMPORT_MODULES =
R"(
import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layers.layer_function_generator import autodoc, templatedoc, _generate_doc_string_
from paddle.fluid.data_feeder import convert_dtype, check_variable_and_dtype, check_type, check_dtype


)";


const char* MODULE_ALL_TEMPLATE =
R"(
__all__ = [%s]

)";

const char* MODULE_ALL_ITEM_TEMPLATE = R"('%s')";

const char* OP_LAYER_TEMPLATE =
R"(
def %s(%s):
    if in_dygraph_mode():
        return core.ops.%s(%s)
    
    helper = LayerHelper("%s", **locals())
    # check_variable_and_dtype()  TODO if needed ?
    %s = helper.create_variable_for_type_inference(dtype=%s.dtype)
    helper.append_op(
        type="%s", inputs={%s}, attrs={%s}, outputs={%s})
    return %s
)";



const char* PYBIND_ITEM_TEMPLATE = R"( %s.def("%s", &%s);)";

// clang-format on
static inline bool FindInsMap(const std::string& op_type,
                              const std::string& in_name) {
  return op_ins_map[op_type].count(in_name);
}

static inline bool FindOutsMap(const std::string& op_type,
                               const std::string& out_name) {
  return op_outs_map[op_type].count(out_name);
}

static inline bool FindParamsMap(const std::string& op_type,
                                 const std::string& param) {
  return op_layer_params_map[op_type].count(param);
}

static inline std::string ToLower(const std::string& str) {
  std::string s = str;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  return s;
}

static inline bool FindPassingOutsMap(const std::string& op_type,
                                      const std::string& out_name) {
  return op_passing_outs_map[op_type].count(out_name);
}

static std::vector<std::string> GenerateOpLayers() {
  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_layers;
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();
  std::string module_all = "";
  for (auto& pair : op_info_map) {
    auto& op_info = pair.second;
    auto op_proto = op_info.proto_;
    if (op_proto == nullptr) {
      continue;
    }
    auto& op_type = op_proto->type();
    if (op_need_generate_layer.count(op_type) == 0) {
      continue;
    }
    // Skip ooerator which is not inherit form OperatorWithKernel, like while,
    // since only OperatorWithKernel can run in dygraph mode.
    if (!all_kernels.count(op_type)) {
      continue;
    }
    std::string layer_args = "";
    std::string ins_initializer = "{";
    std::string ins_initializer_with_null = "";
    std::string py_arg = "";
    std::string core_ops_args = "";
    std::string append_op_inputs = "";
    std::string append_op_outputs = "";
    std::string append_op_attrs = "";
    for (auto& input : op_proto->inputs()) {
      auto& in_name = input.name();
      // skip those dispensable inputs, like ResidualData in conv2d
      if (input.dispensable() && !FindInsMap(op_type, in_name)) {
        continue;
      }

      // const auto in_type = input.duplicable() ? VAR_LIST_TYPE : VAR_TYPE;
      // auto input_arg = paddle::string::Sprintf(ARG_TEMPLATE, in_type,
      // in_name);
      layer_args += ToLower(in_name);
      layer_args += ", ";

      // if (input.dispensable()) {
      //   const auto in_template = input.duplicable()
      //                                ?
      //                                INPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST
      //                                : INPUT_INITIALIZER_TEMPLATE_WITH_NULL;
      //   ins_initializer_with_null +=
      //       paddle::string::Sprintf(in_template, in_name, in_name, in_name);
      // } else {
      //   const auto in_template = input.duplicable()
      //                                ? INPUT_LIST_INITIALIZER_TEMPLATE
      //                                : INPUT_INITIALIZER_TEMPLATE;
      //   ins_initializer +=
      //       paddle::string::Sprintf(in_template, in_name, in_name);
      //   ins_initializer += ",";
      // }
      append_op_inputs +=
          paddle::string::Sprintf(MAP_ITEM_TEMPLATE, in_name, ToLower(in_name));
      append_op_inputs += ", ";
    }
    core_ops_args += layer_args;

    for (auto& attr : op_proto->attrs()) {
      auto& attr_name = attr.name();
      if (!FindParamsMap(op_type, attr_name)) {
        continue;
      }
      layer_args += ToLower(attr_name);
      core_ops_args +=
          paddle::string::Sprintf(ATTR_TEMPLATE, attr_name, ToLower(attr_name));
      core_ops_args += ", ";
      append_op_attrs += paddle::string::Sprintf(MAP_ITEM_TEMPLATE, attr_name,
                                                 ToLower(attr_name));
      append_op_attrs += ", ";
    }

    auto first_input = layer_args[0];

    std::string out = "";
    for (auto& output : op_proto->outputs()) {
      auto& out_name = output.name();
      // skip those dispensable oututs
      if (output.dispensable() && !FindOutsMap(op_type, out_name)) {
        continue;
      }
      if (out != "") {
        continue;  // TODO(zhiqiu): handle multiple out
      }
      append_op_outputs += paddle::string::Sprintf(MAP_ITEM_TEMPLATE, out_name,
                                                   ToLower(out_name));
      out = ToLower(out_name);
    }
    // generate op layer body
    auto op_layer_str = paddle::string::Sprintf(
        OP_LAYER_TEMPLATE, op_type, layer_args, op_type, core_ops_args, op_type,
        out, first_input, op_type, append_op_inputs, append_op_attrs,
        append_op_outputs, out);

    module_all += paddle::string::Sprintf(MODULE_ALL_ITEM_TEMPLATE, op_type);
    module_all += ", ";

    op_layers.emplace_back(std::move(op_layer_str));
  }
  std::string module_all_str =
      paddle::string::Sprintf(MODULE_ALL_TEMPLATE, module_all);
  op_layers.emplace_back(std::move(module_all_str));
  return op_layers;
}

}  // namespace details
}  // namespace paddle

int gen_op_layers(std::string path) {
  std::cout << path << std::endl;
  std::ofstream out(path, std::ios::out);

  auto op_layers = paddle::details::GenerateOpLayers();

  out << paddle::details::IMPORT_MODULES;
  out << paddle::string::join_strings(op_layers, '\n');
  std::cout << paddle::string::join_strings(op_layers, '\n') << std::endl;

  out.close();
  return 0;
}
