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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

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
    {"nll_loss", {"X", "Label", "Weight"}},
    {"bilinear_tensor_product", {"X", "Y", "Weight", "Bias"}},
    {"gather", {"X", "Index", "Axis"}},
    {"roi_pool", {"X", "ROIs", "RoisNum"}},
    {"roi_align", {"X", "ROIs", "RoisNum"}},
    {"collect_fpn_proposals",
     {"MultiLevelRois", "MultiLevelScores", "MultiLevelRoIsNum"}},
    {"distribute_fpn_proposals", {"FpnRois", "RoisNum"}},
    {"warpctc", {"Logits", "Label", "LogitsLength", "LabelLength"}},
    {"hierarchical_sigmoid",
     {"X", "W", "Label", "PathTable", "PathCode", "Bias"}},
    {"moving_average_abs_max_scale", {"X", "InAccum", "InState"}},
    {"multiclass_nms3", {"BBoxes", "Scores", "RoisNum"}},
    {"box_coder", {"PriorBox", "PriorBoxVar", "TargetBox"}},
    {"momentum", {"Param", "Grad", "Velocity", "LearningRate"}},
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
    {"sync_batch_norm",
     {"Y", "MeanOut", "VarianceOut", "SavedMean", "SavedVariance",
      "ReserveSpace"}},
    {"unique", {"Out", "Index", "Indices", "Counts"}},
    {"generate_proposals", {"RpnRois", "RpnRoiProbs", "RpnRoisNum"}},
    {"collect_fpn_proposals", {"FpnRois", "RoisNum"}},
    {"matrix_nms", {"Out", "Index", "RoisNum"}},
    {"distribute_fpn_proposals",
     {"MultiFpnRois", "RestoreIndex", "MultiLevelRoIsNum"}},
    {"moving_average_abs_max_scale", {"OutScale", "OutAccum", "OutState"}},
    {"multiclass_nms3", {"Out", "NmsRoisNum"}},
    {"generate_proposals_v2", {"RpnRois", "RpnRoiProbs", "RpnRoisNum"}},
    {"momentum", {"ParamOut", "VelocityOut"}},
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
    {"sync_batch_norm", {"MeanOut", "VarianceOut"}},
    {"accuracy", {"Correct", "Total"}},
    {"fill_constant", {"Out"}},
    {"matmul", {"Out"}},
    {"c_broadcast", {"Out"}},
    {"c_allreduce_sum", {"Out"}},
    {"c_allreduce_max", {"Out"}},
    {"c_allreduce_min", {"Out"}},
    {"c_allreduce_prod", {"Out"}},
    {"c_reduce_sum", {"Out"}},
    {"c_reduce_max", {"Out"}},
    {"c_reduce_min", {"Out"}},
    {"c_reduce_prod", {"Out"}},
    {"c_reduce", {"Out"}},
    {"c_allgather", {"Out"}},
    {"c_scatter", {"Out"}},
    {"barrier", {"Out"}},
    {"fake_quantize_dequantize_moving_average_abs_max",
     {"Out", "OutScale", "OutAccum", "OutState"}},
    {"fake_quantize_dequantize_abs_max", {"Out", "OutScale"}},
    {"fake_channel_wise_quantize_dequantize_abs_max", {"Out", "OutScale"}},
    {"check_finite_and_unscale", {"Out", "FoundInfinite"}},
    {"update_loss_scaling",
     {"Out", "LossScaling", "OutGoodSteps", "OutBadSteps"}},
    {"moving_average_abs_max_scale", {"OutScale", "OutAccum", "OutState"}},
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

const char* IN_VAR_TYPE = R"(py::handle)";
const char* IN_VAR_LIST_TYPE = R"(py::handle)";

const char* OUT_VAR_TYPE = R"(std::shared_ptr<imperative::VarBase>)";
const char* OUT_VAR_LIST_TYPE = R"(std::vector<std::shared_ptr<imperative::VarBase>>)";

const char* CAST_VAR_TEMPLATE = R"(
  auto %s = CastPyHandleToVarBase("%s", "%s", %d, %s, %s);)";

const char* CAST_VAR_LIST_TEMPLATE = R"(
  auto %s = CastPyHandleToVarBaseList("%s", "%s", %d, %s, %s);)";


const char* ARG_TEMPLATE = R"(const %s& %s)";

const char* RETURN_TUPLE_TYPE = R"(std::tuple<%s>)";
const char* RETURN_TYPE = R"(%s)";
const char* RETURN_TUPLE_TEMPLATE = R"(std::make_tuple(%s))";
const char* RETURN_LIST_TEMPLATE = R"(outs["%s"])";
const char* RETURN_TEMPLATE = R"(outs["%s"][0])";

const char* FUNCTION_ARGS = R"(%s, const py::args& args)";
const char* FUNCTION_ARGS_NO_INPUT = R"(const py::args& args)";

const char* OP_FUNCTION_TEMPLATE =
R"(
%s %s(%s)
{
  %s
  framework::AttributeMap attrs;
  ConstructAttrMapFromPyArgs("%s", %d, &attrs, args);
  {
    py::gil_scoped_release release;
    auto tracer = imperative::GetCurrentTracer();
    imperative::NameVarBaseMap outs = %s;
    imperative::NameVarBaseMap ins = %s;
    %s
    tracer->TraceOp("%s", ins, outs, attrs);
    return %s; 
  }   
})";

const char* PYBIND_ITEM_TEMPLATE = R"(  %s.def("%s", &%s);)";

// clang-format on
static inline bool FindInsMap(const std::string& op_type,
                              const std::string& in_name) {
  return op_ins_map[op_type].count(in_name);
}

static inline bool FindOutsMap(const std::string& op_type,
                               const std::string& out_name) {
  return op_outs_map[op_type].count(out_name);
}

static inline bool FindPassingOutsMap(const std::string& op_type,
                                      const std::string& out_name) {
  return op_passing_outs_map[op_type].count(out_name);
}

static inline std::string TempName(const std::string& name) {
  return name + '_';
}

static std::tuple<std::vector<std::string>, std::vector<std::string>>
GenerateOpFunctions(const std::string& module_name) {
  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  std::vector<std::string> op_function_list, bind_function_list;
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();

  for (auto& pair : op_info_map) {
    auto& op_info = pair.second;
    auto op_proto = op_info.proto_;
    if (op_proto == nullptr) {
      continue;
    }
    auto& op_type = op_proto->type();
    // Skip ooerator which is not inherit form OperatorWithKernel, like while,
    // since only OperatorWithKernel can run in dygraph mode.
    if (!all_kernels.count(op_type)) {
      continue;
    }
    std::string input_args = "";
    std::string ins_initializer = "{";
    std::string ins_initializer_with_null = "";
    std::string py_arg = "";
    int arg_idx = 0;
    int input_args_num = 0;
    std::string ins_cast_str = "";
    for (auto& input : op_proto->inputs()) {
      auto& in_name = input.name();
      // skip those dispensable inputs, like ResidualData in conv2d
      if (input.dispensable() && !FindInsMap(op_type, in_name)) {
        continue;
      }
      const auto in_type = input.duplicable() ? IN_VAR_LIST_TYPE : IN_VAR_TYPE;
      auto input_arg =
          paddle::string::Sprintf(ARG_TEMPLATE, in_type, TempName(in_name));
      input_args += input_arg;
      input_args += ",";
      input_args_num++;
      const auto in_cast_type =
          input.duplicable() ? CAST_VAR_LIST_TEMPLATE : CAST_VAR_TEMPLATE;
      auto dispensable = input.dispensable() ? "true" : "false";
      ins_cast_str +=
          paddle::string::Sprintf(in_cast_type, in_name, op_type, in_name,
                                  arg_idx++, TempName(in_name), dispensable);

      if (input.dispensable()) {
        const auto in_template = input.duplicable()
                                     ? INPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST
                                     : INPUT_INITIALIZER_TEMPLATE_WITH_NULL;
        ins_initializer_with_null +=
            paddle::string::Sprintf(in_template, in_name, in_name, in_name);
      } else {
        const auto in_template = input.duplicable()
                                     ? INPUT_LIST_INITIALIZER_TEMPLATE
                                     : INPUT_INITIALIZER_TEMPLATE;
        ins_initializer +=
            paddle::string::Sprintf(in_template, in_name, in_name);
        ins_initializer += ",";
      }
    }
    if (ins_initializer.back() == ',') {
      ins_initializer.pop_back();
    }
    ins_initializer += "}";

    if (input_args.back() == ',') {
      input_args.pop_back();
    }

    // Generate outs initializer
    std::string outs_initializer = "{";
    std::string outs_initializer_with_null = "";
    std::string return_type = "";
    std::string return_str = "";

    int outs_num = 0;
    for (auto& output : op_proto->outputs()) {
      auto& out_name = output.name();
      // skip those dispensable oututs
      if (output.dispensable() && !FindOutsMap(op_type, out_name)) {
        continue;
      }
      const auto out_type =
          output.duplicable() ? OUT_VAR_LIST_TYPE : OUT_VAR_TYPE;
      const auto return_template =
          output.duplicable() ? RETURN_LIST_TEMPLATE : RETURN_TEMPLATE;
      if (FindPassingOutsMap(op_type, out_name)) {
        if (input_args != "") {
          input_args += ",";
        }
        input_args += out_type;
        input_args += out_name;
        input_args_num++;

        if (output.dispensable()) {
          const auto out_template =
              output.duplicable() ? OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL_LIST
                                  : OUTPUT_INITIALIZER_TEMPLATE_WITH_NULL;
          outs_initializer_with_null +=
              paddle::string::Sprintf(out_template, out_name, out_name);
        } else {
          const auto out_template = output.duplicable()
                                        ? INPUT_LIST_INITIALIZER_TEMPLATE
                                        : INPUT_INITIALIZER_TEMPLATE;
          outs_initializer +=
              paddle::string::Sprintf(out_template, out_name, out_name);
          outs_initializer += ",";
        }
      } else {
        // There are few Operators that have duplicable output, like `Out` in
        // split op. We need to specify the number of variables for the
        // duplicable output, as the argument OutNum;
        if (output.duplicable()) {
          if (input_args != "") {
            input_args += ",";
          }
          auto out_num_str = paddle::string::Sprintf(ARG_OUT_NUM, out_name);
          input_args += ARG_OUT_NUM_TYPE;
          input_args += out_num_str;
          input_args_num++;
          outs_initializer += paddle::string::Sprintf(
              OUT_DUPLICABLE_INITIALIZER_TEMPLATE, out_name, out_num_str);
        } else {
          outs_initializer +=
              paddle::string::Sprintf(OUT_INITIALIZER_TEMPLATE, out_name);
        }
        outs_initializer += ",";
      }

      return_type += out_type;
      return_type += ",";
      return_str += paddle::string::Sprintf(return_template, out_name);
      return_str += ",";
      outs_num += 1;
    }
    if (outs_initializer.back() == ',') {
      outs_initializer.pop_back();
      return_type.pop_back();
      return_str.pop_back();
    }
    outs_initializer += "}";
    if (outs_num == 0) {
      return_type = "void";
    }
    if (outs_num > 1) {
      return_str = paddle::string::Sprintf(RETURN_TUPLE_TEMPLATE, return_str);
      return_type = paddle::string::Sprintf(RETURN_TUPLE_TYPE, return_type);
    }
    std::string function_args = "";
    if (input_args == "") {
      function_args = FUNCTION_ARGS_NO_INPUT;
    } else {
      function_args = paddle::string::Sprintf(FUNCTION_ARGS, input_args);
    }

    std::string func_name = "imperative_" + op_type;
    // generate op funtcion body
    auto op_function_str = paddle::string::Sprintf(
        OP_FUNCTION_TEMPLATE, return_type, func_name, function_args,
        ins_cast_str, op_type, input_args_num, outs_initializer,
        ins_initializer, ins_initializer_with_null + outs_initializer_with_null,
        op_type, return_str);

    // generate pybind item
    auto bind_function_str = paddle::string::Sprintf(
        PYBIND_ITEM_TEMPLATE, module_name, op_type, func_name);

    op_function_list.emplace_back(std::move(op_function_str));
    bind_function_list.emplace_back(std::move(bind_function_str));
  }
  return std::make_tuple(op_function_list, bind_function_list);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "argc must be 2" << std::endl;
    return -1;
  }

  std::vector<std::string> headers{"\"paddle/fluid/imperative/tracer.h\""};

  std::ofstream out(argv[1], std::ios::out);

  out << "#pragma once\n\n";

  for (auto& header : headers) {
    out << "#include  " + header + "\n";
  }

  auto op_funcs = GenerateOpFunctions("m");

  out << "namespace py = pybind11;"
      << "\n";
  out << "namespace paddle {\n"
      << "namespace pybind {\n";
  out << paddle::string::join_strings(std::get<0>(op_funcs), '\n');
  out << "\n\n";

  out << "inline void BindOpFunctions(pybind11::module *module) {\n"
      << "  auto m = module->def_submodule(\"ops\");\n\n";

  out << paddle::string::join_strings(std::get<1>(op_funcs), '\n');
  out << "\n";
  out << "}\n\n"
      << "} // namespace pybind\n"
      << "} // namespace paddle\n";

  out.close();
  return 0;
}
