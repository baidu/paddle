/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/custom_operator.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/extension/include/ext_tensor.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/custom_tensor_utils.h"
#include "paddle/fluid/framework/op_meta_info_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

namespace detail {

// dynamic lib load func
template <typename T>
static T* DynLoad(void* handle, std::string name) {
  T* func = reinterpret_cast<T*>(dlsym(handle, name.c_str()));
#if !defined(_WIN32)
  auto errorno = dlerror();
#else
  auto errorno = GetLastError();
#endif  // !_WIN32
  PADDLE_ENFORCE_NOT_NULL(
      func, platform::errors::NotFound(
                "Failed to load dynamic operator library, error message(%s).",
                errorno));
  return func;
}

inline bool IsGradVar(const std::string& var_name) {
  std::string suffix = kGradVarSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

inline bool IsDuplicableVar(const std::string& var_name) {
  std::string suffix = kTensorVectorSuffix;
  return var_name.rfind(suffix) != std::string::npos;
}

inline std::string NoGrad(const std::string& var_name) {
  std::string suffix = kGradVarSuffix;
  return var_name.substr(0, var_name.size() - kGradVarSuffixSize);
}

inline bool IsMemberOf(const std::vector<std::string>& vec,
                       const std::string& name) {
  return std::find(vec.cbegin(), vec.cend(), name) != vec.cend();
}

std::vector<std::string> ParseAttrStr(const std::string& attr) {
  auto split_pos = attr.find_first_of(":");
  PADDLE_ENFORCE_NE(split_pos, std::string::npos,
                    platform::errors::InvalidArgument(
                        "Invalid attribute string format. Attribute string "
                        "format is `<name>:<type>`."));

  std::vector<std::string> rlt;
  // 1. name
  rlt.emplace_back(string::trim_spaces(attr.substr(0, split_pos)));
  // 2. type
  rlt.emplace_back(string::trim_spaces(attr.substr(split_pos + 1)));

  VLOG(1) << "attr name: " << rlt[0] << ", attr type str: " << rlt[1];

  return rlt;
}

}  // namespace detail

////////////////// Kernel Define ////////////////////

// custom op kernel call function define
static void RunKernelFunc(const framework::ExecutionContext& ctx,
                          const paddle::KernelFunc& func,
                          const std::vector<std::string>& inputs,
                          const std::vector<std::string>& outputs,
                          const std::vector<std::string>& attrs) {
  VLOG(1) << "Custom Operator: Start run KernelFunc.";
  std::vector<paddle::Tensor> custom_ins;
  std::vector<std::vector<paddle::Tensor>> custom_vec_ins;
  for (auto& in_name : inputs) {
    VLOG(1) << "Custom Operator: input name - " << in_name;
    if (detail::IsDuplicableVar(in_name)) {
      // return const std::vector<const Tensor*>
      auto vec_x = ctx.MultiInput<Tensor>(in_name);
      PADDLE_ENFORCE_NE(vec_x.empty(), true,
                        platform::errors::NotFound(
                            "Input vector<tensor> (%s) is empty.", in_name));
      std::vector<paddle::Tensor> custom_vec_in;
      for (size_t i = 0; i < vec_x.size(); ++i) {
        auto* x = vec_x[i];
        PADDLE_ENFORCE_NOT_NULL(
            x, platform::errors::NotFound(
                   "The %d-th tensor in input vector<tensor> (%s) is nullptr.",
                   i, in_name));
        PADDLE_ENFORCE_EQ(x->IsInitialized(), true,
                          platform::errors::InvalidArgument(
                              "The %d-th tensor in input vector<tensor> (%s) "
                              "is not initialized.",
                              i, in_name));
        auto custom_t = paddle::Tensor(
            CustomTensorUtils::ConvertInnerPlaceToEnumPlace(x->place()));
        CustomTensorUtils::ShareDataFrom(static_cast<const void*>(x), custom_t);
        CustomTensorUtils::SetTensorCurrentStream(&custom_t, ctx.GetPlace());
        custom_vec_in.emplace_back(custom_t);
      }
      custom_vec_ins.emplace_back(custom_vec_in);
    } else {
      auto* x = ctx.Input<Tensor>(in_name);
      PADDLE_ENFORCE_NOT_NULL(x, platform::errors::NotFound(
                                     "Input tensor (%s) is nullptr.", in_name));
      PADDLE_ENFORCE_EQ(x->IsInitialized(), true,
                        platform::errors::InvalidArgument(
                            "Input tensor (%s) is not initialized.", in_name));
      auto custom_in = paddle::Tensor(
          CustomTensorUtils::ConvertInnerPlaceToEnumPlace(x->place()));
      CustomTensorUtils::ShareDataFrom(static_cast<const void*>(x), custom_in);
      CustomTensorUtils::SetTensorCurrentStream(&custom_in, ctx.GetPlace());
      custom_ins.emplace_back(custom_in);
    }
  }

  std::vector<boost::any> custom_attrs;
  for (auto& attr_str : attrs) {
    auto attr_name_and_type = detail::ParseAttrStr(attr_str);
    auto attr_name = attr_name_and_type[0];
    auto attr_type_str = attr_name_and_type[1];
    if (attr_type_str == "bool") {
      custom_attrs.emplace_back(ctx.Attr<bool>(attr_name));
    } else if (attr_type_str == "int") {
      custom_attrs.emplace_back(ctx.Attr<int>(attr_name));
    } else if (attr_type_str == "float") {
      custom_attrs.emplace_back(ctx.Attr<float>(attr_name));
    } else if (attr_type_str == "int64_t") {
      custom_attrs.emplace_back(ctx.Attr<int64_t>(attr_name));
    } else if (attr_type_str == "std::string") {
      custom_attrs.emplace_back(ctx.Attr<std::string>(attr_name));
    } else if (attr_type_str == "std::vector<int>") {
      custom_attrs.emplace_back(ctx.Attr<std::vector<int>>(attr_name));
    } else if (attr_type_str == "std::vector<float>") {
      custom_attrs.emplace_back(ctx.Attr<std::vector<float>>(attr_name));
    } else if (attr_type_str == "std::vector<int64_t>") {
      custom_attrs.emplace_back(ctx.Attr<std::vector<int64_t>>(attr_name));
    } else if (attr_type_str == "std::vector<std::string>") {
      custom_attrs.emplace_back(ctx.Attr<std::vector<std::string>>(attr_name));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported `%s` type value as custom attribute now. "
          "Supported data types include `bool`, `int`, `float`, "
          "`int64_t`, `std::string`, `std::vector<int>`, "
          "`std::vector<float>`, `std::vector<int64_t>`, "
          "`std::vector<std::string>`, Please check whether "
          "the attribute data type and data type string are matched.",
          attr_type_str));
    }
  }

  VLOG(1) << "Custom Operator: Run ComputeFunc.";
  try {
    auto outs = func(custom_ins, custom_vec_ins, custom_attrs);

    VLOG(1) << "Custom Operator: Share outputs into ExecutionContext.";
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto out_name = outputs[i];
      if (detail::IsDuplicableVar(out_name)) {
        PADDLE_ENFORCE(i == 0UL && outputs.size() == 1UL,
                       platform::errors::PreconditionNotMet(
                           "If custom operator's outputs contains `paddle::Vec("
                           ")` type, "
                           "it only can hold one output."));
        auto vec_true_outs = ctx.MultiOutput<Tensor>(out_name);
        PADDLE_ENFORCE_EQ(
            vec_true_outs.size(), outs.size(),
            platform::errors::InvalidArgument(
                "The number of element in custom operator outputs is wrong, "
                "expected contains %d Tensors, but actually contains %d "
                "Tensors.",
                vec_true_outs.size(), outs.size()));
        for (size_t j = 0; j < vec_true_outs.size(); ++j) {
          CustomTensorUtils::ShareDataTo(outs.at(j), vec_true_outs.at(j));
        }
      } else {
        auto* true_out = ctx.Output<Tensor>(out_name);
        CustomTensorUtils::ShareDataTo(outs.at(i), true_out);
      }
    }
  } catch (platform::EnforceNotMet& exception) {
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(platform::errors::External("%s", ex.what()));
  } catch (...) {
    PADDLE_THROW(platform::errors::Fatal(
        "Custom operator raises an unknown exception in rumtime."));
  }
}

//////////////////// Operator Define /////////////////

class CustomOperator : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  // Dummy infershape
  // Because it is a pure virtual function, it must be implemented
  void InferShape(framework::InferShapeContext* ctx) const override {
    VLOG(1) << "Custom Operator: Dummy infer shape of custom operator.";
  }

  /**
   * NOTE: [Skip the Kernel Selection]
   * Custom Op only registers one Op kernel on each device, so that the
   * data type selection and promotion that depends on GetExpectedKernelType,
   * as well as the adaptation of various other special situations,
   * need users to implement, to avoid users needs to implement
   * GetExpectedKernelType function when expanding other cases.
   * The RAW type is used here as the data type, indicating that
   * it can only be determined at runtime.
   */
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(proto::VarType::RAW, ctx.GetPlace());
  }

  /**
   * NOTE: [Skip Input Variable Cast for DataType]
   * Because the kernel data type is RAW, we should skip the cast for
   * data type difference when PrepareData.
   */
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const OpKernelType& expected_kernel_type) const override {
    return OpKernelType(expected_kernel_type.data_type_,
                        expected_kernel_type.place_, tensor.layout());
  }
};

class CustomOpMaker : public OpProtoAndCheckerMaker {
 public:
  explicit CustomOpMaker(const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs,
                         const std::vector<std::string>& attrs)
      : inputs_(inputs), outputs_(outputs), attrs_(attrs) {}

  void Make() override {
    for (auto& in_name : inputs_) {
      if (detail::IsDuplicableVar(in_name)) {
        AddInput(in_name, "The input " + in_name + "of Custom operator.")
            .AsDuplicable();
      } else {
        AddInput(in_name, "The input " + in_name + "of Custom operator.");
      }
    }
    for (auto& out_name : outputs_) {
      if (detail::IsDuplicableVar(out_name)) {
        AddOutput(out_name, "The output " + out_name + "of Custom Operator.")
            .AsDuplicable();
      } else {
        AddOutput(out_name, "The output " + out_name + "of Custom Operator.");
      }
    }
    for (auto& attr : attrs_) {
      auto attr_name_and_type = detail::ParseAttrStr(attr);
      auto attr_name = attr_name_and_type[0];
      auto attr_type_str = attr_name_and_type[1];
      if (attr_type_str == "bool") {
        AddAttr<bool>(attr_name, "custom operator bool attribute.")
            .SetDefault(false);
      } else if (attr_type_str == "int") {
        AddAttr<int>(attr_name, "custom operator int attribute.").SetDefault(1);
      } else if (attr_type_str == "float") {
        AddAttr<float>(attr_name, "custom operator float attribute.")
            .SetDefault(1.0f);
      } else if (attr_type_str == "int64_t") {
        AddAttr<int64_t>(attr_name, "custom operator int64_t attribute.")
            .SetDefault(1);
      } else if (attr_type_str == "std::string") {
        AddAttr<std::string>(attr_name, "custom operator int attribute.")
            .SetDefault("");
      } else if (attr_type_str == "std::vector<int>") {
        AddAttr<std::vector<int>>(attr_name,
                                  "custom operator std::vector<int> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<float>") {
        AddAttr<std::vector<float>>(
            attr_name, "custom operator std::vector<float> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<int64_t>") {
        AddAttr<std::vector<int64_t>>(
            attr_name, "custom operator std::vector<int64_t> attribute.")
            .SetDefault({});
      } else if (attr_type_str == "std::vector<std::string>") {
        AddAttr<std::vector<std::string>>(
            attr_name, "custom operator std::vector<std::string> attribute.")
            .SetDefault({});
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported `%s` type value as custom attribute now. "
            "Supported data types include `bool`, `int`, `float`, "
            "`int64_t`, `std::string`, `std::vector<int>`, "
            "`std::vector<float>`, `std::vector<int64_t>`, "
            "`std::vector<std::string>`, Please check whether "
            "the attribute data type and data type string are matched.",
            attr_type_str));
      }
    }
    AddComment(R"DOC(
Custom Operator.

According to the Tensor operation function implemented by the user 
independently of the framework, it is encapsulated into a framework 
operator to adapt to various execution scenarios such as dynamic graph, 
mode static graph mode, and inference mode.

)DOC");
  }

 private:
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  std::vector<std::string> attrs_;
};

template <typename T>
class CustomGradOpMaker;

template <>
class CustomGradOpMaker<OpDesc> : public SingleGradOpMaker<OpDesc> {
 public:
  explicit CustomGradOpMaker(
      const OpDesc& fwd_op, const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      const std::vector<BlockDesc*>& grad_block, const std::string& name,
      const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs)
      : SingleGradOpMaker<OpDesc>(fwd_op, no_grad_set, grad_to_var, grad_block),
        name_(name),
        inputs_(inputs),
        outputs_(outputs) {}

 protected:
  void Apply(GradOpPtr<OpDesc> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(1) << "Custom Operator: GradOpDescMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
      }
    }
    for (auto& out_name : outputs_) {
      VLOG(1) << "Custom Operator: GradOpDescMaker - output: " << out_name;
      if (detail::IsDuplicableVar(out_name)) {
        grad_op->SetOutput(out_name,
                           this->InputGrad(detail::NoGrad(out_name),
                                           /*drop_empty_grad=*/false));
      } else {
        grad_op->SetOutput(out_name, this->InputGrad(detail::NoGrad(out_name)));
      }
    }
    grad_op->SetAttrMap(this->Attrs());
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

template <>
class CustomGradOpMaker<imperative::OpBase>
    : public SingleGradOpMaker<imperative::OpBase> {
 public:
  explicit CustomGradOpMaker(
      const std::string& type,
      const imperative::NameVarBaseMap& var_base_map_in,
      const imperative::NameVarBaseMap& var_base_map_out,
      const AttributeMap& attrs,
      const std::map<std::string, std::string>& inplace_map,
      const std::string& name, const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs)
      : SingleGradOpMaker<imperative::OpBase>(
            type, var_base_map_in, var_base_map_out, attrs, inplace_map),
        name_(name),
        inputs_(inputs),
        outputs_(outputs) {}

 protected:
  // TODO(chenweihang): The code is duplicated with the previous one, because
  // ere OpMaker's Input, Output and other methods are protected. Putting the
  // function implementation outside the class will cause the method to be
  // uncallable,
  // so it is still implemented in the class for the time being.
  void Apply(GradOpPtr<imperative::OpBase> grad_op) const override {
    grad_op->SetType(name_);

    auto fwd_op_inputs = this->InputNames();
    auto fwd_op_outputs = this->OutputNames();

    for (auto& in_name : inputs_) {
      VLOG(1) << "Custom Operator: GradOpBaseMaker - input: " << in_name;
      if (!detail::IsGradVar(in_name)) {
        if (detail::IsMemberOf(fwd_op_inputs, in_name)) {
          grad_op->SetInput(in_name, this->Input(in_name));
        } else if (detail::IsMemberOf(fwd_op_outputs, in_name)) {
          grad_op->SetInput(in_name, this->Output(in_name));
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The input tensor name `%s` is invalid, expected it is the input "
              "or output of forward operator.",
              in_name));
        }
      } else {
        grad_op->SetInput(in_name, this->OutputGrad(detail::NoGrad(in_name)));
      }
    }
    for (auto& out_name : outputs_) {
      VLOG(1) << "Custom Operator: GradOpBaseMaker - output: " << out_name;
      grad_op->SetOutput(out_name, this->InputGrad(detail::NoGrad(out_name)));
    }
    grad_op->SetAttrMap(this->Attrs());
  }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

//////////// Operator and Kernel Register //////////////

void RegisterOperatorKernelWithPlace(const std::string& name,
                                     const paddle::KernelFunc& kernel_func,
                                     const proto::VarType::Type type,
                                     const PlaceType& place,
                                     const std::vector<std::string>& inputs,
                                     const std::vector<std::string>& outputs,
                                     const std::vector<std::string>& attrs) {
  OpKernelType key(type,
                   CustomTensorUtils::ConvertEnumPlaceToInnerPlace(place));
  VLOG(1) << "Custom Operator: op kernel key: " << key;
  OperatorWithKernel::AllOpKernels()[name][key] =
      [kernel_func, inputs, outputs,
       attrs](const framework::ExecutionContext& ctx) {
        VLOG(1) << "Custom Operator: run custom kernel func in lambda.";
        RunKernelFunc(ctx, kernel_func, inputs, outputs, attrs);
      };
}

void RegisterOperatorKernel(const std::string& name,
                            const paddle::KernelFunc& kernel_func,
                            const std::vector<std::string>& inputs,
                            const std::vector<std::string>& outputs,
                            const std::vector<std::string>& attrs) {
  VLOG(1) << "Custom Operator: op name in kernel: " << name;
  // NOTE [ Dummy Op Kernel Key ]
  // TODO(chenweihang): Because execute engine need get device context based
  // op_kernel_key.place_, so we should register kernel for each
  // device. But this is not entirely correct, if user only give a cpu kernel,
  // but call api in gpu device, it will cause error.
  RegisterOperatorKernelWithPlace(name, kernel_func, proto::VarType::RAW,
                                  PlaceType::kCPU, inputs, outputs, attrs);
#ifdef PADDLE_WITH_CUDA
  RegisterOperatorKernelWithPlace(name, kernel_func, proto::VarType::RAW,
                                  PlaceType::kGPU, inputs, outputs, attrs);
#endif
}

void RegisterOperatorWithMetaInfo(
    const std::vector<OpMetaInfo>& op_meta_infos) {
  /* Op register */
  OpInfo info;

  auto& base_op_meta = op_meta_infos.front();

  auto op_name = OpMetaInfoHelper::GetOpName(base_op_meta);
  auto& op_inputs = OpMetaInfoHelper::GetInputs(base_op_meta);
  auto& op_outputs = OpMetaInfoHelper::GetOutputs(base_op_meta);
  auto& op_attrs = OpMetaInfoHelper::GetAttrs(base_op_meta);
  auto& kernel_fn = OpMetaInfoHelper::GetKernelFn(base_op_meta);
  auto& infer_shape_func = OpMetaInfoHelper::GetInferShapeFn(base_op_meta);
  auto& infer_dtype_func = OpMetaInfoHelper::GetInferDtypeFn(base_op_meta);

  VLOG(1) << "Custom Operator: forward, op name: " << op_name;
  VLOG(1) << "Custom Operator: forward, op inputs: "
          << string::join_strings(op_inputs, ',');
  VLOG(1) << "Custom Operator: forward, op outputs: "
          << string::join_strings(op_outputs, ',');
  VLOG(1) << "Custom Operator: forward, op attrs: "
          << string::join_strings(op_attrs, ',');

  // Op
  info.creator_ = [](const std::string& op_name, const VariableNameMap& inputs,
                     const VariableNameMap& outputs,
                     const AttributeMap& attrs) {
    return new CustomOperator(op_name, inputs, outputs, attrs);
  };

  // OpMaker
  info.proto_ = new proto::OpProto;
  info.proto_->set_type(op_name);

  info.checker_ = new OpAttrChecker();
  CustomOpMaker custom_maker(op_inputs, op_outputs, op_attrs);
  custom_maker(info.proto_, info.checker_);
  PADDLE_ENFORCE_EQ(
      info.proto_->IsInitialized(), true,
      platform::errors::PreconditionNotMet(
          "Fail to initialize %s's OpProto, because %s is not initialized.",
          op_name, info.proto_->InitializationErrorString()));

  // InferShape
  if (infer_shape_func == nullptr) {
    // use default InferShape
    info.infer_shape_ = [op_inputs, op_outputs](InferShapeContext* ctx) {
      PADDLE_ENFORCE_EQ(
          op_inputs.size(), 1UL,
          platform::errors::Unavailable(
              "Your custom operator contains multiple inputs. "
              "We only allow a custom operator that contains only one input "
              "and only one output without setting the InferShapeFn. "
              "At this time, the input shape will be directly set to "
              "the output shape.\n"
              "Please set the InferShapeFn of custom "
              "operator by .SetInferShapeFn(PD_INFER_SHAPE(...))"));
      PADDLE_ENFORCE_EQ(
          op_outputs.size(), 1UL,
          platform::errors::Unavailable(
              "Your custom operator contains multiple outputs. "
              "We only allow a custom operator that contains only one input "
              "and only one output without setting the InferShapeFn. "
              "At this time, the input shape will be directly set to "
              "the output shape.\n"
              "Please set the InferShapeFn of custom "
              "operator by .SetInferShapeFn(PD_INFER_SHAPE(...))"));

      VLOG(1) << "Custom Operator: Default InferShape - share ddim.";
      ctx->ShareDim(op_inputs[0], op_outputs[0]);
    };
  } else {
    info.infer_shape_ = [op_inputs, op_outputs, op_attrs,
                         infer_shape_func](InferShapeContext* ctx) {
      std::vector<std::vector<int64_t>> input_shapes;
      std::vector<std::vector<std::vector<int64_t>>> vec_input_shapes;

      VLOG(1) << "Custom Operator: InferShape - get input ddim.";
      for (auto& in_name : op_inputs) {
        if (detail::IsDuplicableVar(in_name)) {
          OP_INOUT_CHECK(ctx->HasInputs(in_name), "Input", in_name, "Custom");
          auto vec_ddim = ctx->GetInputsDim(in_name);
          std::vector<std::vector<int64_t>> vec_shape;
          vec_shape.reserve(vec_ddim.size());
          std::transform(vec_ddim.begin(), vec_ddim.end(),
                         std::back_inserter(vec_shape),
                         [&](const DDim& ddim) -> std::vector<int64_t> {
                           return framework::vectorize(ddim);
                         });
          vec_input_shapes.emplace_back(vec_shape);
        } else {
          OP_INOUT_CHECK(ctx->HasInput(in_name), "Input", in_name, "Custom");
          auto ddim = ctx->GetInputDim(in_name);
          input_shapes.emplace_back(framework::vectorize(ddim));
        }
      }

      std::vector<boost::any> custom_attrs;
      for (auto& attr_str : op_attrs) {
        auto attr_name_and_type = detail::ParseAttrStr(attr_str);
        auto attr_name = attr_name_and_type[0];
        auto attr_type_str = attr_name_and_type[1];
        if (attr_type_str == "bool") {
          custom_attrs.emplace_back(ctx->Attrs().Get<bool>(attr_name));
        } else if (attr_type_str == "int") {
          custom_attrs.emplace_back(ctx->Attrs().Get<int>(attr_name));
        } else if (attr_type_str == "float") {
          custom_attrs.emplace_back(ctx->Attrs().Get<float>(attr_name));
        } else if (attr_type_str == "int64_t") {
          custom_attrs.emplace_back(ctx->Attrs().Get<int64_t>(attr_name));
        } else if (attr_type_str == "std::string") {
          custom_attrs.emplace_back(ctx->Attrs().Get<std::string>(attr_name));
        } else if (attr_type_str == "std::vector<int>") {
          custom_attrs.emplace_back(
              ctx->Attrs().Get<std::vector<int>>(attr_name));
        } else if (attr_type_str == "std::vector<float>") {
          custom_attrs.emplace_back(
              ctx->Attrs().Get<std::vector<float>>(attr_name));
        } else if (attr_type_str == "std::vector<int64_t>") {
          // NOTE(chenweihang): InferShape can't support std::vector<int64_t>
          // attr type, because the input type is std::vector<int64_t>, only
          // can use one rule to parse std::vector<int64_t> parameter
          continue;
        } else if (attr_type_str == "std::vector<std::string>") {
          custom_attrs.emplace_back(
              ctx->Attrs().Get<std::vector<std::string>>(attr_name));
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "Unsupported `%s` type value as custom attribute now. "
              "Supported data types include `bool`, `int`, `float`, "
              "`int64_t`, `std::string`, `std::vector<int>`, "
              "`std::vector<float>`, `std::vector<std::string>`, "
              "Please check whether the attribute data type and "
              "data type string are matched.",
              attr_type_str));
        }
      }

      VLOG(1) << "Custom Operator: InferShape - calc output ddim.";
      auto output_shapes =
          infer_shape_func(input_shapes, vec_input_shapes, custom_attrs);

      VLOG(1) << "Custom Operator: InferShape - set output ddim.";
      for (size_t i = 0; i < op_outputs.size(); ++i) {
        auto out_name = op_outputs[i];
        if (detail::IsDuplicableVar(out_name)) {
          std::vector<DDim> vec_ddim;
          vec_ddim.reserve(output_shapes.size());
          std::transform(output_shapes.begin(), output_shapes.end(),
                         std::back_inserter(vec_ddim),
                         [&](const std::vector<int64_t>& shape) -> DDim {
                           return framework::make_ddim(shape);
                         });
          ctx->SetOutputsDim(out_name, vec_ddim);
        } else {
          ctx->SetOutputDim(out_name, framework::make_ddim(output_shapes[i]));
        }
      }
    };
  }

  // Infer Dtype
  if (infer_dtype_func == nullptr) {
    // use defalut InferDtype
    info.infer_var_type_ = [op_inputs, op_outputs](InferVarTypeContext* ctx) {
      PADDLE_ENFORCE_EQ(
          op_inputs.size(), 1UL,
          platform::errors::Unavailable(
              "Your custom operator contains multiple inputs. "
              "We only allow a custom operator that contains only one input "
              "and only one output without setting the InferDtypeFn. "
              "At this time, the input dtype will be directly set to "
              "the output dtype.\n"
              "Please set the InferDtypeFn of custom "
              "operator by .SetInferDtypeFn(PD_INFER_DTYPE(...))"));
      PADDLE_ENFORCE_EQ(
          op_outputs.size(), 1UL,
          platform::errors::Unavailable(
              "Your custom operator contains multiple outputs. "
              "We only allow a custom operator that contains only one input "
              "and only one output without setting the InferDtypeFn. "
              "At this time, the input dtype will be directly set to "
              "the output dtype.\n"
              "Please set the InferDtypeFn of custom "
              "operator by .SetInferDtypeFn(PD_INFER_DTYPE(...))"));

      VLOG(1) << "Custom Operator: InferDtype - share dtype.";
      auto dtype = ctx->GetInputDataType(op_inputs[0]);
      ctx->SetOutputDataType(op_outputs[0], dtype);
    };
  } else {
    info.infer_var_type_ = [op_inputs, op_outputs,
                            infer_dtype_func](InferVarTypeContext* ctx) {
      std::vector<DataType> input_dtypes;
      std::vector<std::vector<DataType>> vec_input_dtypes;

      VLOG(1) << "Custom Operator: InferDtype - get input dtype.";
      for (auto& in_name : op_inputs) {
        if (detail::IsDuplicableVar(in_name)) {
          std::vector<DataType> vec_custom_dtype;
          for (size_t i = 0; i < ctx->InputSize(in_name); ++i) {
            auto dtype = ctx->GetInputDataType(in_name, i);
            vec_custom_dtype.emplace_back(
                CustomTensorUtils::ConvertInnerDTypeToEnumDType(dtype));
          }
          vec_input_dtypes.emplace_back(vec_custom_dtype);
        } else {
          auto dtype = ctx->GetInputDataType(in_name);
          input_dtypes.emplace_back(
              CustomTensorUtils::ConvertInnerDTypeToEnumDType(dtype));
        }
      }

      VLOG(1) << "Custom Operator: InferDtype - infer output dtype.";
      auto output_dtypes = infer_dtype_func(input_dtypes, vec_input_dtypes);

      VLOG(1) << "Custom Operator: InferDtype - set output dtype.";
      for (size_t i = 0; i < op_outputs.size(); ++i) {
        auto out_name = op_outputs[i];
        if (detail::IsDuplicableVar(out_name)) {
          for (size_t j = 0; j < output_dtypes.size(); ++j) {
            auto dtype = CustomTensorUtils::ConvertEnumDTypeToInnerDType(
                output_dtypes[i]);
            ctx->SetOutputDataType(out_name, dtype, j);
          }
        } else {
          ctx->SetOutputDataType(
              out_name, CustomTensorUtils::ConvertEnumDTypeToInnerDType(
                            output_dtypes[i]));
        }
      }
    };
  }

  // Kernel func
  RegisterOperatorKernel(op_name, kernel_fn, op_inputs, op_outputs, op_attrs);

  // If grad op or double grad op exists
  std::string cur_op_name = op_name;
  for (size_t i = 1; i < op_meta_infos.size(); ++i) {
    auto& cur_grad_op = op_meta_infos[i];

    auto& grad_op_name = OpMetaInfoHelper::GetOpName(cur_grad_op);
    auto& grad_op_inputs = OpMetaInfoHelper::GetInputs(cur_grad_op);
    auto& grad_op_outputs = OpMetaInfoHelper::GetOutputs(cur_grad_op);
    auto& grad_op_attrs = OpMetaInfoHelper::GetAttrs(cur_grad_op);
    auto& grad_kernel_fn = OpMetaInfoHelper::GetKernelFn(cur_grad_op);

    VLOG(1) << "Custom Operator: backward, op name: " << grad_op_name;
    VLOG(1) << "Custom Operator: backward, op inputs: "
            << string::join_strings(grad_op_inputs, ',');
    VLOG(1) << "Custom Operator: backward, op outputs: "
            << string::join_strings(grad_op_outputs, ',');

    // GradOpDescMaker
    info.grad_op_maker_ = [grad_op_name, grad_op_inputs, grad_op_outputs](
        const OpDesc& fwd_op,
        const std::unordered_set<std::string>& no_grad_set,
        std::unordered_map<std::string, std::string>* grad_to_var,
        const std::vector<BlockDesc*>& grad_block) {
      CustomGradOpMaker<paddle::framework::OpDesc> maker(
          fwd_op, no_grad_set, grad_to_var, grad_block, grad_op_name,
          grad_op_inputs, grad_op_outputs);
      return maker();
    };

    // GradOpBaseMaker
    info.dygraph_grad_op_maker_ = [grad_op_name, grad_op_inputs,
                                   grad_op_outputs](
        const std::string& type,
        const imperative::NameVarBaseMap& var_base_map_in,
        const imperative::NameVarBaseMap& var_base_map_out,
        const framework::AttributeMap& attrs,
        const framework::AttributeMap& default_attrs,
        const std::map<std::string, std::string>& inplace_map) {
      CustomGradOpMaker<paddle::imperative::OpBase> maker(
          type, var_base_map_in, var_base_map_out, attrs, inplace_map,
          grad_op_name, grad_op_inputs, grad_op_outputs);
      maker.SetDygraphDefaultAttrsMap(default_attrs);
      return maker();
    };

    /* Grad op register */
    OpInfo grad_info;

    // Grad Op
    grad_info.creator_ = [](
        const std::string& type, const VariableNameMap& inputs,
        const VariableNameMap& outputs, const AttributeMap& attrs) {
      return new CustomOperator(type, inputs, outputs, attrs);
    };

    // Grad InferShape
    grad_info.infer_shape_ = [grad_op_inputs,
                              grad_op_outputs](InferShapeContext* ctx) {
      // 1. if forward input exists, gradient's shape is same with forward input
      // default
      //    [Suitable for most situations]
      // 2. if forward input not exists, and only contains one grad input and
      // output,
      //    use grad input shape as grad output shape
      //    [Suitable for the situation that forward input is not used as
      //    backward input]
      // TODO(chenweihang): support set grad op infershape func if needed
      for (auto& out_name : grad_op_outputs) {
        auto fwd_name = detail::NoGrad(out_name);
        if (detail::IsDuplicableVar(fwd_name)) {
          // Duplicable forward var must as backward input
          ctx->ShareDim(fwd_name, out_name);
        } else {
          if (ctx->HasInput(fwd_name)) {
            ctx->ShareDim(fwd_name, out_name);
          } else {
            PADDLE_ENFORCE_EQ(
                grad_op_inputs.size() == 1UL && grad_op_outputs.size() == 1UL,
                true,
                platform::errors::Unavailable(
                    "Custom grad operator infershape error. "
                    "If a custom grad operator contains only one input and "
                    "only one output, the input shape will be directly set to "
                    "the output shape. Otherwise, Please set the forward input "
                    "as the grad operator's input."));
            ctx->ShareDim(grad_op_inputs[0], out_name);
          }
        }
      }
    };

    // Kernel func
    RegisterOperatorKernel(grad_op_name, grad_kernel_fn, grad_op_inputs,
                           grad_op_outputs, grad_op_attrs);

    // update current info
    OpInfoMap::Instance().Insert(cur_op_name, info);
    cur_op_name = grad_op_name;
    info = grad_info;
  }
  // insert last info
  OpInfoMap::Instance().Insert(cur_op_name, info);
}

void RegisterOperatorWithMetaInfoMap(
    const paddle::OpMetaInfoMap& op_meta_info_map) {
  auto& meta_info_map = op_meta_info_map.GetMap();
  VLOG(1) << "Custom Operator: size of op meta info map - "
          << meta_info_map.size();
  // pair: {op_type, OpMetaInfo}
  for (auto& pair : meta_info_map) {
    VLOG(1) << "Custom Operator: pair first -> op name: " << pair.first;
    RegisterOperatorWithMetaInfo(pair.second);
  }
}

////////////////////// User APIs ///////////////////////

// load op api
void LoadOpMetaInfoAndRegisterOp(const std::string& dso_name) {
  void* handle = paddle::platform::dynload::GetOpDsoHandle(dso_name);

  typedef OpMetaInfoMap& get_op_meta_info_map_t();
  auto* get_op_meta_info_map =
      detail::DynLoad<get_op_meta_info_map_t>(handle, "PD_GetOpMetaInfoMap");
  auto& op_meta_info_map = get_op_meta_info_map();

  RegisterOperatorWithMetaInfoMap(op_meta_info_map);
}

}  // namespace framework
}  // namespace paddle
