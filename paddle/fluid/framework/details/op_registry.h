/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <tuple>
#include <vector>
#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace framework {
namespace details {

enum OpInfoFillType {
  kOperator = 0,
  kOpProtoAndCheckerMaker = 1,
  kGradOpDescMaker = 2,
  kVarTypeInference = 3,
  kShapeInference = 4,
  kEstimateFlops = 5,
  kUnknown = -1
};

template <typename T>
struct OpInfoFillTypeID {
  static constexpr OpInfoFillType ID() {
    return std::is_base_of<OperatorBase, T>::value
               ? kOperator
               : (std::is_base_of<OpProtoAndCheckerMaker, T>::value
                      ? kOpProtoAndCheckerMaker
                      : (std::is_base_of<GradOpDescMakerBase, T>::value
                             ? kGradOpDescMaker
                             : (std::is_base_of<VarTypeInference, T>::value
                                    ? kVarTypeInference
                                    : (std::is_base_of<InferShapeBase, T>::value
                                           ? kShapeInference
                                           : (std::is_base_of<EstimateFlopsBase,
                                                              T>::value
                                                  ? kEstimateFlops
                                                  : kUnknown)))));
  }
};

template <typename T, OpInfoFillType = OpInfoFillTypeID<T>::ID()>
struct OpInfoFiller;

template <size_t I, bool at_end, typename... ARGS>
class OperatorRegistrarRecursive;

template <size_t I, typename... ARGS>
class OperatorRegistrarRecursive<I, false, ARGS...> {
 public:
  using T = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
  OperatorRegistrarRecursive(const char* op_type, OpInfo* info) {
    OpInfoFiller<T> fill;
    fill(op_type, info);
    constexpr auto size = sizeof...(ARGS);
    OperatorRegistrarRecursive<I + 1, I + 1 == size, ARGS...> reg(op_type,
                                                                  info);
    (void)(reg);
  }
};

template <size_t I, typename... ARGS>
class OperatorRegistrarRecursive<I, true, ARGS...> {
 public:
  OperatorRegistrarRecursive(const char* op_type, OpInfo* info) {}
};

template <typename T>
struct OpInfoFiller<T, kOperator> {
  void operator()(const char* op_type, OpInfo* info) const {
    info->creator_ = [](const std::string& type, const VariableNameMap& inputs,
                        const VariableNameMap& outputs,
                        const AttributeMap& attrs) {
      return new T(type, inputs, outputs, attrs);
    };
  }
};

template <typename T>
struct OpInfoFiller<T, kOpProtoAndCheckerMaker> {
  void operator()(const char* op_type, OpInfo* info) const {
    info->proto_ = new proto::OpProto;
    info->checker_ = new OpAttrChecker();
    T maker;
    maker(info->proto_, info->checker_);
    info->proto_->set_type(op_type);
    PADDLE_ENFORCE(
        info->proto_->IsInitialized(),
        "Fail to initialize %s's OpProto, because %s is not initialized",
        op_type, info->proto_->InitializationErrorString());
  }
};

template <typename T>
struct OpInfoFiller<T, kGradOpDescMaker> {
  void operator()(const char* op_type, OpInfo* info) const {
    info->grad_op_maker_ = [](
        const OpDesc& fwd_op,
        const std::unordered_set<std::string>& no_grad_set,
        std::unordered_map<std::string, std::string>* grad_to_var,
        const std::vector<BlockDesc*>& grad_block) {
      T maker(fwd_op, no_grad_set, grad_to_var, grad_block);
      return maker();
    };
  }
};

template <typename T>
struct OpInfoFiller<T, kVarTypeInference> {
  void operator()(const char* op_type, OpInfo* info) const {
    info->infer_var_type_ = [](const OpDesc& fwd_op, BlockDesc* block) {
      T inference;
      inference(fwd_op, block);
    };
  }
};

template <typename T>
struct OpInfoFiller<T, kShapeInference> {
  void operator()(const char* op_type, OpInfo* info) const {
    info->infer_shape_ = [](InferShapeContext* ctx) {
      T inference;
      inference(ctx);
    };
  }
};

template <typename T>
struct OpInfoFiller<T, kEstimateFlops> {
  void operator()(const char* op_tpe, OpInfo* info) const {
    info->estimate_flops_ = [](InferShapeContext* ctx) {
      T estimate_flops;
      return estimate_flops(ctx);
    };
  }
};

}  // namespace details

}  // namespace framework
}  // namespace paddle
