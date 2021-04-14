/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace framework {
class OperatorBase;
class OpDesc;
class InferShapeContext;
class InferVarTypeContext;
class BlockDesc;
class Variable;
class InferNoNeedBufferVarsFN;

using VariableNameMap = std::map<std::string, std::vector<std::string>>;
// TODO(panyx0718): Replace vector with something like gtl::Vector.
using VariableValueMap = std::map<std::string, std::vector<Variable*>>;

// The order should be as same as framework.proto
using Attribute = boost::variant<
    boost::blank, int, float, std::string, std::vector<int>, std::vector<float>,
    std::vector<std::string>, bool, std::vector<bool>, BlockDesc*, int64_t,
    std::vector<BlockDesc*>, std::vector<int64_t>, std::vector<double>>;

using AttributeMap = std::unordered_map<std::string, Attribute>;

#ifdef PADDLE_WITH_ASCEND_CL
using NPUAttribute =
    boost::variant<boost::blank, int, float, std::string, std::vector<int>,
                   std::vector<float>, std::vector<std::string>, bool,
                   std::vector<bool>, BlockDesc*, int64_t,
                   std::vector<BlockDesc*>, std::vector<int64_t>,
                   std::vector<double>, std::vector<std::vector<int64_t>>>;

using NPUAttributeMap = std::unordered_map<std::string, NPUAttribute>;
#endif

using OpCreator = std::function<OperatorBase*(
    const std::string& /*type*/, const VariableNameMap& /*inputs*/,
    const VariableNameMap& /*outputs*/, const AttributeMap& /*attrs*/)>;

using GradOpMakerFN = std::function<std::vector<std::unique_ptr<OpDesc>>(
    const OpDesc&, const std::unordered_set<std::string>& /*no_grad_set*/,
    std::unordered_map<std::string, std::string>* /*grad_to_var*/,
    const std::vector<BlockDesc*>& grad_block)>;

using DygraphGradOpMakerFN =
    std::function<std::shared_ptr<imperative::GradOpNode>(
        const std::string& /*op_type*/,
        const imperative::NameVarBaseMap& /*var_base_map_in*/,
        const imperative::NameVarBaseMap& /*var_base_map_out*/,
        const framework::AttributeMap& /*attributes*/,
        const std::map<std::string, std::string>& /*inplace_map*/)>;

using InferVarTypeFN =
    std::function<void(framework::InferVarTypeContext* /*context*/)>;

using InferShapeFN = std::function<void(InferShapeContext*)>;

using InplacePair = std::unordered_map<std::string, std::string>;
using InferInplaceOpFN = std::function<InplacePair(bool /*use_cuda*/)>;

}  // namespace framework
}  // namespace paddle
