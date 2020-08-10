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

#include <string>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {

class OperatorBase;

using InferShapeVarPtr = boost::variant<VarDesc *, Variable *>;

class InferShapeContext {
 public:
  virtual ~InferShapeContext() = default;
  virtual bool HasInput(const std::string &name) const = 0;
  virtual bool HasOutput(const std::string &name) const = 0;

  virtual std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string &name) const = 0;
  virtual std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string &name) const = 0;

  virtual bool HasInputs(const std::string &name) const = 0;
  virtual bool HasOutputs(const std::string &name) const = 0;

  virtual DDim GetInputDim(const std::string &name) const = 0;
  virtual std::vector<DDim> GetInputsDim(const std::string &name) const = 0;
  virtual std::vector<DDim> GetReaderDims(const std::string &name) const;

  virtual void SetOutputDim(const std::string &name, const DDim &dim) = 0;
  virtual void SetOutputsDim(const std::string &name,
                             const std::vector<DDim> &dims) = 0;
  virtual void SetReaderDims(const std::string &name,
                             const std::vector<DDim> &dims);
  virtual std::string GetInputNameByIdx(size_t idx) const = 0;
  virtual std::string GetOutputNameByIdx(size_t idx) const = 0;
  virtual AttrReader Attrs() const = 0;
  virtual std::vector<std::string> Inputs(const std::string &name) const = 0;
  virtual std::vector<std::string> Outputs(const std::string &name) const = 0;

  virtual void ShareDim(const std::string &in, const std::string &out,
                        size_t i = 0, size_t j = 0) = 0;

  virtual void ShareLoD(const std::string &in, const std::string &out,
                        size_t i = 0, size_t j = 0) const = 0;
  // share the lod information of all the tensor from in to out.
  // out_vars[i].lod = in_vars[i].lod
  virtual void ShareAllLoD(const std::string &in,
                           const std::string &out) const = 0;

  virtual int32_t GetLoDLevel(const std::string &in, size_t i = 0) const = 0;

  virtual void SetLoDLevel(const std::string &out, int32_t lod_level,
                           size_t j = 0) const = 0;

  virtual bool IsRuntime() const = 0;

  virtual std::vector<InferShapeVarPtr> GetInputVarPtrs(
      const std::string &name) = 0;
  virtual std::vector<InferShapeVarPtr> GetOutputVarPtrs(
      const std::string &name) = 0;

 protected:
  virtual std::vector<DDim> GetRepeatedDims(const std::string &name) const = 0;
  virtual void SetRepeatedDims(const std::string &name,
                               const std::vector<DDim> &dims) = 0;
};

}  // namespace framework
}  // namespace paddle
