//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include "paddle/fluid/framework/async_variable.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type_traits.h"

namespace paddle {
namespace framework {

class Variable {
 public:
  Variable() { holder_.reset(new AsyncVariable()); }

  template <typename T>
  const T& Get() const {
    return holder_->Get<T>();
  }

  bool IsInitialized() const { return holder_->IsAvailable(); }

  template <typename T>
  T* GetMutable() {
    return holder_->GetMutable<T>();
  }

  template <typename T>
  bool IsType() const {
    return holder_ && holder_->Type() == VarTypeTrait<T>::kId;
  }

  void Clear() { holder_->Clear(); }

  int Type() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, platform::errors::NotFound("Variable is not initialized."));
    return holder_->Type();
  }

  void NotifyAvailable() {
    if (holder_ == nullptr) {
      VLOG(8) << "Warning: Calling NotifyAvailable on null AsyncVariable";
    }
    holder_->NotifyAvailable();
  }

 private:
  // This method hides type T, so it doesn't appear as a template parameter of
  // Variable.
  framework::TensorInplaceVersion* InplaceVersionCounter();

 public:
  uint32_t CurrentInplaceVersion();
  void BumpInplaceVersion();

 private:
  // pointers to an AsyncVariable object indeed.
  std::shared_ptr<AsyncVariable> holder_;
};

inline framework::TensorInplaceVersion* Variable::InplaceVersionCounter() {
  framework::TensorInplaceVersion* version_counter_ptr(nullptr);
  if (IsType<framework::LoDTensor>()) {
    version_counter_ptr =
        &GetMutable<framework::LoDTensor>()->InplaceVersionCounter();
  } else if (IsType<framework::Tensor>()) {
    version_counter_ptr =
        &GetMutable<framework::Tensor>()->InplaceVersionCounter();

  } else if (IsType<framework::SelectedRows>()) {
    version_counter_ptr = &GetMutable<framework::SelectedRows>()
                               ->mutable_value()
                               ->InplaceVersionCounter();
  } else {
    VLOG(4) << "Only supports Tensor, LoDTensor, SelectedRows to have "
               "TensorInplaceVersion, but received type "
            << platform::demangle(framework::ToTypeName(Type()));
  }
  return version_counter_ptr;
}

inline uint32_t Variable::CurrentInplaceVersion() {
  auto version_counter_ptr = InplaceVersionCounter();
  if (version_counter_ptr) {
    return version_counter_ptr->CurrentVersion();
  } else {
    return 0;
  }
}

inline void Variable::BumpInplaceVersion() {
  auto version_counter_ptr = InplaceVersionCounter();
  if (version_counter_ptr) {
    return version_counter_ptr->Bump();
  } else {
    VLOG(4) << "Only supports Tensor, LoDTensor, SelectedRows to have "
               "TensorInplaceVersion, but received type "
            << platform::demangle(framework::ToTypeName(Type()));
  }
}
}  // namespace framework
}  // namespace paddle
