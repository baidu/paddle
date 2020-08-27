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

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "glog/logging.h"               // For VLOG()
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/details/op_registry.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/shape_inference.h"

namespace paddle {
namespace framework {

class Registrar {
 public:
  // In our design, various kinds of classes, e.g., operators and kernels,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_OP macros to
  // call this method. So, as long as the callee code calls USE_OP, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename... ARGS>
struct OperatorRegistrar : public Registrar {
  explicit OperatorRegistrar(const char* op_type) {
    if (OpInfoMap::Instance().Has(op_type)) {
      PADDLE_THROW("'%s' is registered more than once.", op_type);
    }
    static_assert(sizeof...(ARGS) != 0,
                  "OperatorRegistrar should be invoked at least by OpClass");
    OpInfo info;
    details::OperatorRegistrarRecursive<0, false, ARGS...>(op_type, &info);
    OpInfoMap::Instance().Insert(op_type, info);
  }
};

class OpRegistry {
 public:
  /**
   * @brief Return an OperatorBase constructed by type, inputs, outputs, attrs.
   *        In dygraph mode, inputs, output, attrs will be set to empty map to
   *        improve the execution efficiency of dygraph.
   *        Dygraph mode will use:
   *        framework::OpRegistry::CreateOp(type, {}, {}, {}, false).
   *
   * @param[str] type               The operator type.
   * @param[map] inputs             Inputs map of the operator.
   * @param[map] outputs            Outputs map of the operator.
   * @param[unordered_map] attrs    Attributes map of the operator.
   * @param[bool] attr_check
   *            Whether do the attribute check before OperatorBase construction.
   *            Default is true.
   *            Attr_check is used to control the check of attribute map.
   *            The check of attribute map have two purposes:
   *            1. check whether the attribute item is valid or not.
   *            2. add attribute item which has default value
   *            if it is not in attrs.
   *            In dygraph mode, attrs is an empty unordered_map,
   *            attr_check is set to false, otherwise it will be failed
   *            when check function called.
   */
  static std::unique_ptr<OperatorBase> CreateOp(
      const std::string& type, const VariableNameMap& inputs,
      const VariableNameMap& outputs, AttributeMap attrs,
      const std::vector<std::string>& input_names = {},
      const std::vector<std::string>& output_names = {},
      bool attr_check = true);

  static std::unique_ptr<OperatorBase> CreateOp(
      const proto::OpDesc& op_desc,
      const std::vector<std::string>& input_names = {},
      const std::vector<std::string>& output_names = {});

  static std::unique_ptr<OperatorBase> CreateOp(const OpDesc& op_desc);
  static std::unique_ptr<OperatorBase> CreateOp(const std::string& type);
};

template <typename PlaceType, bool at_end, size_t I, typename... KernelType>
struct OpKernelRegistrarFunctor;

template <typename PlaceType, typename T, typename Func>
inline void RegisterKernelClass(const char* op_type, const char* library_type,
                                int customized_type_value, Func func) {
  std::string library(library_type);
  std::string data_layout = "ANYLAYOUT";
  if (library == "MKLDNN") {
    data_layout = "MKLDNNLAYOUT";
  }
  OpKernelType key(ToDataType(std::type_index(typeid(T))), PlaceType(),
                   StringToDataLayout(data_layout),
                   StringToLibraryType(library_type), customized_type_value);
  OperatorWithKernel::AllOpKernels()[op_type][key] = func;
}

template <typename PlaceType, size_t I, typename... KernelTypes>
struct OpKernelRegistrarFunctor<PlaceType, false, I, KernelTypes...> {
  using KERNEL_TYPE =
      typename std::tuple_element<I, std::tuple<KernelTypes...>>::type;

  void operator()(const char* op_type, const char* library_type,
                  int customized_type_value) const {
    using T = typename KERNEL_TYPE::ELEMENT_TYPE;
    RegisterKernelClass<PlaceType, T>(
        op_type, library_type, customized_type_value,

        [](const framework::ExecutionContext& ctx) {
          KERNEL_TYPE().Compute(ctx);
        });
    constexpr auto size = std::tuple_size<std::tuple<KernelTypes...>>::value;
    OpKernelRegistrarFunctor<PlaceType, I + 1 == size, I + 1, KernelTypes...>
        func;
    func(op_type, library_type, customized_type_value);
  }
};

template <typename PlaceType, size_t I, typename... KernelType>
struct OpKernelRegistrarFunctor<PlaceType, true, I, KernelType...> {
  void operator()(const char* op_type, const char* library_type,
                  int customized_type_value) const {}
};

// User can register many kernel in one place. The data type could be
// different.
template <typename PlaceType, typename... KernelType>
class OpKernelRegistrar : public Registrar {
 public:
  explicit OpKernelRegistrar(const char* op_type, const char* library_type,
                             int customized_type_value) {
    OpKernelRegistrarFunctor<PlaceType, false, 0, KernelType...> func;
    func(op_type, library_type, customized_type_value);
  }
};

template <typename PlaceType, bool at_end, size_t I, typename... KernelType>
struct OpKernelRegistrarFunctorEx;

template <typename PlaceType, typename... DataTypeAndKernelType>
class OpKernelRegistrarEx : public Registrar {
 public:
  explicit OpKernelRegistrarEx(const char* op_type, const char* library_type,
                               int customized_type_value) {
    OpKernelRegistrarFunctorEx<PlaceType, false, 0, DataTypeAndKernelType...>
        func;
    func(op_type, library_type, customized_type_value);
  }
};

template <typename PlaceType, size_t I, typename... DataTypeAndKernelType>
struct OpKernelRegistrarFunctorEx<PlaceType, true, I,
                                  DataTypeAndKernelType...> {
  void operator()(const char* op_type, const char* library_type,
                  int customized_type_value) const {}
};

template <typename PlaceType, size_t I, typename... DataTypeAndKernelType>
struct OpKernelRegistrarFunctorEx<PlaceType, false, I,
                                  DataTypeAndKernelType...> {
  using Functor =
      typename std::tuple_element<I + 1,
                                  std::tuple<DataTypeAndKernelType...>>::type;
  using T =
      typename std::tuple_element<I,
                                  std::tuple<DataTypeAndKernelType...>>::type;

  void operator()(const char* op_type, const char* library_type,
                  int customized_type_value) const {
    RegisterKernelClass<PlaceType, T>(op_type, library_type,
                                      customized_type_value, Functor());

    constexpr auto size =
        std::tuple_size<std::tuple<DataTypeAndKernelType...>>::value;
    OpKernelRegistrarFunctorEx<PlaceType, I + 2 >= size, I + 2,
                               DataTypeAndKernelType...>
        func;
    func(op_type, library_type, customized_type_value);
  }
};

// clang-format off
/**
 * check if MACRO is used in GLOBAL NAMESPACE.
 */
#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

/*
  The variadic arguments should be class types derived from one of the
  following classes:
    OpProtoAndCheckerMaker
    GradOpDescMakerBase
    VarTypeInference
    InferShapeBase
*/
#define REGISTER_OPERATOR(op_type, op_class, ...)                        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                        \
      __reg_op__##op_type,                                               \
      "REGISTER_OPERATOR must be called in global namespace");           \
  static ::paddle::framework::OperatorRegistrar<op_class, ##__VA_ARGS__> \
      __op_registrar_##op_type##__(#op_type);                            \
  int TouchOpRegistrar_##op_type() {                                     \
    __op_registrar_##op_type##__.Touch();                                \
    return 0;                                                            \
  }

#define REGISTER_OP_WITHOUT_GRADIENT(op_type, op_class, op_maker_class) \
  REGISTER_OPERATOR(op_type, op_class, op_maker_class, \
        paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,   \
        paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

/**
 * Macro to register OperatorKernel.
 */
#define REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(op_type, library_type,             \
                                            place_class, customized_name,      \
                                            customized_type_value, ...)        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                              \
      __reg_op_kernel_##op_type##_##library_type##_##customized_name##__,      \
                                 "REGISTER_OP_KERNEL must be called in "       \
                                 "global namespace");                          \
  static ::paddle::framework::OpKernelRegistrar<place_class,                   \
                                                __VA_ARGS__>                   \
      __op_kernel_registrar_##op_type##_##library_type##_##customized_name##__(\
          #op_type, #library_type, customized_type_value);                     \
  int TouchOpKernelRegistrar_##op_type##_##library_type##_##customized_name() {\
    __op_kernel_registrar_##op_type##_##library_type##_##customized_name##__   \
        .Touch();                                                              \
    return 0;                                                                  \
  }

#define REGISTER_OP_KERNEL(op_type, library_type, place_class, ...)   \
  REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(                                \
      op_type, library_type, place_class, DEFAULT_TYPE,               \
      ::paddle::framework::OpKernelType::kDefaultCustomizedTypeValue, \
      __VA_ARGS__)

#define REGISTER_OP_CUDA_KERNEL(op_type, ...) \
  REGISTER_OP_KERNEL(op_type, CUDA, ::paddle::platform::CUDAPlace, __VA_ARGS__)

#define REGISTER_OP_CPU_KERNEL(op_type, ...) \
  REGISTER_OP_KERNEL(op_type, CPU, ::paddle::platform::CPUPlace, __VA_ARGS__)

#define REGISTER_OP_XPU_KERNEL(op_type, ...) \
  REGISTER_OP_KERNEL(op_type, XPU, ::paddle::platform::XPUPlace, __VA_ARGS__)

#define REGISTER_OP_KERNEL_EX(op_type, library_type, place_class,  \
                              customized_name,                     \
                              customized_type_value,               \
                              ...)                                 \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                  \
      __reg_op_kernel_##op_type##_##library_type##_##customized_name##__, \
                                 "REGISTER_OP_KERNEL_EX must be called in "  \
                                 "global namespace");  \
  static ::paddle::framework::OpKernelRegistrarEx<place_class,  \
                                                  __VA_ARGS__>  \
      __op_kernel_registrar_##op_type##_##library_type##_##customized_name##__(\
          #op_type, #library_type, customized_type_value);  \
  int TouchOpKernelRegistrar_##op_type##_##library_type##_##customized_name() {\
    __op_kernel_registrar_##op_type##_##library_type##_##customized_name##__   \
        .Touch();                                                              \
    return 0;                                                                  \
  }

#define REGISTER_OP_CUDA_KERNEL_FUNCTOR(op_type, ...)                 \
  REGISTER_OP_KERNEL_EX(                                              \
      op_type, CUDA, ::paddle::platform::CUDAPlace, DEFAULT_TYPE,     \
      ::paddle::framework::OpKernelType::kDefaultCustomizedTypeValue, \
      __VA_ARGS__)

#define REGISTER_OP_CPU_KERNEL_FUNCTOR(op_type, ...)                  \
  REGISTER_OP_KERNEL_EX(                                              \
      op_type, CPU, ::paddle::platform::CPUPlace, DEFAULT_TYPE,       \
      ::paddle::framework::OpKernelType::kDefaultCustomizedTypeValue, \
      __VA_ARGS__)

#define REGISTER_OP_XPU_KERNEL_FUNCTOR(op_type, ...)                  \
  REGISTER_OP_KERNEL_EX(                                              \
      op_type, XPU, ::paddle::platform::XPUPlace, DEFAULT_TYPE,       \
      ::paddle::framework::OpKernelType::kDefaultCustomizedTypeValue, \
      __VA_ARGS__)

/**
 * Macro to mark what Operator and Kernel
 * we will use and tell the compiler to
 * link them into target.
 */
#define USE_OP_ITSELF(op_type)                             \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                          \
      __use_op_itself_##op_type,                           \
      "USE_OP_ITSELF must be called in global namespace"); \
  extern int TouchOpRegistrar_##op_type();                 \
  UNUSED static int use_op_itself_##op_type##_ = TouchOpRegistrar_##op_type()

#define USE_OP_DEVICE_KERNEL_WITH_CUSTOM_TYPE(op_type,                     \
                                              LIBRARY_TYPE,                \
                                              customized_name)             \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      __use_op_kernel_##op_type##_##LIBRARY_TYPE##_##customized_name##__,  \
      "USE_OP_DEVICE_KERNEL must be in global namespace");                 \
  extern int                                                               \
      TouchOpKernelRegistrar_##op_type##_##LIBRARY_TYPE##_##customized_name(); \
  UNUSED static int use_op_kernel_##op_type##_##LIBRARY_TYPE##_##customized_name##_ = /* NOLINT */ \
      TouchOpKernelRegistrar_##op_type##_##LIBRARY_TYPE##_##customized_name()

#define USE_OP_DEVICE_KERNEL(op_type, LIBRARY_TYPE) \
  USE_OP_DEVICE_KERNEL_WITH_CUSTOM_TYPE(op_type, LIBRARY_TYPE, DEFAULT_TYPE)

// TODO(fengjiayi): The following macros
// seems ugly, do we have better method?

#ifndef PADDLE_WITH_CUDA
#define USE_OP_KERNEL(op_type) USE_OP_DEVICE_KERNEL(op_type, CPU)
#else
#define USE_OP_KERNEL(op_type)        \
  USE_OP_DEVICE_KERNEL(op_type, CPU); \
  USE_OP_DEVICE_KERNEL(op_type, CUDA)
#endif

#define USE_NO_KERNEL_OP(op_type) USE_OP_ITSELF(op_type);

#define USE_CPU_ONLY_OP(op_type) \
  USE_OP_ITSELF(op_type);        \
  USE_OP_DEVICE_KERNEL(op_type, CPU);

#define USE_CUDA_ONLY_OP(op_type) \
  USE_OP_ITSELF(op_type);         \
  USE_OP_DEVICE_KERNEL(op_type, CUDA)

#define USE_OP(op_type)   \
  USE_OP_ITSELF(op_type); \
  USE_OP_KERNEL(op_type)
// clang-format on

}  // namespace framework
}  // namespace paddle
