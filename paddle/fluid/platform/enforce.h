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

#ifdef __GNUC__
#include <cxxabi.h>  // for __cxa_demangle
#endif               // __GNUC__

#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr
#include <unistd.h>  // sleep, usleep
#else                // _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // GetModuleFileName, Sleep
#endif

#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <thrust/system/cuda/error.h>
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <thrust/system_error.h>
#include "paddle/fluid/platform/cuda_error.pb.h"
#endif

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/dynload/curand.h"
#include "paddle/fluid/platform/dynload/cusolver.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include <error.h>
#include "paddle/fluid/platform/dynload/nccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP

#include "paddle/fluid/platform/dynload/hiprand.h"
#include "paddle/fluid/platform/dynload/miopen.h"
#include "paddle/fluid/platform/dynload/rocblas.h"
#ifdef PADDLE_WITH_RCCL
#include <rccl.h>
#include "paddle/fluid/platform/dynload/rccl.h"
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand.h>
#include <miopen/miopen.h>
#include <rocblas.h>
#include <thrust/system/hip/error.h>
#endif

// Note: these headers for simplify demangle type string
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace platform {
class ErrorSummary;
}  // namespace platform
}  // namespace paddle

#ifdef PADDLE_WITH_CUDA
DECLARE_int64(gpu_allocator_retry_time);
#endif
DECLARE_int32(call_stack_level);

namespace paddle {
namespace platform {

/** HELPER MACROS AND FUNCTIONS **/

#ifndef PADDLE_MAY_THROW
#define PADDLE_MAY_THROW noexcept(false)
#endif

// Because most enforce conditions would evaluate to true, we can use
// __builtin_expect to instruct the C++ compiler to generate code that
// always forces branch prediction of true.
// This generates faster binary code. __builtin_expect is since C++11.
// For more details, please check https://stackoverflow.com/a/43870188/724872.
#if !defined(_WIN32)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
// there is no equivalent intrinsics in msvc.
#define UNLIKELY(condition) (condition)
#endif

#if !defined(_WIN32)
#define LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#else
// there is no equivalent intrinsics in msvc.
#define LIKELY(condition) (condition)
#endif

#if defined _WIN32 && defined PADDLE_ON_INFERENCE && defined PADDLE_NO_PYTHON
#define HANDLE_THE_ERROR try {
#define END_HANDLE_THE_ERROR            \
  }                                     \
  catch (const std::exception& e) {     \
    std::cout << e.what() << std::endl; \
    throw;                              \
  }
#else
#define HANDLE_THE_ERROR
#define END_HANDLE_THE_ERROR
#endif

#ifdef __GNUC__
inline std::string demangle(std::string name) {
  int status = -4;  // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
  return (status == 0) ? res.get() : name;
}
#else
inline std::string demangle(std::string name) { return name; }
#endif

namespace details {
template <typename T>
inline constexpr bool IsArithmetic() {
  return std::is_arithmetic<T>::value;
}

template <typename T1, typename T2, bool kIsArithmetic /* = true */>
struct TypeConverterImpl {
  using Type1 = typename std::common_type<T1, T2>::type;
  using Type2 = Type1;
};

template <typename T1, typename T2>
struct TypeConverterImpl<T1, T2, false> {
  using Type1 = T1;
  using Type2 = T2;
};

template <typename T1, typename T2>
struct TypeConverter {
 private:
  static constexpr bool kIsArithmetic =
      IsArithmetic<T1>() && IsArithmetic<T2>();

 public:
  using Type1 = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type1;
  using Type2 = typename TypeConverterImpl<T1, T2, kIsArithmetic>::Type2;
};

template <typename T1, typename T2>
using CommonType1 = typename std::add_lvalue_reference<
    typename std::add_const<typename TypeConverter<T1, T2>::Type1>::type>::type;

template <typename T1, typename T2>
using CommonType2 = typename std::add_lvalue_reference<
    typename std::add_const<typename TypeConverter<T1, T2>::Type2>::type>::type;

// Here, we use SFINAE to check whether T can be converted to std::string
template <typename T>
struct CanToString {
 private:
  using YesType = uint8_t;
  using NoType = uint16_t;

  template <typename U>
  static YesType Check(decltype(std::cout << std::declval<U>())) {
    return 0;
  }

  template <typename U>
  static NoType Check(...) {
    return 0;
  }

 public:
  static constexpr bool kValue =
      std::is_same<YesType, decltype(Check<T>(std::cout))>::value;
};

template <bool kCanToString /* = true */>
struct BinaryCompareMessageConverter {
  template <typename T>
  static std::string Convert(const char* expression, const T& value) {
    return expression + std::string(":") + string::to_string(value);
  }
};

template <>
struct BinaryCompareMessageConverter<false> {
  template <typename T>
  static const char* Convert(const char* expression, const T& value) {
    return expression;
  }
};
}  // namespace details

template <typename T>
inline std::string ReplaceComplexTypeStr(std::string str,
                                         const std::string& type_name) {
  auto demangle_type_str = demangle(typeid(T).name());
  size_t start_pos = 0;
  while ((start_pos = str.find(demangle_type_str, start_pos)) !=
         std::string::npos) {
    str.replace(start_pos, demangle_type_str.length(), type_name);
    start_pos += type_name.length();
  }
  return str;
}

#define __REPLACE_COMPLEX_TYPE_STR__(__TYPENAME, __STR)                       \
  do {                                                                        \
    __STR = paddle::platform::ReplaceComplexTypeStr<__TYPENAME>(__STR,        \
                                                                #__TYPENAME); \
  } while (0)

inline std::string SimplifyDemangleStr(std::string str) {
  // the older is important, you have to put complex types in front
  __REPLACE_COMPLEX_TYPE_STR__(paddle::framework::AttributeMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::framework::Attribute, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::imperative::NameVariableWrapperMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(paddle::imperative::NameVarBaseMap, str);
  __REPLACE_COMPLEX_TYPE_STR__(std::string, str);
  return str;
}

inline std::string GetCurrentTraceBackString() {
  std::ostringstream sout;

  sout << "\n\n--------------------------------------\n";
  sout << "C++ Traceback (most recent call last):";
  sout << "\n--------------------------------------\n";
#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
  static constexpr int TRACE_STACK_LIMIT = 100;

  void* call_stack[TRACE_STACK_LIMIT];
  auto size = backtrace(call_stack, TRACE_STACK_LIMIT);
  auto symbols = backtrace_symbols(call_stack, size);
  Dl_info info;
  int idx = 0;
  for (int i = size - 1; i >= 0; --i) {
    if (dladdr(call_stack[i], &info) && info.dli_sname) {
      auto demangled = demangle(info.dli_sname);
      std::string path(info.dli_fname);
      // C++ traceback info are from core.so
      if (path.substr(path.length() - 3).compare(".so") == 0) {
        sout << string::Sprintf("%-3d %s\n", idx++,
                                SimplifyDemangleStr(demangled));
      }
    }
  }
  free(symbols);
#else
  sout << "Not support stack backtrace yet.\n";
#endif
  return sout.str();
}

template <typename StrType>
inline std::string GetErrorSumaryString(StrType&& what, const char* file,
                                        int line) {
  std::ostringstream sout;
  if (FLAGS_call_stack_level > 1) {
    sout << "\n----------------------\nError Message "
            "Summary:\n----------------------\n";
  }
  sout << string::Sprintf("%s (at %s:%d)", std::forward<StrType>(what), file,
                          line)
       << std::endl;
  return sout.str();
}

template <typename StrType>
inline std::string GetTraceBackString(StrType&& what, const char* file,
                                      int line) {
  if (FLAGS_call_stack_level > 1) {
    // FLAGS_call_stack_level>1 means showing c++ call stack
    return GetCurrentTraceBackString() + GetErrorSumaryString(what, file, line);
  } else {
    return GetErrorSumaryString(what, file, line);
  }
}

inline std::string SimplifyErrorTypeFormat(const std::string& str) {
  std::ostringstream sout;
  size_t type_end_pos = str.find(":", 0);
  if (type_end_pos == std::string::npos) {
    sout << str;
  } else {
    // Remove "Error:", add "()""
    sout << "(" << str.substr(0, type_end_pos - 5) << ")"
         << str.substr(type_end_pos + 1);
  }
  return sout.str();
}

inline bool is_error(bool stat) { return !stat; }

// Note: This Macro can only be used within enforce.h
#define __THROW_ERROR_INTERNAL__(__ERROR_SUMMARY)                      \
  do {                                                                 \
    HANDLE_THE_ERROR                                                   \
    throw ::paddle::platform::EnforceNotMet(__ERROR_SUMMARY, __FILE__, \
                                            __LINE__);                 \
    END_HANDLE_THE_ERROR                                               \
  } while (0)

/** ENFORCE EXCEPTION AND MACROS **/

struct EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(std::exception_ptr e, const char* file, int line) {
    try {
      std::rethrow_exception(e);
    } catch (platform::EnforceNotMet& e) {
      code_ = e.code();
      err_str_ = GetTraceBackString(e.what(), file, line);
      simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    } catch (std::exception& e) {
      err_str_ = GetTraceBackString(e.what(), file, line);
      simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
    }
  }

  EnforceNotMet(const std::string& str, const char* file, int line)
      : err_str_(GetTraceBackString(str, file, line)) {
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  }

  EnforceNotMet(const ErrorSummary& error, const char* file, int line)
      : code_(error.code()),
        err_str_(GetTraceBackString(error.to_string(), file, line)) {
    simple_err_str_ = SimplifyErrorTypeFormat(err_str_);
  }

  const char* what() const noexcept override {
    if (FLAGS_call_stack_level > 1) {
      return err_str_.c_str();
    } else {
      return simple_err_str_.c_str();
    }
  }

  error::Code code() const { return code_; }

  const std::string& error_str() const { return err_str_; }

  const std::string& simple_error_str() const { return simple_err_str_; }

  void set_error_str(std::string str) {
    if (FLAGS_call_stack_level > 1) {
      err_str_ = str;
    } else {
      simple_err_str_ = str;
    }
  }

 private:
  // Used to determine the final type of exception thrown
  error::Code code_ = error::LEGACY;
  // Complete error message
  // e.g. InvalidArgumentError: ***
  std::string err_str_;
  // Simple errror message used when no C++ stack and python compile stack
  // e.g. (InvalidArgument) ***
  std::string simple_err_str_;
};

#define PADDLE_THROW(...)                                                   \
  do {                                                                      \
    HANDLE_THE_ERROR                                                        \
    throw ::paddle::platform::EnforceNotMet(                                \
        ::paddle::platform::ErrorSummary(__VA_ARGS__), __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                                                    \
  } while (0)

#if defined(__CUDA_ARCH__)
// For cuda, the assertions can affect performance and it is therefore
// recommended to disable them in production code
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#assertion
#define PADDLE_ENFORCE(_IS_NOT_ERROR, __FORMAT, ...)                         \
  do {                                                                       \
    if (!(_IS_NOT_ERROR)) {                                                  \
      printf("Error: %s:%d Assertion `%s` failed. " __FORMAT "\n", __FILE__, \
             __LINE__, #_IS_NOT_ERROR, ##__VA_ARGS__);                       \
      asm("trap;");                                                          \
    }                                                                        \
  } while (0)
#else
#define PADDLE_ENFORCE(COND, ...)                                              \
  do {                                                                         \
    auto __cond__ = (COND);                                                    \
    if (UNLIKELY(::paddle::platform::is_error(__cond__))) {                    \
      __THROW_ERROR_INTERNAL__(::paddle::platform::ErrorSummary(__VA_ARGS__)); \
    }                                                                          \
  } while (0)
#endif

/*
 * Some enforce helpers here, usage:
 *    int a = 1;
 *    int b = 2;
 *    PADDLE_ENFORCE_EQ(a, b);
 *
 *    will raise an expression described as follows:
 *    "Expected input a == b, but received a(1) != b(2)."
 *      with detailed stack information.
 *
 *    extra messages is also supported, for example:
 *    PADDLE_ENFORCE(a, b, "some simple enforce failed between %d numbers", 2)
 */

#define PADDLE_ENFORCE_NOT_NULL(__VAL, ...)                                   \
  do {                                                                        \
    if (UNLIKELY(nullptr == (__VAL))) {                                       \
      auto __summary__ = ::paddle::platform::ErrorSummary(__VA_ARGS__);       \
      auto __message__ = ::paddle::string::Sprintf(                           \
          "%s\n  [Hint: " #__VAL " should not be null.]",                     \
          __summary__.error_message());                                       \
      __THROW_ERROR_INTERNAL__(                                               \
          ::paddle::platform::ErrorSummary(__summary__.code(), __message__)); \
    }                                                                         \
  } while (0)

#define __PADDLE_BINARY_COMPARE(__VAL1, __VAL2, __CMP, __INV_CMP, ...)        \
  do {                                                                        \
    auto __val1 = (__VAL1);                                                   \
    auto __val2 = (__VAL2);                                                   \
    using __TYPE1__ = decltype(__val1);                                       \
    using __TYPE2__ = decltype(__val2);                                       \
    using __COMMON_TYPE1__ =                                                  \
        ::paddle::platform::details::CommonType1<__TYPE1__, __TYPE2__>;       \
    using __COMMON_TYPE2__ =                                                  \
        ::paddle::platform::details::CommonType2<__TYPE1__, __TYPE2__>;       \
    bool __is_not_error = (static_cast<__COMMON_TYPE1__>(__val1))__CMP(       \
        static_cast<__COMMON_TYPE2__>(__val2));                               \
    if (UNLIKELY(!__is_not_error)) {                                          \
      auto __summary__ = ::paddle::platform::ErrorSummary(__VA_ARGS__);       \
      constexpr bool __kCanToString__ =                                       \
          ::paddle::platform::details::CanToString<__TYPE1__>::kValue &&      \
          ::paddle::platform::details::CanToString<__TYPE2__>::kValue;        \
      auto __message__ = ::paddle::string::Sprintf(                           \
          "%s\n  [Hint: Expected %s " #__CMP                                  \
          " %s, but received %s " #__INV_CMP " %s.]",                         \
          __summary__.error_message(), #__VAL1, #__VAL2,                      \
          ::paddle::platform::details::BinaryCompareMessageConverter<         \
              __kCanToString__>::Convert(#__VAL1, __val1),                    \
          ::paddle::platform::details::BinaryCompareMessageConverter<         \
              __kCanToString__>::Convert(#__VAL2, __val2));                   \
      __THROW_ERROR_INTERNAL__(                                               \
          ::paddle::platform::ErrorSummary(__summary__.code(), __message__)); \
    }                                                                         \
  } while (0)

#define PADDLE_ENFORCE_EQ(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, ==, !=, __VA_ARGS__)
#define PADDLE_ENFORCE_NE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, !=, ==, __VA_ARGS__)
#define PADDLE_ENFORCE_GT(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, >, <=, __VA_ARGS__)
#define PADDLE_ENFORCE_GE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, >=, <, __VA_ARGS__)
#define PADDLE_ENFORCE_LT(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, <, >=, __VA_ARGS__)
#define PADDLE_ENFORCE_LE(__VAL0, __VAL1, ...) \
  __PADDLE_BINARY_COMPARE(__VAL0, __VAL1, <=, >, __VA_ARGS__)

/** EXTENDED TOOL FUNCTIONS WITH CHECKING **/

/*
 * Summary: This macro is used to get Variable or internal type
 *   data (such as LoDTensor or SelectedRows) of the Input and
 *   Output in op, generally used when call scope.FindVar(Input/
 *   Output("Name")) or ctx.Input<LoDTensor>().
 *   Firstly this macro check whether the obtained pointer is null,
 *   and then return data if it is not null.
 *
 * Note: This macro is only suitable for specific scenarios and
 *   does not intended to be widely used. If it cannot meet the
 *   requirements, please use other PADDLE_ENFORCE** check macro.
 *
 * Parameters:
 *     __PTR: pointer
 *     __ROLE: (string), Input or Output
 *     __NAME: (string), Input or Output name
 *     __OP_TYPE: (string), the op type
 *  
 * Return: The data pointed to by the pointer.
 *
 * Examples:
 *    GET_DATA_SAFELY(ctx.Input<LoDTensor>("X"), "Input", "X", "Mul");
 */
#define GET_DATA_SAFELY(__PTR, __ROLE, __NAME, __OP_TYPE)                     \
  (([&]() -> std::add_lvalue_reference<decltype(*(__PTR))>::type {            \
    auto* __ptr = (__PTR);                                                    \
    if (UNLIKELY(nullptr == __ptr)) {                                         \
      auto __summary__ = paddle::platform::errors::NotFound(                  \
          "Unable to get %s data of %s %s in operator %s. "                   \
          "Possible reasons are:\n"                                           \
          "  1. The %s is not the %s of operator %s;\n"                       \
          "  2. The %s has no corresponding variable passed in;\n"            \
          "  3. The %s corresponding variable is not initialized.",           \
          paddle::platform::demangle(                                         \
              typeid(std::add_lvalue_reference<decltype(*__ptr)>::type)       \
                  .name()),                                                   \
          __ROLE, __NAME, __OP_TYPE, __NAME, __ROLE, __OP_TYPE, __NAME,       \
          __NAME);                                                            \
      auto __message__ = ::paddle::string::Sprintf(                           \
          "%s\n  [Hint: pointer " #__PTR " should not be null.]",             \
          __summary__.error_message());                                       \
      __THROW_ERROR_INTERNAL__(                                               \
          ::paddle::platform::ErrorSummary(__summary__.code(), __message__)); \
    }                                                                         \
    return *__ptr;                                                            \
  })())

/*
 * Summary: This macro is used to check whether op has specified
 * Input or Output Variables. Because op's Input and Output
 * checking are written similarly, so abstract this macro.
 *
 * Parameters:
 *     __EXPR: (bool), the bool expression
 *     __ROLE: (string), Input or Output
 *     __NAME: (string), Input or Output name
 *     __OP_TYPE: (string), the op type
 *
 * Examples:
 *    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Mul");
 */
#define OP_INOUT_CHECK(__EXPR, __ROLE, __NAME, __OP_TYPE)                   \
  do {                                                                      \
    PADDLE_ENFORCE_EQ(__EXPR, true, paddle::platform::errors::NotFound(     \
                                        "No %s(%s) found for %s operator.", \
                                        __ROLE, __NAME, __OP_TYPE));        \
  } while (0)

/*
 * Summary: This BOOST_GET(_**) series macros are used to call boost::get
 *   safely. boost::get is not a completely safe api, although it will not
 *   go wrong in most cases, but in extreme cases, it may fail and directly
 *   throw a boost::bad_get exception, without any stack information.
 *   This kind of problems is difficult to debug, so add these macros to
 *   enrich boost::get error information. At the same time, we restrict
 *   the direct use of boost::get by CI rule.
 *
 * Parameters:
 *     __TYPE: the target variable type
 *     __VALUE: the target variable to get
 *
 * Examples:
 *     - unsafe writing: int x = boost::get<int>(y);
 *     - safe writing: int x = BOOST_GET(int, y);
 *
 * Note: GCC 4.8 cannot select right overloaded function here, so need
 *    to define different functions and macros here, after we upgreade
 *    CI gcc version, we can only define one BOOST_GET macro.
 */
namespace details {

#define DEFINE_SAFE_BOOST_GET(__InputType, __OutputType, __OutputTypePtr,      \
                              __FuncName)                                      \
  template <typename OutputType, typename InputType>                           \
  auto __FuncName(__InputType input, const char* expression, const char* file, \
                  int line)                                                    \
      ->typename std::conditional<std::is_pointer<InputType>::value,           \
                                  __OutputTypePtr, __OutputType>::type {       \
    try {                                                                      \
      return boost::get<OutputType>(input);                                    \
    } catch (boost::bad_get&) {                                                \
      HANDLE_THE_ERROR                                                         \
      throw ::paddle::platform::EnforceNotMet(                                 \
          ::paddle::platform::errors::InvalidArgument(                         \
              "boost::get failed, cannot get value "                           \
              "(%s) by type %s, its type is %s.",                              \
              expression,                                                      \
              paddle::platform::demangle(typeid(OutputType).name()),           \
              paddle::platform::demangle(input.type().name())),                \
          file, line);                                                         \
      END_HANDLE_THE_ERROR                                                     \
    }                                                                          \
  }

DEFINE_SAFE_BOOST_GET(InputType&, OutputType&, OutputType*, SafeBoostGet);
DEFINE_SAFE_BOOST_GET(const InputType&, const OutputType&, const OutputType*,
                      SafeBoostGetConst);
DEFINE_SAFE_BOOST_GET(InputType&&, OutputType, OutputType*,
                      SafeBoostGetMutable);

}  // namespace details

#define BOOST_GET(__TYPE, __VALUE)                                     \
  ::paddle::platform::details::SafeBoostGet<__TYPE>(__VALUE, #__VALUE, \
                                                    __FILE__, __LINE__)
#define BOOST_GET_CONST(__TYPE, __VALUE)                                    \
  ::paddle::platform::details::SafeBoostGetConst<__TYPE>(__VALUE, #__VALUE, \
                                                         __FILE__, __LINE__)
#define BOOST_GET_MUTABLE(__TYPE, __VALUE)                                    \
  ::paddle::platform::details::SafeBoostGetMutable<__TYPE>(__VALUE, #__VALUE, \
                                                           __FILE__, __LINE__)

/** OTHER EXCEPTION AND ENFORCE **/

struct EOFException : public std::exception {
  std::string err_str_;
  EOFException(const char* err_msg, const char* file, int line) {
    err_str_ = string::Sprintf("%s at [%s:%d]", err_msg, file, line);
  }

  const char* what() const noexcept override { return err_str_.c_str(); }
};

#define PADDLE_THROW_EOF()                                                     \
  do {                                                                         \
    HANDLE_THE_ERROR                                                           \
    throw ::paddle::platform::EOFException("There is no next data.", __FILE__, \
                                           __LINE__);                          \
    END_HANDLE_THE_ERROR                                                       \
  } while (0)

#define PADDLE_THROW_BAD_ALLOC(...)                                          \
  do {                                                                       \
    HANDLE_THE_ERROR                                                         \
    throw ::paddle::memory::allocation::BadAlloc(                            \
        ::paddle::platform::ErrorSummary(__VA_ARGS__).to_string(), __FILE__, \
        __LINE__);                                                           \
    END_HANDLE_THE_ERROR                                                     \
  } while (0)

/** CUDA PADDLE ENFORCE FUNCTIONS AND MACROS **/
#ifdef PADDLE_WITH_CUDA

/***** CUDA ERROR *****/
inline bool is_error(cudaError_t e) { return e != cudaSuccess; }

inline std::string GetCudaErrorWebsite(int32_t cuda_version) {
  std::ostringstream webstr;
  webstr << "https://docs.nvidia.com/cuda/";
  if (cuda_version != -1) {
    double version = cuda_version / 10;
    webstr << "archive/" << std::fixed << std::setprecision(1) << version;
  }
  webstr << "/cuda-runtime-api/group__CUDART__TYPES.html"
            "#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038";
  return webstr.str();
}

inline std::string build_nvidia_error_msg(cudaError_t e) {
#if CUDA_VERSION >= 10000 && CUDA_VERSION < 11000
  int32_t cuda_version = 100;
#elif CUDA_VERSION >= 9000
  int32_t cuda_version = 90;
#else
  int32_t cuda_version = -1;
#endif
  std::ostringstream sout;
  sout << " Cuda error(" << e << "), " << cudaGetErrorString(e) << ".";
  static platform::proto::cudaerrorDesc cudaerror;
  static bool _initSucceed = false;
  if (cudaerror.ByteSizeLong() == 0) {
    std::string filePath;
#if !defined(_WIN32)
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(GetCudaErrorWebsite), &info)) {
      std::string strModule(info.dli_fname);
      const size_t last_slash_idx = strModule.find_last_of("/");
      std::string compare_path = strModule.substr(strModule.length() - 6);
      if (std::string::npos != last_slash_idx) {
        strModule.erase(last_slash_idx, std::string::npos);
      }
      if (compare_path.compare("avx.so") == 0) {
        filePath = strModule +
                   "/../include/third_party/cudaerror/data/cudaErrorMessage.pb";
      } else {
        filePath =
            strModule + "/../../thirl_party/cudaerror/data/cudaErrorMessage.pb";
      }
    }
#else
    char buf[100];
    MEMORY_BASIC_INFORMATION mbi;
    HMODULE h_module =
        (::VirtualQuery(GetCudaErrorWebsite, &mbi, sizeof(mbi)) != 0)
            ? (HMODULE)mbi.AllocationBase
            : NULL;
    GetModuleFileName(h_module, buf, 100);
    std::string strModule(buf);
    const size_t last_slash_idx = strModule.find_last_of("\\");
    std::string compare_path = strModule.substr(strModule.length() - 7);
    if (std::string::npos != last_slash_idx) {
      strModule.erase(last_slash_idx, std::string::npos);
    }
    if (compare_path.compare("avx.pyd") == 0) {
      filePath =
          strModule +
          "\\..\\include\\third_party\\cudaerror\\data\\cudaErrorMessage.pb";
    } else {
      filePath =
          strModule + "\\..\\third_party\\cudaerror\\data\\cudaErrorMessage.pb";
    }
#endif
    std::ifstream fin(filePath, std::ios::in | std::ios::binary);
    _initSucceed = cudaerror.ParseFromIstream(&fin);
  }
  if (_initSucceed) {
    for (int i = 0; i < cudaerror.allmessages_size(); ++i) {
      if (cuda_version == cudaerror.allmessages(i).version()) {
        for (int j = 0; j < cudaerror.allmessages(i).messages_size(); ++j) {
          if (e == cudaerror.allmessages(i).messages(j).errorcode()) {
            sout << "\n  [Advise: "
                 << cudaerror.allmessages(i).messages(j).errormessage() << "]";
            return sout.str();
          }
        }
      }
    }
  }
  sout << "\n  [Advise: Please search for the error code(" << e
       << ") on website( " << GetCudaErrorWebsite(cuda_version)
       << " ) to get Nvidia's official solution about CUDA Error.]";
  return sout.str();
}

/** curand ERROR **/
inline bool is_error(curandStatus_t stat) {
  return stat != CURAND_STATUS_SUCCESS;
}

inline const char* curandGetErrorString(curandStatus_t stat) {
  switch (stat) {
    case CURAND_STATUS_SUCCESS:
      return "`CURAND_STATUS_SUCCESS`. No errors.";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "`CURAND_STATUS_VERSION_MISMATCH`. Header file and linked library "
             "version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "`CURAND_STATUS_NOT_INITIALIZED`. Generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "`CURAND_STATUS_ALLOCATION_FAILED`. Memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR:
      return "`CURAND_STATUS_TYPE_ERROR`. Generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "`CURAND_STATUS_OUT_OF_RANGE`. Argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "`CURAND_STATUS_LENGTH_NOT_MULTIPLE`. Length requested is not a "
             "multple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "`CURAND_STATUS_DOUBLE_PRECISION_REQUIRED`. GPU does not have "
             "double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "`CURAND_STATUS_LAUNCH_FAILURE`. Kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "`CURAND_STATUS_PREEXISTING_FAILURE`. Preexisting failure on "
             "library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "`CURAND_STATUS_INITIALIZATION_FAILED`. Initialization of CUDA "
             "failed.";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "`CURAND_STATUS_ARCH_MISMATCH`. Architecture mismatch, GPU does "
             "not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "`CURAND_STATUS_INTERNAL_ERROR`. Internal library error.";
    default:
      return "Unknown curand status";
  }
}

inline std::string build_nvidia_error_msg(curandStatus_t stat) {
  std::string msg(" Curand error, ");
  return msg + curandGetErrorString(stat) + " ";
}

/***** CUDNN ERROR *****/
inline bool is_error(cudnnStatus_t stat) {
  return stat != CUDNN_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cudnnStatus_t stat) {
  std::string msg(" Cudnn error, ");
  return msg + platform::dynload::cudnnGetErrorString(stat) + " ";
}

/***** CUBLAS ERROR *****/
inline bool is_error(cublasStatus_t stat) {
  return stat != CUBLAS_STATUS_SUCCESS;
}

inline const char* cublasGetErrorString(cublasStatus_t stat) {
  switch (stat) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "`CUBLAS_STATUS_NOT_INITIALIZED`. The cuBLAS library was not "
             "initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "`CUBLAS_STATUS_ALLOC_FAILED`. Resource allocation failed inside "
             "the cuBLAS library.";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "`CUBLAS_STATUS_INVALID_VALUE`. An unsupported value or parameter "
             "was passed to the function (a negative vector size, for "
             "example).";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "`CUBLAS_STATUS_ARCH_MISMATCH`. The function requires a feature "
             "absent from the device architecture; usually caused by the lack "
             "of support for double precision.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "`CUBLAS_STATUS_MAPPING_ERROR`. An access to GPU memory space "
             "failed, which is usually caused by a failure to bind a texture.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "`CUBLAS_STATUS_EXECUTION_FAILED`. The GPU program failed to "
             "execute. This is often caused by a launch failure of the kernel "
             "on the GPU, which can be caused by multiple reasons.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "`CUBLAS_STATUS_INTERNAL_ERROR`. An internal cuBLAS operation "
             "failed. This error is usually caused by a cudaMemcpyAsync() "
             "failure.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "`CUBLAS_STATUS_NOT_SUPPORTED`. The functionality requested is "
             "not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "`CUBLAS_STATUS_LICENSE_ERROR`. The functionality requested "
             "requires some license and an error was detected when trying to "
             "check the current licensing.";
    default:
      return "Unknown cublas status";
  }
}

inline std::string build_nvidia_error_msg(cublasStatus_t stat) {
  std::string msg(" Cublas error, ");
  return msg + cublasGetErrorString(stat) + " ";
}

/***** CUSOLVER ERROR *****/
inline bool is_error(cusolverStatus_t stat) {
  return stat != CUSOLVER_STATUS_SUCCESS;
}

inline const char* cusolverGetErrorString(cusolverStatus_t stat) {
  switch (stat) {
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "`CUSOLVER_STATUS_NOT_INITIALIZED`. The cuSolver library was not "
             "initialized. This is usually caused by the lack of a prior call, "
             "an error in the CUDA Runtime API called by the cuSolver routine, "
             "or an error in the hardware setup.";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "`CUSOLVER_STATUS_ALLOC_FAILED`. Resource allocation failed "
             "inside the cuSolver library. This is usually caused by a "
             "cudaMalloc() failure.";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "`CUSOLVER_STATUS_INVALID_VALUE`. An unsupported value or "
             "parameter was passed to the function (a negative vector size, "
             "for example).";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "`CUSOLVER_STATUS_ARCH_MISMATCH`. The function requires a feature "
             "absent from the device architecture; usually caused by the lack "
             "of support for atomic operations or double precision.";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "`CUSOLVER_STATUS_EXECUTION_FAILED`. The GPU program failed to "
             "execute. This is often caused by a launch failure of the kernel "
             "on the GPU, which can be caused by multiple reasons.";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "`CUSOLVER_STATUS_INTERNAL_ERROR`. An internal cuSolver operation "
             "failed. This error is usually caused by a cudaMemcpyAsync() "
             "failure.";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`. The matrix type is "
             "not supported by this function. This is usually caused by "
             "passing an invalid matrix descriptor to the function.";
    default:
      return "Unknown cusolver status";
  }
}

inline std::string build_nvidia_error_msg(cusolverStatus_t stat) {
  std::string msg(" Cublas error, ");
  return msg + cusolverGetErrorString(stat) + " ";
}

/****** NCCL ERROR ******/
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_nvidia_error_msg(ncclResult_t nccl_result) {
  std::string msg(" Nccl error, ");
  if (errno == ENOSPC || errno == EAGAIN) {
    std::string detail(strerror(errno));
    detail += "\nPlease try one of the following solutions:";
    detail += "\n1. export NCCL_SHM_DISABLE=1;";
    detail += "\n2. export NCCL_P2P_LEVEL=SYS;";
    detail +=
        "\n3. Increase shared memory by setting the -shm-size "
        "option when starting docker container, e.g., setting "
        " -shm-size=2g.\n";
    return msg + platform::dynload::ncclGetErrorString(nccl_result) +
           ", detail: " + detail + " ";
  }
  return msg + platform::dynload::ncclGetErrorString(nccl_result) + " ";
}
#endif  // not(__APPLE__) and PADDLE_WITH_NCCL

namespace details {

template <typename T>
struct CudaStatusType {};

#define DEFINE_CUDA_STATUS_TYPE(type, success_value) \
  template <>                                        \
  struct CudaStatusType<type> {                      \
    using Type = type;                               \
    static constexpr Type kSuccess = success_value;  \
  }

DEFINE_CUDA_STATUS_TYPE(cudaError_t, cudaSuccess);
DEFINE_CUDA_STATUS_TYPE(curandStatus_t, CURAND_STATUS_SUCCESS);
DEFINE_CUDA_STATUS_TYPE(cudnnStatus_t, CUDNN_STATUS_SUCCESS);
DEFINE_CUDA_STATUS_TYPE(cublasStatus_t, CUBLAS_STATUS_SUCCESS);
DEFINE_CUDA_STATUS_TYPE(cusolverStatus_t, CUSOLVER_STATUS_SUCCESS);

#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
DEFINE_CUDA_STATUS_TYPE(ncclResult_t, ncclSuccess);
#endif

}  // namespace details

#define PADDLE_ENFORCE_CUDA_SUCCESS(COND)                        \
  do {                                                           \
    auto __cond__ = (COND);                                      \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);             \
    constexpr auto __success_type__ =                            \
        ::paddle::platform::details::CudaStatusType<             \
            __CUDA_STATUS_TYPE__>::kSuccess;                     \
    if (UNLIKELY(__cond__ != __success_type__)) {                \
      auto __summary__ = ::paddle::platform::errors::External(   \
          ::paddle::platform::build_nvidia_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                     \
    }                                                            \
  } while (0)

inline void retry_sleep(unsigned milliseconds) {
#ifdef _WIN32
  Sleep(milliseconds);
#else
  if (milliseconds < 1000) {
    // usleep argument must be less than 1,000,000. Reference:
    // https://pubs.opengroup.org/onlinepubs/7908799/xsh/usleep.html
    usleep(milliseconds * 1000);
  } else {
    // clip to sleep in seconds because we can not and don't have to
    // sleep for exact milliseconds
    sleep(milliseconds / 1000);
  }
#endif
}

#define PADDLE_RETRY_CUDA_SUCCESS(COND)                                 \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    int retry_count = 1;                                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                    \
    constexpr auto __success_type__ =                                   \
        ::paddle::platform::details::CudaStatusType<                    \
            __CUDA_STATUS_TYPE__>::kSuccess;                            \
    while (UNLIKELY(__cond__ != __success_type__) && retry_count < 5) { \
      retry_sleep(FLAGS_gpu_allocator_retry_time);                      \
      __cond__ = (COND);                                                \
      ++retry_count;                                                    \
    }                                                                   \
    if (UNLIKELY(__cond__ != __success_type__)) {                       \
      auto __summary__ = ::paddle::platform::errors::External(          \
          ::paddle::platform::build_nvidia_error_msg(__cond__));        \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#undef DEFINE_CUDA_STATUS_TYPE
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
/***** HIP ERROR *****/
inline bool is_error(hipError_t e) { return e != hipSuccess; }

inline std::string GetCudaErrorWebsite(int32_t cuda_version) {
  std::ostringstream webstr;
  webstr << "rocm not need web check";
  return webstr.str();
}

inline std::string build_nvidia_error_msg(hipError_t e) {
  int32_t cuda_version = 100;

  std::ostringstream sout;
  sout << " hip error(" << e << "), " << hipGetErrorString(e) << ".";
  static platform::proto::cudaerrorDesc cudaerror;
  static bool _initSucceed = false;
  if (cudaerror.ByteSizeLong() == 0) {
    std::string filePath;
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(GetCudaErrorWebsite), &info)) {
      std::string strModule(info.dli_fname);
      const size_t last_slash_idx = strModule.find_last_of("/");
      std::string compare_path = strModule.substr(strModule.length() - 6);
      if (std::string::npos != last_slash_idx) {
        strModule.erase(last_slash_idx, std::string::npos);
      }
      if (compare_path.compare("avx.so") == 0) {
        filePath = strModule +
                   "/../include/third_party/cudaerror/data/cudaErrorMessage.pb";
      } else {
        filePath =
            strModule + "/../../thirl_party/cudaerror/data/cudaErrorMessage.pb";
      }
    }
    std::ifstream fin(filePath, std::ios::in | std::ios::binary);
    _initSucceed = cudaerror.ParseFromIstream(&fin);
  }
  if (_initSucceed) {
    for (int i = 0; i < cudaerror.allmessages_size(); ++i) {
      if (cuda_version == cudaerror.allmessages(i).version()) {
        for (int j = 0; j < cudaerror.allmessages(i).messages_size(); ++j) {
          if (e == cudaerror.allmessages(i).messages(j).errorcode()) {
            sout << "\n  [Advise: "
                 << cudaerror.allmessages(i).messages(j).errormessage() << "]";
            return sout.str();
          }
        }
      }
    }
  }
  sout << "\n  [Advise: Please search for the error code(" << e
       << ") on website( " << GetCudaErrorWebsite(cuda_version)
       << " ) to get Nvidia's official solution about CUDA Error.]";
  return sout.str();
}

inline void throw_on_error(hipError_t e, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(msg);
#else
  LOG(FATAL) << msg;
#endif
}

/** hiprand ERROR **/
inline bool is_error(hiprandStatus_t stat) {
  return stat != HIPRAND_STATUS_SUCCESS;
}

inline const char* hiprandGetErrorString(hiprandStatus_t stat) {
  switch (stat) {
    case HIPRAND_STATUS_SUCCESS:
      return "HIPRAND_STATUS_SUCCESS";
    case HIPRAND_STATUS_VERSION_MISMATCH:
      return "HIPRAND_STATUS_VERSION_MISMATCH";
    case HIPRAND_STATUS_NOT_INITIALIZED:
      return "HIPRAND_STATUS_NOT_INITIALIZED";
    case HIPRAND_STATUS_ALLOCATION_FAILED:
      return "HIPRAND_STATUS_ALLOCATION_FAILED";
    case HIPRAND_STATUS_TYPE_ERROR:
      return "HIPRAND_STATUS_TYPE_ERROR";
    case HIPRAND_STATUS_OUT_OF_RANGE:
      return "HIPRAND_STATUS_OUT_OF_RANGE";
    case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
    case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case HIPRAND_STATUS_LAUNCH_FAILURE:
      return "HIPRAND_STATUS_LAUNCH_FAILURE";
    case HIPRAND_STATUS_PREEXISTING_FAILURE:
      return "HIPRAND_STATUS_PREEXISTING_FAILURE";
    case HIPRAND_STATUS_INITIALIZATION_FAILED:
      return "HIPRAND_STATUS_INITIALIZATION_FAILED";
    case HIPRAND_STATUS_ARCH_MISMATCH:
      return "HIPRAND_STATUS_ARCH_MISMATCH";
    case HIPRAND_STATUS_INTERNAL_ERROR:
      return "HIPRAND_STATUS_INTERNAL_ERROR";
    case HIPRAND_STATUS_NOT_IMPLEMENTED:
      return "HIPRAND_STATUS_NOT_IMPLEMENTED";
    default:
      return "Unknown hiprand status";
  }
}

inline std::string build_nvidia_error_msg(hiprandStatus_t stat) {
  std::string msg(" Hiprand error, ");
  return msg + hiprandGetErrorString(stat) + " ";
}

inline void throw_on_error(hiprandStatus_t stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw thrust::system_error(hipErrorLaunchFailure, thrust::hip_category(),
                             msg);
#else
  LOG(FATAL) << msg;
#endif
}

/***** MIOPEN ERROR *****/
inline bool is_error(miopenStatus_t stat) {
  return stat != miopenStatusSuccess;
}

inline std::string build_nvidia_error_msg(miopenStatus_t stat) {
  std::string msg(" Miopen error, ");
  return msg + platform::dynload::miopenGetErrorString(stat) + " ";
}

inline void throw_on_error(miopenStatus_t stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(msg);
#else
  LOG(FATAL) << msg;
#endif
}

/***** ROCBLAS ERROR *****/
inline bool is_error(rocblas_status stat) {
  return stat != rocblas_status_success;
}

inline const char* rocblasGetErrorString(rocblas_status stat) {
  switch (stat) {
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_invalid_value:
      return "rocblas_status_invalid_value";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return "Unknown rocblas status";
  }
}

inline std::string build_nvidia_error_msg(rocblas_status stat) {
  std::string msg(" Rocblas error, ");
  return msg + rocblasGetErrorString(stat) + " ";
}

inline void throw_on_error(rocblas_status stat, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(msg);
#else
  LOG(FATAL) << msg;
#endif
}

/****** RCCL ERROR ******/
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_nvidia_error_msg(ncclResult_t nccl_result) {
  std::string msg(" Rccl error, ");
  return msg + platform::dynload::ncclGetErrorString(nccl_result) + " ";
}

inline void throw_on_error(ncclResult_t nccl_result, const std::string& msg) {
#ifndef REPLACE_ENFORCE_GLOG
  throw std::runtime_error(msg);
#else
  LOG(FATAL) << msg;
#endif
}
#endif  // not(__APPLE__) and PADDLE_WITH_RCCL

namespace details {

template <typename T>
struct CudaStatusType {};

#define DEFINE_CUDA_STATUS_TYPE(type, success_value) \
  template <>                                        \
  struct CudaStatusType<type> {                      \
    using Type = type;                               \
    static constexpr Type kSuccess = success_value;  \
  }

DEFINE_CUDA_STATUS_TYPE(hipError_t, hipSuccess);
DEFINE_CUDA_STATUS_TYPE(hiprandStatus_t, HIPRAND_STATUS_SUCCESS);
DEFINE_CUDA_STATUS_TYPE(miopenStatus_t, miopenStatusSuccess);
DEFINE_CUDA_STATUS_TYPE(rocblas_status, rocblas_status_success);

#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
DEFINE_CUDA_STATUS_TYPE(ncclResult_t, ncclSuccess);
#endif

}  // namespace details

#define PADDLE_ENFORCE_CUDA_SUCCESS(COND)                        \
  do {                                                           \
    auto __cond__ = (COND);                                      \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);             \
    constexpr auto __success_type__ =                            \
        ::paddle::platform::details::CudaStatusType<             \
            __CUDA_STATUS_TYPE__>::kSuccess;                     \
    if (UNLIKELY(__cond__ != __success_type__)) {                \
      auto __summary__ = ::paddle::platform::errors::External(   \
          ::paddle::platform::build_nvidia_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                     \
    }                                                            \
  } while (0)

inline void retry_sleep(unsigned millisecond) {
#ifdef _WIN32
  Sleep(millisecond);
#else
  sleep(millisecond);
#endif
}
#undef DEFINE_CUDA_STATUS_TYPE
#endif  // PADDLE_WITH_HIP

}  // namespace platform
}  // namespace paddle
