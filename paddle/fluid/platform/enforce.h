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
#include <thrust/system_error.h>
#include "paddle/fluid/platform/external_error.pb.h"
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hiprand.h>
#include <miopen/miopen.h>
#include <rocblas.h>
#include <thrust/system/hip/error.h>
#include <thrust/system_error.h>  // NOLINT
#endif

#ifdef PADDLE_WITH_ASCEND_CL
#include "acl/acl.h"
#include "paddle/fluid/platform/dynload/eccl.h"
#include "paddle/fluid/platform/dynload/eccl_types.h"
#endif  // PADDLE_WITH_ASCEND_CL

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
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include <error.h>  // NOLINT
#include "paddle/fluid/platform/dynload/rccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_HIP

// Note: these headers for simplify demangle type string
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/type_defs.h"
// Note: this header for simplify HIP and CUDA type string
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/type_defs.h"
#endif

namespace paddle {
namespace platform {
class ErrorSummary;
}  // namespace platform
}  // namespace paddle

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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
#elif defined(__HIPCC__)
#define PADDLE_ENFORCE(_IS_NOT_ERROR, __FORMAT, ...)                         \
  do {                                                                       \
    if (!(_IS_NOT_ERROR)) {                                                  \
      printf("Error: %s:%d Assertion `%s` failed. " __FORMAT "\n", __FILE__, \
             __LINE__, #_IS_NOT_ERROR, ##__VA_ARGS__);                       \
      abort();                                                               \
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

/**************************************************************************/
/**************************** NVIDIA ERROR ********************************/
#ifdef PADDLE_WITH_CUDA

namespace details {

template <typename T>
struct ExternalApiType {};

#define DEFINE_EXTERNAL_API_TYPE(type, success_value, proto_type) \
  template <>                                                     \
  struct ExternalApiType<type> {                                  \
    using Type = type;                                            \
    static constexpr Type kSuccess = success_value;               \
    static constexpr const char* kTypeString = #proto_type;       \
    static constexpr platform::proto::ApiType kProtoType =        \
        platform::proto::ApiType::proto_type;                     \
  }

DEFINE_EXTERNAL_API_TYPE(cudaError_t, cudaSuccess, CUDA);
DEFINE_EXTERNAL_API_TYPE(curandStatus_t, CURAND_STATUS_SUCCESS, CURAND);
DEFINE_EXTERNAL_API_TYPE(cudnnStatus_t, CUDNN_STATUS_SUCCESS, CUDNN);
DEFINE_EXTERNAL_API_TYPE(cublasStatus_t, CUBLAS_STATUS_SUCCESS, CUBLAS);
DEFINE_EXTERNAL_API_TYPE(cusolverStatus_t, CUSOLVER_STATUS_SUCCESS, CUSOLVER);

#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
DEFINE_EXTERNAL_API_TYPE(ncclResult_t, ncclSuccess, NCCL);
#endif

}  // namespace details

template <typename T>
inline const char* GetErrorMsgUrl(T status) {
  using __CUDA_STATUS_TYPE__ = decltype(status);
  platform::proto::ApiType proto_type =
      details::ExternalApiType<__CUDA_STATUS_TYPE__>::kProtoType;
  switch (proto_type) {
    case platform::proto::ApiType::CUDA:
      return "https://docs.nvidia.com/cuda/cuda-runtime-api/"
             "group__CUDART__TYPES.html#group__CUDART__TYPES_"
             "1g3f51e3575c2178246db0a94a430e0038";
      break;
    case platform::proto::ApiType::CURAND:
      return "https://docs.nvidia.com/cuda/curand/"
             "group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437";
      break;
    case platform::proto::ApiType::CUDNN:
      return "https://docs.nvidia.com/deeplearning/cudnn/api/"
             "index.html#cudnnStatus_t";
      break;
    case platform::proto::ApiType::CUBLAS:
      return "https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t";
      break;
    case platform::proto::ApiType::CUSOLVER:
      return "https://docs.nvidia.com/cuda/cusolver/"
             "index.html#cuSolverSPstatus";
      break;
    case platform::proto::ApiType::NCCL:
      return "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/"
             "types.html#ncclresult-t";
      break;
    default:
      return "Unknown type of External API, can't get error message URL!";
      break;
  }
}

template <typename T>
inline std::string GetExternalErrorMsg(T status) {
  std::ostringstream sout;
  bool _initSucceed = false;
  platform::proto::ExternalErrorDesc externalError;
  if (externalError.ByteSizeLong() == 0) {
    std::string filePath;
#if !defined(_WIN32)
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(GetCurrentTraceBackString), &info)) {
      std::string strModule(info.dli_fname);
      const size_t last_slash_idx = strModule.find_last_of("/");
      std::string compare_path = strModule.substr(strModule.length() - 6);
      if (std::string::npos != last_slash_idx) {
        strModule.erase(last_slash_idx, std::string::npos);
      }
      if (compare_path.compare("avx.so") == 0) {
        filePath =
            strModule +
            "/../include/third_party/externalError/data/externalErrorMsg.pb";
      } else {
        filePath = strModule +
                   "/../../third_party/externalError/data/externalErrorMsg.pb";
      }
    }
#else
    char buf[512];
    MEMORY_BASIC_INFORMATION mbi;
    HMODULE h_module =
        (::VirtualQuery(GetCurrentTraceBackString, &mbi, sizeof(mbi)) != 0)
            ? (HMODULE)mbi.AllocationBase
            : NULL;
    GetModuleFileName(h_module, buf, 512);
    std::string strModule(buf);
    const size_t last_slash_idx = strModule.find_last_of("\\");
    std::string compare_path = strModule.substr(strModule.length() - 7);
    if (std::string::npos != last_slash_idx) {
      strModule.erase(last_slash_idx, std::string::npos);
    }
    if (compare_path.compare("avx.pyd") == 0) {
      filePath = strModule +
                 "\\..\\include\\third_"
                 "party\\externalerror\\data\\externalErrorMsg.pb";
    } else {
      filePath =
          strModule +
          "\\..\\..\\third_party\\externalerror\\data\\externalErrorMsg.pb";
    }
#endif
    std::ifstream fin(filePath, std::ios::in | std::ios::binary);
    _initSucceed = externalError.ParseFromIstream(&fin);
  }
  using __CUDA_STATUS_TYPE__ = decltype(status);
  platform::proto::ApiType proto_type =
      details::ExternalApiType<__CUDA_STATUS_TYPE__>::kProtoType;
  if (_initSucceed) {
    for (int i = 0; i < externalError.errors_size(); ++i) {
      if (proto_type == externalError.errors(i).type()) {
        for (int j = 0; j < externalError.errors(i).messages_size(); ++j) {
          if (status == externalError.errors(i).messages(j).code()) {
            sout << "\n  [Hint: "
                 << externalError.errors(i).messages(j).message() << "]";
            return sout.str();
          }
        }
      }
    }
  }

  sout << "\n  [Hint: Please search for the error code(" << status
       << ") on website (" << GetErrorMsgUrl(status)
       << ") to get Nvidia's official solution and advice about "
       << details::ExternalApiType<__CUDA_STATUS_TYPE__>::kTypeString
       << " Error.]";
  return sout.str();
}

template std::string GetExternalErrorMsg<cudaError_t>(cudaError_t);
template std::string GetExternalErrorMsg<curandStatus_t>(curandStatus_t);
template std::string GetExternalErrorMsg<cudnnStatus_t>(cudnnStatus_t);
template std::string GetExternalErrorMsg<cublasStatus_t>(cublasStatus_t);
template std::string GetExternalErrorMsg<cusolverStatus_t>(cusolverStatus_t);
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
template std::string GetExternalErrorMsg<ncclResult_t>(ncclResult_t);
#endif

/*************** CUDA ERROR ***************/
inline bool is_error(cudaError_t e) { return e != cudaSuccess; }

inline std::string build_nvidia_error_msg(cudaError_t e) {
  std::ostringstream sout;
  sout << "CUDA error(" << e << "), " << cudaGetErrorString(e) << ". "
       << GetExternalErrorMsg(e);
  return sout.str();
}

/*************** CURAND ERROR ***************/
inline bool is_error(curandStatus_t stat) {
  return stat != CURAND_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(curandStatus_t stat) {
  std::ostringstream sout;
  sout << "CURAND error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUDNN ERROR ***************/
inline bool is_error(cudnnStatus_t stat) {
  return stat != CUDNN_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cudnnStatus_t stat) {
  std::ostringstream sout;
  sout << "CUDNN error(" << stat << "), "
       << platform::dynload::cudnnGetErrorString(stat) << ". "
       << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUBLAS ERROR ***************/
inline bool is_error(cublasStatus_t stat) {
  return stat != CUBLAS_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cublasStatus_t stat) {
  std::ostringstream sout;
  sout << "CUBLAS error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/*************** CUSOLVER ERROR ***************/
inline bool is_error(cusolverStatus_t stat) {
  return stat != CUSOLVER_STATUS_SUCCESS;
}

inline std::string build_nvidia_error_msg(cusolverStatus_t stat) {
  std::ostringstream sout;
  sout << "CUSOLVER error(" << stat << "). " << GetExternalErrorMsg(stat);
  return sout.str();
}

/**************** NCCL ERROR ****************/
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_nvidia_error_msg(ncclResult_t nccl_result) {
  std::ostringstream sout;
  sout << "NCCL error(" << nccl_result << "), "
       << platform::dynload::ncclGetErrorString(nccl_result) << ". ";
  if (errno == ENOSPC || errno == EAGAIN) {
    std::string detail(strerror(errno));
    detail += "\nPlease try one of the following solutions:";
    detail += "\n1. export NCCL_SHM_DISABLE=1;";
    detail += "\n2. export NCCL_P2P_LEVEL=SYS;";
    detail +=
        "\n3. Increase shared memory by setting the -shm-size "
        "option when starting docker container, e.g., setting "
        " -shm-size=2g.\n";
    sout << " Detail: " + detail;
  }
  sout << GetExternalErrorMsg(nccl_result);
  return sout.str();
}
#endif  // not(__APPLE__) and PADDLE_WITH_NCCL

#define PADDLE_ENFORCE_CUDA_SUCCESS(COND)                        \
  do {                                                           \
    auto __cond__ = (COND);                                      \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);             \
    constexpr auto __success_type__ =                            \
        ::paddle::platform::details::ExternalApiType<            \
            __CUDA_STATUS_TYPE__>::kSuccess;                     \
    if (UNLIKELY(__cond__ != __success_type__)) {                \
      auto __summary__ = ::paddle::platform::errors::External(   \
          ::paddle::platform::build_nvidia_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                     \
    }                                                            \
  } while (0)

#define PADDLE_ENFORCE_CUDA_LAUNCH_SUCCESS(OP)                                 \
  do {                                                                         \
    auto res = cudaGetLastError();                                             \
    if (UNLIKELY(res != cudaSuccess)) {                                        \
      auto msg = ::paddle::platform::build_nvidia_error_msg(res);              \
      PADDLE_THROW(platform::errors::Fatal("CUDA error after kernel (%s): %s", \
                                           OP, msg));                          \
    }                                                                          \
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
        ::paddle::platform::details::ExternalApiType<                   \
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

#undef DEFINE_EXTERNAL_API_TYPE
#endif  // PADDLE_WITH_CUDA

/**************************************************************************/
/***************************** HIP ERROR **********************************/
#ifdef PADDLE_WITH_HIP

/***** HIP ERROR *****/
inline bool is_error(hipError_t e) { return e != hipSuccess; }

inline std::string build_rocm_error_msg(hipError_t e) {
  std::ostringstream sout;
  sout << " Hip error(" << e << "), " << hipGetErrorString(e) << ".";
  return sout.str();
}

/***** HIPRAND ERROR *****/
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

inline std::string build_rocm_error_msg(hiprandStatus_t stat) {
  std::string msg(" Hiprand error, ");
  return msg + hiprandGetErrorString(stat) + " ";
}

/***** MIOPEN ERROR *****/
inline bool is_error(miopenStatus_t stat) {
  return stat != miopenStatusSuccess;
}

inline std::string build_rocm_error_msg(miopenStatus_t stat) {
  std::string msg(" Miopen error, ");
  return msg + platform::dynload::miopenGetErrorString(stat) + " ";
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
      return "Unknown cublas status";
  }
}

inline std::string build_rocm_error_msg(rocblas_status stat) {
  std::string msg(" Rocblas error, ");
  return msg + rocblasGetErrorString(stat) + " ";
}

/****** RCCL ERROR ******/
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
inline bool is_error(ncclResult_t nccl_result) {
  return nccl_result != ncclSuccess;
}

inline std::string build_rocm_error_msg(ncclResult_t nccl_result) {
  std::string msg(" Rccl error, ");
  return msg + platform::dynload::ncclGetErrorString(nccl_result) + " ";
}
#endif  // not(__APPLE__) and PADDLE_WITH_NCCL

namespace details {

template <typename T>
struct ExternalApiType {};

#define DEFINE_EXTERNAL_API_TYPE(type, success_value) \
  template <>                                         \
  struct ExternalApiType<type> {                      \
    using Type = type;                                \
    static constexpr Type kSuccess = success_value;   \
  }

DEFINE_EXTERNAL_API_TYPE(hipError_t, hipSuccess);
DEFINE_EXTERNAL_API_TYPE(hiprandStatus_t, HIPRAND_STATUS_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(miopenStatus_t, miopenStatusSuccess);
DEFINE_EXTERNAL_API_TYPE(rocblas_status, rocblas_status_success);

#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
DEFINE_EXTERNAL_API_TYPE(ncclResult_t, ncclSuccess);
#endif

}  // namespace details

#define PADDLE_ENFORCE_CUDA_SUCCESS(COND)                      \
  do {                                                         \
    auto __cond__ = (COND);                                    \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);           \
    constexpr auto __success_type__ =                          \
        ::paddle::platform::details::ExternalApiType<          \
            __CUDA_STATUS_TYPE__>::kSuccess;                   \
    if (UNLIKELY(__cond__ != __success_type__)) {              \
      auto __summary__ = ::paddle::platform::errors::External( \
          ::paddle::platform::build_rocm_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                   \
    }                                                          \
  } while (0)

inline void retry_sleep(unsigned millisecond) {
#ifdef _WIN32
  Sleep(millisecond);
#else
  sleep(millisecond);
#endif
}

#define PADDLE_RETRY_CUDA_SUCCESS(COND)                                 \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    int retry_count = 1;                                                \
    using __CUDA_STATUS_TYPE__ = decltype(__cond__);                    \
    constexpr auto __success_type__ =                                   \
        ::paddle::platform::details::ExternalApiType<                   \
            __CUDA_STATUS_TYPE__>::kSuccess;                            \
    while (UNLIKELY(__cond__ != __success_type__) && retry_count < 5) { \
      retry_sleep(FLAGS_gpu_allocator_retry_time);                      \
      __cond__ = (COND);                                                \
      ++retry_count;                                                    \
    }                                                                   \
    if (UNLIKELY(__cond__ != __success_type__)) {                       \
      auto __summary__ = ::paddle::platform::errors::External(          \
          ::paddle::platform::build_rocm_error_msg(__cond__));          \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#undef DEFINE_EXTERNAL_API_TYPE
#endif  // PADDLE_WITH_HIP

#ifdef PADDLE_WITH_ASCEND_CL
namespace details {
template <typename T>
struct NPUStatusType {};

#define DEFINE_NPU_STATUS_TYPE(type, success_value) \
  template <>                                       \
  struct NPUStatusType<type> {                      \
    using Type = type;                              \
    static constexpr Type kSuccess = success_value; \
  }

DEFINE_NPU_STATUS_TYPE(aclError, ACL_ERROR_NONE);
DEFINE_NPU_STATUS_TYPE(EcclResult, SUCCESS);
}  // namespace details

inline std::string build_npu_error_msg(aclError stat) {
  std::ostringstream sout;
  sout << " ACL error, the error code is : " << stat << ". ";
  return sout.str();
}

inline std::string build_npu_error_msg(EcclResult stat) {
  std::ostringstream sout;
  sout << " HCCL error, the error code is : " << stat << ". ";
  return sout.str();
}

#define PADDLE_ENFORCE_NPU_SUCCESS(COND)                       \
  do {                                                         \
    auto __cond__ = (COND);                                    \
    using __NPU_STATUS_TYPE__ = decltype(__cond__);            \
    constexpr auto __success_type__ =                          \
        ::paddle::platform::details::NPUStatusType<            \
            __NPU_STATUS_TYPE__>::kSuccess;                    \
    if (UNLIKELY(__cond__ != __success_type__)) {              \
      auto __summary__ = ::paddle::platform::errors::External( \
          ::paddle::platform::build_npu_error_msg(__cond__));  \
      __THROW_ERROR_INTERNAL__(__summary__);                   \
    }                                                          \
  } while (0)
#endif  // PADDLE_WITH_ASCEND_CL

}  // namespace platform
}  // namespace paddle
